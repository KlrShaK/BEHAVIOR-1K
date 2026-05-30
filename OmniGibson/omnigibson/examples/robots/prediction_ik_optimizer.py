"""
Optimization-based IK utilities for prediction actuation.

This module is intentionally independent from the example's execution loop:
it samples candidate arm joint states, evaluates the objective, restores the
robot state, and returns an arm joint target for the caller to command.
"""

from dataclasses import dataclass

import numpy as np
import torch as th

import omnigibson.utils.transform_utils_np as NT

try:
    from scipy.optimize import least_squares
except ImportError:
    least_squares = None


def scipy_available():
    return least_squares is not None


def tensor_to_numpy(array):
    if isinstance(array, th.Tensor):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def world_position_to_robot(robot, world_position):
    robot_position, robot_orientation = robot.get_position_orientation()
    robot_position = tensor_to_numpy(robot_position)
    robot_orientation = tensor_to_numpy(robot_orientation)
    world_position = np.asarray(tensor_to_numpy(world_position), dtype=np.float32)
    return NT.quat2mat(robot_orientation).T @ (world_position - robot_position)


def current_gripper_tcp_position(robot, arm):
    finger_links = robot.finger_links.get(arm, [])
    if not finger_links:
        return robot.get_relative_eef_position(arm=arm).detach().cpu().numpy()

    finger_positions = [
        world_position_to_robot(robot, finger_link.get_position_orientation()[0])
        for finger_link in finger_links
    ]
    return np.mean(finger_positions, axis=0)


@dataclass(frozen=True)
class OptimizedIKConfig:
    max_delta: float
    position_tolerance: float
    max_joint_delta: float = 0.20
    max_nfev: int = 80
    arm_link_behind_gripper_margin: float = 0.0
    trajectory_max_vertical_angle_deg: float | None = 30.0
    horizontal_gripper_max_vertical_deg: float = 30.0
    arm_link_scale: float = 0.01
    eef_forward_vertical_scale: float = 0.05
    eef_vertical_axis_scale: float = 0.10
    vertical_motion_scale: float = 0.01
    forward_xy_weight: float = 0.20
    joint_motion_weight: float = 0.05
    finite_difference_step: float = 1e-3


@dataclass(frozen=True)
class OptimizedIKSolveOptions:
    penalize_arm_links_ahead: bool = True


@dataclass(frozen=True)
class RobotJointState:
    positions: th.Tensor
    velocities: th.Tensor


@dataclass(frozen=True)
class ArmStateSample:
    tcp_position: np.ndarray
    eef_rotation: np.ndarray
    link_x_positions: np.ndarray


@dataclass(frozen=True)
class OptimizedIKResult:
    joint_positions: np.ndarray
    requested_distance: float
    target_position: np.ndarray
    tcp_position: np.ndarray
    tcp_error: float
    max_link_ahead_excess: float
    joint_delta_norm: float
    cost: float
    nfev: int
    optimality: float


class RobotKinematicsSampler:
    def __init__(self, robot, arm):
        self.robot = robot
        self.arm = arm
        self.arm_control_idx = robot.arm_control_idx[arm]

    def snapshot(self):
        return RobotJointState(
            positions=self.robot.get_joint_positions().clone(),
            velocities=self.robot.get_joint_velocities().clone(),
        )

    def restore(self, state):
        self.robot.set_joint_positions(state.positions, drive=False)
        self.robot.set_joint_velocities(state.velocities, drive=False)

    def arm_joint_positions_from(self, state):
        return tensor_to_numpy(state.positions[self.arm_control_idx]).astype(np.float64)

    def arm_joint_limits(self):
        position_limits = self.robot.control_limits["position"]
        lower = tensor_to_numpy(position_limits[0][self.arm_control_idx]).astype(np.float64)
        upper = tensor_to_numpy(position_limits[1][self.arm_control_idx]).astype(np.float64)
        return lower, upper

    def sample(self, q_arm):
        self.robot.set_joint_positions(
            th.tensor(q_arm, dtype=th.float32),
            indices=self.arm_control_idx,
            drive=False,
        )
        tcp_position = current_gripper_tcp_position(self.robot, self.arm)
        eef_orientation = self.robot.get_relative_eef_orientation(arm=self.arm).detach().cpu().numpy()
        eef_rotation = NT.quat2mat(np.asarray(eef_orientation, dtype=np.float32))
        link_x_positions = np.asarray(
            [
                world_position_to_robot(self.robot, link.get_position_orientation()[0])[0]
                for link in self.robot.arm_links.get(self.arm, [])
            ],
            dtype=np.float32,
        )
        return ArmStateSample(
            tcp_position=tcp_position,
            eef_rotation=eef_rotation,
            link_x_positions=link_x_positions,
        )


class OptimizedIKSolver:
    def __init__(self, config):
        if least_squares is None:
            raise ImportError("SciPy is required for optimized IK; rerun with --ik-solver omnigibson.")
        self.config = config

    def solve(self, robot, arm, target_robot, options=None):
        options = OptimizedIKSolveOptions() if options is None else options
        sampler = RobotKinematicsSampler(robot, arm)
        initial_state = sampler.snapshot()
        start_tcp = np.asarray(current_gripper_tcp_position(robot, arm), dtype=np.float32)
        requested_distance = float(np.linalg.norm(np.asarray(target_robot, dtype=np.float32) - start_tcp))
        target = self._clamped_target(start_tcp, target_robot)
        q_start = sampler.arm_joint_positions_from(initial_state)
        lower_bounds, upper_bounds = self._joint_bounds(sampler, q_start)
        desired_forward_xy = self._desired_forward_xy(start_tcp, target)

        def residual(q_arm):
            sample = sampler.sample(q_arm)
            return self._residual(
                sample=sample,
                target=target,
                start_tcp=start_tcp,
                q_arm=q_arm,
                q_start=q_start,
                desired_forward_xy=desired_forward_xy,
                options=options,
            )

        def jacobian(q_arm):
            return self._finite_difference_jacobian(
                residual=residual,
                q_arm=q_arm,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
            )

        try:
            result = least_squares(
                residual,
                np.clip(q_start, lower_bounds, upper_bounds),
                jac=jacobian,
                bounds=(lower_bounds, upper_bounds),
                max_nfev=self.config.max_nfev,
                xtol=1e-4,
                ftol=1e-4,
                gtol=1e-4,
            )
            accepted_q = result.x.astype(np.float32)
            accepted_sample = sampler.sample(accepted_q)
            return self._build_result(
                optimizer_result=result,
                accepted_q=accepted_q,
                accepted_sample=accepted_sample,
                target=target,
                requested_distance=requested_distance,
                q_start=q_start,
            )
        finally:
            sampler.restore(initial_state)

    def _clamped_target(self, start_tcp, target_robot):
        target = np.asarray(target_robot, dtype=np.float32)
        delta = target - start_tcp
        distance = np.linalg.norm(delta)
        if distance <= self.config.max_delta:
            return target
        return start_tcp + delta / distance * self.config.max_delta

    def _joint_bounds(self, sampler, q_start):
        q_lower, q_upper = sampler.arm_joint_limits()
        lower = np.maximum(q_lower, q_start - self.config.max_joint_delta)
        upper = np.minimum(q_upper, q_start + self.config.max_joint_delta)
        collapsed_bounds = lower >= upper
        if np.any(collapsed_bounds):
            lower[collapsed_bounds] = q_start[collapsed_bounds] - 1e-6
            upper[collapsed_bounds] = q_start[collapsed_bounds] + 1e-6
        return lower, upper

    def _desired_forward_xy(self, start_tcp, target):
        direction = target[:2] - start_tcp[:2]
        norm = np.linalg.norm(direction)
        return None if norm <= 1e-6 else direction / norm

    def _residual(self, sample, target, start_tcp, q_arm, q_start, desired_forward_xy, options):
        values = []
        values.extend((sample.tcp_position - target) / self.config.position_tolerance)
        if options.penalize_arm_links_ahead:
            values.extend(self._arm_link_residual(sample))
        values.extend(self._horizontal_gripper_residual(sample, desired_forward_xy))
        values.extend(self._vertical_motion_residual(sample, start_tcp))
        values.extend(
            self.config.joint_motion_weight
            * (q_arm - q_start)
            / max(self.config.max_joint_delta, 1e-6)
        )
        return np.asarray(values, dtype=np.float64)

    def _finite_difference_jacobian(self, residual, q_arm, lower_bounds, upper_bounds):
        q_arm = np.asarray(q_arm, dtype=np.float64)
        base_residual = residual(q_arm)
        jacobian = np.empty((base_residual.size, q_arm.size), dtype=np.float64)
        step_size = self.config.finite_difference_step

        for joint_idx in range(q_arm.size):
            q_minus = q_arm.copy()
            q_plus = q_arm.copy()
            q_minus[joint_idx] = max(lower_bounds[joint_idx], q_arm[joint_idx] - step_size)
            q_plus[joint_idx] = min(upper_bounds[joint_idx], q_arm[joint_idx] + step_size)

            actual_step = q_plus[joint_idx] - q_minus[joint_idx]
            if actual_step <= 1e-12:
                jacobian[:, joint_idx] = 0.0
                continue

            jacobian[:, joint_idx] = (residual(q_plus) - residual(q_minus)) / actual_step

        return jacobian

    def _arm_link_residual(self, sample):
        if sample.link_x_positions.size == 0:
            return []
        excess = np.maximum(
            0.0,
            sample.link_x_positions - sample.tcp_position[0] - self.config.arm_link_behind_gripper_margin,
        )
        return excess / self.config.arm_link_scale

    def _horizontal_gripper_residual(self, sample, desired_forward_xy):
        eef_forward = sample.eef_rotation[:, 2]
        eef_vertical = sample.eef_rotation[:, 1]
        robot_z = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        max_forward_vertical = np.sin(np.deg2rad(self.config.horizontal_gripper_max_vertical_deg))

        values = [
            max(0.0, abs(float(eef_forward[2])) - max_forward_vertical)
            / self.config.eef_forward_vertical_scale
        ]
        values.extend((eef_vertical - robot_z) / self.config.eef_vertical_axis_scale)

        if desired_forward_xy is not None:
            forward_xy = eef_forward[:2]
            forward_xy_norm = np.linalg.norm(forward_xy)
            if forward_xy_norm > 1e-6:
                values.extend(self.config.forward_xy_weight * (forward_xy / forward_xy_norm - desired_forward_xy))
        return values

    def _vertical_motion_residual(self, sample, start_tcp):
        max_angle = self.config.trajectory_max_vertical_angle_deg
        if max_angle is None or max_angle >= 90.0:
            return []

        delta = sample.tcp_position - start_tcp
        xy_distance = np.linalg.norm(delta[:2])
        max_dz = np.tan(np.deg2rad(max_angle)) * xy_distance
        return [max(0.0, abs(float(delta[2])) - max_dz) / self.config.vertical_motion_scale]

    def _build_result(self, optimizer_result, accepted_q, accepted_sample, target, requested_distance, q_start):
        link_excess = np.maximum(
            0.0,
            accepted_sample.link_x_positions
            - accepted_sample.tcp_position[0]
            - self.config.arm_link_behind_gripper_margin,
        )
        max_link_excess = float(np.max(link_excess)) if link_excess.size > 0 else 0.0
        return OptimizedIKResult(
            joint_positions=accepted_q,
            requested_distance=requested_distance,
            target_position=target,
            tcp_position=accepted_sample.tcp_position,
            tcp_error=float(np.linalg.norm(accepted_sample.tcp_position - target)),
            max_link_ahead_excess=max_link_excess,
            joint_delta_norm=float(np.linalg.norm(accepted_q - q_start)),
            cost=float(optimizer_result.cost),
            nfev=int(optimizer_result.nfev),
            optimality=float(optimizer_result.optimality),
        )
