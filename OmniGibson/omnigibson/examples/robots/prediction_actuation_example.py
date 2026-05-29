"""
Replay prediction trajectories from a captured RGB-D bundle.

Loads a capture directory containing metadata.json and prediction/*.npz files,
restores the robot pose / joints from metadata, and lets the user press E then
1, 2, ... to execute the corresponding prediction with IK.
"""

import argparse
import json
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch as th

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.controllers import ControllerView
from omnigibson.macros import gm
import omnigibson.utils.transform_utils_np as NT
from omnigibson.utils.usd_utils import RigidContactAPI


gm.USE_GPU_DYNAMICS = False


@dataclass
class Prediction:
    index: int
    path: Path
    label: str
    best_candidate: int
    best_loss: float
    trajectory_camera: np.ndarray


class PredictionKeyHandler:
    """
    Keyboard sequence handler for E + digit prediction execution.
    """

    def __init__(self, num_predictions):
        self.pending_index = None
        self.reset_requested = False
        self.quit_requested = False
        self._armed_until = 0.0
        self._num_predictions = num_predictions
        self._digit_to_index = {
            lazy.carb.input.KeyboardInput.KEY_1: 1,
            lazy.carb.input.KeyboardInput.KEY_2: 2,
            lazy.carb.input.KeyboardInput.KEY_3: 3,
            lazy.carb.input.KeyboardInput.KEY_4: 4,
            lazy.carb.input.KeyboardInput.KEY_5: 5,
            lazy.carb.input.KeyboardInput.KEY_6: 6,
            lazy.carb.input.KeyboardInput.KEY_7: 7,
            lazy.carb.input.KeyboardInput.KEY_8: 8,
            lazy.carb.input.KeyboardInput.KEY_9: 9,
        }

        appwindow = lazy.omni.appwindow.get_default_app_window()
        input_interface = lazy.carb.input.acquire_input_interface()
        keyboard = appwindow.get_keyboard()
        self._input_interface = input_interface
        self._keyboard = keyboard
        self._subscription = input_interface.subscribe_to_keyboard_events(keyboard, self._on_keyboard_event)

    def close(self):
        self._input_interface.unsubscribe_to_keyboard_events(self._keyboard, self._subscription)

    def _on_keyboard_event(self, event, *args, **kwargs):
        if event.type != lazy.carb.input.KeyboardEventType.KEY_PRESS:
            return True

        if event.input == lazy.carb.input.KeyboardInput.E:
            self._armed_until = time.time() + 2.0
            print("Prediction execution armed. Press 1-9 within 2 seconds.")
            return True

        if event.input == lazy.carb.input.KeyboardInput.R:
            self.pending_index = None
            self.reset_requested = True
            return True

        if event.input == lazy.carb.input.KeyboardInput.ESCAPE:
            self.quit_requested = True
            return True

        prediction_index = self._digit_to_index.get(event.input)
        if prediction_index is None:
            return True

        if time.time() > self._armed_until:
            return True

        if prediction_index > self._num_predictions:
            print(f"No prediction {prediction_index}; only {self._num_predictions} available.")
            self._armed_until = 0.0
            return True

        self.pending_index = prediction_index
        self._armed_until = 0.0
        return True


def decode_ascii_array(array):
    return "".join(chr(int(x)) for x in array)


def load_metadata(capture_dir):
    metadata_path = capture_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")
    with metadata_path.open("r") as f:
        return json.load(f)


def load_predictions(capture_dir):
    prediction_paths = sorted((capture_dir / "prediction").glob("*.npz"))
    if not prediction_paths:
        raise FileNotFoundError(f"No prediction .npz files found under {capture_dir / 'prediction'}")

    predictions = []
    for i, path in enumerate(prediction_paths, start=1):
        data = np.load(path, allow_pickle=True)
        total_losses = data["guide_losses-total_loss"][0]
        best_candidate = int(np.argmin(total_losses))
        label = decode_ascii_array(data["label_text_singlestr"][0])
        predictions.append(
            Prediction(
                index=i,
                path=path,
                label=label,
                best_candidate=best_candidate,
                best_loss=float(total_losses[best_candidate]),
                trajectory_camera=np.asarray(data["pred_trajectories"][0, best_candidate], dtype=np.float32),
            )
        )
    return predictions


def camera_frame_transform(frame):
    if frame == "usd":
        return np.eye(4, dtype=np.float32)
    if frame == "opencv":
        # Model / intrinsics convention: x right, y down, z forward.
        # USD camera prim convention: x right, y up, -z forward.
        return np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
    raise ValueError(f"Unsupported camera frame convention: {frame}")


def transform_points(transform, points):
    ones = np.ones((len(points), 1), dtype=np.float32)
    points_h = np.concatenate([points, ones], axis=1)
    return (transform @ points_h.T).T[:, :3]


def prediction_to_robot_trajectory(prediction, metadata, camera_frame):
    t_world_camera = np.asarray(metadata["camera"]["T_world_camera"], dtype=np.float32)
    t_world_robot = np.asarray(metadata["robot"]["T_world_robot"], dtype=np.float32)
    t_robot_world = np.linalg.inv(t_world_robot)
    t_camera_usd_from_prediction = camera_frame_transform(camera_frame)

    trajectory_world = transform_points(
        t_world_camera @ t_camera_usd_from_prediction,
        prediction.trajectory_camera,
    )
    return transform_points(t_robot_world, trajectory_world)


def quat_to_axisangle(quat):
    return np.asarray(NT.quat2axisangle(np.asarray(quat, dtype=np.float32)), dtype=np.float32)


def horizontal_target_axisangle(current_robot, target_robot):
    direction = np.asarray(target_robot, dtype=np.float32) - np.asarray(current_robot, dtype=np.float32)
    direction[2] = 0.0
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        return None

    forward = direction / norm
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    side = np.cross(up, forward)
    side_norm = np.linalg.norm(side)
    if side_norm < 1e-6:
        side = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        side = side / side_norm
    vertical = np.cross(forward, side)
    rotation = np.column_stack((side, vertical, forward))
    return quat_to_axisangle(NT.mat2quat(rotation))


def build_environment(metadata, scene_model, empty_scene):
    robot_metadata = metadata["robot"]
    scene_cfg = (
        {"type": "Scene"}
        if empty_scene
        else {"type": "InteractiveTraversableScene", "scene_model": scene_model}
    )
    robot_cfg = {
        "model": robot_metadata.get("model", "fetch"),
        "position": robot_metadata["pose"]["position"],
        "orientation": robot_metadata["pose"]["orientation_xyzw"],
        "obs_modalities": [],
        "action_type": "continuous",
        "action_normalize": False,
        "controller_config": {
            "arm_0": {
                "name": "InverseKinematicsController",
                "mode": "pose_absolute_ori",
                "command_input_limits": None,
            },
            "gripper_0": {
                "name": "MultiFingerGripperController",
            },
        },
    }
    env = og.Environment(configs={"scene": scene_cfg, "robots": [robot_cfg]})
    env.reset()
    return env, env.robots[0]


def restore_robot_state(robot, metadata):
    robot_metadata = metadata["robot"]
    robot.set_position_orientation(
        position=th.tensor(robot_metadata["pose"]["position"], dtype=th.float32),
        orientation=th.tensor(robot_metadata["pose"]["orientation_xyzw"], dtype=th.float32),
    )

    joint_positions = robot.get_joint_positions()
    saved_joint_positions = robot_metadata.get("joint_positions", {})
    for i, joint_name in enumerate(robot.joints.keys()):
        if joint_name in saved_joint_positions:
            joint_positions[i] = float(saved_joint_positions[joint_name])
    robot.set_joint_positions(joint_positions, drive=False)

    for _ in range(5):
        og.sim.step()


def compute_no_op_action(robot):
    action = th.zeros(robot.action_dim, dtype=th.float32)
    controller_action_idx = robot.controller_action_idx
    for controller_name, (group_key, controller_idx) in robot.controllers.items():
        action_idx = controller_action_idx[controller_name]
        try:
            no_op = ControllerView.compute_no_op_action(
                group_key,
                controller_idx,
            ).flatten()
        except ValueError:
            continue

        # Some controllers expose a mode-dependent command_dim but return their
        # full pose command from compute_no_op_action(). Keep the values that fit
        # the actual action slice, which is what robot.apply_action() consumes.
        if no_op.numel() > action_idx.numel():
            no_op = no_op[: action_idx.numel()]
        elif no_op.numel() < action_idx.numel():
            no_op = th.cat([no_op, th.zeros(action_idx.numel() - no_op.numel(), dtype=no_op.dtype)])

        action[action_idx] = no_op.to(dtype=action.dtype)
    return action


def set_gripper_action(robot, action, arm, command):
    if command is None:
        return
    if f"gripper_{arm}" not in robot.controllers:
        return
    action[robot.gripper_action_idx[arm]] = float(command)


def get_end_effector_contacts(robot, arm):
    end_effector_paths = {robot.eef_links[arm].prim_path}
    end_effector_paths.update(link.prim_path for link in robot.finger_links.get(arm, []))
    robot_link_paths = set(robot.link_prim_paths)
    contact_pairs = RigidContactAPI.get_contact_pairs(
        scene_idx=robot.scene.idx,
        query_set=end_effector_paths,
        with_set=None,
        current_only=True,
    )
    return sorted(
        {
            other_contact
            for _, other_contact in contact_pairs
            if other_contact not in robot_link_paths
        }
    )


def format_contact_paths(contact_paths):
    preview = contact_paths[:3]
    suffix = "" if len(contact_paths) <= len(preview) else f" and {len(contact_paths) - len(preview)} more"
    return ", ".join(preview) + suffix


def current_eef_position(robot, arm):
    return robot.get_relative_eef_position(arm=arm).detach().cpu().numpy()


def world_position_to_robot(robot, world_position):
    robot_position, robot_orientation = robot.get_position_orientation()
    robot_position = robot_position.detach().cpu().numpy()
    robot_orientation = robot_orientation.detach().cpu().numpy()
    world_position = np.asarray(world_position.detach().cpu().numpy(), dtype=np.float32)
    return NT.quat2mat(robot_orientation).T @ (world_position - robot_position)


def current_gripper_tcp_position(robot, arm):
    finger_links = robot.finger_links.get(arm, [])
    if not finger_links:
        return current_eef_position(robot, arm)

    finger_positions = [
        world_position_to_robot(robot, finger_link.get_position_orientation()[0])
        for finger_link in finger_links
    ]
    return np.mean(finger_positions, axis=0)


def current_control_point_position(robot, arm, target_is_tcp):
    return current_gripper_tcp_position(robot, arm) if target_is_tcp else current_eef_position(robot, arm)


def arm_links_ahead_of_gripper(robot, arm, margin):
    gripper_x = current_gripper_tcp_position(robot, arm)[0]
    violations = []
    for link in robot.arm_links.get(arm, []):
        link_x = world_position_to_robot(robot, link.get_position_orientation()[0])[0]
        excess = link_x - gripper_x - margin
        if excess > 0.0:
            violations.append((link.prim_path, float(link_x), float(excess)))
    return sorted(violations, key=lambda item: item[2], reverse=True)


def format_link_violations(violations):
    preview = [f"{path} x={link_x:.3f} excess={excess:.3f}" for path, link_x, excess in violations[:3]]
    suffix = "" if len(violations) <= len(preview) else f" and {len(violations) - len(preview)} more"
    return "; ".join(preview) + suffix


def closest_trajectory_index(robot, arm, trajectory_robot, recovery_x_clearance=0.0):
    current = current_gripper_tcp_position(robot, arm)
    candidate_indices = np.arange(len(trajectory_robot))
    has_safe_x_candidate = False
    if recovery_x_clearance > 0.0:
        target_x = current[0] - recovery_x_clearance
        behind_mask = trajectory_robot[:, 0] <= target_x
        if np.any(behind_mask):
            candidate_indices = candidate_indices[behind_mask]
            has_safe_x_candidate = True
        else:
            behind_mask = trajectory_robot[:, 0] <= current[0]
            if np.any(behind_mask):
                candidate_indices = candidate_indices[behind_mask]
                has_safe_x_candidate = True
    else:
        behind_mask = trajectory_robot[:, 0] <= current[0]
        if np.any(behind_mask):
            candidate_indices = candidate_indices[behind_mask]
            has_safe_x_candidate = True

    candidate_points = trajectory_robot[candidate_indices]
    distances = np.linalg.norm(candidate_points - current, axis=1)
    closest_candidate = int(np.argmin(distances))
    closest_index = int(candidate_indices[closest_candidate])
    return closest_index, float(distances[closest_candidate]), current, has_safe_x_candidate


def constrain_waypoint_vertical_angle(robot, arm, waypoint, max_vertical_angle_deg):
    if max_vertical_angle_deg is None or max_vertical_angle_deg >= 90.0:
        return waypoint, False

    current = current_gripper_tcp_position(robot, arm)
    constrained = np.array(waypoint, dtype=np.float32, copy=True)
    xy_distance = np.linalg.norm(constrained[:2] - current[:2])
    max_dz = np.tan(np.deg2rad(max_vertical_angle_deg)) * xy_distance
    dz = constrained[2] - current[2]
    if abs(dz) <= max_dz:
        return constrained, False

    constrained[2] = current[2] + np.sign(dz) * max_dz
    return constrained, True


def step_toward(
    robot,
    arm,
    target_robot,
    max_delta,
    gripper_command=None,
    target_orientation_robot=None,
    check_contacts=False,
    keep_arm_links_behind_gripper=True,
    arm_link_behind_gripper_margin=0.0,
    target_is_tcp=True,
):
    target = th.tensor(target_robot, dtype=th.float32)
    current_control_point = th.tensor(
        current_control_point_position(robot, arm, target_is_tcp),
        dtype=th.float32,
    )
    delta = target - current_control_point
    distance = th.norm(delta).item()
    if distance > max_delta:
        delta = delta / distance * max_delta

    action = compute_no_op_action(robot)
    previous_joint_positions = robot.get_joint_positions().clone()
    arm_action_idx = robot.arm_action_idx[arm]
    arm_command_dim = action[arm_action_idx].numel()
    if arm_command_dim == 6:
        if target_orientation_robot is None:
            target_orientation_robot = horizontal_target_axisangle(
                current_control_point.detach().cpu().numpy(),
                target_robot,
            )
        orientation = (
            robot.get_relative_eef_orientation(arm=arm)
            if target_orientation_robot is None
            else th.tensor(target_orientation_robot, dtype=th.float32)
        )
        if orientation.numel() == 4:
            orientation = th.tensor(quat_to_axisangle(orientation.detach().cpu().numpy()), dtype=th.float32)
        action[arm_action_idx] = th.cat([delta, orientation])
    elif arm_command_dim == 3:
        action[arm_action_idx] = delta
    else:
        raise ValueError(f"Unsupported arm action dimension for {arm}: {arm_command_dim}")
    set_gripper_action(robot, action, arm, gripper_command)
    robot.apply_action(action)
    og.sim.step()
    if keep_arm_links_behind_gripper:
        link_violations = arm_links_ahead_of_gripper(robot, arm, arm_link_behind_gripper_margin)
        if link_violations:
            robot.set_joint_positions(previous_joint_positions, drive=False)
            og.sim.step()
            return distance, [], link_violations
    return distance, get_end_effector_contacts(robot, arm) if check_contacts else [], []


def drive_to_waypoint(
    robot,
    arm,
    waypoint,
    args,
    gripper_command=None,
    target_orientation_robot=None,
    stop_on_contact=False,
    target_is_tcp=True,
    key_handler=None,
):
    for _ in range(args.max_steps_per_waypoint):
        if key_handler is not None and key_handler.reset_requested:
            return "reset", []
        distance, contact_paths, link_violations = step_toward(
            robot=robot,
            arm=arm,
            target_robot=waypoint,
            max_delta=args.max_delta,
            gripper_command=gripper_command,
            target_orientation_robot=target_orientation_robot,
            check_contacts=stop_on_contact,
            keep_arm_links_behind_gripper=args.keep_arm_links_behind_gripper,
            arm_link_behind_gripper_margin=args.arm_link_behind_gripper_margin,
            target_is_tcp=target_is_tcp,
        )
        if link_violations:
            return "link_ahead", link_violations
        if stop_on_contact and contact_paths:
            return "contact", contact_paths
        if key_handler is not None and key_handler.reset_requested:
            return "reset", []
        if distance <= args.position_tolerance:
            return "reached", []
    return "timeout", []


def execute_prediction(robot, arm, prediction, trajectory_robot, args, key_handler=None):
    print(
        f"Started e{prediction.index}: {prediction.path.name} "
        f"(candidate {prediction.best_candidate}, loss {prediction.best_loss:.4f})"
    )
    print(f"Stage: prediction trajectory has {len(trajectory_robot)} gripper TCP waypoints.")
    if args.open_gripper_command is None:
        print("Stage: open gripper skipped; no --open-gripper-command provided.")
    else:
        print("Stage: open gripper command active while moving to first waypoint.")

    first_waypoint = trajectory_robot[0]
    print("Stage: moving gripper TCP to first waypoint.")
    status, contact_paths = drive_to_waypoint(
        robot,
        arm,
        first_waypoint,
        args,
        gripper_command=args.open_gripper_command,
        stop_on_contact=True,
        key_handler=key_handler,
    )
    if status == "reset":
        print("Reset requested; interrupting prediction before first waypoint completion.")
        return "reset"
    next_waypoint_index = 1
    recovery_waypoint_index = None
    if status == "reached":
        print("Stage: moved gripper TCP to first waypoint.")
    else:
        if status == "contact":
            print(
                f"Warning: stopped moving to first waypoint after gripper / EEF contact with "
                f"{format_contact_paths(contact_paths)}."
            )
            print("Stage: contact checks disabled after approach; contact is expected during drawer actuation.")
        elif status == "link_ahead":
            print(
                "Warning: stopped moving to first waypoint because an arm link moved ahead of the gripper TCP: "
                f"{format_link_violations(contact_paths)}."
            )
        else:
            print("Warning: did not reach the first prediction waypoint within the step budget.")
        closest_index, closest_distance, current_position, has_safe_x_candidate = closest_trajectory_index(
            robot,
            arm,
            trajectory_robot,
            recovery_x_clearance=args.approach_recovery_x_clearance,
        )
        print(
            f"Stage: first waypoint not reached; closest available gripper TCP waypoint is "
            f"{closest_index} ({closest_distance:.3f} m away, current x={current_position[0]:.3f}, "
            f"waypoint x={trajectory_robot[closest_index, 0]:.3f}, "
            f"required x<={current_position[0] - args.approach_recovery_x_clearance:.3f})."
        )
        if not has_safe_x_candidate:
            print(
                "Warning: no trajectory waypoint is behind the current gripper TCP x; "
                "skipping recovery move to avoid pushing into the cabinet."
            )
            next_waypoint_index = len(trajectory_robot)
        elif closest_index > 0 and closest_distance > args.position_tolerance:
            recovery_waypoint_index = closest_index
            print(
                f"Stage: closest available waypoint {closest_index} will be executed after close gripper."
            )
        else:
            print("Stage: continuing from the current gripper TCP pose.")
        if has_safe_x_candidate:
            next_waypoint_index = min(closest_index + 1, len(trajectory_robot))

    if args.close_gripper_command is not None:
        print("Stage: close gripper.")
        for _ in range(args.gripper_settle_steps):
            if key_handler is not None and key_handler.reset_requested:
                print("Reset requested; interrupting prediction during close gripper.")
                return "reset"
            action = compute_no_op_action(robot)
            set_gripper_action(robot, action, arm, args.close_gripper_command)
            robot.apply_action(action)
            og.sim.step()
            if key_handler is not None and key_handler.reset_requested:
                print("Reset requested; interrupting prediction during close gripper.")
                return "reset"
    else:
        print("Stage: close gripper skipped; no --close-gripper-command provided.")

    if recovery_waypoint_index is not None:
        print(f"Stage: moving gripper TCP to closest available waypoint {recovery_waypoint_index}.")
        recovery_waypoint, constrained = constrain_waypoint_vertical_angle(
            robot,
            arm,
            trajectory_robot[recovery_waypoint_index],
            args.trajectory_max_vertical_angle_deg,
        )
        if constrained:
            print(
                f"Stage: constrained recovery gripper TCP vertical motion to "
                f"{args.trajectory_max_vertical_angle_deg:.1f} deg from the XY plane."
            )
        status, contact_paths = drive_to_waypoint(
            robot,
            arm,
            recovery_waypoint,
            args,
            gripper_command=args.close_gripper_command,
            stop_on_contact=False,
            key_handler=key_handler,
        )
        if status == "reset":
            print("Reset requested; interrupting prediction during recovery waypoint.")
            return "reset"
        if status == "reached":
            print(f"Stage: moved gripper TCP to closest available waypoint {recovery_waypoint_index}.")
        elif status == "link_ahead":
            print(
                "Stopped prediction: recovery would place an arm link ahead of the gripper TCP: "
                f"{format_link_violations(contact_paths)}."
            )
            return
        else:
            print(
                f"Warning: closest available waypoint {recovery_waypoint_index} was not reached; "
                "continuing from the current gripper TCP pose."
            )

    if next_waypoint_index >= len(trajectory_robot):
        print("Stage: no remaining waypoints after closest available point.")
    else:
        print(
            f"Stage: trying to follow the rest of predicted trajectory from waypoint "
            f"{next_waypoint_index} with stride {args.waypoint_stride}."
        )

    for waypoint in trajectory_robot[next_waypoint_index :: args.waypoint_stride]:
        current_position = current_gripper_tcp_position(robot, arm)
        if waypoint[0] > current_position[0] + args.max_forward_x_after_approach:
            print(
                f"Stage: skipping waypoint at x={waypoint[0]:.3f}; current TCP x={current_position[0]:.3f} "
                "and moving further into the cabinet is disabled after approach."
            )
            continue
        waypoint, constrained = constrain_waypoint_vertical_angle(
            robot,
            arm,
            waypoint,
            args.trajectory_max_vertical_angle_deg,
        )
        if constrained:
            print(
                f"Stage: constrained waypoint vertical motion to "
                f"{args.trajectory_max_vertical_angle_deg:.1f} deg from the XY plane."
            )
        status, contact_paths = drive_to_waypoint(
            robot,
            arm,
            waypoint,
            args,
            gripper_command=args.close_gripper_command,
            stop_on_contact=False,
            key_handler=key_handler,
        )
        if status == "reset":
            print("Reset requested; interrupting prediction while following trajectory.")
            return "reset"
        if status == "link_ahead":
            print(
                "Stopped prediction: trajectory step would place an arm link ahead of the gripper TCP: "
                f"{format_link_violations(contact_paths)}."
            )
            return

    print("Stage: end of predicted trajectory.")
    print(f"Finished e{prediction.index}.")
    return "finished"


def reset_arm_to_home(robot, arm, home_position_robot, home_orientation_robot, args):
    print("Resetting arm to captured home position.")
    status, _ = drive_to_waypoint(
        robot=robot,
        arm=arm,
        waypoint=home_position_robot,
        args=args,
        gripper_command=args.open_gripper_command,
        target_orientation_robot=home_orientation_robot,
        target_is_tcp=False,
    )
    if status == "reached":
        print("Arm reset finished.")
    else:
        print("Warning: arm reset did not reach home position within the step budget.")


def run_arm_reset(robot, arm, home_position_robot, home_orientation_robot, args):
    reset_arm_to_home(
        robot=robot,
        arm=arm,
        home_position_robot=home_position_robot,
        home_orientation_robot=home_orientation_robot,
        args=args,
    )
    print("Ready for another prediction after arm reset.")


def print_prediction_menu(predictions):
    print("")
    print("Available predictions:")
    for prediction in predictions:
        print(
            f"  e{prediction.index}: {prediction.path.name} "
            f"label={prediction.label} best={prediction.best_candidate} loss={prediction.best_loss:.4f}"
        )
    print("")
    print("Press E then a number to execute a prediction. Press R to reset the arm. Press ESC to quit.")


def main():
    parser = argparse.ArgumentParser(description="Actuate robot arm from saved RGB-D prediction trajectories.")
    parser.add_argument(
        "--capture-dir",
        type=Path,
        default=Path("omnigibson_6"),
        help="Capture directory containing metadata.json and prediction/*.npz.",
    )
    parser.add_argument("--scene-model", default="Rs_int", help="Interactive scene model to load.")
    parser.add_argument(
        "--empty-scene",
        action="store_true",
        help="Use an empty scene instead of an interactive scene.",
    )
    parser.add_argument(
        "--camera-frame",
        choices=("opencv", "usd"),
        default="opencv",
        help="Frame convention used by prediction 3D points.",
    )
    parser.add_argument("--max-delta", type=float, default=0.025, help="Max IK delta command per sim step in meters.")
    parser.add_argument(
        "--position-tolerance",
        type=float,
        default=0.015,
        help="Waypoint tolerance in robot-frame meters.",
    )
    parser.add_argument("--max-steps-per-waypoint", type=int, default=80)
    parser.add_argument(
        "--approach-recovery-x-clearance",
        type=float,
        default=0.05,
        help="Minimum robot-frame x retreat, in meters, when resuming after failed first-waypoint approach.",
    )
    parser.add_argument(
        "--max-forward-x-after-approach",
        type=float,
        default=0.0,
        help="Allowed robot-frame x increase after approach recovery; 0 prevents pushing farther into the cabinet.",
    )
    parser.add_argument(
        "--trajectory-max-vertical-angle-deg",
        type=float,
        default=30.0,
        help="Max vertical angle from the robot XY plane while executing predicted trajectory waypoints.",
    )
    parser.add_argument(
        "--disable-arm-link-behind-gripper-guard",
        dest="keep_arm_links_behind_gripper",
        action="store_false",
        help="Disable rejecting IK steps where arm links move ahead of the gripper TCP in robot-frame x.",
    )
    parser.add_argument(
        "--arm-link-behind-gripper-margin",
        type=float,
        default=0.0,
        help="Required robot-frame x margin, in meters, for arm links to remain behind the gripper TCP.",
    )
    parser.add_argument(
        "--waypoint-stride",
        type=int,
        default=4,
        help="Execute every Nth predicted trajectory waypoint.",
    )
    parser.add_argument("--gripper-settle-steps", type=int, default=20)
    parser.add_argument("--open-gripper-command", type=float, default=None, help="Optional gripper command, e.g. 1.0.")
    parser.add_argument(
        "--close-gripper-command",
        type=float,
        default=None,
        help="Optional gripper command, e.g. -1.0.",
    )
    args = parser.parse_args()

    metadata = load_metadata(args.capture_dir)
    predictions = load_predictions(args.capture_dir)
    _env, robot = build_environment(
        metadata=metadata,
        scene_model=args.scene_model,
        empty_scene=args.empty_scene,
    )
    restore_robot_state(robot, metadata)

    camera_pose = metadata.get("camera", {}).get("pose", {})
    if "position" in camera_pose and "orientation_xyzw" in camera_pose:
        og.sim.viewer_camera.set_position_orientation(
            position=th.tensor(camera_pose["position"], dtype=th.float32),
            orientation=th.tensor(camera_pose["orientation_xyzw"], dtype=th.float32),
        )

    arm = metadata["robot"].get("default_arm", robot.default_arm)
    home_eef_pose = metadata["robot"]["eef_poses"][arm]["robot_relative"]
    home_position_robot = np.asarray(home_eef_pose["position"], dtype=np.float32)
    home_orientation_robot = quat_to_axisangle(home_eef_pose["orientation_xyzw"])
    trajectories_robot = {
        prediction.index: prediction_to_robot_trajectory(prediction, metadata, args.camera_frame)
        for prediction in predictions
    }

    print_prediction_menu(predictions)
    key_handler = PredictionKeyHandler(num_predictions=len(predictions))

    try:
        while not key_handler.quit_requested:
            if key_handler.reset_requested:
                key_handler.reset_requested = False
                key_handler.pending_index = None
                try:
                    run_arm_reset(robot, arm, home_position_robot, home_orientation_robot, args)
                except Exception:
                    print("Failed to reset arm; simulator remains open.")
                    traceback.print_exc()
                print_prediction_menu(predictions)
            elif key_handler.pending_index is not None:
                prediction_index = key_handler.pending_index
                key_handler.pending_index = None
                prediction = predictions[prediction_index - 1]
                try:
                    result = execute_prediction(
                        robot=robot,
                        arm=arm,
                        prediction=prediction,
                        trajectory_robot=trajectories_robot[prediction_index],
                        args=args,
                        key_handler=key_handler,
                    )
                    if result == "reset":
                        key_handler.reset_requested = False
                        key_handler.pending_index = None
                        run_arm_reset(robot, arm, home_position_robot, home_orientation_robot, args)
                    else:
                        print(f"Ready for another prediction after e{prediction.index}.")
                except Exception:
                    print(f"Failed to execute e{prediction.index}; simulator remains open.")
                    traceback.print_exc()
                print_prediction_menu(predictions)
            else:
                og.sim.step()
    finally:
        key_handler.close()
        og.shutdown()


if __name__ == "__main__":
    main()
