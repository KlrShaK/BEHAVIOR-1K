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
                "mode": "position_fixed_ori",
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


def step_toward(robot, arm, target_robot, max_delta, gripper_command=None):
    target = th.tensor(target_robot, dtype=th.float32)
    current = robot.get_relative_eef_position(arm=arm)
    delta = target - current
    distance = th.norm(delta).item()
    if distance > max_delta:
        delta = delta / distance * max_delta

    action = compute_no_op_action(robot)
    action[robot.arm_action_idx[arm]] = delta
    set_gripper_action(robot, action, arm, gripper_command)
    robot.apply_action(action)
    og.sim.step()
    return distance


def drive_to_waypoint(robot, arm, waypoint, args, gripper_command=None):
    for _ in range(args.max_steps_per_waypoint):
        distance = step_toward(
            robot=robot,
            arm=arm,
            target_robot=waypoint,
            max_delta=args.max_delta,
            gripper_command=gripper_command,
        )
        if distance <= args.position_tolerance:
            return True
    return False


def execute_prediction(robot, arm, prediction, trajectory_robot, args):
    print(
        f"Executing e{prediction.index}: {prediction.path.name} "
        f"(candidate {prediction.best_candidate}, loss {prediction.best_loss:.4f})"
    )

    first_waypoint = trajectory_robot[0]
    reached = drive_to_waypoint(robot, arm, first_waypoint, args, gripper_command=args.open_gripper_command)
    if not reached:
        print("Warning: did not reach the first prediction waypoint within the step budget.")

    if args.close_gripper_command is not None:
        for _ in range(args.gripper_settle_steps):
            action = compute_no_op_action(robot)
            set_gripper_action(robot, action, arm, args.close_gripper_command)
            robot.apply_action(action)
            og.sim.step()

    for waypoint in trajectory_robot[1:: args.waypoint_stride]:
        drive_to_waypoint(robot, arm, waypoint, args, gripper_command=args.close_gripper_command)

    print(f"Finished e{prediction.index}.")


def reset_arm_to_home(robot, arm, home_position_robot, args):
    print("Resetting arm to captured home position.")
    reached = drive_to_waypoint(
        robot=robot,
        arm=arm,
        waypoint=home_position_robot,
        args=args,
        gripper_command=args.open_gripper_command,
    )
    if reached:
        print("Arm reset finished.")
    else:
        print("Warning: arm reset did not reach home position within the step budget.")


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
    home_position_robot = np.asarray(
        metadata["robot"]["eef_poses"][arm]["robot_relative"]["position"],
        dtype=np.float32,
    )
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
                try:
                    reset_arm_to_home(
                        robot=robot,
                        arm=arm,
                        home_position_robot=home_position_robot,
                        args=args,
                    )
                    print("Ready for another prediction after arm reset.")
                except Exception:
                    print("Failed to reset arm; simulator remains open.")
                    traceback.print_exc()
                print_prediction_menu(predictions)
            elif key_handler.pending_index is not None:
                prediction_index = key_handler.pending_index
                key_handler.pending_index = None
                prediction = predictions[prediction_index - 1]
                try:
                    execute_prediction(
                        robot=robot,
                        arm=arm,
                        prediction=prediction,
                        trajectory_robot=trajectories_robot[prediction_index],
                        args=args,
                    )
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
