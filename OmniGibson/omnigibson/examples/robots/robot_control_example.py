"""
Example script demo'ing robot control.

Options for random actions, as well as selection of robot action space
"""

import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch as th

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.macros import gm
from omnigibson.robots import REGISTERED_ROBOTS
from omnigibson.sensors import VisionSensor
from omnigibson.utils.ui_utils import KeyboardRobotController, choose_from_options

CONTROL_MODES = dict(
    random="Use autonomous random actions (default)",
    teleop="Use keyboard control",
)

SCENES = dict(
    Rs_int="Realistic interactive home environment (default)",
    empty="Empty environment with no objects",
)

# Don't use GPU dynamics for performance boost
gm.USE_GPU_DYNAMICS = False

CAPTURE_RESOLUTION = (1920, 1080)


def find_capture_camera(robot, preferred_name="eyes"):
    """
    Finds the robot-mounted camera to use for RGB-D captures.
    """
    vision_sensors = [(name, sensor) for name, sensor in robot.sensors.items() if isinstance(sensor, VisionSensor)]
    if not vision_sensors:
        raise RuntimeError(f"Robot {robot.name} has no VisionSensor cameras available for capture.")

    for name, sensor in vision_sensors:
        if preferred_name in name:
            return name, sensor

    return vision_sensors[0]


def next_capture_idx(output_root):
    """
    Returns the next 1-indexed capture id based on existing omnigibson_<idx> folders.
    """
    existing = []
    for path in output_root.glob("omnigibson_*"):
        if not path.is_dir():
            continue
        try:
            existing.append(int(path.name.removeprefix("omnigibson_")))
        except ValueError:
            pass
    return max(existing, default=0) + 1


def capture_camera_observation(camera, output_root, capture_idx, resolution=CAPTURE_RESOLUTION):
    """
    Captures RGB, depth, and intrinsics from @camera into:
        omnigibson_<idx>/colour/<idx>.png
        omnigibson_<idx>/depth/<idx>.png
        omnigibson_<idx>/camera_intrinsic.json
    """
    capture_dir = output_root / f"omnigibson_{capture_idx}"
    colour_dir = capture_dir / "colour"
    depth_dir = capture_dir / "depth"
    colour_dir.mkdir(parents=True, exist_ok=False)
    depth_dir.mkdir(parents=True, exist_ok=False)

    if "rgb" not in camera.modalities or "depth_linear" not in camera.modalities:
        raise RuntimeError(
            f"Capture camera {camera.name} must have rgb and depth_linear enabled. "
            f"Current modalities: {camera.modalities}"
        )

    if (camera.image_width, camera.image_height) != resolution:
        print(
            f"Warning: expected capture resolution {resolution}, got "
            f"{camera.image_width}x{camera.image_height}."
        )

    # Let the latest teleop step render before reading the annotators.
    for _ in range(2):
        og.sim.render()

    obs, _ = camera.get_obs()
    rgb = obs["rgb"][..., :3].cpu().numpy()
    depth = obs["depth_linear"].cpu().numpy()
    K = camera.intrinsic_matrix.cpu().numpy()

    Image.fromarray(rgb).save(colour_dir / f"{capture_idx}.png")

    depth_mm = np.clip(depth * 1000.0, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    Image.fromarray(depth_mm).save(depth_dir / f"{capture_idx}.png")

    intrinsics = {
        "width": int(camera.image_width),
        "height": int(camera.image_height),
        "intrinsic_matrix": K.reshape(-1, order="F").astype(float).tolist(),
    }
    with (capture_dir / "camera_intrinsic.json").open("w") as f:
        json.dump(intrinsics, f, indent=2)

    print(f"Saved RGB-D capture {capture_idx} to {capture_dir}")


def choose_controllers(robot, random_selection=False):
    """
    For a given robot, iterates over all components of the robot, and returns the requested controller type for each
    component.

    :param robot: BaseRobot, robot class from which to infer relevant valid controller options
    :param random_selection: bool, if the selection is random (for automatic demo execution). Default False

    :return dict: Mapping from individual robot component (e.g.: base, arm, etc.) to selected controller names
    """
    # Create new dict to store responses from user
    controller_choices = dict()

    # Grab the default controller config so we have the registry of all possible controller options
    default_config = robot._default_controller_config

    # Iterate over all components in robot
    controller_names = robot.controller_order
    for controller_name in controller_names:
        controller_options = default_config[controller_name]
        # Select controller
        options = list(sorted(controller_options.keys()))
        choice = choose_from_options(
            options=options,
            name=f"{controller_name} controller",
            random_selection=random_selection,
        )

        # Add to user responses
        controller_choices[controller_name] = choice

    return controller_choices


def main(random_selection=False, headless=False, short_exec=False, quickstart=False):
    """
    Robot control demo with selection
    Queries the user to select a robot, the controllers, a scene and a type of input (random actions or teleop)
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

    # Choose scene to load
    scene_model = "Rs_int"
    if not quickstart:
        scene_model = choose_from_options(options=SCENES, name="scene", random_selection=random_selection)

    # Choose robot to create
    robot_name = "fetch"
    if not quickstart:
        robot_name = choose_from_options(
            options=list(sorted(REGISTERED_ROBOTS)), name="robot", random_selection=random_selection
        )

    scene_cfg = dict()
    if scene_model == "empty":
        scene_cfg["type"] = "Scene"
    else:
        scene_cfg["type"] = "InteractiveTraversableScene"
        scene_cfg["scene_model"] = scene_model

    # Add the robot we want to load
    robot0_cfg = dict()
    robot0_cfg["model"] = robot_name
    robot0_cfg["obs_modalities"] = ["rgb", "depth_linear"]
    robot0_cfg["sensor_config"] = {
        "VisionSensor": {
            "modalities": ["rgb", "depth_linear"],
            "sensor_kwargs": {
                "image_width": CAPTURE_RESOLUTION[0],
                "image_height": CAPTURE_RESOLUTION[1],
            },
        }
    }
    robot0_cfg["action_type"] = "continuous"
    robot0_cfg["action_normalize"] = True

    # Compile config
    cfg = dict(scene=scene_cfg, robots=[robot0_cfg])

    # Create the environment
    env = og.Environment(configs=cfg)

    # Choose robot controller to use
    robot = env.robots[0]
    controller_choices = {
        "base": "DifferentialDriveController",
        "arm_0": "InverseKinematicsController",
        "gripper_0": "MultiFingerGripperController",
        "camera": "JointController",
    }
    if not quickstart:
        controller_choices = choose_controllers(robot=robot, random_selection=random_selection)

    # Choose control mode
    if random_selection:
        control_mode = "random"
    elif quickstart:
        control_mode = "teleop"
    else:
        control_mode = choose_from_options(options=CONTROL_MODES, name="control mode")

    # Update the control mode of the robot
    controller_config = {component: {"name": name} for component, name in controller_choices.items()}
    robot.reload_controllers(controller_config=controller_config)

    # Because the controllers have been updated, we need to update the initial state so the correct controller state
    # is preserved
    env.scene.update_initial_file()

    # Update the simulator's viewer camera's pose so it points towards the robot
    og.sim.viewer_camera.set_position_orientation(
        position=th.tensor([1.46949, -3.97358, 2.21529]),
        orientation=th.tensor([0.56829048, 0.09569975, 0.13571846, 0.80589577]),
    )

    # Reset environment and robot
    env.reset()
    robot.reset()

    # Create teleop controller
    action_generator = KeyboardRobotController(robot=robot)

    capture_enabled = control_mode == "teleop"
    capture_output_root = None
    capture_camera = None
    capture_idx = 1
    capture_requested = False
    last_capture_time = 0.0

    if capture_enabled:
        capture_output_root = Path.cwd() / "robo_images"
        capture_output_root.mkdir(parents=True, exist_ok=True)
        capture_camera_name, capture_camera = find_capture_camera(robot)
        capture_idx = next_capture_idx(capture_output_root)
        # Initialize camera params outside of the keyboard callback path.
        _ = capture_camera.intrinsic_matrix
        print(f"RGB-D capture camera: {capture_camera_name}")
        print(f"Press Y to save RGB, depth, and intrinsics to {capture_output_root}/omnigibson_<n>")

    # Register custom binding to reset the environment
    action_generator.register_custom_keymapping(
        key=lazy.carb.input.KeyboardInput.R,
        description="Reset the robot",
        callback_fn=lambda: env.reset(),
    )

    def request_camera_capture():
        nonlocal capture_requested, last_capture_time

        now = time.time()
        if now - last_capture_time < 0.75:
            return
        last_capture_time = now
        capture_requested = True

    if capture_enabled:
        # Register custom binding to capture the robot camera.
        action_generator.register_custom_keymapping(
            key=lazy.carb.input.KeyboardInput.Y,
            description="Save RGB, depth, and camera intrinsics",
            callback_fn=request_camera_capture,
        )

    # Print out relevant keyboard info if using keyboard teleop
    if control_mode == "teleop":
        action_generator.print_keyboard_teleop_info()

    # Other helpful user info
    print("Running demo.")
    print("Press ESC to quit")

    # Loop control until user quits
    max_steps = -1 if not short_exec else 100
    step = 0

    random_action = None
    while step != max_steps:
        if control_mode == "random":
            # Sample new random action every 30 steps
            if step % 30 == 0:
                random_action = action_generator.get_random_action() * 0.05
            action = random_action
        else:
            action = action_generator.get_teleop_action()

        # Avoid pulling full-resolution RGB-D observations on every control step.
        robot.apply_action(action)
        og.sim.step()

        if capture_enabled and capture_requested:
            capture_requested = False
            try:
                capture_idx = max(capture_idx, next_capture_idx(capture_output_root))
                capture_camera_observation(
                    camera=capture_camera,
                    output_root=capture_output_root,
                    capture_idx=capture_idx,
                )
                capture_idx += 1
            except Exception as e:
                print(f"Failed to save RGB-D capture: {e}")
                capture_idx = next_capture_idx(capture_output_root)
        step += 1

    # Always shut down the environment cleanly at the end
    og.shutdown()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Teleoperate a robot in a BEHAVIOR scene.")

    parser.add_argument(
        "--quickstart",
        action="store_true",
        help="Whether the example should be loaded with default settings for a quick start.",
    )
    args = parser.parse_args()
    main(quickstart=args.quickstart)
