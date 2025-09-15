"""
Tatbot remote inference (no ROS) using openpi-client over WebSocket.

Requirements (NUC client):
- `pip install -e packages/openpi-client`
- Tatbot + LeRobot runtime available (tatbot arms + cameras)

Usage (example):
  uv run python examples/tatbot/infer.py \
    --host 192.168.1.50 --port 8000 \
    --ip_address_l 192.168.1.71 --ip_address_r 192.168.1.72 \
    --arm_l_config ~/tatbot/configs/left.yaml \
    --arm_r_config ~/tatbot/configs/right.yaml \
    --home_pos_l 0 -1.5 1.5 0 0 0 0.5 \
    --home_pos_r 0 -1.5 1.5 0 0 0 0.5 \
    --left_cam realsense1 --right_cam realsense2 --high_cam overhead
"""

from __future__ import annotations

import dataclasses
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import tyro

from openpi_client import image_tools
from openpi_client import websocket_client_policy

# Tatbot + LeRobot
from lerobot.robots import make_robot_from_config
from lerobot.robots.tatbot.config_tatbot import TatbotConfig


def prep_image(img: np.ndarray, size: int = 224) -> np.ndarray:
    """Convert to uint8 and letterbox to `size` for bandwidth/latency.
    Server also resizes, but pre-resize reduces transport time.
    """
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(img, size, size))


@dataclasses.dataclass
class Args:
    # Policy server
    host: str
    port: int = 8000
    default_prompt: Optional[str] = None

    # Tatbot connection
    ip_address_l: str
    ip_address_r: str
    arm_l_config: Path
    arm_r_config: Path
    goal_time: float = 0.06
    connection_timeout: float = 3.0
    home_pos_l: list[float] = dataclasses.field(default_factory=lambda: [0, -1.5, 1.5, 0, 0, 0, 0.5])
    home_pos_r: list[float] = dataclasses.field(default_factory=lambda: [0, -1.5, 1.5, 0, 0, 0, 0.5])

    # Camera mapping to ALOHA keys
    left_cam: str = "realsense1"
    right_cam: str = "realsense2"
    high_cam: Optional[str] = None  # If None, duplicates left_cam as cam_high

    # Loop
    max_hz: float = 10.0
    max_steps: int = 2000


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, force=True)

    # Create policy client
    client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    meta = client.get_server_metadata()
    logging.info(f"Connected to policy server. Metadata: {meta}")

    # Configure Tatbot
    cfg = TatbotConfig(
        ip_address_l=args.ip_address_l,
        ip_address_r=args.ip_address_r,
        arm_l_config_filepath=str(args.arm_l_config.expanduser()),
        arm_r_config_filepath=str(args.arm_r_config.expanduser()),
        goal_time=args.goal_time,
        connection_timeout=args.connection_timeout,
        home_pos_l=args.home_pos_l,
        home_pos_r=args.home_pos_r,
        rs_cameras={},  # camera connections are managed inside tatbot setup
        ip_cameras={},
    )
    robot = make_robot_from_config(cfg)
    robot.connect()
    logging.info("Tatbot connected")

    dt = 1.0 / max(1e-6, args.max_hz)
    step = 0
    try:
        while step < args.max_steps:
            t0 = time.perf_counter()

            # Grab observation from Tatbot
            obs = robot.get_observation()

            # Extract cameras
            def get_cam(name: Optional[str]) -> Optional[np.ndarray]:
                if name is None:
                    return None
                frame = obs.get(name)
                if frame is None:
                    return None
                return prep_image(frame)

            img_left = get_cam(args.left_cam)
            img_right = get_cam(args.right_cam)
            img_high = get_cam(args.high_cam) or img_left

            # Extract 14‑D state (left 7 + right 7)
            joints = []
            for side in ("left", "right"):
                for j in ("joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "gripper"):
                    key = f"{side}.{j}.pos"
                    v = obs.get(key, 0.0)
                    joints.append(float(v))
            state = np.asarray(joints, dtype=np.float32)

            prompt = args.default_prompt or "perform the current task"

            payload = {
                "observation.images.cam_high": img_high,
                "observation.images.cam_left_wrist": img_left if img_left is not None else img_high,
                "observation.images.cam_right_wrist": img_right if img_right is not None else img_high,
                "observation.state": state,
                "prompt": prompt,
            }

            # Call policy
            out = client.infer(payload)
            action_chunk = np.asarray(out["actions"])  # [H, 14]

            # Execute open-loop action chunk on Tatbot
            for a in action_chunk:
                # Map 14‑D joint vector back to Tatbot action dict
                vals = list(map(float, a.tolist()))
                left = vals[:7]
                right = vals[7:14]
                action_dict = {f"left.{n}.pos": left[i] for i, n in enumerate(["joint_0","joint_1","joint_2","joint_3","joint_4","joint_5","gripper"])}
                action_dict.update({f"right.{n}.pos": right[i] for i, n in enumerate(["joint_0","joint_1","joint_2","joint_3","joint_4","joint_5","gripper"])} )
                robot.send_action(action_dict, goal_time=args.goal_time)

            step += len(action_chunk)

            # Maintain loop rate
            elapsed = time.perf_counter() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    finally:
        try:
            robot.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    tyro.cli(main)

