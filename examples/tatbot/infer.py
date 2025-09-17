"""
Tatbot remote inference (no ROS) using openpi-client over WebSocket.
"""
import dataclasses
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import tyro
from PIL import Image

from openpi_client import image_tools
from openpi_client import websocket_client_policy

# Tatbot + LeRobot
from lerobot.robots import make_robot_from_config
from lerobot.robots.tatbot.config_tatbot import TatbotConfig
from lerobot.cameras.realsense import RealSenseCameraConfig


def prep_image(img: np.ndarray, size: int = 224) -> np.ndarray:
    """Convert to uint8 and letterbox to `size` for bandwidth/latency.
    Server also resizes, but pre-resize reduces transport time.
    """
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(img, size, size))


@dataclasses.dataclass
class Args:
    # Policy server
    host: str = "192.168.1.51"  # policy server
    port: int = 8003
    prompt: str = "stay still while opening and closing the grippers"
    # prompt: str = "left: left arm inkdip into inkcap_left_large to fill with true_blue ink, right: right arm stroke after inkdip in inkcap_right_large"
    stroke_image: Path = Path("/nfs/tatbot/designs/wow/stroke_bright_red_right_0000.png")

    # Tatbot connection
    ip_address_l: str = "192.168.1.3"
    ip_address_r: str = "192.168.1.2"
    arm_l_config: Path = Path("/home/hog/tatbot/config/trossen/arm_l.yaml")
    arm_r_config: Path = Path("/home/hog/tatbot/config/trossen/arm_r.yaml")
    goal_time: float = 2.0
    connection_timeout: float = 3.0
    home_pos_l: list[float] = dataclasses.field(default_factory=lambda: [-0.333, 0.639, 0.667, -1.034, 0.541, 2.240, 0.04])
    home_pos_r: list[float] = dataclasses.field(default_factory=lambda: [0.333, 0.639, 0.667, -1.034, -0.541, -2.240, 0.04])

    # Camera mapping to ALOHA keys
    left_cam: str = "realsense1"
    right_cam: str = "realsense2"
    rs_left_serial: str = "230422273017"
    rs_right_serial: str = "218622278376"
    rs_fps: int = 30
    rs_width: int = 640
    rs_height: int = 480

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
    rs_cams = {}
    def cfg_rs(serial: Optional[str]) -> Optional[RealSenseCameraConfig]:
        if serial is None:
            return None
        return RealSenseCameraConfig(
            fps=args.rs_fps,
            width=args.rs_width,
            height=args.rs_height,
            serial_number_or_name=serial,
        )
    if (c := cfg_rs(args.rs_left_serial)) is not None:
        rs_cams[args.left_cam] = c
    if (c := cfg_rs(args.rs_right_serial)) is not None:
        rs_cams[args.right_cam] = c
    cfg = TatbotConfig(
        ip_address_l=args.ip_address_l,
        ip_address_r=args.ip_address_r,
        arm_l_config_filepath=str(args.arm_l_config.expanduser()),
        arm_r_config_filepath=str(args.arm_r_config.expanduser()),
        goal_time=args.goal_time,
        connection_timeout=args.connection_timeout,
        home_pos_l=args.home_pos_l,
        home_pos_r=args.home_pos_r,
        rs_cameras=rs_cams,
        ip_cameras={},
    )
    robot = make_robot_from_config(cfg)
    robot.connect()
    logging.info("Tatbot connected")
    # Warm up camera streams to reduce initial timeouts
    try:
        for _ in range(3):
            _ = robot.get_observation()
            time.sleep(0.2)
    except Exception as e:
        logging.warning(f"Camera warmup warning: {e}")

    # Load optional stroke (overhead) image once and reuse per-step for cam_high
    stroke_img_arr: Optional[np.ndarray] = None
    try:
        p = args.stroke_image.expanduser()
        if p.exists():
            im = Image.open(p).convert("RGB")
            stroke_img_arr = np.asarray(im, dtype=np.uint8)
            logging.info(f"Loaded stroke image for cam_high: {p}")
        else:
            logging.info(f"Stroke image not found at {p}; will use left_cam for cam_high")
    except Exception as e:
        logging.warning(f"Failed to load stroke image: {e}")
        stroke_img_arr = None

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
            img_high = prep_image(stroke_img_arr)
            
            # Extract 14‑D state (left 7 + right 7)
            joints = []
            for side in ("left", "right"):
                for j in ("joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "gripper"):
                    key = f"{side}.{j}.pos"
                    v = obs.get(key, 0.0)
                    joints.append(float(v))
            state = np.asarray(joints, dtype=np.float32)

            # Build payload for server ALOHAInputs: images (CHW), state, prompt
            def to_chw(x: np.ndarray) -> np.ndarray:
                # input is HWC uint8; convert to CHW
                return np.transpose(x, (2, 0, 1))

            images = {
                "cam_high": to_chw(img_high),
                "cam_left_wrist": to_chw(img_left) if img_left is not None else to_chw(img_high),
                "cam_right_wrist": to_chw(img_right) if img_right is not None else to_chw(img_high),
            }

            payload = {
                "images": images,
                "state": state,
                "prompt": args.prompt,
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
