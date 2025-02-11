import dataclasses
import logging
from pathlib import Path
from typing import Dict, Any, Literal, Optional

import cv2
import numpy as np
import tyro
from pymycobot.mycobot import MyCobot
from typing_extensions import override

from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME, LeRobotDataset

from examples.mycobot import constants as _c

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None

class MyCobotRecorder:
    def __init__(self, 
                 port: str = "/dev/ttyAMA0",
                 baudrate: int = 1000000,
                 camera_id: int = 0,
                 render_height: int = 224,
                 render_width: int = 224) -> None:
        self.robot = MyCobot(port, baudrate)
        self.camera = cv2.VideoCapture(camera_id)
        self._render_height = render_height
        self._render_width = render_width
        
        # Initialize robot
        self.reset()
        logger.info("MyCobotRecorder initialized")

    def reset(self) -> None:
        """Reset robot to home position"""
        home_angles = _c.DEFAULT_RESET_POSITION
        self.robot.send_angles(home_angles, 50)

    def get_observation(self) -> Dict[str, Any]:
        """Get current observation including joint states and camera image"""
        joint_positions = self.robot.get_angles() or _c.DEFAULT_RESET_POSITION
        gripper_pos = self.robot.get_gripper_value() or 0
        gripper_pos = np.clip(gripper_pos/100, 0, 1)
        
        ret, frame = self.camera.read()
        if not ret:
            logger.warning("Failed to get camera frame")
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
        frame = cv2.resize(frame, (self._render_width, self._render_height))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        return {
            "state": np.array(joint_positions + [gripper_pos], dtype=np.float32),
            "image": frame
        }

    def __del__(self) -> None:
        if hasattr(self, 'camera'):
            self.camera.release()

def create_empty_dataset(
    repo_id: str,
    dataset_config: DatasetConfig = DatasetConfig()
) -> LeRobotDataset:
    motors = [
        "joint1", "joint2", "joint3", 
        "joint4", "joint5", "joint6",
        "gripper"
    ]

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [motors],
        },
        "observation.images.camera": {
            "dtype": "image",
            "shape": (3, 224, 224),
            "names": ["channels", "height", "width"],
        }
    }

    if Path(LEROBOT_HOME / repo_id).exists():
        Path(LEROBOT_HOME / repo_id).unlink(missing_ok=True)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=50,
        robot_type="mycobot",
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )

@dataclasses.dataclass
class Args:
    repo_id: str
    task: str = "DEBUG"
    port: str = "/dev/ttyAMA0" 
    baudrate: int = 1000000
    camera_id: int = 0
    push_to_hub: bool = False
    max_episode_steps: int = 500

def main(args: Args) -> None:
    recorder = MyCobotRecorder(
        port=args.port,
        baudrate=args.baudrate,
        camera_id=args.camera_id
    )
    
    dataset = create_empty_dataset(args.repo_id)
    logger.info(f"Recording episode for task: {args.task}")
    
    try:
        for i in range(args.max_episode_steps):
            obs = recorder.get_observation()
            frame = {
                "observation.state": obs["state"],
                "observation.images.camera": obs["image"]
            }
            dataset.add_frame(frame)
            
    except KeyboardInterrupt:
        logger.info("Recording stopped by user")
    
    dataset.save_episode(task=args.task)
    dataset.consolidate()
    
    if args.push_to_hub:
        dataset.push_to_hub()

if __name__ == "__main__":
    tyro.cli(main) 