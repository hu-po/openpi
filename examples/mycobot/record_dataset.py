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
                 port: str = _c.DEFAULT_PORT,
                 baudrate: int = _c.DEFAULT_BAUDRATE,
                 camera_id: int = _c.DEFAULT_CAMERA_ID,
                 render_height: int = _c.IMAGE_HEIGHT,
                 render_width: int = _c.IMAGE_WIDTH) -> None:
        self.robot = MyCobot(port, baudrate)
        self.camera = cv2.VideoCapture(camera_id)
        self._render_height = render_height
        self._render_width = render_width
        
        # Initialize robot
        self.reset()
        logger.info("MyCobotRecorder initialized")

    def reset(self) -> None:
        self.robot.send_angles(_c.HOME_POSITION, 50)

    def get_observation(self) -> Dict[str, Any]:
        """Get current observation including joint states and camera image"""
        joint_positions = self.robot.get_angles() or _c.HOME_POSITION
        
        ret, frame = self.camera.read()
        if not ret:
            logger.warning("Failed to get camera frame")
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
        frame = cv2.resize(frame, (self._render_width, self._render_height))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        return {
            "state": np.array(joint_positions, dtype=np.float32),
            "image": frame
        }

    def __del__(self) -> None:
        if hasattr(self, 'camera'):
            self.camera.release()

def create_empty_dataset(
    repo_id: str,
    dataset_config: DatasetConfig = DatasetConfig()
) -> LeRobotDataset:
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(_c.JOINT_NAMES),),
            "names": [_c.JOINT_NAMES],
        },
        "observation.images.camera": {
            "dtype": "image",
            "shape": (3, _c.IMAGE_HEIGHT, _c.IMAGE_WIDTH),
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
    repo_id: str = 'hu-po/mycobot'
    task: str = "DEBUG"
    port: str = _c.DEFAULT_PORT
    baudrate: int = _c.DEFAULT_BAUDRATE
    camera_id: int = _c.DEFAULT_CAMERA_ID
    push_to_hub: bool = True
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
        logger.info("Pushing dataset to Hugging Face Hub...")
        dataset.push_to_hub('hu-po/mycobot') # private=True)
        logger.info("Dataset pushed to Hugging Face Hub successfully.")


if __name__ == "__main__":
    tyro.cli(main) 