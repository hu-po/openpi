import dataclasses
import logging
from pathlib import Path
from typing import Dict, Any, Literal, Optional
import uuid

import cv2
import numpy as np
import tyro
from pymycobot.mycobot import MyCobot
from typing_extensions import override

from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME, LeRobotDataset

from examples.mycobot import constants as _c
from examples.mycobot.env import MyCobotEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None

def create_empty_dataset(repo_id: str, dataset_config: DatasetConfig) -> LeRobotDataset:
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

def record_episode(env: MyCobotEnv, dataset: LeRobotDataset, max_steps: int) -> None:
    try:
        for _ in range(max_steps):
            obs = env.get_observation()
            frame = {
                "observation.state": obs["state"],
                "observation.images.camera": np.transpose(obs["images"]["cam_main"], (1, 2, 0))
            }
            dataset.add_frame(frame)
    except KeyboardInterrupt:
        logger.info("Recording stopped by user")

@dataclasses.dataclass
class Args:
    base_repo_id: str = 'hu-po/mycobot'
    task: str = "DEBUG"
    port: str = _c.DEFAULT_PORT
    baudrate: int = _c.DEFAULT_BAUDRATE
    camera_id: int = _c.DEFAULT_CAMERA_ID
    push_to_hub: bool = True
    max_episode_steps: int = 500

def main(args: Args) -> None:
    # Generate unique repo ID using UUID
    unique_id = str(uuid.uuid4())[:8]
    repo_id = f"{args.base_repo_id}-{unique_id}"
    logger.info(f"Generated unique repo ID: {repo_id}")

    env = MyCobotEnv(
        port=args.port,
        baudrate=args.baudrate,
        camera_id=args.camera_id
    )
    
    dataset = create_empty_dataset(repo_id, DatasetConfig())
    logger.info(f"Recording episode for task: {args.task}")
    
    record_episode(env, dataset, args.max_episode_steps)
    
    dataset.save_episode(task=args.task)
    dataset.consolidate()
    
    if args.push_to_hub:
        logger.info("Pushing dataset to Hugging Face Hub...")
        dataset.push_to_hub(repo_id)
        logger.info(f"Dataset pushed to Hugging Face Hub successfully as: {repo_id}")

if __name__ == "__main__":
    tyro.cli(main) 