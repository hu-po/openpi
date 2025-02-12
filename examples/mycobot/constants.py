import logging
import numpy as np
from typing import Callable, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Hardware configuration
DEFAULT_PORT: str = "/dev/ttyAMA0"
DEFAULT_BAUDRATE: int = 1000000
DEFAULT_CAMERA_ID: int = 0
DEFAULT_SPEED: int = 50  # Default movement speed (percentage)

# Time step for control loop
DT: float = 0.02  # 50Hz

# Camera sizes
IMAGE_HEIGHT: int = 224
IMAGE_WIDTH: int = 224

# Tablet configuration
DEFAULT_TABLET_DEVICE: str = "Wacom Intuos Pro L Pen"
DEFAULT_CANVAS_SIZE: Tuple[int, int] = (1024, 1024)
DEFAULT_MAX_STEPS: int = 1000
DEFAULT_CAPTURE_DURATION: float = 5.0

# Joint names and default positions
JOINT_NAMES: list[str] = [
    "waist",          # joint1 - base rotation
    "shoulder",       # joint2 - shoulder joint
    "elbow",         # joint3 - elbow joint
    "wrist_flex",    # joint4 - wrist up/down
    "wrist_rot",     # joint5 - wrist rotation
    "gripper_rot"    # joint6 - gripper/end effector rotation
]

# robot positions (from calibration process, these are in DEGREES)
SLEEP_POSITION: list[float] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
HOME_POSITION: list[float] = [13.35, -90.61, -76.28, 68.99, -76.99, 54.05]
JOINT_LIMITS: dict[str, tuple[float, float]] = {
    "waist": (-165.0, 165.0),
    "shoulder": (-165.0, 165.0),
    "elbow": (-165.0, 165.0),
    "wrist_flex": (-165.0, 165.0),
    "wrist_rot": (-165.0, 165.0),
    "gripper_rot": (-175.0, 175.0),
}

# Normalize/unnormalize functions for converting between raw and normalized values
def normalize_joint_position(x: float, joint_name: str) -> float:
    """Normalize joint position from degrees to [-1, 1]"""
    min_val, max_val = JOINT_LIMITS[joint_name]
    return 2.0 * (x - min_val) / (max_val - min_val) - 1.0

def unnormalize_joint_position(x: float, joint_name: str) -> float:
    """Unnormalize joint value from [-1, 1] to degrees"""
    min_val, max_val = JOINT_LIMITS[joint_name]
    return 0.5 * (x + 1.0) * (max_val - min_val) + min_val

# Type aliases for normalize/unnormalize functions
JOINT_POSITION_NORMALIZE_FN: Callable[[float, str], float] = normalize_joint_position
JOINT_POSITION_UNNORMALIZE_FN: Callable[[float, str], float] = unnormalize_joint_position