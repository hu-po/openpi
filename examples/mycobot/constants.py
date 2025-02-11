import logging
import numpy as np
from typing import Callable

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Hardware configuration
DEFAULT_PORT: str = "/dev/ttyAMA0"
DEFAULT_BAUDRATE: int = 1000000
DEFAULT_CAMERA_ID: int = 0

# Time step for control loop
DT: float = 0.02  # 50Hz

# camera sizes
IMAGE_HEIGHT: int = 224
IMAGE_WIDTH: int = 224

# Joint names and default positions
JOINT_NAMES: list[str] = [
    "waist",          # joint1 - base rotation
    "shoulder",       # joint2 - shoulder joint
    "elbow",         # joint3 - elbow joint
    "wrist_flex",    # joint4 - wrist up/down
    "wrist_rot",     # joint5 - wrist rotation
    "gripper_rot"    # joint6 - gripper/end effector rotation
]
# HOME_POSITION_DEGREES: list[float] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
HOME_POSITION_DEGREES: list[float] = [13.35, -90.61, -76.28, 68.99, -76.99, 54.05]
HOME_POSITION: list[float] = [deg * np.pi / 180.0 for deg in HOME_POSITION_DEGREES]

# Joint limits for MyCobot
JOINT_LIMITS_DEGREES: dict[str, tuple[float, float]] = {
    "waist": (-165.0, 165.0),
    "shoulder": (-165.0, 165.0),
    "elbow": (-165.0, 165.0),
    "wrist_flex": (-165.0, 165.0),
    "wrist_rot": (-165.0, 165.0),
    "gripper_rot": (-175.0, 175.0),
}
JOINT_LIMITS: dict[str, tuple[float, float]] = {
    joint: (deg * np.pi / 180.0 for deg in JOINT_LIMITS_DEGREES[joint])
    for joint in JOINT_LIMITS_DEGREES
}

# Normalize/unnormalize functions for converting between raw and normalized values
def normalize_joint_position(x: float, joint_name: str) -> float:
    """Normalize joint position from radians to [-1, 1]"""
    min_val, max_val = JOINT_LIMITS[joint_name]
    return 2.0 * (x - min_val) / (max_val - min_val) - 1.0

def unnormalize_joint_position(x: float, joint_name: str) -> float:
    """Unnormalize joint position from [-1, 1] to radians"""
    min_val, max_val = JOINT_LIMITS[joint_name]
    return 0.5 * (x + 1.0) * (max_val - min_val) + min_val

# Type aliases for normalize/unnormalize functions
JOINT_POSITION_NORMALIZE_FN: Callable[[float, str], float] = normalize_joint_position
JOINT_POSITION_UNNORMALIZE_FN: Callable[[float, str], float] = unnormalize_joint_position