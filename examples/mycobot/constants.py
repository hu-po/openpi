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

# Joint names and default positions
JOINT_NAMES: list[str] = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
DEFAULT_RESET_POSITION: list[float] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# Joint limits for MyCobot (radians)
JOINT_LIMITS: dict[str, tuple[float, float]] = {
    "joint1": (-165.0 * np.pi / 180.0, 165.0 * np.pi / 180.0),
    "joint2": (-165.0 * np.pi / 180.0, 165.0 * np.pi / 180.0),
    "joint3": (-165.0 * np.pi / 180.0, 165.0 * np.pi / 180.0),
    "joint4": (-165.0 * np.pi / 180.0, 165.0 * np.pi / 180.0),
    "joint5": (-165.0 * np.pi / 180.0, 165.0 * np.pi / 180.0),
    "joint6": (-175.0 * np.pi / 180.0, 175.0 * np.pi / 180.0),
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