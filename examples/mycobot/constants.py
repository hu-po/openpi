import logging
import numpy as np
from typing import Callable, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Time step for control loop
DT: float = 0.02  # 50Hz

# Robot settings
ROBOT_PORT: str = "/dev/ttyAMA0"
ROBOT_BAUDRATE: int = 1000000
ROBOT_SPEED: int = 60  # Default movement speed (percentage)
ROBOT_MOVE_TIMEOUT: int = 3  # seconds
ROBOT_MODE: int = 1  # 0 = joint mode, 1 = Cartesian mode
ROBOT_SCALE: float = 20.0  # Default scale for movement patterns in mm

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
HOME_POSITION: list[float] = [1.23, -88.94, -63.80, 60.29, 87.71, -5.80]
ORIGIN_POSITION: list[float] = [-0.08, -108.98, -64.86, 88.5, 88.24, 1.49]
JOINT_LIMITS: dict[str, tuple[float, float]] = {
    "waist": (-34.98, 36.12),
    "shoulder": (-120.14, -88.94),
    "elbow": (-78.39, -13.09),
    "wrist_flex": (28.03, 88.94),
    "wrist_rot": (86.48, 89.47),
    "gripper_rot": (-5.88, -5.80),
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

# Camera settings
CAMERA_ID: int = 0
CAMERA_IMAGE_HEIGHT: int = 224
CAMERA_IMAGE_WIDTH: int = 224

# Tablet configuration
TABLET_DEVICE_NAME: str = "Wacom Intuos Pro L Pen"
TABLET_MAX_STEPS: int = 1000
TABLET_CAPTURE_DURATION: float = 5.0

# canvas is a specific sub-region of the tablet
TABLET_CANVAS_ORIGIN: Tuple[int, int] = (0, 0) # center of canvas in tablet coordinates
TABLET_CANVAS_SIZE_TABLETSPACE: Tuple[int, int] = (224, 224) # size of canvas in tablet coordinates
TABLET_CANVAS_SIZE_PIXELSPACE: Tuple[int, int] = (1024, 1024) # size of canvas in pixel space
TABLET_PEN_WIDTH: int = 10 # width of pen in tablet coordinates

# ANSI color codes for axis visualization
AXIS_COLORS: dict[str, str] = {
    "x": "\033[31m",  # Red
    "y": "\033[32m",  # Green
    "z": "\033[34m",  # Blue
    "reset": "\033[0m"  # Reset color
}