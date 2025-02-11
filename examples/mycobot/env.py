import logging
import time
from typing import Dict, Any, Literal, Optional
import cv2
import numpy as np
from pymycobot.mycobot import MyCobot
from openpi_client.runtime import environment as _environment
from openpi_client import image_tools
from typing_extensions import override

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MyCobotEnv(_environment.Environment):
    def __init__(self, 
                 port: str = "/dev/ttyAMA0",
                 baudrate: int = 1000000,
                 camera_id: int = 0,
                 action_space: Literal["joint_velocity", "joint_position"] = "joint_velocity",
                 gripper_action_space: Literal["position"] = "position",
                 render_height: int = 224,
                 render_width: int = 224) -> None:
        self.robot = MyCobot(port, baudrate)
        self.camera = cv2.VideoCapture(camera_id)
        self.action_space = action_space
        self.gripper_action_space = gripper_action_space
        self._render_height = render_height
        self._render_width = render_width
        self._ts = None
        
        # Initialize robot
        self.reset()
        logger.info("MyCobotEnv initialized")

    @override
    def reset(self) -> None:
        """Reset robot to home position"""
        home_angles = [0, 0, 0, 0, 0, 0]
        self.robot.send_angles(home_angles, 50)
        time.sleep(3)  # Wait for movement to complete
        self._ts = self.get_observation()
        
    @override
    def is_episode_complete(self) -> bool:
        return False

    @override
    def get_observation(self) -> Dict[str, Any]:
        """Get current observation including joint states and camera image"""
        joint_positions = self.robot.get_angles() or [0]*6
        gripper_pos = self.robot.get_gripper_value() or 0
        gripper_pos = np.clip(gripper_pos/100, 0, 1)
        
        ret, frame = self.camera.read()
        if not ret:
            logger.warning("Failed to get camera frame")
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
        # Process image similar to Aloha example
        frame = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(frame, self._render_height, self._render_width)
        )
        frame = np.transpose(frame, (2, 0, 1))  # Convert to CxHxW format
        
        return {
            "state": joint_positions + [gripper_pos],
            "images": {
                "camera": frame
            }
        }

    @override
    def apply_action(self, action: Dict[str, np.ndarray]) -> None:
        """Execute action on robot"""
        joint_action = action["actions"][:-1]
        gripper_action = action["actions"][-1]
        
        if self.action_space == "joint_velocity":
            curr_angles = self.robot.get_angles() or [0]*6
            target_angles = [curr + vel * 0.1 for curr, vel in zip(curr_angles, joint_action)]
            self.robot.send_angles(target_angles, 50)
        else:
            self.robot.send_angles(joint_action.tolist(), 50)
            
        gripper_pos = int(gripper_action * 100)
        self.robot.set_gripper_value(gripper_pos, 50)

    def __del__(self) -> None:
        if hasattr(self, 'camera'):
            self.camera.release() 