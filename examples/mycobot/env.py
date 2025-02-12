import logging
import time
import numpy as np
from typing import Dict, Any, Literal
from typing_extensions import override

from openpi_client import image_tools
from openpi_client.runtime import environment as _environment

from examples.mycobot import constants as _c
from examples.mycobot import hardware as _hw

logger = logging.getLogger(__name__)

class MyCobotEnv(_environment.Environment):
    def __init__(self, action_space: Literal["joint_velocity", "joint_position"] = "joint_velocity") -> None:
        self.robot = _hw.Robot()
        self.camera = _hw.Camera()
        self.tablet = _hw.Tablet()
        self.action_space = action_space
        self._render_height = _c.IMAGE_HEIGHT
        self._render_width = _c.IMAGE_WIDT
        self._ts = None
        self.reset()
        logger.info("MyCobotEnv initialized")
    
    @override
    def reset(self) -> None:
        self.robot.go_home()
        time.sleep(3)
        self._ts = self.get_observation()
    
    @override
    def is_episode_complete(self) -> bool:
        return False
    
    @override
    def get_observation(self) -> Dict[str, Any]:
        joint_positions = self.robot.get_angles() or [0] * 6
        logger.info(f"Joint positions: {joint_positions}")
        normalized_joints = [_c.JOINT_POSITION_NORMALIZE_FN(pos, name) for pos, name in zip(joint_positions, _c.JOINT_NAMES)]
        ret, frame = self.camera.read()
        if not ret:
            logger.warning("Failed to get camera frame")
            frame = np.zeros((_c.IMAGE_HEIGHT, _c.IMAGE_WIDTH, 3), dtype=np.uint8)
        frame = image_tools.convert_to_uint8(image_tools.resize_with_pad(frame, self._render_height, self._render_width))
        frame = np.transpose(frame, (2, 0, 1))
        return {"state": normalized_joints, "images": {"cam_main": frame}}
    
    @override
    def apply_action(self, action: Dict[str, np.ndarray]) -> None:
        joint_action = action["actions"]
        if self.action_space == "joint_velocity":
            curr_angles = self.robot.get_angles() or [0] * 6
            target_angles = [curr + vel * _c.DT for curr, vel in zip(curr_angles, joint_action)]
            unnorm_angles = [_c.JOINT_POSITION_UNNORMALIZE_FN(pos, name) for pos, name in zip(target_angles, _c.JOINT_NAMES)]
            self.robot.send_angles(unnorm_angles, 50)
        else:
            unnorm_angles = [_c.JOINT_POSITION_UNNORMALIZE_FN(pos, name) for pos, name in zip(joint_action, _c.JOINT_NAMES)]
            self.robot.send_angles(unnorm_angles, 50)
        time.sleep(_c.DT)

    def __del__(self) -> None:
        if hasattr(self, "camera"):
            del self.camera
        if hasattr(self, "robot"):
            del self.robot
        if hasattr(self, "tablet"):
            del self.tablet