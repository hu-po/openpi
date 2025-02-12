import logging
import cv2
import evdev
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Optional, Tuple, Union
from pymycobot.mycobot import MyCobot
from dataclasses import dataclass
import tyro
import termios
import tty
import sys

from examples.mycobot import constants as _c

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Args:
    cmd: str = "reset"
    """Command to run (i.e. test, test_camera, test_robot, test_tablet, reset, record_home"""
    debug: bool = False
    """Debug mode"""

class Camera:
    def __init__(
        self,
        camera_id: int = _c.DEFAULT_CAMERA_ID
    ) -> None:
        self._cam = cv2.VideoCapture(camera_id)
        if not self._cam.isOpened():
            logger.warning("Camera failed to open; fallback to blank frames")
        logger.info("MyCobotCamera initialized")

    def read(self) -> Tuple[bool, np.ndarray]:
        ret, frame = self._cam.read()
        if not ret:
            logger.warning("Failed to read frame; returning blank frame")
            frame = np.zeros((_c.IMAGE_HEIGHT, _c.IMAGE_WIDTH, 3), dtype=np.uint8)
        return ret, frame

    def release(self) -> None:
        self._cam.release()

    @staticmethod
    def test() -> None:
        camera = Camera()
        ret, frame = camera.read()
        logger.info(f"Camera test - frame shape: {frame.shape}")
        cv2.imwrite('camera_test.png', frame)
        logger.info("Saved test frame to camera_test.png")
        camera.release()

class Robot:
    def __init__(
        self,
        port: str = _c.DEFAULT_PORT,
        baudrate: int = _c.DEFAULT_BAUDRATE
    ) -> None:
        self._robot = MyCobot(
            port=port,
            baudrate=baudrate
        )
        logger.info("MyCobotRobot initialized")

    def send_angles(self, angles: List[float], speed: Union[int, float]) -> None:
        self._robot.send_angles(angles, speed)

    def get_angles(self) -> List[float]:
        return self._robot.get_angles()

    def release_all_servos(self) -> None:
        self._robot.release_all_servos()

    def reset(self) -> None:
        logger.info("Moving to home position...")
        self.send_angles(_c.HOME_POSITION, _c.DEFAULT_SPEED)
        time.sleep(3)
        logger.info("Releasing servos...")
        self.release_all_servos()
        logger.info("Done")

    def record_home(self) -> None:
        logger.info("Recording home position")
        logger.info("Move robot to home position and press SPACE")
        logger.info("Press q to quit at any time")
        self.release_all_servos()
        
        class Raw:
            def __init__(self, stream):
                self.stream = stream
                self.fd = self.stream.fileno()
            def __enter__(self):
                self.original_stty = termios.tcgetattr(self.fd)
                tty.setraw(self.fd)
            def __exit__(self, type_, value, traceback):
                termios.tcsetattr(self.fd, termios.TCSANOW, self.original_stty)

        while True:
            with Raw(sys.stdin):
                key = sys.stdin.read(1)
                if key == "q":
                    logger.info("Recording cancelled")
                    return
                elif key == " ":
                    angles = self.get_angles()
                    if angles:
                        logger.info(f"Recorded home position: {angles}")
                        logger.info("\nHome position recorded. Copy this into your constants.py:")
                        logger.info("HOME_POSITION = [")
                        logger.info(f"    {angles},")
                        logger.info("]")
                        return
                    else:
                        logger.warning("Failed to get angles, try again")

    @staticmethod
    def test() -> None:
        robot = Robot()
        current_angles = robot.get_angles()
        logger.info(f"Robot test - current angles: {current_angles}")
        if current_angles:
            new_angles = [a + 10 for a in current_angles]
            robot.send_angles(new_angles, _c.DEFAULT_SPEED)
            time.sleep(2)
            robot.send_angles(current_angles, _c.DEFAULT_SPEED)
        robot.release_all_servos()
        logger.info("Robot test complete")

class Tablet:
    def __init__(self,
                 device_name: str = _c.DEFAULT_TABLET_DEVICE,
                 canvas_size: Tuple[int, int] = _c.DEFAULT_CANVAS_SIZE,
                 max_steps: int = _c.DEFAULT_MAX_STEPS) -> None:
        self.canvas_size = canvas_size
        self.max_steps = max_steps
        self.buffer = np.zeros(self.canvas_size, dtype=np.uint8)
        self.state: Dict[str, int] = {'x': 0, 'y': 0, 'pressure': 0, 'tilt_x': 0, 'tilt_y': 0}
        self.step_count = 0
        
        devices = evdev.list_devices()
        if not devices:
            logger.error("No input devices found. Try running with sudo or add user to input group")
            self.device = None
            return
        
        for path in devices:
            try:
                dev = evdev.InputDevice(path)
                if device_name in dev.name:
                    self.device = dev
                    logger.info(f"Connected to {dev.name} at {dev.path}")
                    break
            except (PermissionError, OSError) as e:
                logger.error(f"Permission denied for device {path}: {e}")
        else:
            logger.error("Wacom pen device not found")
            self.device = None
            return

        caps = dict(self.device.capabilities()[evdev.ecodes.EV_ABS])
        self.x_info = caps[evdev.ecodes.ABS_X]
        self.y_info = caps[evdev.ecodes.ABS_Y]
        self.p_info = caps[evdev.ecodes.ABS_PRESSURE]
        self.tilt_x_info = caps[evdev.ecodes.ABS_TILT_X]
        self.tilt_y_info = caps[evdev.ecodes.ABS_TILT_Y]
        
        logger.info(f"X range: {self.x_info.min} to {self.x_info.max}")
        logger.info(f"Y range: {self.y_info.min} to {self.y_info.max}")
        logger.info(f"Pressure range: {self.p_info.min} to {self.p_info.max}")
        logger.info(f"Tilt X range: {self.tilt_x_info.min} to {self.tilt_x_info.max}")
        logger.info(f"Tilt Y range: {self.tilt_y_info.min} to {self.tilt_y_info.max}")

    @staticmethod
    def _normalize(value: int, min_val: int, max_val: int) -> float:
        return min(max((value - min_val) / (max_val - min_val), 0), 1)

    def _map_coordinates(self, x: int, y: int, p: int) -> Tuple[int, int, float]:
        norm_x = self._normalize(x, self.x_info.min, self.x_info.max)
        norm_y = self._normalize(y, self.y_info.min, self.y_info.max)
        norm_p = self._normalize(p, self.p_info.min, self.p_info.max)
        mapped_x = int(norm_x * (self.canvas_size[0] - 1))
        mapped_y = int(norm_y * (self.canvas_size[1] - 1))
        intensity = norm_p
        return mapped_x, mapped_y, intensity

    def draw_point(self, x: int, y: int, intensity: float) -> None:
        if 0 <= x < self.canvas_size[0] and 0 <= y < self.canvas_size[1]:
            y = self.canvas_size[1] - 1 - y
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    new_x, new_y = x + dx, y + dy
                    if 0 <= new_x < self.canvas_size[0] and 0 <= new_y < self.canvas_size[1]:
                        self.buffer[new_y, new_x] = min(255, int((1.0 - intensity) * 255))

    def capture(self, duration: float = _c.DEFAULT_CAPTURE_DURATION) -> None:
        if not self.device:
            logger.error("No tablet device initialized")
            return
        try:
            start_time = time.time()
            for event in self.device.read_loop():
                if time.time() - start_time > duration:
                    logger.info(f"Time limit ({duration} seconds) reached")
                    break
                if event.type == evdev.ecodes.EV_ABS:
                    self._handle_event(event)
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        finally:
            self.display_result()

    def _handle_event(self, event: evdev.events.InputEvent) -> None:
        if event.code == evdev.ecodes.ABS_X:
            self.state['x'] = event.value
        elif event.code == evdev.ecodes.ABS_Y:
            self.state['y'] = event.value
        elif event.code == evdev.ecodes.ABS_TILT_X:
            self.state['tilt_x'] = event.value
        elif event.code == evdev.ecodes.ABS_TILT_Y:
            self.state['tilt_y'] = event.value
        elif event.code == evdev.ecodes.ABS_PRESSURE:
            self.state['pressure'] = event.value
            if self.state['pressure'] > 0:
                x, y, intensity = self._map_coordinates(
                    self.state['x'], self.state['y'], self.state['pressure'])
                self.draw_point(x, y, intensity)
                self.step_count += 1
                if self.step_count >= self.max_steps:
                    logger.info("Maximum steps reached")
                    raise KeyboardInterrupt

    def display_result(self) -> None:
        logger.info(f"Buffer size: {self.buffer.shape}")
        logger.info(f"Buffer type: {self.buffer.dtype}")
        logger.info(f"Buffer min: {self.buffer.min()}")
        logger.info(f"Buffer max: {self.buffer.max()}")
        plt.switch_backend('Agg')
        plt.imsave('tablet_output.png', self.buffer, cmap='Greys')
        logger.info("Saved output to tablet_output.png")

    @staticmethod
    def test() -> None:
        tablet = Tablet()
        if tablet.device:
            logger.info("Starting tablet capture for 5 seconds...")
            tablet.capture(duration=5.0)
            logger.info("Tablet test complete - check tablet_output.png")
        else:
            logger.error("Tablet test failed - no device found")

def main(args: Args) -> None:
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
    if args.cmd == "test":
        logger.info("Running all hardware tests...")
        Camera.test()
        Robot.test()
        Tablet.test()
    elif args.cmd == "test_camera":
        Camera.test()
    elif args.cmd == "test_robot":
        Robot.test()
    elif args.cmd == "test_tablet":
        Tablet.test()
    elif args.cmd == "reset_robot":
        robot = Robot()
        robot.reset()
    elif args.cmd == "record_home":
        robot = Robot()
        robot.record_home()
    else:
        logger.error(f"Unknown command: {args.cmd}")
        logger.info("Available commands: test, test_camera, test_robot, test_tablet, reset_robot, record_home")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)