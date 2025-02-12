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
    cmd: str = "test"
    """Command to run (i.e. test, test_camera, test_robot, test_tablet, sleep, calibrate, calibrate_zero, square)"""
    debug: bool = False
    """Debug mode"""

class Camera:
    def __init__(
        self,
        camera_id: int = _c.CAMERA_ID
    ) -> None:
        self._cam = cv2.VideoCapture(camera_id)
        if not self._cam.isOpened():
            logger.warning("🚫 Camera failed to open; fallback to blank frames")
        logger.info("📸 MyCobotCamera initialized")

    def read(self) -> Tuple[bool, np.ndarray]:
        ret, frame = self._cam.read()
        if not ret:
            logger.warning("Failed to read frame; returning blank frame")
            frame = np.zeros((_c.CAMERA_IMAGE_HEIGHT, _c.CAMERA_IMAGE_WIDTH, 3), dtype=np.uint8)
        return ret, frame

    def release(self) -> None:
        self._cam.release()

def test_camera() -> None:
    camera = Camera()
    ret, frame = camera.read()
    logger.info(f"Camera test - frame shape: {frame.shape}")
    cv2.imwrite('camera_test.png', frame)
    logger.info("Saved test frame to camera_test.png")
    camera.release()

class Robot:
    def __init__(
        self,
        port: str = _c.ROBOT_PORT,
        baudrate: int = _c.ROBOT_BAUDRATE
    ) -> None:
        self._robot = MyCobot(port=port, baudrate=baudrate)
        self._robot.set_color(0, 255, 0)
        logger.info("🤖 MyCobotRobot initialized")

    def get_angles(self) -> List[float]:
        return self._robot.get_angles()

    def send_angles(self, angles: List[float], speed: Union[int, float] = _c.ROBOT_SPEED) -> None:
        self._robot.send_angles(angles, speed)

    def go_home(self) -> None:
        logger.info("Moving to home position...")
        self._robot.set_color(255, 255, 0)
        self._robot.sync_send_angles(_c.HOME_POSITION, _c.ROBOT_SPEED, timeout=_c.ROBOT_MOVE_TIMEOUT)
        self._robot.set_color(0, 255, 0)
        logger.info("Done")

    def go_sleep(self) -> None:
        logger.info("Moving to sleep position...")
        self._robot.send_angles(_c.SLEEP_POSITION, _c.ROBOT_SPEED)
        time.sleep(3)
        logger.info("Releasing servos...")
        self._robot.release_all_servos()
        logger.info("Done")

    def __del__(self) -> None:
        logger.info("Terminating Robot")
        self.go_sleep()
        self._robot.set_color(0, 0, 0)

def test_robot() -> None:
    robot = Robot()
    robot.go_home()
    logger.info("Robot test complete")

def calibrate() -> None:
    robot = Robot()
    positions: List[List[float]] = []
    logger.info("Recording up to 5 positions")
    logger.info("Move robot and press SPACE to record each position")
    logger.info("Press q to finish recording")
    robot._robot.release_all_servos() # floppy mode
    
    class Raw:
        def __init__(self, stream):
            self.stream = stream
            self.fd = self.stream.fileno()
        def __enter__(self):
            self.original_stty = termios.tcgetattr(self.fd)
            tty.setraw(self.fd)
        def __exit__(self, type_, value, traceback):
            termios.tcsetattr(self.fd, termios.TCSANOW, self.original_stty)

    while len(positions) < 5:
        with Raw(sys.stdin):
            key = sys.stdin.read(1)
            if key == "q":
                break
            elif key == " ":
                angles = robot.get_angles()
                if angles:
                    positions.append(angles)
                    logger.info(f"Position {len(positions)} recorded: {angles}")
                else:
                    logger.warning("Failed to get angles, try again")

    if not positions:
        logger.info("No positions recorded")
        return

    logger.info("\nRecorded positions:")
    for i, pos in enumerate(positions, 1):
        pos_str = f"[{', '.join(f'{angle:.2f}' for angle in pos)}]"
        logger.info(f"Position {i}: {pos_str}")

    positions_array = np.array(positions)
    min_angles = positions_array.min(axis=0)
    max_angles = positions_array.max(axis=0)
    
    logger.info("\nJoint limits for constants.py:")
    logger.info("JOINT_LIMITS: dict[str, tuple[float, float]] = {")
    for name, min_ang, max_ang in zip(_c.JOINT_NAMES, min_angles, max_angles):
        logger.info(f'    "{name}": ({min_ang:.2f}, {max_ang:.2f}),')
    logger.info("}")

def calibrate_zero() -> None:
    robot = Robot()
    logger.info("Starting zero-point calibration for all servos")
    logger.info("Move each joint to desired zero position and press SPACE")
    logger.info("Press q to exit calibration")
    robot._robot.release_all_servos()  # floppy mode
    
    class Raw:
        def __init__(self, stream):
            self.stream = stream
            self.fd = self.stream.fileno()
        def __enter__(self):
            self.original_stty = termios.tcgetattr(self.fd)
            tty.setraw(self.fd)
        def __exit__(self, type_, value, traceback):
            termios.tcsetattr(self.fd, termios.TCSANOW, self.original_stty)

    for servo_id in range(1, 7):
        logger.info(f"\nCalibrating servo {servo_id} ({_c.JOINT_NAMES[servo_id-1]})")
        while True:
            with Raw(sys.stdin):
                key = sys.stdin.read(1)
                if key == "q":
                    logger.info("Calibration aborted")
                    return
                elif key == " ":
                    try:
                        robot._robot.set_servo_calibration(servo_id)
                        logger.info(f"Zero point set for servo {servo_id}")
                        break
                    except Exception as e:
                        logger.error(f"Failed to calibrate servo {servo_id}: {e}")
                        continue

    logger.info("\nCalibration complete for all servos")

def square() -> None:
    robot = Robot()
    robot.go_home()
    coords = robot._robot.get_coords()
    logger.info(f"Coords: {coords}")
    # move +10mm in x direction
    robot._robot.send_coord(1, coords[0] + 10, _c.ROBOT_SPEED)
    time.sleep(0.1)
    coords = robot._robot.get_coords()
    logger.info(f"Coords: {coords}")
    # move +10mm in y direction
    robot._robot.send_coord(2, coords[1] + 10, _c.ROBOT_SPEED)
    time.sleep(0.1)
    coords = robot._robot.get_coords()
    logger.info(f"Coords: {coords}")
    # move -10mm in x direction
    robot._robot.send_coord(1, coords[0] - 10, _c.ROBOT_SPEED)
    time.sleep(0.1)
    coords = robot._robot.get_coords()
    logger.info(f"Coords: {coords}")
    # move -10mm in y direction
    robot._robot.send_coord(2, coords[1] - 10, _c.ROBOT_SPEED)
    time.sleep(0.1)
    coords = robot._robot.get_coords()
    logger.info(f"Coords: {coords}")

def spiral(waypoints: int = 100, max_radius: float = 30.0) -> None:
    """Draw a spiral pattern starting from home position.
    
    Args:
        waypoints: Number of points to use for spiral (default: 100)
        max_radius: Maximum radius of spiral in mm (default: 50.0)
    """
    robot = Robot()
    robot.go_home()
    coords = robot._robot.get_coords()
    logger.info(f"Starting spiral from coords: {coords}")
    
    for i in range(waypoints):
        # Calculate angle and radius for this point
        theta = i * 4 * np.pi / waypoints
        radius = (i / waypoints) * max_radius
        
        # Calculate x,y offset using parametric equations
        x_offset = radius * np.cos(theta)
        y_offset = radius * np.sin(theta)
        
        # Move to new position
        robot._robot.send_coord(1, coords[0] + x_offset, _c.ROBOT_SPEED)
        robot._robot.send_coord(2, coords[1] + y_offset, _c.ROBOT_SPEED)
        time.sleep(0.1)
        
        # Log current position
        new_coords = robot._robot.get_coords()
        logger.info(f"Spiral point {i+1}/{waypoints}: {new_coords}")
    
    # Return to start position
    robot.go_home()


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
                    logger.info(f"✏️ Connected to {dev.name} at {dev.path}")
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

def test_tablet() -> None:
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
        test_camera()
        test_robot()
        test_tablet()
    elif args.cmd == "test_camera":
        test_camera()
    elif args.cmd == "test_robot":
        test_robot()
    elif args.cmd == "test_tablet":
        test_tablet()
    elif args.cmd == "sleep":
        robot = Robot()
        del robot
    elif args.cmd == "calibrate":
        calibrate()
    elif args.cmd == "calibrate_zero":
        calibrate_zero()
    elif args.cmd == "square":
        square()
    elif args.cmd == "spiral":
        spiral()
    else:
        logger.error(f"Unknown command: {args.cmd}")
        logger.info("Available commands: test, test_camera, test_robot, test_tablet, sleep, calibrate, calibrate_zero, square, spiral")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    args = tyro.cli(Args)
    main(args)