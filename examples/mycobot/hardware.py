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
# Filter out verbose pymycobot debug logs
logging.getLogger("pymycobot.generate").setLevel(logging.WARNING)

@dataclass
class Args:
    cmd: str = "test"
    """Command to run (i.e. test, test_camera, test_robot, test_tablet, sleep, calibrate, calibrate_zero, square)"""
    debug: bool = False
    """Debug mode"""
    speed: int = _c.ROBOT_SPEED
    """Movement speed"""
    mode: int = _c.ROBOT_MODE
    """Movement mode"""
    scale: float = _c.ROBOT_SCALE
    """Scale for movement patterns in mm"""

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
        baudrate: int = _c.ROBOT_BAUDRATE,
        speed: int = _c.ROBOT_SPEED,
        mode: int = _c.ROBOT_MODE
    ) -> None:
        self._robot = MyCobot(port=port, baudrate=baudrate)
        self._robot.set_color(0, 255, 0)
        self._speed = speed
        self._mode = mode
        logger.info("🤖 MyCobotRobot initialized")

    def get_angles(self) -> List[float]:
        return self._robot.get_angles()

    def send_angles(self, angles: List[float], speed: Optional[Union[int, float]] = None) -> None:
        speed = speed or self._speed
        self._robot.send_angles(angles, speed)

    def send_coords(self, coords: List[float], speed: Optional[Union[int, float]] = None, mode: Optional[int] = None) -> None:
        speed = speed or self._speed
        mode = mode or self._mode
        self._robot.sync_send_coords(coords, speed, mode=mode, timeout=_c.ROBOT_MOVE_TIMEOUT)
        
    def go_home(self, speed: Optional[Union[int, float]] = None) -> None:
        logger.info("Moving to home position...")
        speed = speed or self._speed
        self._robot.set_color(255, 255, 0)
        self._robot.sync_send_angles(_c.HOME_POSITION, speed, timeout=_c.ROBOT_MOVE_TIMEOUT)
        self._robot.set_color(0, 255, 0)
        logger.info("Done")

    def go_sleep(self, speed: Optional[Union[int, float]] = None) -> None:
        logger.info("Moving to sleep position...")
        speed = speed or self._speed
        self._robot.send_angles(_c.SLEEP_POSITION, speed)
        time.sleep(3)
        logger.info("Releasing servos...")
        self._robot.release_all_servos()
        logger.info("Done")

    def go_origin(self, speed: Optional[Union[int, float]] = None) -> None:
        logger.info("Moving to origin position...")
        speed = speed or self._speed
        self._robot.set_color(255, 255, 0)
        self._robot.sync_send_angles(_c.ORIGIN_POSITION, speed, timeout=_c.ROBOT_MOVE_TIMEOUT)
        self._robot.set_color(0, 255, 0)
        logger.info("Done")

    def __del__(self) -> None:
        logger.info("Terminating Robot")
        self.go_sleep()
        self._robot.set_color(0, 0, 0)

    def print_position(self) -> None:
        coords = self._robot.get_coords()
        if coords:
            x, y, z, rx, ry, rz = coords
            print(f"🎯 <x, y, z, rx, ry, rz>")
            print(f"  pos<{_c.AXIS_COLORS['x']}{x:.0f}{_c.AXIS_COLORS['reset']}, {_c.AXIS_COLORS['y']}{y:.0f}{_c.AXIS_COLORS['reset']}, {_c.AXIS_COLORS['z']}{z:.0f}{_c.AXIS_COLORS['reset']}> (mm)")
            print(f"  rot<{_c.AXIS_COLORS['x']}{rx:.0f}{_c.AXIS_COLORS['reset']}, {_c.AXIS_COLORS['y']}{ry:.0f}{_c.AXIS_COLORS['reset']}, {_c.AXIS_COLORS['z']}{rz:.0f}{_c.AXIS_COLORS['reset']}> (deg)")
        else:
            print("\033[31m❌ Could not get robot position\033[0m")

def test_robot(speed: int = _c.ROBOT_SPEED, mode: int = _c.ROBOT_MODE) -> None:
    robot = Robot(speed=speed, mode=mode)
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

def square(scale: float = _c.ROBOT_SCALE, speed: int = _c.ROBOT_SPEED, mode: int = _c.ROBOT_MODE) -> None:
    robot = Robot(speed=speed, mode=mode)
    robot.go_home()
    robot.print_position()
    coords = robot._robot.get_coords()
    axis_names = ["x", "y", "z"]
    for axis in [1, 2, 3]:
        axis_name = axis_names[axis-1]
        logger.info(f"Moving +{scale}mm along {_c.AXIS_COLORS[axis_name]}{axis_name}{_c.AXIS_COLORS['reset']}-axis")
        new_coords = coords.copy()
        new_coords[axis-1] += scale
        robot.send_coords(new_coords)
        coords = robot._robot.get_coords()
        robot.print_position()
    for axis in [1, 2, 3]:
        axis_name = axis_names[axis-1]
        logger.info(f"Moving -{scale}mm along {_c.AXIS_COLORS[axis_name]}{axis_name}{_c.AXIS_COLORS['reset']}-axis")
        new_coords = coords.copy()
        new_coords[axis-1] -= scale
        robot.send_coords(new_coords)
        coords = robot._robot.get_coords()
        robot.print_position()

def spiral(scale: float = _c.ROBOT_SCALE, speed: int = _c.ROBOT_SPEED, mode: int = _c.ROBOT_MODE) -> None:
    robot = Robot(speed=speed, mode=mode)
    robot.go_home()
    robot.print_position()
    coords = robot._robot.get_coords()
    # Generate spiral points (4 turns = 8π radians)
    t = np.linspace(0, 8*np.pi, 100)
    radius = scale * t / (8*np.pi)  # Radius grows linearly with angle
    x = coords[0] + radius * np.cos(t)
    y = coords[1] + radius * np.sin(t)
    # Keep z, rx, ry, rz constant from origin position
    for xi, yi in zip(x, y):
        new_coords = [xi, yi, coords[2], coords[3], coords[4], coords[5]]
        robot.send_coords(new_coords)
        robot.print_position()
    
    # Return to origin
    robot.send_coords(coords)
    logger.info("Spiral movement complete")

class Tablet:
    def __init__(
        self,
        device_name: str = _c.TABLET_DEVICE_NAME,
        canvas_size: Tuple[int, int] = _c.TABLET_CANVAS_SIZE,
        max_steps: int = _c.TABLET_MAX_STEPS,
    ) -> None:
        self.canvas_size = canvas_size
        self.max_steps = max_steps
        self.buffer = np.zeros(self.canvas_size, dtype=np.uint8)
        self.state: Dict[str, int] = {
            "x": 0, "y": 0, "pressure": 0, "tilt_x": 0, "tilt_y": 0
        }
        self.step_count = 0
        self.device: Optional[evdev.InputDevice] = None
        self._open_device(device_name)
        self._capture_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def __enter__(self):
        """
        Context manager entrypoint: automatically start background capturing.
        """
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit: stop capturing thread, ensure cleanup.
        """
        self.stop()

    def start(self):
        """
        Manually start background capture if not using 'with Tablet() as t:'.
        """
        if not self.device:
            logger.warning("Cannot start - no device found/initialized.")
            return

        # Clear the stop event (so that the capture loop runs)
        self._stop_event.clear()

        if self._capture_thread is None or not self._capture_thread.is_alive():
            self._capture_thread = threading.Thread(
                target=self._capture_loop,
                name="TabletCaptureThread",
                daemon=True,  # or False, depends on preference
            )
            self._capture_thread.start()
            logger.info("Tablet capture thread started.")

    def stop(self):
        """
        Manually stop background capture if not using the context manager.
        """
        self._stop_event.set()
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=2.0)
            self._capture_thread = None
        logger.info("Tablet capture thread stopped.")
        self.display_result()

    def _open_device(self, device_name: str) -> None:
        """
        Helper to find/open the evdev device matching 'device_name'.
        """
        devices = evdev.list_devices()
        if not devices:
            logger.error("No input devices found.")
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
            logger.error("Wacom pen device not found.")
            return

        caps = dict(self.device.capabilities()[evdev.ecodes.EV_ABS])
        self.x_info = caps[evdev.ecodes.ABS_X]
        self.y_info = caps[evdev.ecodes.ABS_Y]
        self.p_info = caps[evdev.ecodes.ABS_PRESSURE]
        self.tilt_x_info = caps.get(evdev.ecodes.ABS_TILT_X, None)
        self.tilt_y_info = caps.get(evdev.ecodes.ABS_TILT_Y, None)
        
        logger.info(f"X range: {self.x_info.min} to {self.x_info.max}")
        logger.info(f"Y range: {self.y_info.min} to {self.y_info.max}")
        logger.info(f"Pressure range: {self.p_info.min} to {self.p_info.max}")
        if self.tilt_x_info:
            logger.info(f"Tilt X range: {self.tilt_x_info.min} to {self.tilt_x_info.max}")
        if self.tilt_y_info:
            logger.info(f"Tilt Y range: {self.tilt_y_info.min} to {self.tilt_y_info.max}")

    def _capture_loop(self):
        """
        The background thread: reads tablet events in a loop and updates buffer.
        """
        logger.info("Entering tablet capture loop.")
        while not self._stop_event.is_set():
            try:
                # We'll use read() which returns next event or blocks until available
                for event in self.device.read():
                    if self._stop_event.is_set():
                        break
                    if event.type == evdev.ecodes.EV_ABS:
                        self._handle_event(event)
            except OSError as e:
                # read() can raise OSError if device is disconnected, permissions changed, etc.
                logger.error(f"Tablet device read error: {e}")
                break
            except Exception as e:
                logger.error(f"Unexpected error in capture loop: {e}", exc_info=True)
                break

        logger.info("Exiting tablet capture loop.")

    def _handle_event(self, event: evdev.events.InputEvent) -> None:
        # same logic you currently have
        if event.code == evdev.ecodes.ABS_X:
            self.state["x"] = event.value
        elif event.code == evdev.ecodes.ABS_Y:
            self.state["y"] = event.value
        elif event.code == evdev.ecodes.ABS_PRESSURE:
            self.state["pressure"] = event.value
            if self.state["pressure"] > 0:
                x, y, intensity = self._map_coordinates(
                    self.state["x"], self.state["y"], self.state["pressure"]
                )
                self.draw_point(x, y, intensity)
                self.step_count += 1
                if self.step_count >= self.max_steps:
                    logger.info("Maximum steps reached.")
                    # Instead of raising KeyboardInterrupt, let's just stop:
                    self.stop()
        elif event.code == evdev.ecodes.ABS_TILT_X and self.tilt_x_info:
            self.state["tilt_x"] = event.value
        elif event.code == evdev.ecodes.ABS_TILT_Y and self.tilt_y_info:
            self.state["tilt_y"] = event.value

    @staticmethod
    def _normalize(value: int, min_val: int, max_val: int) -> float:
        # same as before
        return min(max((value - min_val) / (max_val - min_val), 0), 1)

    def _map_coordinates(self, x: int, y: int, p: int):
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
                    if (0 <= new_x < self.canvas_size[0]) and (0 <= new_y < self.canvas_size[1]):
                        self.buffer[new_y, new_x] = min(255, int((1.0 - intensity) * 255))

    def display_result(self) -> None:
        plt.switch_backend("Agg")
        plt.imsave("tablet_output.png", self.buffer, cmap="Greys")
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
        test_robot(speed=args.speed, mode=args.mode)
        test_tablet()
    elif args.cmd == "test_camera":
        test_camera()
    elif args.cmd == "test_robot":
        test_robot(speed=args.speed, mode=args.mode)
    elif args.cmd == "test_tablet":
        test_tablet()
    elif args.cmd == "sleep":
        robot = Robot(speed=args.speed, mode=args.mode)
        del robot
    elif args.cmd == "calibrate":
        calibrate()
    elif args.cmd == "calibrate_zero":
        calibrate_zero()
    elif args.cmd == "square":
        square(scale=args.scale, speed=args.speed, mode=args.mode)
    elif args.cmd == "spiral":
        spiral(scale=args.scale, speed=args.speed, mode=args.mode)
    else:
        logger.error(f"Unknown command: {args.cmd}")
        logger.info("Available commands: test, test_camera, test_robot, test_tablet, sleep, calibrate, calibrate_zero, square, spiral")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    args = tyro.cli(Args)
    main(args)