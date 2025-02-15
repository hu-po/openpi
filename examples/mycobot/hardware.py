import logging
import cv2
import evdev
import numpy as np
import matplotlib.pyplot as plt
import time
import threading
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

class Raw:
    def __init__(self, stream):
        self.stream = stream
        self.fd = self.stream.fileno()
    def __enter__(self):
        self.original_stty = termios.tcgetattr(self.fd)
        tty.setraw(self.fd)
    def __exit__(self, type_, value, traceback):
        termios.tcsetattr(self.fd, termios.TCSANOW, self.original_stty)

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

class Tablet:
    def __init__(
        self,
        device_name: str = _c.TABLET_DEVICE_NAME,
        canvas_size_tabletspace: Tuple[int, int] = _c.TABLET_CANVAS_SIZE_TABLETSPACE,
        canvas_size_pixelspace: Tuple[int, int] = _c.TABLET_CANVAS_SIZE_PIXELSPACE,
        max_steps: int = _c.TABLET_MAX_STEPS,
    ) -> None:
        self.canvas_size_tabletspace = canvas_size_tabletspace
        self.canvas_size_pixelspace = canvas_size_pixelspace
        self.max_steps = max_steps
        self.canvas = np.zeros(self.canvas_size_pixelspace, dtype=np.uint8)
        self.state: Dict[str, int] = {
            "x": 0, "y": 0, "pressure": 0, "tilt_x": 0, "tilt_y": 0
        }
        self.device: Optional[evdev.InputDevice] = None
        if not self._open_device(device_name):
            raise RuntimeError("No tablet device found")
        logger.info("Tablet state initialized: %s", self.state)

    def _open_device(self, device_name: str) -> bool:
        devices = evdev.list_devices()
        if not devices:
            logger.error("No input devices found")
            return False

        logger.info("Available input devices:")
        for path in devices:
            try:
                dev = evdev.InputDevice(path)
                logger.info(f"  {dev.path}: {dev.name}")
                if device_name in dev.name:
                    self.device = dev
                    logger.info(f"✏️ Connected to {dev.name} at {dev.path}")
                    break
            except (PermissionError, OSError) as e:
                logger.error(f"Permission denied for device {path}: {e}")

        if not self.device:
            logger.error(f"Wacom pen device '{device_name}' not found")
            return False

        # Log capabilities
        caps = dict(self.device.capabilities()[evdev.ecodes.EV_ABS])
        self.x_info = caps[evdev.ecodes.ABS_X]
        self.y_info = caps[evdev.ecodes.ABS_Y]
        self.p_info = caps[evdev.ecodes.ABS_PRESSURE]
        logger.info(f"X range: {self.x_info.min} to {self.x_info.max}")
        logger.info(f"Y range: {self.y_info.min} to {self.y_info.max}")
        logger.info(f"Pressure range: {self.p_info.min} to {self.p_info.max}")
        return True

    def update(self) -> None:
        """Poll for new events and update state"""
        if not self.device:
            return
        
        try:
            events = self.device.read()
            for event in events:
                if event.type == evdev.ecodes.EV_ABS:
                    if event.code == evdev.ecodes.ABS_X:
                        self.state["x"] = event.value
                        logger.debug(f"X updated: {event.value}")
                    elif event.code == evdev.ecodes.ABS_Y:
                        self.state["y"] = event.value
                        logger.debug(f"Y updated: {event.value}")
                    elif event.code == evdev.ecodes.ABS_PRESSURE:
                        self.state["pressure"] = event.value
                        logger.debug(f"Pressure updated: {event.value}")
        except (OSError, BlockingIOError) as e:
            # Non-blocking read with no data available
            pass

    def print_position(self) -> None:
        """Print current tablet position and pressure information"""
        logger.info(f"Position: ({self.state['x']}, {self.state['y']})")
        logger.info(f"Pressure: {self.state['pressure']}")
        if self.state['pressure'] > 0:
            logger.info("✏️  Pen touching surface")
        else:
            logger.info("✏️  Pen hovering")

    def print_canvas(self) -> None:
        """Print ASCII visualization of the canvas state"""
        h, w = self.canvas.shape
        scale = 255 // 9  # Scale pixel values to 0-9 range
        header = "Canvas State:"
        separator = "-" * w
        logger.info(f"\n{header}")
        logger.info(separator)
        for i in range(h):
            row = []
            for j in range(w):
                val = min(9, self.canvas[i,j] // scale)
                row.append(str(val))
            logger.info("".join(row))

# Test and calibration functions
def test_camera() -> None:
    camera = Camera()
    ret, frame = camera.read()
    logger.info(f"Camera test - frame shape: {frame.shape}")
    cv2.imwrite('camera_test.png', frame)
    logger.info("Saved test frame to camera_test.png")
    camera.release()

def test_robot(speed: int = _c.ROBOT_SPEED, mode: int = _c.ROBOT_MODE) -> None:
    robot = Robot(speed=speed, mode=mode)
    robot.go_home()
    logger.info("Robot test complete")

def calibrate_robot() -> None:
    robot = Robot()
    positions: List[List[float]] = []
    logger.info("Recording up to 5 positions")
    logger.info("Move robot and press SPACE to record each position")
    logger.info("Press q to finish recording")
    robot._robot.release_all_servos() # floppy mode
    
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

def calibrate_canvas() -> None:
    """Guide user through calibrating the TABLET and ROBOT variables in constants.py"""
    logger.info("Starting canvas calibration...")
    robot = Robot()
    tablet = Tablet()

    # Move to home position first
    robot.go_home()
    time.sleep(1)

    # Calibrate origin
    logger.info("\n=== Calibrating Canvas Origin ===")
    logger.info("1. Robot will enter floppy mode")
    logger.info("2. Move pen to center of canvas")
    logger.info("3. Press SPACE when ready (q to quit)")
    logger.info("4. Touch pen to tablet surface to see position")
    robot._robot.release_all_servos()

    while True:
        robot.print_position()
        print("\nTablet Position:")
        tablet.update()  # Poll for new events
        tablet.print_position()
        
        with Raw(sys.stdin):
            key = sys.stdin.read(1)
            if key == "q":
                logger.info("Calibration aborted")
                return
            elif key == " ":
                # Record positions...
                angles = robot.get_angles()
                if not angles:
                    logger.error("Failed to get robot angles. Try again.")
                    continue
                
                # Record tablet position
                tablet_x = tablet.state["x"]
                tablet_y = tablet.state["y"]
                
                if tablet.state["pressure"] == 0:
                    logger.warning("No pressure detected - pen may not be touching tablet")
                    continue
                    
                logger.info("\nCanvas Origin Calibration Results:")
                logger.info(f"ORIGIN_POSITION = {angles}")
                logger.info(f"TABLET_CANVAS_ORIGIN = ({tablet_x}, {tablet_y})")
                break
        time.sleep(0.1)  # Small delay to prevent CPU spinning

    # Move to recorded origin
    robot.send_angles(angles)
    time.sleep(1)

    # Calibrate X-axis limit
    logger.info("\n=== Calibrating Canvas X-Axis Limit ===")
    logger.info("1. Robot will enter floppy mode")
    logger.info("2. Move pen to right edge of desired canvas area")
    logger.info("3. Press SPACE when ready (q to quit)")
    logger.info("4. Touch pen to tablet surface to see position")
    robot._robot.release_all_servos()

    while True:
        print("\033[2J\033[H")  # Clear screen and move cursor to top
        robot.print_position()
        print("\nTablet Position:")
        tablet.update()
        print(f"\nX-axis distance from origin: {abs(tablet.state['x'] - tablet_x):.0f}")
        
        with Raw(sys.stdin):
            key = sys.stdin.read(1)
            if key == "q":
                logger.info("Calibration aborted")
                return
            elif key == " ":
                if tablet.state["pressure"] == 0:
                    logger.warning("No pressure detected - pen may not be touching tablet")
                    continue
                    
                x_limit = abs(tablet.state["x"] - tablet_x)
                logger.info(f"\nX-axis size in tablet space: {x_limit}")
                break
        time.sleep(0.1)

    # Move back to origin
    robot.send_angles(angles)
    time.sleep(1)

    # Calibrate Y-axis limit
    logger.info("\n=== Calibrating Canvas Y-Axis Limit ===")
    logger.info("1. Robot will enter floppy mode")
    logger.info("2. Move pen to top edge of desired canvas area")
    logger.info("3. Press SPACE when ready (q to quit)")
    logger.info("4. Touch pen to tablet surface to see position")
    robot._robot.release_all_servos()

    while True:
        print("\033[2J\033[H")  # Clear screen and move cursor to top
        robot.print_position()
        print("\nTablet Position:")
        tablet.update()
        print(f"\nY-axis distance from origin: {abs(tablet.state['y'] - tablet_y):.0f}")
        
        with Raw(sys.stdin):
            key = sys.stdin.read(1)
            if key == "q":
                logger.info("Calibration aborted")
                return
            elif key == " ":
                if tablet.state["pressure"] == 0:
                    logger.warning("No pressure detected - pen may not be touching tablet")
                    continue
                    
                y_limit = abs(tablet.state["y"] - tablet_y)
                logger.info(f"\nY-axis size in tablet space: {y_limit}")
                break
        time.sleep(0.1)

    # Final results
    logger.info("\n=== Calibration Results ===")
    logger.info("Add these values to constants.py:")
    logger.info(f"ORIGIN_POSITION = {angles}")
    logger.info(f"TABLET_CANVAS_ORIGIN = ({tablet_x}, {tablet_y})")
    logger.info(f"TABLET_CANVAS_SIZE_TABLETSPACE = ({x_limit}, {y_limit})")

    # Return to origin
    robot.send_angles(angles)

def test_canvas() -> None:
    """Test canvas calibration by rastering across the tablet surface"""
    logger.info("Starting canvas test...")
    robot = Robot()
    tablet = Tablet()
    # Move to origin
    logger.info("Moving to canvas origin...")
    robot.send_angles(_c.ORIGIN_POSITION)
    time.sleep(1)

    # Get current coordinates
    coords = robot._robot.get_coords()
    if not coords:
        logger.error("Failed to get robot coordinates")
        return

    # Calculate grid points
    num_points = 5  # 5x5 grid
    x_spacing = _c.TABLET_CANVAS_SIZE_TABLETSPACE[0] / (num_points - 1)
    y_spacing = _c.TABLET_CANVAS_SIZE_TABLETSPACE[1] / (num_points - 1)

    logger.info(f"Testing {num_points}x{num_points} grid points...")
    
    def wait_for_stable_reading(timeout: float = 2.0) -> Tuple[int, int]:
        """Wait for stable tablet reading with timeout"""
        start_time = time.time()
        last_x, last_y = 0, 0
        stable_count = 0
        
        while time.time() - start_time < timeout:
            tablet.update()
            if tablet.state["pressure"] > 0:
                current_x, current_y = tablet.state["x"], tablet.state["y"]
                if abs(current_x - last_x) < 10 and abs(current_y - last_y) < 10:
                    stable_count += 1
                    if stable_count > 5:  # Require 5 stable readings
                        return current_x, current_y
                else:
                    stable_count = 0
                last_x, last_y = current_x, current_y
            time.sleep(0.05)
        return 0, 0  # Return zeros if no stable reading

    # Raster pattern
    for i in range(num_points):
        for j in range(num_points):
            target_x = _c.TABLET_CANVAS_ORIGIN[0] + (j * x_spacing)
            target_y = _c.TABLET_CANVAS_ORIGIN[1] + (i * y_spacing)
            new_coords = coords.copy()
            new_coords[0] += j * x_spacing * _c.ROBOT_SCALE / _c.TABLET_CANVAS_SIZE_TABLETSPACE[0]
            new_coords[1] += i * y_spacing * _c.ROBOT_SCALE / _c.TABLET_CANVAS_SIZE_TABLETSPACE[1]
            robot.send_coords(new_coords)
            actual_x, actual_y = wait_for_stable_reading()
            logger.info(f"Grid point ({i},{j})")
            logger.info(f"Target tablet pos: ({target_x:.1f}, {target_y:.1f})")
            logger.info(f"Actual tablet pos: ({actual_x:.1f}, {actual_y:.1f})")
            logger.info(f"Pressure: {tablet.state['pressure']}")
            
    # Return to origin
    robot.send_angles(_c.ORIGIN_POSITION)
    logger.info("Canvas test complete")

def test_tablet() -> None:
    logger.info("Starting tablet test...")
    robot = Robot()
    tablet = Tablet()
    
    logger.info("1. Robot will enter floppy mode")
    logger.info("2. Move pen around tablet surface")
    logger.info("3. Press SPACE to show current canvas")
    logger.info("4. Press q to quit")
    robot.go_home()
    robot._robot.release_all_servos()

    while True:
        print("\033[2J\033[H", end="")  # Keep this print for screen clearing
        
        logger.info("=== Robot Status ===")
        robot.print_position()
        
        logger.info("=== Tablet Status ===")
        tablet.update()
        tablet.print_position()
        
        with Raw(sys.stdin):
            key = sys.stdin.read(1)
            if key == "q":
                logger.info("Test complete")
                return
            elif key == " ":
                tablet.print_canvas()
        
        time.sleep(0.1)

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
    elif args.cmd == "calibrate_robot":
        calibrate_robot()
    elif args.cmd == "calibrate_zero":
        calibrate_zero()
    elif args.cmd == "square":
        square(scale=args.scale, speed=args.speed, mode=args.mode)
    elif args.cmd == "calibrate_canvas":
        calibrate_canvas()
    elif args.cmd == "test_canvas":
        test_canvas()
    else:
        logger.error(f"Unknown command: {args.cmd}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    args = tyro.cli(Args)
    main(args)