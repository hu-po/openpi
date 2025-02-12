

import evdev
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Tablet:
    def __init__(self,
                 device_name: str = "Wacom Intuos Pro L Pen",
                 canvas_size: Tuple[int, int] = (1024, 1024),
                 max_steps: int = 1000) -> None:
        self.canvas_size = canvas_size
        self.max_steps = max_steps
        self.buffer = np.zeros(self.canvas_size, dtype=np.uint8)
        self.state: Dict[str, int] = {'x': 0, 'y': 0, 'pressure': 0, 'tilt_x': 0, 'tilt_y': 0}
        self.step_count = 0
        
        # Find and initialize device
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

        # Setup tablet dimensions
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
            # Flip y so that (0,0) is at the bottom-left
            y = self.canvas_size[1] - 1 - y
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    new_x, new_y = x + dx, y + dy
                    if 0 <= new_x < self.canvas_size[0] and 0 <= new_y < self.canvas_size[1]:
                        # Write the value into our "buffer" array
                        self.buffer[new_y, new_x] = min(255, int((1.0 - intensity) * 255))

    def capture(self, duration: float = 5.0) -> None:
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
        plt.switch_backend('Agg')  # Use the most basic backend
        plt.imsave('tablet_output.png', self.buffer, cmap='Greys')
        logger.info("Saved output to tablet_output.png")

if __name__ == "__main__":
    tablet = Tablet()
    tablet.capture(duration=5.0)
