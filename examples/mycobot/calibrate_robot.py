import time
import logging
from typing import Final, List
import sys
import tty
import termios
from pymycobot.mycobot import MyCobot
from pymycobot.genre import Angle

from examples.mycobot import constants as _c

logging.basicConfig(level=logging.INFO)
logger: Final = logging.getLogger(__name__)

class Raw:
    def __init__(self, stream):
        self.stream = stream
        self.fd = self.stream.fileno()
    def __enter__(self):
        self.original_stty = termios.tcgetattr(self.fd)
        tty.setraw(self.fd)
    def __exit__(self, type_, value, traceback):
        termios.tcsetattr(self.fd, termios.TCSANOW, self.original_stty)

def record_cube_corners() -> None:
    mycobot = MyCobot(
        port=_c.DEFAULT_PORT,
        baudrate=_c.DEFAULT_BAUDRATE,
    )
    corners: List[List[float]] = []
    
    logger.info("First recording home position")
    logger.info("Move robot to home position and press SPACE")
    logger.info("Press q to quit at any time")
    
    mycobot.release_all_servos()
    
    while True:
        with Raw(sys.stdin):
            key = sys.stdin.read(1)
            if key == "q":
                logger.info("Recording cancelled")
                return
            elif key == " ":
                angles = mycobot.get_angles()
                if angles:
                    corners.append(angles)
                    logger.info(f"Recorded home position: {angles}")
                    break
                else:
                    logger.warning("Failed to get angles, try again")
    
    logger.info("Starting cube corner recording sequence")
    logger.info("Robot will be put in free mode. Manually move to each corner and press SPACE")
    logger.info("Press q to quit at any time")
    
    while len(corners) < 9:  # 1 home + 8 corners
        with Raw(sys.stdin):
            key = sys.stdin.read(1)
            if key == "q":
                logger.info("Recording cancelled")
                return
            elif key == " ":
                angles = mycobot.get_angles()
                if angles:
                    corners.append(angles)
                    logger.info(f"Recorded corner {len(corners)-1}/8: {angles}")
                else:
                    logger.warning("Failed to get angles, try again")
    
    logger.info("\nAll corners recorded. Copy this into your constants.py:")
    logger.info("HOME_POSITION = [")
    logger.info(f"    {corners[0]},")
    logger.info("]")
    logger.info("\nCUBE_CORNERS = [")
    for corner in corners[1:]:
        logger.info(f"    {corner},")
    logger.info("]")

if __name__ == "__main__":
    record_cube_corners() 