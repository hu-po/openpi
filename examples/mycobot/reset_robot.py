import logging
from typing import Final
from pymycobot.mycobot import MyCobot

from examples.mycobot import constants as _c

logging.basicConfig(level=logging.INFO)
logger: Final = logging.getLogger(__name__)

def reset_robot(mycobot: MyCobot) -> None:
    home_angles: Final = _c.DEFAULT_RESET_POSITION
    logger.info("Moving to home position...")
    mycobot.send_angles(home_angles, 50)  # slower speed for safety
    logger.info("Releasing servos...")
    mycobot.release_all_servos()
    logger.info("Done")

if __name__ == "__main__":
    mycobot = MyCobot(
        port=_c.DEFAULT_PORT,
        baudrate=_c.DEFAULT_BAUDRATE,
    )
    reset_robot(mycobot)
