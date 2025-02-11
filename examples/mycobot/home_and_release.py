import logging
from typing import Final
from pymycobot.mycobot import MyCobot

logging.basicConfig(level=logging.INFO)
logger: Final = logging.getLogger(__name__)

def home_and_release(mycobot: MyCobot) -> None:
    home_angles: Final = [0, 0, 0, 0, 0, 0]
    logger.info("Moving to home position...")
    mycobot.send_angles(home_angles, 50)  # slower speed for safety
    logger.info("Releasing servos...")
    mycobot.release_all_servos()
    logger.info("Done")

if __name__ == "__main__":
    mycobot = MyCobot("/dev/ttyAMA0", 1000000)
    home_and_release(mycobot) 