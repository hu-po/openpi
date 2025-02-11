import time
import logging
from typing import List, Tuple
from pymycobot.mycobot import MyCobot
from pymycobot.genre import Coord

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_raster_points(
    x_start: float,
    x_end: float,
    y_start: float,
    y_end: float,
    step_size: float,
    z_height: float
) -> List[Tuple[float, float, float, float, float, float]]:
    """Generate points for raster scan pattern."""
    points = []
    y_current = y_start
    direction = 1  # 1 for left-to-right, -1 for right-to-left
    
    while y_current <= y_end:
        if direction == 1:
            x_current = x_start
            while x_current <= x_end:
                points.append((x_current, y_current, z_height, 0, 0, 0))
                x_current += step_size
        else:
            x_current = x_end
            while x_current >= x_start:
                points.append((x_current, y_current, z_height, 0, 0, 0))
                x_current -= step_size
        
        y_current += step_size
        direction *= -1  # Switch direction
    
    return points

def raster_scan(mycobot: MyCobot, speed: int = 50) -> None:
    """Execute raster scan pattern."""
    # Define scan parameters
    x_start, x_end = -100, 100  # mm
    y_start, y_end = -100, 100  # mm
    step_size = 20  # mm
    z_height = 150  # mm working height
    
    # Generate scan points
    points = generate_raster_points(x_start, x_end, y_start, y_end, step_size, z_height)
    
    # Move to safe starting position
    logger.info("Moving to starting position...")
    mycobot.send_angles([0, 0, 0, 0, 0, 0], speed)
    time.sleep(3)
    
    # Execute raster pattern
    logger.info("Starting raster scan pattern...")
    for i, point in enumerate(points):
        logger.info(f"Moving to point {i+1}/{len(points)}: {point}")
        mycobot.send_coords(list(point), speed, mode=1)  # mode=1 for linear movement
        
        # Wait until motion is complete
        while mycobot.is_moving():
            time.sleep(0.1)
    
    logger.info("Raster scan complete")
    
    # Return to home position
    logger.info("Returning to home position...")
    mycobot.send_angles([0, 0, 0, 0, 0, 0], speed)

if __name__ == "__main__":
    # Initialize robot
    mycobot = MyCobot("/dev/ttyAMA0", 1000000)
    
    try:
        # Power on and set up
        mycobot.power_on()
        time.sleep(1)
        
        # Execute raster scan
        raster_scan(mycobot)
        
    except Exception as e:
        logger.error(f"Error during raster scan: {e}")
        
    finally:
        # Ensure robot is in a safe position
        mycobot.send_angles([0, 0, 0, 0, 0, 0], 50)
        time.sleep(1) 