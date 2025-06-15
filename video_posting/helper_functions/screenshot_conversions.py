"""
Helper functions to convert coordinates between screenshot space and actual screen space.
This is useful when you have coordinates from a screenshot and need to find the corresponding
position on the actual screen for mouse movements.
"""
import pyautogui
import numpy as np
from PIL import Image

def get_screen_size():
    """Get the current screen resolution"""
    width, height = pyautogui.size()
    return width, height

def get_screenshot_size(screenshot_path):
    """Get the dimensions of a screenshot"""
    with Image.open(screenshot_path) as img:
        width, height = img.size
    return width, height

def screenshot_to_screen_coords(screenshot_path, screenshot_x, screenshot_y):
    """
    Convert coordinates from screenshot space to actual screen space.
    
    Args:
        screenshot_path (str): Path to the screenshot file
        screenshot_x (int): X coordinate from the screenshot
        screenshot_y (int): Y coordinate from the screenshot
    
    Returns:
        tuple: (screen_x, screen_y) coordinates that correspond to the actual screen position
    """
    # Get screen and screenshot dimensions
    screen_width, screen_height = get_screen_size()
    screenshot_width, screenshot_height = get_screenshot_size(screenshot_path)
    
    # Calculate scaling factors
    scale_x = screen_width / screenshot_width
    scale_y = screen_height / screenshot_height
    
    # Convert coordinates
    screen_x = int(screenshot_x * scale_x)
    screen_y = int(screenshot_y * scale_y)
    
    return screen_x, screen_y

def screen_to_screenshot_coords(screenshot_path, screen_x, screen_y):
    """
    Convert coordinates from actual screen space to screenshot space.
    
    Args:
        screenshot_path (str): Path to the screenshot file
        screen_x (int): X coordinate from the actual screen
        screen_y (int): Y coordinate from the actual screen
    
    Returns:
        tuple: (screenshot_x, screenshot_y) coordinates that correspond to the screenshot position
    """
    # Get screen and screenshot dimensions
    screen_width, screen_height = get_screen_size()
    screenshot_width, screenshot_height = get_screenshot_size(screenshot_path)
    
    # Calculate scaling factors
    scale_x = screenshot_width / screen_width
    scale_y = screenshot_height / screen_height
    
    # Convert coordinates
    screenshot_x = int(screen_x * scale_x)
    screenshot_y = int(screen_y * scale_y)
    
    return screenshot_x, screenshot_y

def verify_coordinate_conversion(screenshot_path, screenshot_x, screenshot_y):
    """
    Verify the coordinate conversion by converting back and forth.
    This helps ensure the conversion is working correctly.
    
    Args:
        screenshot_path (str): Path to the screenshot file
        screenshot_x (int): X coordinate from the screenshot
        screenshot_y (int): Y coordinate from the screenshot
    
    Returns:
        bool: True if the conversion is accurate (within 1 pixel), False otherwise
    """
    # Convert to screen coordinates
    screen_x, screen_y = screenshot_to_screen_coords(screenshot_path, screenshot_x, screenshot_y)
    
    # Convert back to screenshot coordinates
    back_x, back_y = screen_to_screenshot_coords(screenshot_path, screen_x, screen_y)
    
    # Check if we're within 1 pixel of the original coordinates
    x_diff = abs(screenshot_x - back_x)
    y_diff = abs(screenshot_y - back_y)
    
    print(f"Original screenshot coordinates: ({screenshot_x}, {screenshot_y})")
    print(f"Converted to screen coordinates: ({screen_x}, {screen_y})")
    print(f"Converted back to screenshot coordinates: ({back_x}, {back_y})")
    print(f"Difference: ({x_diff}, {y_diff}) pixels")
    
    return x_diff <= 1 and y_diff <= 1

# Example usage:
if __name__ == "__main__":
    # Example screenshot path (replace with actual path)
    screenshot_path = "../temporary_files/calendar_screenshot_20240321_123456.png"
    
    # Example coordinates from screenshot (replace with actual coordinates)
    screenshot_x = 100
    screenshot_y = 200
    
    # Convert coordinates
    screen_x, screen_y = screenshot_to_screen_coords(screenshot_path, screenshot_x, screenshot_y)
    print(f"\nConverting coordinates:")
    print(f"Screenshot coordinates ({screenshot_x}, {screenshot_y})")
    print(f"correspond to screen coordinates ({screen_x}, {screen_y})")
    
    # Verify the conversion
    print("\nVerifying conversion:")
    is_accurate = verify_coordinate_conversion(screenshot_path, screenshot_x, screenshot_y)
    print(f"Conversion is {'accurate' if is_accurate else 'inaccurate'}") 