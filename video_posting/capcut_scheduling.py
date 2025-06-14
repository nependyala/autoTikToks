"""
Script to open CapCut publish calendar, take a screenshot, and highlight schedule text.
Takes a CSV file as input to get the filename to search for.
"""
import webbrowser
import time
import os
import shutil
from datetime import datetime
import pyautogui
import cv2
import numpy as np
import pytesseract
from PIL import Image
import random
import pandas as pd
import sys
from helper_functions.screenshot_conversions import screenshot_to_screen_coords

def clear_temp_folder(temp_dir, processing_dir=None):
    """
    Clear all files in the temporary folder.
    If processing_dir is provided, only clear that subfolder.
    """
    target_dir = processing_dir if processing_dir else temp_dir
    if os.path.exists(target_dir):
        for filename in os.listdir(target_dir):
            file_path = os.path.join(target_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
    print(f"Cleared {'processing' if processing_dir else 'temporary'} files folder")

def create_temp_folder():
    """Create temporary_files folder and processing subfolder if they don't exist"""
    temp_dir = os.path.join(os.path.dirname(__file__), 'temporary_files')
    processing_dir = os.path.join(temp_dir, 'processing')
    
    # Create both directories
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(processing_dir, exist_ok=True)
    
    print(f"Temporary files folder created/verified at: {temp_dir}")
    print(f"Processing folder created/verified at: {processing_dir}")
    
    return temp_dir, processing_dir

def check_for_error(image_path):
    """
    Check if the screenshot contains error messages like 503 or 'unable to load'.
    Returns True if error is found, False otherwise.
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return True  # Assume error if we can't read the image
    
    # Convert to RGB for pytesseract
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Get all text from the image
    data = pytesseract.image_to_data(img_rgb, output_type=pytesseract.Output.DICT)
    
    # Combine all text into one string
    all_text = ' '.join([text.lower() for text in data['text'] if text.strip()])
    
    # Check for various error messages
    error_phrases = [
        '503',  # Service Unavailable
        'service unavailable',
        'unable to load',
        'this page isn\'t working',
        'page isn\'t working',
        'isn\'t working',
        'error loading page',
        'failed to load'
    ]
    
    has_error = any(phrase in all_text for phrase in error_phrases)
    if has_error:
        print(f"Found error message in page: {[phrase for phrase in error_phrases if phrase in all_text]}")
    return has_error

def find_and_highlight_text(image_path, search_text, region='right', prioritize_center=False):
    """
    Find text in the specified region of the image, draw a red box around it,
    and return the coordinates of the box for clicking.
    If prioritize_center is True and multiple matches are found, prioritizes the one with right edge closest to screen center.
    Text search is case insensitive.
    
    Args:
        image_path (str): Path to the image file
        search_text (str): Text to search for (case insensitive)
        region (str): 'right' for right half, 'middle' for middle 50% of the screen
        prioritize_center (bool): Whether to prioritize boxes closer to screen center
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return None
    
    # Get image dimensions
    height, width = img.shape[:2]
    screen_center_x = width // 2
    
    # Select region based on parameter
    if region == 'right':
        search_area = img[:, width//2:]
        x_offset = width//2
    elif region == 'middle':
        # Get middle 50% of the screen
        start_x = width//4
        end_x = (width * 3)//4
        search_area = img[:, start_x:end_x]
        x_offset = start_x
    else:
        print(f"Invalid region: {region}")
        return None
    
    # Convert to RGB for pytesseract
    search_area_rgb = cv2.cvtColor(search_area, cv2.COLOR_BGR2RGB)
    
    # Get text data from search area
    data = pytesseract.image_to_data(search_area_rgb, output_type=pytesseract.Output.DICT)
    
    # Convert search text to lowercase for case-insensitive comparison
    search_text_lower = search_text.lower()
    
    # Find search text
    text_boxes = []
    for i, text in enumerate(data['text']):
        # Convert text to lowercase for case-insensitive comparison
        text_lower = text.lower()
        if search_text_lower in text_lower:
            x = data['left'][i] + x_offset  # Adjust x coordinate for region
            y = data['top'][i]
            w = data['width'][i]
            h = data['height'][i]
            
            if prioritize_center:
                # Calculate distance of right edge to screen center
                right_edge = x + w
                distance_to_center = abs(right_edge - screen_center_x)
                text_boxes.append((x, y, w, h, distance_to_center))
            else:
                text_boxes.append((x, y, w, h))
    
    if not text_boxes:
        print(f"No '{search_text}' text found in the {region} region (case insensitive search)")
        return None
    
    if prioritize_center:
        # Sort boxes by distance to center (closest first)
        text_boxes.sort(key=lambda box: box[4])
        print(f"Found {len(text_boxes)} instances of '{search_text}' text")
        print(f"Selected box with right edge {text_boxes[0][4]} pixels from screen center")
        
        # Draw red boxes around all found text
        for i, (x, y, w, h, _) in enumerate(text_boxes):
            # Make the closest box to center a thicker red line
            thickness = 3 if i == 0 else 2
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), thickness)
        
        # Return the box closest to center (without the distance value)
        click_box = text_boxes[0][:4]
    else:
        # Draw red boxes around found text and store the first box for clicking
        click_box = None
        for i, (x, y, w, h) in enumerate(text_boxes):
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            if i == 0:  # Store the first box for clicking
                click_box = (x, y, w, h)
        print(f"Found {len(text_boxes)} instances of '{search_text}' text")
    
    # Save the annotated image with the original filename
    cv2.imwrite(image_path, img)
    print(f"Saved annotated image as: {image_path}")
    
    return click_box

def get_random_point_in_box(box):
    """Get a random point inside a box (x, y, width, height)"""
    x, y, w, h = box
    # Add some padding to avoid clicking exactly on the edge
    padding = 5
    random_x = random.randint(x + padding, x + w - padding)
    random_y = random.randint(y + padding, y + h - padding)
    return random_x, random_y

def take_screenshot(processing_dir, filename_prefix='screenshot'):
    """Take a screenshot and save it to the processing directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(processing_dir, f'{filename_prefix}_{timestamp}.png')
    screenshot = pyautogui.screenshot()
    screenshot.save(filename)
    print(f"Took screenshot: {filename}")
    return filename

def take_screenshot_and_click(processing_dir, search_text, region='right', filename_prefix='screenshot', prioritize_center=False):
    """
    Take a screenshot, find specified text, and click inside the box
    
    Args:
        processing_dir (str): Directory to save screenshots
        search_text (str): Text to search for
        region (str): 'right' or 'middle' region to search in
        filename_prefix (str): Prefix for the screenshot filename
        prioritize_center (bool): Whether to prioritize boxes closer to screen center
    """
    # Take and save screenshot
    filename = take_screenshot(processing_dir, filename_prefix)
    
    # Find and highlight text, get the box coordinates
    click_box = find_and_highlight_text(filename, search_text, region, prioritize_center)
    
    if click_box:
        # Get a random point inside the box
        screenshot_x, screenshot_y = get_random_point_in_box(click_box)
        print(f"Selected point in screenshot: ({screenshot_x}, {screenshot_y})")
        
        # Convert to screen coordinates
        screen_x, screen_y = screenshot_to_screen_coords(filename, screenshot_x, screenshot_y)
        print(f"Converted to screen coordinates: ({screen_x}, {screen_y})")
        
        # Move mouse and click
        print("Moving mouse to position and clicking...")
        pyautogui.moveTo(screen_x, screen_y, duration=0.5)  # Smooth movement
        pyautogui.click()
        print("Click completed")
        return True
    else:
        print(f"Could not find '{search_text}' text to click")
        return False

def close_current_tab():
    """Close the current browser tab"""
    print("Closing current tab...")
    pyautogui.hotkey('command', 'w')  # Use 'ctrl' instead of 'command' on Windows
    time.sleep(1)  # Wait for tab to close

def try_load_schedule_page(processing_dir, max_attempts=3):
    """
    Try to load the schedule page, checking for errors and reloading if necessary.
    Returns True if successful, False if max attempts reached.
    """
    publish_calendar_url = "https://www.capcut.com/publish-calendar?from_page=work_space&start_tab=video&enter_from=page_header"
    
    for attempt in range(max_attempts):
        print(f"\nAttempt {attempt + 1} of {max_attempts} to load schedule page...")
        
        if attempt > 0:
            # Close the previous tab if this isn't the first attempt
            close_current_tab()
        
        # Open new tab
        webbrowser.open_new_tab(publish_calendar_url)
        print("Opening CapCut publish calendar...")
        
        # Wait for page to load
        print("Waiting for page to load...")
        time.sleep(3)
        
        # Take screenshot to check for errors
        error_check_filename = take_screenshot(processing_dir, 'error_check')
        
        # Check for error
        if check_for_error(error_check_filename):
            print("Found error on page")
            if attempt < max_attempts - 1:
                print("Will try again with a new tab...")
                continue
            else:
                print("Max attempts reached. Giving up.")
                return False
        else:
            print("Page loaded successfully!")
            return True
    
    return False

def get_filename_from_csv(csv_path):
    """
    Read the first entry from a CSV file and return just the filename (not full path),
    truncated to 15 characters, and the platform. The CSV should have 'filepath' and 'platform' columns.
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # Check if required columns exist
        required_columns = ['filepath', 'platform']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"CSV file is missing required columns: {missing_columns}")
        
        # Get the first entry's filepath and platform
        if len(df) == 0:
            raise ValueError("CSV file is empty")
        
        full_path = df.iloc[0]['filepath']
        platform = df.iloc[0]['platform']
        
        # Extract just the filename from the full path
        filename = os.path.basename(full_path)
        
        # Truncate to first 15 characters
        truncated_filename = filename[:15]
        
        print(f"Found filename to search for: {filename}")
        print(f"Using truncated filename (first 15 chars): {truncated_filename}")
        print(f"Found platform: {platform}")
        return truncated_filename, platform
        
    except Exception as e:
        print(f"Error reading CSV file: {str(e)}")
        sys.exit(1)

def type_text(text):
    """Type the given text with a small delay between characters"""
    print(f"Typing: {text}")
    # Add a small delay before starting to type
    time.sleep(0.5)
    # Type each character with a small delay
    for char in text:
        pyautogui.write(char)
        time.sleep(0.1)  # 100ms delay between characters
    print("Finished typing")

def main():
    # Check if CSV file path is provided
    if len(sys.argv) != 2:
        print("Usage: python3 capcut_scheduling.py <schedule_csv_path>")
        print("Example: python3 capcut_scheduling.py temporary_files/schedules/schedule.csv")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    filename_to_search, platform_to_search = get_filename_from_csv(csv_path)
    
    # Create temporary files folder and processing subfolder
    temp_dir, processing_dir = create_temp_folder()
    
    # Clear only the processing folder first
    clear_temp_folder(temp_dir, processing_dir)
    
    # Try to load the schedule page
    if try_load_schedule_page(processing_dir):
        # First step: Find and click 'schedule' in right half
        print("\nStep 1: Looking for 'schedule' text...")
        if take_screenshot_and_click(processing_dir, 'schedule', 'right', 'schedule_screenshot'):
            # Wait a second after clicking schedule
            print("Waiting 1 second after clicking schedule...")
            time.sleep(1)
            
            # Second step: Find and click 'upload' in middle region
            print("\nStep 2: Looking for 'upload' text...")
            if take_screenshot_and_click(processing_dir, 'upload', 'middle', 'upload_screenshot'):
                # Wait a moment after clicking upload
                print("Waiting 1 second after clicking upload...")
                time.sleep(1)
                
                # Take final screenshot after upload click
                print("\nTaking final screenshot after upload...")
                post_upload_screenshot = take_screenshot(processing_dir, 'post_upload_screenshot')
                
                # Third step: Try to find and click 'Search' in right half
                print("\nStep 3: Looking for 'Search' text...")
                search_found = take_screenshot_and_click(processing_dir, 'Search', 'right', 'search_screenshot')
                
                if search_found:
                    print("Successfully clicked Search button")
                    # Wait a moment for the search field to be ready
                    time.sleep(0.5)
                    
                    # Fourth step: Type the filename
                    print("\nStep 4: Typing filename...")
                    type_text(filename_to_search)
                    
                    # Wait a moment after typing
                    print("Waiting 1 second after typing...")
                    time.sleep(1)
                    
                    # Take screenshot after typing
                    print("\nTaking screenshot after typing...")
                    post_typing_screenshot = take_screenshot(processing_dir, 'post_typing_screenshot')
                else:
                    print("Search text not found, skipping search and typing steps")
                    # Use the post-upload screenshot for filename search
                    post_typing_screenshot = post_upload_screenshot
                
                # Fifth step: Find and click the filename in middle region
                # This is the only search that uses center prioritization
                print("\nStep 5: Looking for filename in middle region...")
                if take_screenshot_and_click(processing_dir, filename_to_search, 'middle', 'filename_click_screenshot', prioritize_center=True):
                    print("Successfully clicked on filename")
                    
                    # Sixth step: Try to find and click 'Open' in right half, or press Enter if not found
                    print("\nStep 6: Looking for 'Open' text in right half...")
                    if take_screenshot_and_click(processing_dir, 'Open', 'right', 'open_click_screenshot'):
                        print("Successfully clicked Open button")
                    else:
                        print("Open button not found, pressing Enter instead...")
                        time.sleep(0.5)  # Wait a moment before pressing Enter
                        pyautogui.press('enter')
                        print("Pressed Enter")
                    
                    # Wait a moment for the platform selection to appear
                    print("Waiting 1 second for platform selection...")
                    time.sleep(1)
                    
                    # Seventh step: Take screenshot and look for platform
                    print(f"\nStep 7: Looking for platform '{platform_to_search}' in right region...")
                    if take_screenshot_and_click(processing_dir, platform_to_search, 'right', 'platform_click_screenshot'):
                        print(f"Successfully clicked {platform_to_search} platform")
                    else:
                        print(f"Failed to find {platform_to_search} platform")
                else:
                    print("Failed to find filename in search results")
            else:
                print("Failed to find 'upload' text, stopping process")
        else:
            print("Failed to find 'schedule' text, stopping process")
    else:
        print("Failed to load schedule page after multiple attempts")
        # Close the tab if we failed to load
        close_current_tab()
        sys.exit(1)

if __name__ == "__main__":
    main() 