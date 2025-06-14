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
import re

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

def take_screenshot_and_click(processing_dir, search_text, region='right', filename_prefix='screenshot', prioritize_center=False, max_attempts=5):
    """
    Take a screenshot, find specified text, and click inside the box.
    Will retry up to max_attempts times if text is not found.
    
    Args:
        processing_dir (str): Directory to save screenshots
        search_text (str): Text to search for
        region (str): 'right' or 'middle' region to search in
        filename_prefix (str): Prefix for the screenshot filename
        prioritize_center (bool): Whether to prioritize boxes closer to screen center
        max_attempts (int): Maximum number of attempts to find and click the text
    """
    for attempt in range(max_attempts):
        print(f"\nAttempt {attempt + 1} of {max_attempts} to find '{search_text}'...")
        
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
            pyautogui.moveTo(screen_x, screen_y, duration=0.125)
            pyautogui.click()
            print("Click completed")
            return True
        else:
            print(f"Could not find '{search_text}' text on attempt {attempt + 1}")
            if attempt < max_attempts - 1:
                print("Retrying immediately...")
            else:
                print(f"Failed to find '{search_text}' after {max_attempts} attempts")
    
    return False

def close_current_tab():
    """Close the current browser tab"""
    print("Closing current tab...")
    pyautogui.hotkey('command', 'w')  # Use 'ctrl' instead of 'command' on Windows
    time.sleep(0.5)  # Wait for tab to close

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
        time.sleep(1.5)
        
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

def extract_date_from_datetime(datetime_str):
    """Extract just the YYYY-MM-DD portion from a datetime string"""
    try:
        # Parse the datetime string
        dt = pd.to_datetime(datetime_str)
        # Return just the date portion in YYYY-MM-DD format
        return dt.strftime('%Y-%m-%d')
    except Exception as e:
        print(f"Error parsing datetime: {str(e)}")
        return None

def extract_time_from_datetime(datetime_str):
    """Extract time from datetime string and format as H:MM AM/PM (no leading zero for hours)"""
    try:
        # Parse the datetime string
        dt = pd.to_datetime(datetime_str)
        # Format as H:MM AM/PM (using %-I instead of %I to remove leading zero)
        return dt.strftime('%-I:%M %p')
    except Exception as e:
        print(f"Error parsing datetime: {str(e)}")
        return None

def get_filename_from_csv(csv_path):
    """
    Read the first entry from a CSV file and return:
    - filename (truncated to 15 chars)
    - platform
    - scheduled_date (just the date portion in YYYY-MM-DD format)
    - scheduled_time (formatted as HH:MM AM/PM)
    The CSV should have 'filepath', 'platform', and 'scheduled_date' columns.
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # Check if required columns exist
        required_columns = ['filepath', 'platform', 'scheduled_date']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"CSV file is missing required columns: {missing_columns}")
        
        # Get the first entry's data
        if len(df) == 0:
            raise ValueError("CSV file is empty")
        
        full_path = df.iloc[0]['filepath']
        platform = df.iloc[0]['platform']
        scheduled_date = extract_date_from_datetime(df.iloc[0]['scheduled_date'])
        scheduled_time = extract_time_from_datetime(df.iloc[0]['scheduled_date'])
        
        if not scheduled_date or not scheduled_time:
            raise ValueError("Could not parse scheduled_date from CSV")
        
        # Extract just the filename from the full path
        filename = os.path.basename(full_path)
        
        # Truncate to first 15 characters
        truncated_filename = filename[:15]
        
        print(f"Found filename to search for: {filename}")
        print(f"Using truncated filename (first 15 chars): {truncated_filename}")
        print(f"Found platform: {platform}")
        print(f"Found scheduled date: {scheduled_date}")
        print(f"Found scheduled time: {scheduled_time}")
        return truncated_filename, platform, scheduled_date, scheduled_time
        
    except Exception as e:
        print(f"Error reading CSV file: {str(e)}")
        sys.exit(1)

def type_text(text):
    """Type the given text with a small delay between characters"""
    print(f"Typing: {text}")
    # Add a small delay before starting to type
    time.sleep(0.25)
    
    # Split the time into parts (HH:MM and AM/PM)
    time_parts = text.split(' ')
    if len(time_parts) == 2:
        # Type the time part (HH:MM)
        for char in time_parts[0]:
            pyautogui.write(char)
            time.sleep(0.05)
        
        # Add a space
        pyautogui.write(' ')
        time.sleep(0.05)
        
        # Type AM/PM with longer delay after
        for char in time_parts[1]:
            pyautogui.write(char)
            time.sleep(0.05)
        
        # Add 1 second delay after typing AM/PM
        print("Waiting 1 second after typing AM/PM...")
        time.sleep(1.0)
    else:
        # If not a time format, just type normally
        for char in text:
            pyautogui.write(char)
            time.sleep(0.05)
        time.sleep(0.25)
    
    print("Finished typing")

def find_and_click_date(image_path, region='right'):
    """
    Find a date in YYYY-MM-DD format in the image, draw a red box around it,
    and click to the right of the DD portion.
    Returns True if successful, False otherwise.
    Also returns the coordinates and dimensions of the date box if found.
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return False, None, None, None, None
    
    # Get image dimensions
    height, width = img.shape[:2]
    
    # Select region based on parameter
    if region == 'right':
        search_area = img[:, width//2:]
        x_offset = width//2
    else:
        print(f"Invalid region: {region}")
        return False, None, None, None, None
    
    # Convert to RGB for pytesseract
    search_area_rgb = cv2.cvtColor(search_area, cv2.COLOR_BGR2RGB)
    
    # Get text data from search area
    data = pytesseract.image_to_data(search_area_rgb, output_type=pytesseract.Output.DICT)
    
    # Find date in YYYY-MM-DD format
    date_pattern = r'\d{4}-\d{2}-\d{2}'
    date_boxes = []
    
    for i, text in enumerate(data['text']):
        if re.search(date_pattern, text):
            x = data['left'][i] + x_offset  # Adjust x coordinate for region
            y = data['top'][i]
            w = data['width'][i]
            h = data['height'][i]
            date_boxes.append((x, y, w, h, text))
    
    if not date_boxes:
        print("No date in YYYY-MM-DD format found in the right region")
        return False, None, None, None, None
    
    # Use the first date found
    x, y, w, h, date_text = date_boxes[0]
    
    # Draw red box around the date
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    # Calculate click position (to the right of DD)
    # Find the position of the last hyphen in the date
    last_hyphen_pos = date_text.rfind('-')
    if last_hyphen_pos == -1:
        print("Could not find hyphen in date text")
        return False, None, None, None, None
    
    # Calculate the width of the text up to the last hyphen
    text_width = w * (last_hyphen_pos + 1) / len(date_text)
    click_x = x + int(text_width) + 5  # 5 pixels to the right of DD
    click_y = y + h//2  # Middle of the height
    
    # Draw a small circle at the click position
    cv2.circle(img, (click_x, click_y), 3, (0, 255, 0), -1)
    
    # Save the annotated image
    cv2.imwrite(image_path, img)
    print(f"Saved annotated image as: {image_path}")
    
    # Convert to screen coordinates and click
    screen_x, screen_y = screenshot_to_screen_coords(image_path, click_x, click_y)
    print(f"Clicking at screen coordinates: ({screen_x}, {screen_y})")
    pyautogui.moveTo(screen_x, screen_y, duration=0.25)
    pyautogui.click()
    return True, y, h, x, w

def find_and_click_time(image_path, scheduled_time, region='right', date_box_y=None, date_box_height=None, date_box_x=None, date_box_width=None, max_attempts=3):
    """
    Click on the right border of the time box, then type the scheduled time.
    """
    if date_box_y is None or date_box_height is None or date_box_x is None or date_box_width is None:
        print("Error: Need date box coordinates to search for time")
        return False
        
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return False
    
    # Get image dimensions
    height, width = img.shape[:2]
    
    # Calculate time box dimensions
    time_box_width = int(date_box_width * 1.5)  # 50% larger
    time_box_height = int(date_box_height * 1.5)  # 50% larger
    
    # Position time box two date box widths to the right of the date box
    x_start = date_box_x + (date_box_width * 2) + 5  # Two date box widths to the right, plus 5px gap
    y_start = date_box_y - (time_box_height - date_box_height) // 2  # Center vertically with date box
    
    # Ensure the box stays within image bounds
    x_start = max(0, x_start)
    y_start = max(0, y_start)
    x_end = min(width, x_start + time_box_width)
    y_end = min(height, y_start + time_box_height)
    
    print(f"\nTime box coordinates:")
    print(f"Top: {y_start}")
    print(f"Bottom: {y_end}")
    print(f"Left: {x_start}")
    print(f"Right: {x_end}")
    print(f"Height: {y_end - y_start}")
    print(f"Width: {x_end - x_start}")
    print(f"Date box: ({date_box_x}, {date_box_y}, {date_box_width}, {date_box_height})")
    
    # Create a copy of the image for visualization
    debug_img = img.copy()
    
    # Draw the time box in blue
    cv2.rectangle(debug_img, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
    
    # Draw the date box in green for reference
    cv2.rectangle(debug_img, (date_box_x, date_box_y), (date_box_x + date_box_width, date_box_y + date_box_height), (0, 255, 0), 2)
    
    # Save the debug visualization
    debug_path = image_path.replace('.png', '_debug.png')
    cv2.imwrite(debug_path, debug_img)
    print(f"\nSaved debug visualization as: {debug_path}")
    
    # Click on the right border of the time box
    click_x = x_end - 2  # 2 pixels from the right edge
    click_y = y_start + (time_box_height // 2)  # Middle of the height
    
    # Draw a small circle at the click position
    cv2.circle(img, (click_x, click_y), 3, (0, 255, 0), -1)
    
    # Save the annotated image
    cv2.imwrite(image_path, img)
    print(f"Saved annotated image as: {image_path}")
    
    # Convert to screen coordinates and click
    screen_x, screen_y = screenshot_to_screen_coords(image_path, click_x, click_y)
    print(f"Clicking at screen coordinates: ({screen_x}, {screen_y})")
    pyautogui.moveTo(screen_x, screen_y, duration=0.125)
    pyautogui.click()
    
    # Wait longer before clearing the field
    time.sleep(0.25)  # Increased from 0.125
    
    # Press backspace 8 times to clear the field
    print("Pressing backspace 8 times to clear field...")
    for _ in range(8):
        pyautogui.press('backspace')
        time.sleep(0.1)  # Increased from 0.05
    
    # Wait after clearing
    time.sleep(0.25)  # Added delay after clearing
    
    # Type the scheduled time (now includes 1s delay after AM/PM)
    print(f"Typing scheduled time: {scheduled_time}")
    type_text(scheduled_time)
    
    # Press Enter to confirm (no need for extra delay since we already waited 1s after AM/PM)
    print("Pressing Enter to confirm time...")
    pyautogui.press('enter')
    
    # Wait after Enter to ensure it's saved
    time.sleep(0.5)
    
    # Scroll down significantly more
    print("Scrolling down to find publish button...")
    time.sleep(0.25)  # Increased from 0.125
    for _ in range(3):  # Scroll 3 times to ensure we get far enough down
        pyautogui.scroll(-1000)  # Negative for down on macOS
        time.sleep(0.1)  # Increased from 0.05
    
    return True

def main():
    # Check if CSV file path is provided
    if len(sys.argv) != 2:
        print("Usage: python3 capcut_scheduling.py <schedule_csv_path>")
        print("Example: python3 capcut_scheduling.py temporary_files/schedules/schedule.csv")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    filename_to_search, platform_to_search, scheduled_date, scheduled_time = get_filename_from_csv(csv_path)
    
    # Create temporary files folder and processing subfolder
    temp_dir, processing_dir = create_temp_folder()
    
    # Clear only the processing folder first
    clear_temp_folder(temp_dir, processing_dir)
    
    # Try to load the schedule page
    if try_load_schedule_page(processing_dir):
        # First step: Find and click 'schedule' in right half
        print("\nStep 1: Looking for 'schedule' text...")
        if not take_screenshot_and_click(processing_dir, 'schedule', 'right', 'schedule_screenshot', max_attempts=5):
            print("Failed to find 'schedule' text after 5 attempts, stopping process")
            close_current_tab()
            sys.exit(1)
            
        # Wait a moment after clicking schedule
        print("Waiting 0.25 seconds after clicking schedule...")
        time.sleep(0.25)
        
        # Second step: Find and click 'upload' in middle region
        print("\nStep 2: Looking for 'upload' text...")
        if not take_screenshot_and_click(processing_dir, 'upload', 'middle', 'upload_screenshot', max_attempts=5):
            print("Failed to find 'upload' text after 5 attempts, stopping process")
            close_current_tab()
            sys.exit(1)
            
        # Wait a moment after clicking upload
        print("Waiting 0.25 seconds after clicking upload...")
        time.sleep(0.25)
        
        # Take final screenshot after upload click
        print("\nTaking final screenshot after upload...")
        post_upload_screenshot = take_screenshot(processing_dir, 'post_upload_screenshot')
        
        # Third step: Try to find and click 'Search' in right half
        print("\nStep 3: Looking for 'Search' text...")
        search_found = take_screenshot_and_click(processing_dir, 'Search', 'right', 'search_screenshot', max_attempts=5)
        
        if search_found:
            print("Successfully clicked Search button")
            # Wait a moment for the search field to be ready
            time.sleep(0.125)
            
            # Fourth step: Type the filename
            print("\nStep 4: Typing filename...")
            type_text(filename_to_search)
            
            # Wait a moment after typing
            print("Waiting 0.25 seconds after typing...")
            time.sleep(0.25)
            
            # Take screenshot after typing
            print("\nTaking screenshot after typing...")
            post_typing_screenshot = take_screenshot(processing_dir, 'post_typing_screenshot')
        else:
            print("Search text not found after 5 attempts, skipping search and typing steps")
            # Use the post-upload screenshot for filename search
            post_typing_screenshot = post_upload_screenshot
        
        # Fifth step: Find and click the filename in middle region
        # This is the only search that uses center prioritization
        print("\nStep 5: Looking for filename in middle region...")
        if not take_screenshot_and_click(processing_dir, filename_to_search, 'middle', 'filename_click_screenshot', prioritize_center=True, max_attempts=5):
            print("Failed to find filename in search results after 5 attempts, stopping process")
            close_current_tab()
            sys.exit(1)
            
        print("Successfully clicked on filename")
        
        # Press Enter to open the file
        print("Pressing Enter to open file...")
        time.sleep(0.125)
        pyautogui.press('enter')
        
        # Wait a moment for the platform selection to appear
        print("Waiting 0.25 seconds for platform selection...")
        time.sleep(0.25)
        
        # Sixth step: Take screenshot and look for platform
        print(f"\nStep 6: Looking for platform '{platform_to_search}' in right region...")
        if not take_screenshot_and_click(processing_dir, platform_to_search, 'right', 'platform_click_screenshot', max_attempts=5):
            print(f"Failed to find {platform_to_search} platform after 5 attempts, stopping process")
            close_current_tab()
            sys.exit(1)
            
        print(f"Successfully clicked {platform_to_search} platform")
        
        # Seventh step: Take screenshot and look for date field
        print("\nStep 7: Looking for date field...")
        date_screenshot = take_screenshot(processing_dir, 'date_screenshot')
        date_found, date_y, date_height, date_x, date_width = find_and_click_date(date_screenshot)
        if date_found:
            print("Successfully clicked date field")
            
            # Wait a moment before clearing the field
            time.sleep(0.125)  # Halved from 0.25
            
            # Select all and delete with a single backspace
            print("Selecting all text and deleting...")
            pyautogui.hotkey('command', 'a')  # Select all
            time.sleep(0.05)  # Halved from 0.1
            pyautogui.press('backspace')  # Delete selection
            
            # Type the scheduled date
            print(f"Typing scheduled date: {scheduled_date}")
            type_text(scheduled_date)
            
            # Press Enter to confirm the date
            print("Pressing Enter to confirm date...")
            time.sleep(0.125)  # Halved from 0.25
            pyautogui.press('enter')
            
            # Wait a moment for the time field to appear
            print("Waiting 0.25 seconds for time field...")  # Halved from 0.5
            time.sleep(0.25)  # Halved from 0.5
            
            # Ninth step: Take screenshot and look for time field
            print("\nStep 9: Looking for time field...")
            time_screenshot = take_screenshot(processing_dir, 'time_screenshot')
            if find_and_click_time(time_screenshot, scheduled_time, date_box_y=date_y, date_box_height=date_height, 
                                 date_box_x=date_x, date_box_width=date_width):
                print("Successfully entered time")
            else:
                print("Failed to enter time")
                close_current_tab()
                sys.exit(1)
        else:
            print("Failed to find date field")
    else:
        print("Failed to load schedule page after multiple attempts")
        # Close the tab if we failed to load
        close_current_tab()
        sys.exit(1)

if __name__ == "__main__":
    main() 