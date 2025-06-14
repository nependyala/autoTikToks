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
    - caption
    The CSV should have 'filepath', 'platform', 'scheduled_date', and 'caption' columns.
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # Check if required columns exist
        required_columns = ['filepath', 'platform', 'scheduled_date', 'caption']
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
        caption = df.iloc[0]['caption']
        
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
        print(f"Found caption: {caption}")
        return truncated_filename, platform, scheduled_date, scheduled_time, caption
        
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

def find_color_boundary(img, start_x, start_y, width, direction='down', threshold=30):
    """
    Find where the color changes significantly in a given direction.
    Returns the y-coordinate where the color change occurs.
    """
    # Get the color at the start point
    start_color = img[start_y, start_x]
    
    # Check a few points across the width to be more robust
    check_points = [start_x + (width * i // 4) for i in range(5)]  # 5 points across the width
    
    for y in range(start_y, img.shape[0] if direction == 'down' else 0, 1 if direction == 'down' else -1):
        # Check if any of our points have a significant color change
        for x in check_points:
            if x >= img.shape[1]:
                continue
            current_color = img[y, x]
            # Calculate color difference
            color_diff = np.abs(current_color.astype(int) - start_color.astype(int))
            if np.mean(color_diff) > threshold:
                return y
    
    return None

def find_and_click_lower_title(image_path, region='middle', caption=None):
    """
    Find "Title" text in the middle 50% of the screen horizontally and middle 50% vertically,
    draw red boxes around all instances, and click randomly in the lower instance.
    If only one Title is found, click halfway between Title and "Who can view this video",
    then type the caption, click between who and ensure, press Return, wait, press Return again,
    then find and highlight "comment", "allow", and "disclose", and click at a point that connects all three in a T-shape.
    Returns True if successful, False otherwise.
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return False
    
    # Get image dimensions
    height, width = img.shape[:2]
    
    # Calculate search area boundaries
    # Horizontal: middle 50% (25% to 75%)
    start_x = width//4
    end_x = (width * 3)//4
    
    # Vertical: middle 50% of height
    start_y = height//4
    end_y = (height * 3)//4
    
    # Create search area (middle 50% both horizontally and vertically)
    search_area = img[start_y:end_y, start_x:end_x]
    x_offset = start_x
    y_offset = start_y
    
    # Create a debug visualization
    debug_img = img.copy()
    
    # Draw green box around search area
    cv2.rectangle(debug_img, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
    
    # Convert to RGB for pytesseract
    search_area_rgb = cv2.cvtColor(search_area, cv2.COLOR_BGR2RGB)
    
    # Get text data from search area
    data = pytesseract.image_to_data(search_area_rgb, output_type=pytesseract.Output.DICT)
    
    # Print all found text for debugging
    print("\nAll text found in middle region:")
    for i, text in enumerate(data['text']):
        if text.strip():  # Only print non-empty text
            x = data['left'][i] + x_offset
            y = data['top'][i] + y_offset  # Add y_offset to get correct y position
            w = data['width'][i]
            h = data['height'][i]
            print(f"Text: '{text}' at position ({x}, {y}) with size {w}x{h}")
            # Draw blue box around all found text
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (255, 0, 0), 1)
    
    # Find all instances of "Title"
    title_boxes = []
    for i, text in enumerate(data['text']):
        if text.lower() == 'title':
            x = data['left'][i] + x_offset
            y = data['top'][i] + y_offset  # Add y_offset to get correct y position
            w = data['width'][i]
            h = data['height'][i]
            title_boxes.append((x, y, w, h))
            print(f"Found 'Title' at position ({x}, {y}) with size {w}x{h}")
    
    if not title_boxes:
        print("\nNo 'Title' text found in the middle region")
        # Save debug image even when no title is found
        debug_path = image_path.replace('.png', '_debug.png')
        cv2.imwrite(debug_path, debug_img)
        print(f"Saved debug visualization as: {debug_path}")
        return False
    
    if len(title_boxes) == 1:
        print("\nFound one instance of 'Title', looking for 'Who can view this video'...")
        # Find the sequence of words that make up "Who can view this video"
        who_words = []
        current_sequence = []
        expected_words = ["who", "can", "view", "this", "video"]
        
        # Sort all text boxes by y-coordinate first, then x-coordinate
        all_boxes = []
        for i, text in enumerate(data['text']):
            if text.strip():
                x = data['left'][i] + x_offset
                y = data['top'][i] + y_offset
                w = data['width'][i]
                h = data['height'][i]
                all_boxes.append((text.lower(), x, y, w, h))
        
        # Sort by y-coordinate (with some tolerance for same line)
        y_tolerance = 10  # pixels
        all_boxes.sort(key=lambda box: (box[2] // y_tolerance, box[1]))
        
        # Look for the sequence of words
        for text, x, y, w, h in all_boxes:
            if not current_sequence:
                if text == expected_words[0]:  # Found "who"
                    current_sequence.append((text, x, y, w, h))
            else:
                # Check if this word is next in sequence and on same line
                expected_word = expected_words[len(current_sequence)]
                if text == expected_word and abs(y - current_sequence[-1][2]) < y_tolerance:
                    current_sequence.append((text, x, y, w, h))
                    if len(current_sequence) == len(expected_words):
                        who_words = current_sequence
                        break
                else:
                    # Reset if sequence breaks
                    current_sequence = []
                    if text == expected_words[0]:
                        current_sequence.append((text, x, y, w, h))
        
        if not who_words:
            print("Could not find complete 'Who can view this video' sequence")
            debug_path = image_path.replace('.png', '_debug.png')
            cv2.imwrite(debug_path, debug_img)
            print(f"Saved debug visualization as: {debug_path}")
            return False
        
        # Get the Title box
        title_box = title_boxes[0]
        
        # Calculate the Who box as the bounding box of all words
        who_x = min(x for _, x, _, _, _ in who_words)
        who_y = min(y for _, _, y, _, _ in who_words)
        who_w = max(x + w for _, x, _, w, _ in who_words) - who_x
        who_h = max(y + h for _, _, y, _, h in who_words) - who_y
        who_box = (who_x, who_y, who_w, who_h)
        
        # Draw boxes for visualization
        cv2.rectangle(debug_img, (title_box[0], title_box[1]), 
                     (title_box[0] + title_box[2], title_box[1] + title_box[3]), (0, 0, 255), 2)
        cv2.rectangle(debug_img, (who_box[0], who_box[1]), 
                     (who_box[0] + who_box[2], who_box[1] + who_box[3]), (0, 0, 255), 2)
        
        # Draw individual word boxes in blue for debugging
        for _, x, y, w, h in who_words:
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (255, 0, 0), 1)
        
        # Calculate click position:
        # - Same x-coordinate as Title (middle of Title box)
        # - Halfway between Title and Who boxes vertically
        title_center_x = title_box[0] + (title_box[2] // 2)
        title_bottom = title_box[1] + title_box[3]
        who_top = who_box[1]
        
        # Click halfway between Title bottom and Who top
        click_x = title_center_x
        click_y = title_bottom + ((who_top - title_bottom) // 2)
        
        # Draw a small circle at the click position
        cv2.circle(debug_img, (click_x, click_y), 3, (0, 255, 0), -1)
        
        # Save the debug visualization
        debug_path = image_path.replace('.png', '_debug.png')
        cv2.imwrite(debug_path, debug_img)
        print(f"\nSaved debug visualization as: {debug_path}")
        
        # Convert to screen coordinates and click
        screen_x, screen_y = screenshot_to_screen_coords(image_path, click_x, click_y)
        print(f"Clicking at screen coordinates: ({screen_x}, {screen_y})")
        pyautogui.moveTo(screen_x, screen_y, duration=0.125)
        pyautogui.click()
        
        # Wait a moment before typing
        time.sleep(0.25)
        
        # Type the caption if provided
        if caption:
            print(f"\nTyping caption: {caption}")
            type_text(caption)
        
        # Wait a moment for the dropdown to appear
        print("Waiting 0.25 seconds for dropdown to appear...")
        time.sleep(0.25)
        
        # Take a new screenshot to find everyone
        print("\nTaking screenshot to find 'everyone'...")
        post_who_click_screenshot = take_screenshot(os.path.dirname(image_path), 'post_who_click_screenshot')
        
        # Find everyone
        img = cv2.imread(post_who_click_screenshot)
        if img is not None:
            # Get image dimensions
            height, width = img.shape[:2]
            
            # Calculate search area (middle 50% both horizontally and vertically)
            start_x = width//4
            end_x = (width * 3)//4
            start_y = height//4
            end_y = (height * 3)//4
            
            # Create a debug visualization
            debug_img = img.copy()
            
            # Draw green box around search area
            cv2.rectangle(debug_img, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
            
            # Convert to RGB for pytesseract
            search_area = img[start_y:end_y, start_x:end_x]
            search_area_rgb = cv2.cvtColor(search_area, cv2.COLOR_BGR2RGB)
            
            # Get text data from search area
            data = pytesseract.image_to_data(search_area_rgb, output_type=pytesseract.Output.DICT)
            
            # Find "who" and "ensure"
            who_box = None
            ensure_box = None
            
            for i, text in enumerate(data['text']):
                text_lower = text.lower()
                if text_lower == 'who':
                    x = data['left'][i] + start_x
                    y = data['top'][i] + start_y
                    w = data['width'][i]
                    h = data['height'][i]
                    who_box = (x, y, w, h)
                    print(f"Found 'who' at position ({x}, {y}) with size {w}x{h}")
                elif text_lower == 'ensure':
                    x = data['left'][i] + start_x
                    y = data['top'][i] + start_y
                    w = data['width'][i]
                    h = data['height'][i]
                    ensure_box = (x, y, w, h)
                    print(f"Found 'ensure' at position ({x}, {y}) with size {w}x{h}")
            
            if who_box and ensure_box:
                # Draw red boxes around both words
                cv2.rectangle(debug_img, (who_box[0], who_box[1]), 
                             (who_box[0] + who_box[2], who_box[1] + who_box[3]), (0, 0, 255), 2)
                cv2.rectangle(debug_img, (ensure_box[0], ensure_box[1]), 
                             (ensure_box[0] + ensure_box[2], ensure_box[1] + ensure_box[3]), (0, 0, 255), 2)
                
                # Calculate click position:
                # - Horizontally: exactly at the end of "who"
                # - Vertically: halfway between "who" and "ensure"
                click_x = who_box[0] + who_box[2]  # End of "who" box
                who_bottom = who_box[1] + who_box[3]
                ensure_top = ensure_box[1]
                click_y = who_bottom + ((ensure_top - who_bottom) // 2)
                
                # Draw a small circle at the click position
                cv2.circle(debug_img, (click_x, click_y), 3, (0, 255, 0), -1)
                
                # Save the debug visualization
                debug_path = post_who_click_screenshot.replace('.png', '_debug.png')
                cv2.imwrite(debug_path, debug_img)
                print(f"Saved debug visualization as: {debug_path}")
                
                # Wait a moment before clicking
                print("Waiting 0.25 seconds before clicking...")
                time.sleep(0.25)
                
                # Convert to screen coordinates and click
                screen_x, screen_y = screenshot_to_screen_coords(post_who_click_screenshot, click_x, click_y)
                print(f"Clicking at screen coordinates: ({screen_x}, {screen_y})")
                pyautogui.moveTo(screen_x, screen_y, duration=0.25)  # Slower movement
                pyautogui.click()
                
                # Wait a moment after clicking
                print("Waiting 0.25 seconds after clicking...")
                time.sleep(0.25)
                
                # Press Return with a longer delay
                print("Pressing Return to confirm selection...")
                pyautogui.press('return')
                
                # Wait longer after pressing Return
                print("Waiting 0.75 seconds after pressing Return...")
                time.sleep(0.75)
                
                # Press Return again
                print("Pressing Return again...")
                pyautogui.press('return')
                
                # Wait a moment for the page to update
                print("Waiting 0.5 seconds for page to update...")
                time.sleep(0.5)
                
                # Take a new screenshot to find comment
                print("\nTaking screenshot to find 'comment'...")
                post_return_screenshot = take_screenshot(os.path.dirname(image_path), 'post_return_screenshot')
                
                # Find comment, allow, and disclose
                img = cv2.imread(post_return_screenshot)
                if img is not None:
                    # Get image dimensions
                    height, width = img.shape[:2]
                    
                    # Calculate search area (middle 50% both horizontally and vertically)
                    start_x = width//4
                    end_x = (width * 3)//4
                    start_y = height//4
                    end_y = (height * 3)//4
                    
                    # Create a debug visualization
                    debug_img = img.copy()
                    
                    # Draw green box around search area
                    cv2.rectangle(debug_img, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
                    
                    # Convert to RGB for pytesseract
                    search_area = img[start_y:end_y, start_x:end_x]
                    search_area_rgb = cv2.cvtColor(search_area, cv2.COLOR_BGR2RGB)
                    
                    # Get text data from search area
                    data = pytesseract.image_to_data(search_area_rgb, output_type=pytesseract.Output.DICT)
                    
                    # Find "comment", "allow", and "disclose" (case insensitive)
                    comment_box = None
                    allow_box = None
                    disclose_box = None
                    
                    for i, text in enumerate(data['text']):
                        text_lower = text.lower()
                        x = data['left'][i] + start_x
                        y = data['top'][i] + start_y
                        w = data['width'][i]
                        h = data['height'][i]
                        
                        if text_lower == 'comment':
                            comment_box = (x, y, w, h)
                            print(f"Found 'comment' at position ({x}, {y}) with size {w}x{h}")
                        elif text_lower == 'allow':
                            allow_box = (x, y, w, h)
                            print(f"Found 'allow' at position ({x}, {y}) with size {w}x{h}")
                        elif text_lower == 'disclose':
                            disclose_box = (x, y, w, h)
                            print(f"Found 'disclose' at position ({x}, {y}) with size {w}x{h}")
                    
                    # Draw red boxes around found words
                    if comment_box:
                        cv2.rectangle(debug_img, (comment_box[0], comment_box[1]), 
                                     (comment_box[0] + comment_box[2], comment_box[1] + comment_box[3]), (0, 0, 255), 2)
                    if allow_box:
                        cv2.rectangle(debug_img, (allow_box[0], allow_box[1]), 
                                     (allow_box[0] + allow_box[2], allow_box[1] + allow_box[3]), (0, 0, 255), 2)
                    if disclose_box:
                        cv2.rectangle(debug_img, (disclose_box[0], disclose_box[1]), 
                                     (disclose_box[0] + disclose_box[2], disclose_box[1] + disclose_box[3]), (0, 0, 255), 2)
                    
                    # Calculate T-shaped connection point if all three words are found
                    if comment_box and allow_box and disclose_box:
                        # Get the bottom of allow
                        allow_bottom = allow_box[1] + allow_box[3]
                        
                        # Get the left of comment
                        comment_left = comment_box[0]
                        
                        # Get the top of disclose
                        disclose_top = disclose_box[1]
                        
                        # Calculate the point that would connect all three in a T-shape
                        # This point should be:
                        # - Horizontally: at comment's left edge
                        # - Vertically: halfway between allow's bottom and disclose's top
                        click_x = comment_left
                        click_y = allow_bottom + ((disclose_top - allow_bottom) // 2)
                        
                        # Draw a small circle at the click position
                        cv2.circle(debug_img, (click_x, click_y), 3, (0, 255, 0), -1)
                        
                        # Draw lines to visualize the T-shape
                        # Vertical line from allow to disclose
                        cv2.line(debug_img, (click_x, allow_bottom), (click_x, disclose_top), (0, 255, 0), 1)
                        # Horizontal line to comment
                        cv2.line(debug_img, (click_x, click_y), (comment_left, click_y), (0, 255, 0), 1)
                        
                        # Save the debug visualization
                        debug_path = post_return_screenshot.replace('.png', '_debug.png')
                        cv2.imwrite(debug_path, debug_img)
                        print(f"Saved debug visualization as: {debug_path}")
                        
                        # Convert to screen coordinates and click
                        screen_x, screen_y = screenshot_to_screen_coords(post_return_screenshot, click_x, click_y)
                        print(f"Clicking at screen coordinates: ({screen_x}, {screen_y})")
                        pyautogui.moveTo(screen_x, screen_y, duration=0.125)
                        pyautogui.click()
                    else:
                        print("Could not find all three words (comment, allow, disclose) in search area")
                        # Save debug image even when words aren't found
                        debug_path = post_return_screenshot.replace('.png', '_debug.png')
                        cv2.imwrite(debug_path, debug_img)
                        print(f"Saved debug visualization as: {debug_path}")
                
                return True
            else:
                print("Could not find both 'who' and 'ensure' in search area")
                # Save debug image even when words aren't found
                debug_path = post_who_click_screenshot.replace('.png', '_debug.png')
                cv2.imwrite(debug_path, debug_img)
                print(f"Saved debug visualization as: {debug_path}")
                return False
        
        return True
    
    if len(title_boxes) < 2:
        print(f"\nFound only {len(title_boxes)} instance(s) of 'Title' text, expected 2")
        # Save debug image even when not enough titles are found
        debug_path = image_path.replace('.png', '_debug.png')
        cv2.imwrite(debug_path, debug_img)
        print(f"Saved debug visualization as: {debug_path}")
        return False
    
    # Sort boxes by y coordinate (top to bottom)
    title_boxes.sort(key=lambda box: box[1])
    
    # Draw red boxes around all found text
    for i, (x, y, w, h) in enumerate(title_boxes):
        # Make the lower box a thicker red line
        thickness = 3 if i == 1 else 2
        cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 0, 255), thickness)
    
    # Get the lower box (second one)
    lower_box = title_boxes[1]
    
    # Get a random point inside the lower box
    padding = 5  # Avoid clicking exactly on the edge
    click_x = random.randint(lower_box[0] + padding, lower_box[0] + lower_box[2] - padding)
    click_y = random.randint(lower_box[1] + padding, lower_box[1] + lower_box[3] - padding)
    
    # Draw a small circle at the click position
    cv2.circle(debug_img, (click_x, click_y), 3, (0, 255, 0), -1)
    
    # Save the debug visualization
    debug_path = image_path.replace('.png', '_debug.png')
    cv2.imwrite(debug_path, debug_img)
    print(f"\nSaved debug visualization as: {debug_path}")
    
    # Convert to screen coordinates and click
    screen_x, screen_y = screenshot_to_screen_coords(image_path, click_x, click_y)
    print(f"Clicking at screen coordinates: ({screen_x}, {screen_y})")
    pyautogui.moveTo(screen_x, screen_y, duration=0.125)
    pyautogui.click()
    
    # Immediately press Return after clicking
    print("Pressing Return to confirm selection...")
    pyautogui.press('return')
    
    # Wait half a second after pressing Return
    print("Waiting 0.5 seconds after pressing Return...")
    time.sleep(0.5)
    
    # Press Return again
    print("Pressing Return again...")
    pyautogui.press('return')
    
    # Wait a moment for the page to update
    print("Waiting 0.5 seconds for page to update...")
    time.sleep(0.5)
    
    # Take a new screenshot to find comment
    print("\nTaking screenshot to find 'comment'...")
    post_return_screenshot = take_screenshot(os.path.dirname(image_path), 'post_return_screenshot')
    
    # Find comment, allow, and disclose
    img = cv2.imread(post_return_screenshot)
    if img is not None:
        # Get image dimensions
        height, width = img.shape[:2]
        
        # Calculate search area (middle 50% both horizontally and vertically)
        start_x = width//4
        end_x = (width * 3)//4
        start_y = height//4
        end_y = (height * 3)//4
        
        # Create a debug visualization
        debug_img = img.copy()
        
        # Draw green box around search area
        cv2.rectangle(debug_img, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
        
        # Convert to RGB for pytesseract
        search_area = img[start_y:end_y, start_x:end_x]
        search_area_rgb = cv2.cvtColor(search_area, cv2.COLOR_BGR2RGB)
        
        # Get text data from search area
        data = pytesseract.image_to_data(search_area_rgb, output_type=pytesseract.Output.DICT)
        
        # Find "comment", "allow", and "disclose" (case insensitive)
        comment_box = None
        allow_box = None
        disclose_box = None
        
        for i, text in enumerate(data['text']):
            text_lower = text.lower()
            x = data['left'][i] + start_x
            y = data['top'][i] + start_y
            w = data['width'][i]
            h = data['height'][i]
            
            if text_lower == 'comment':
                comment_box = (x, y, w, h)
                print(f"Found 'comment' at position ({x}, {y}) with size {w}x{h}")
            elif text_lower == 'allow':
                allow_box = (x, y, w, h)
                print(f"Found 'allow' at position ({x}, {y}) with size {w}x{h}")
            elif text_lower == 'disclose':
                disclose_box = (x, y, w, h)
                print(f"Found 'disclose' at position ({x}, {y}) with size {w}x{h}")
        
        # Draw red boxes around found words
        if comment_box:
            cv2.rectangle(debug_img, (comment_box[0], comment_box[1]), 
                         (comment_box[0] + comment_box[2], comment_box[1] + comment_box[3]), (0, 0, 255), 2)
        if allow_box:
            cv2.rectangle(debug_img, (allow_box[0], allow_box[1]), 
                         (allow_box[0] + allow_box[2], allow_box[1] + allow_box[3]), (0, 0, 255), 2)
        if disclose_box:
            cv2.rectangle(debug_img, (disclose_box[0], disclose_box[1]), 
                         (disclose_box[0] + disclose_box[2], disclose_box[1] + disclose_box[3]), (0, 0, 255), 2)
        
        # Calculate T-shaped connection point if all three words are found
        if comment_box and allow_box and disclose_box:
            # Get the bottom of allow
            allow_bottom = allow_box[1] + allow_box[3]
            
            # Get the left of comment
            comment_left = comment_box[0]
            
            # Get the top of disclose
            disclose_top = disclose_box[1]
            
            # Calculate the point that would connect all three in a T-shape
            # This point should be:
            # - Horizontally: at comment's left edge
            # - Vertically: halfway between allow's bottom and disclose's top
            click_x = comment_left
            click_y = allow_bottom + ((disclose_top - allow_bottom) // 2)
            
            # Draw a small circle at the click position
            cv2.circle(debug_img, (click_x, click_y), 3, (0, 255, 0), -1)
            
            # Draw lines to visualize the T-shape
            # Vertical line from allow to disclose
            cv2.line(debug_img, (click_x, allow_bottom), (click_x, disclose_top), (0, 255, 0), 1)
            # Horizontal line to comment
            cv2.line(debug_img, (click_x, click_y), (comment_left, click_y), (0, 255, 0), 1)
            
            # Save the debug visualization
            debug_path = post_return_screenshot.replace('.png', '_debug.png')
            cv2.imwrite(debug_path, debug_img)
            print(f"Saved debug visualization as: {debug_path}")
            
            # Convert to screen coordinates and click
            screen_x, screen_y = screenshot_to_screen_coords(post_return_screenshot, click_x, click_y)
            print(f"Clicking at screen coordinates: ({screen_x}, {screen_y})")
            pyautogui.moveTo(screen_x, screen_y, duration=0.125)
            pyautogui.click()
        else:
            print("Could not find all three words (comment, allow, disclose) in search area")
            # Save debug image even when words aren't found
            debug_path = post_return_screenshot.replace('.png', '_debug.png')
            cv2.imwrite(debug_path, debug_img)
            print(f"Saved debug visualization as: {debug_path}")
    
    return True

def main():
    # Check if CSV file path is provided
    if len(sys.argv) != 2:
        print("Usage: python3 capcut_scheduling.py <schedule_csv_path>")
        print("Example: python3 capcut_scheduling.py temporary_files/schedules/schedule.csv")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    filename_to_search, platform_to_search, scheduled_date, scheduled_time, caption = get_filename_from_csv(csv_path)
    
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
        
        # Take one screenshot after upload that we'll reuse
        print("\nTaking screenshot after upload...")
        post_upload_screenshot = take_screenshot(processing_dir, 'post_upload_screenshot')
        
        # Third step: Try to find and click 'Search' using the same screenshot
        print("\nStep 3: Looking for 'Search' text...")
        click_box = find_and_highlight_text(post_upload_screenshot, 'Search', 'right')
        search_found = False
        
        if click_box:
            # Get a random point inside the box
            screenshot_x, screenshot_y = get_random_point_in_box(click_box)
            print(f"Selected point in screenshot: ({screenshot_x}, {screenshot_y})")
            
            # Convert to screen coordinates and click
            screen_x, screen_y = screenshot_to_screen_coords(post_upload_screenshot, screenshot_x, screenshot_y)
            print(f"Clicking at screen coordinates: ({screen_x}, {screen_y})")
            pyautogui.moveTo(screen_x, screen_y, duration=0.125)
            pyautogui.click()
            search_found = True
            print("Successfully clicked Search button")
            
            # Wait a moment for the search field to be ready
            time.sleep(0.125)
            
            # Fourth step: Type the filename
            print("\nStep 4: Typing filename...")
            type_text(filename_to_search)
            
            # Wait a moment after typing
            print("Waiting 0.25 seconds after typing...")
            time.sleep(0.25)
            
            # Take one screenshot after typing
            print("\nTaking screenshot after typing...")
            post_typing_screenshot = take_screenshot(processing_dir, 'post_typing_screenshot')
        else:
            print("Search text not found, skipping search and typing steps")
            # Use the post-upload screenshot for filename search
            post_typing_screenshot = post_upload_screenshot
        
        # Fifth step: Find and click the filename in middle region using the post-typing screenshot
        print("\nStep 5: Looking for filename in middle region...")
        max_attempts = 5
        filename_found = False
        
        for attempt in range(max_attempts):
            print(f"\nAttempt {attempt + 1} of {max_attempts} to find filename...")
            
            # Take a new screenshot for each attempt
            if attempt > 0:
                print("Taking new screenshot for retry...")
                post_typing_screenshot = take_screenshot(processing_dir, f'post_typing_screenshot_attempt_{attempt + 1}')
            
            click_box = find_and_highlight_text(post_typing_screenshot, filename_to_search, 'middle', prioritize_center=True)
            
            if click_box:
                # Get a random point inside the box
                screenshot_x, screenshot_y = get_random_point_in_box(click_box)
                print(f"Selected point in screenshot: ({screenshot_x}, {screenshot_y})")
                
                # Convert to screen coordinates and click
                screen_x, screen_y = screenshot_to_screen_coords(post_typing_screenshot, screenshot_x, screenshot_y)
                print(f"Clicking at screen coordinates: ({screen_x}, {screen_y})")
                pyautogui.moveTo(screen_x, screen_y, duration=0.125)
                pyautogui.click()
                
                print("Successfully clicked on filename")
                filename_found = True
                break
            else:
                print(f"Failed to find filename on attempt {attempt + 1}")
                if attempt < max_attempts - 1:
                    print("Waiting 0.5 seconds before retry...")
                    time.sleep(0.5)
        
        if not filename_found:
            print(f"Failed to find filename after {max_attempts} attempts, stopping process")
            close_current_tab()
            sys.exit(1)
        
        # Press Enter to open the file
        print("Pressing Enter to open file...")
        time.sleep(0.125)
        pyautogui.press('enter')
        
        # Wait half a second after pressing Enter
        print("Waiting 0.5 seconds after pressing Enter...")
        time.sleep(0.5)
        
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
        
        # Wait half a second after clicking platform
        print("Waiting 0.5 seconds after clicking platform...")
        time.sleep(0.5)
        
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
                
                # Take screenshot after scrolling
                print("\nTaking screenshot after scrolling...")
                post_scroll_screenshot = take_screenshot(processing_dir, 'post_scroll_screenshot')
                
                # Find and click the lower Title text in middle region
                print("\nLooking for lower Title text in middle region...")
                if not find_and_click_lower_title(post_scroll_screenshot, region='middle', caption=caption):
                    print("Failed to find lower Title text")
                    close_current_tab()
                    sys.exit(1)
                    
                print("Successfully clicked lower Title text and typed caption")
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