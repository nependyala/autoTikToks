"""
Script to add a single video entry to the schedule CSV.
If the schedule CSV doesn't exist, it will be created with the first entry.
If it exists, the new entry will be added at the top.
All schedule files are saved in the temporary_files/schedules folder.
"""
import pandas as pd
import os
from datetime import datetime, timedelta
import sys

def get_schedules_dir():
    """Get the path to the schedules directory"""
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.dirname(__file__))  # Go up one level from helper_functions
    schedules_dir = os.path.join(script_dir, 'temporary_files', 'schedules')
    
    # Create the directory if it doesn't exist
    os.makedirs(schedules_dir, exist_ok=True)
    return schedules_dir

def validate_date(date_str):
    """Validate and standardize date format"""
    try:
        # Try parsing the date
        date_obj = pd.to_datetime(date_str)
        # Return in YYYY-MM-DD HH:MM format
        return date_obj.strftime("%Y-%m-%d %H:%M")
    except:
        raise ValueError(f"Invalid date format: {date_str}. Please use a valid date format.")

def validate_filepath(filepath):
    """Validate that the file exists"""
    if not os.path.exists(filepath):
        raise ValueError(f"File not found: {filepath}")
    return filepath

def validate_platform(platform):
    """Validate platform is one of the supported options"""
    valid_platforms = ['tiktok', 'instagram', 'youtube']
    if platform.lower() not in valid_platforms:
        raise ValueError(f"Invalid platform: {platform}. Must be one of {valid_platforms}")
    return platform.lower()

def add_to_schedule(clip_filepath, posting_date, caption, platform, schedule_filename):
    """
    Add a single video entry to the schedule CSV.
    Creates the CSV if it doesn't exist, adds entry at top if it does.
    All schedule files are saved in the temporary_files/schedules folder.
    
    Args:
        clip_filepath (str): Path to the video file
        posting_date (str): Date and time to post the video
        caption (str): Caption for the video
        platform (str): Platform to post to (tiktok, instagram, youtube)
        schedule_filename (str): Name of the schedule CSV file (e.g., 'schedule.csv')
    """
    try:
        # Get the schedules directory and full path for the schedule file
        schedules_dir = get_schedules_dir()
        schedule_csv_path = os.path.join(schedules_dir, schedule_filename)
        
        # Validate inputs
        validated_date = validate_date(posting_date)
        validated_filepath = validate_filepath(clip_filepath)
        validated_platform = validate_platform(platform)
        
        # Create new entry
        new_entry = pd.DataFrame([{
            'filepath': validated_filepath,
            'scheduled_date': validated_date,
            'caption': caption,
            'platform': validated_platform,
            'status': 'pending'
        }])
        
        if os.path.exists(schedule_csv_path):
            # Read existing schedule
            existing_schedule = pd.read_csv(schedule_csv_path)
            # Rename 'date' column to 'scheduled_date' if it exists
            if 'date' in existing_schedule.columns:
                existing_schedule = existing_schedule.rename(columns={'date': 'scheduled_date'})
            # Combine new entry with existing schedule (new entry at top)
            updated_schedule = pd.concat([new_entry, existing_schedule], ignore_index=True)
            print(f"Added new entry to existing schedule: {schedule_filename}")
        else:
            # Create new schedule with this entry
            updated_schedule = new_entry
            print(f"Created new schedule file: {schedule_filename}")
        
        # Save updated schedule
        updated_schedule.to_csv(schedule_csv_path, index=False)
        
        # Print confirmation
        print("\nEntry added successfully:")
        print(f"File: {validated_filepath}")
        print(f"Scheduled Date: {validated_date}")
        print(f"Platform: {validated_platform}")
        print(f"Caption: {caption}")
        print(f"Schedule saved in: {schedules_dir}")
        
    except Exception as e:
        print(f"Error adding to schedule: {str(e)}")
        sys.exit(1)

def create_dummy_schedule():
    """Create a dummy schedule with 5 entries, all at 7 PM, starting one day later"""
    # Get current date and add one day
    current_date = datetime.now() + timedelta(days=1)
    
    # Create a list to store schedule entries
    schedule_entries = []
    
    # Original caption
    caption = 'satisfying asmr ai fruit slicing #aigenerated #satisfying #asmr'
    
    # Original file paths
    files = [
        'video_posting/clips/red_apple.mp4',
        'video_posting/clips/purple_pear.mp4',
        'video_posting/clips/green_apple.mp4',
        'video_posting/clips/purple_plum.mp4',
        'video_posting/clips/blue_pineapple.mp4'
    ]
    
    # Create 5 dummy entries
    for i in range(5):
        # Set time to 7 PM
        scheduled_time = current_date.replace(hour=19, minute=0, second=0, microsecond=0)
        
        # Add i days for each entry (starting from tomorrow)
        scheduled_time = scheduled_time + timedelta(days=i)
        
        # Create entry
        entry = {
            'scheduled_file': files[i],
            'scheduled_date': scheduled_time.strftime('%Y-%m-%d'),
            'scheduled_time': scheduled_time.strftime('%H:%M:%S'),
            'platform': 'TikTok',
            'caption': caption
        }
        schedule_entries.append(entry)
    
    # Create DataFrame
    df = pd.DataFrame(schedule_entries)
    
    # Create schedules directory if it doesn't exist
    schedules_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'temporary_files', 'schedules')
    os.makedirs(schedules_dir, exist_ok=True)
    
    # Always use schedule.csv as the filename
    filepath = os.path.join(schedules_dir, 'schedule.csv')
    
    # Save to CSV
    df.to_csv(filepath, index=False)
    print(f"\nCreated dummy schedule with 5 entries, all at 7 PM starting tomorrow:")
    print("\nScheduled files:")
    for entry in schedule_entries:
        print(f"- {entry['scheduled_file']} on {entry['scheduled_date']} at {entry['scheduled_time']} for {entry['platform']}")
        print(f"  Caption: {entry['caption']}")
    print(f"\nSchedule saved to: {filepath}")
    return filepath

def main():
    if len(sys.argv) == 1:
        # No arguments provided, create dummy schedule
        create_dummy_schedule()
    elif len(sys.argv) != 6:
        print("Usage: python3 create_schedule.py [<clip_filepath> <posting_date> <caption> <platform> <schedule_filename>]")
        print("If no arguments are provided, a dummy schedule will be created.")
        print("Example: python3 create_schedule.py /path/to/video.mp4 '2024-03-21 15:00' 'Fun video!' tiktok schedule.csv")
        sys.exit(1)
    else:
        # Normal operation with arguments
        clip_filepath = sys.argv[1]
        posting_date = sys.argv[2]
        caption = sys.argv[3]
        platform = sys.argv[4]
        schedule_filename = sys.argv[5]
        
        add_to_schedule(clip_filepath, posting_date, caption, platform, schedule_filename)

if __name__ == "__main__":
    main() 