"""
Script to process CSV files with video scheduling details.
Removed browser automation logic for CapCut posting.
"""
import os
import shutil
from datetime import datetime
import pandas as pd
import sys

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

def get_filename_from_csv(csv_path, entry_index=0):
    """
    Get the filename and metadata for a specific entry from the CSV file.
    Returns a dictionary containing all entry data.
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # Check if the entry index is valid
        if entry_index >= len(df):
            print(f"Entry index {entry_index} is out of range (total entries: {len(df)})")
            return None
            
        # Get the entry data
        entry = df.iloc[entry_index]
        
        # Create a dictionary with all entry data
        entry_data = {
            'scheduled_file': entry['scheduled_file'],
            'scheduled_date': entry['scheduled_date'],
            'scheduled_time': entry['scheduled_time'],
            'platform': entry['platform'],
            'caption': entry['caption']
        }
        
        return entry_data
        
    except Exception as e:
        print(f"Error reading CSV entry: {str(e)}")
        return None

def remove_first_entry(csv_path):
    """
    Remove the first entry from the CSV file.
    Returns True if there are more entries to process, False if this was the last entry.
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        if len(df) == 0:
            print("No more entries in CSV file")
            return False
        
        # Remove the first entry
        df = df.iloc[1:]
        
        # Save the updated CSV
        df.to_csv(csv_path, index=False)
        print("Removed first entry from CSV")
        
        # Return True if there are more entries
        return len(df) > 0
        
    except Exception as e:
        print(f"Error removing entry from CSV: {str(e)}")
        return False

def verify_entry_deleted(csv_path, original_entry_count):
    """
    Verify that an entry was actually deleted from the CSV.
    Returns True if deletion was successful, False otherwise.
    """
    try:
        df = pd.read_csv(csv_path)
        current_count = len(df)
        if current_count == original_entry_count - 1:
            print(f"Verified entry deletion: {original_entry_count} -> {current_count} entries")
            return True
        else:
            print(f"Warning: Entry count mismatch. Expected {original_entry_count - 1}, got {current_count}")
            return False
    except Exception as e:
        print(f"Error verifying entry deletion: {str(e)}")
        return False

def process_single_entry(csv_path, temp_dir, processing_dir, entry_index):
    """Process a single entry from the schedule CSV (read-only)"""
    # Get original entry count before processing
    df = pd.read_csv(csv_path)
    original_entry_count = len(df)
    
    # Always process the first entry (index 0)
    if entry_index != 0:
        print("Warning: entry_index should always be 0, using 0 instead")
        entry_index = 0
    
    print(f"\nProcessing entry {entry_index + 1} of {original_entry_count}")
    
    try:
        # Get current entry data (always index 0)
        entry_data = get_filename_from_csv(csv_path, 0)  # Always use index 0
        if not entry_data:
            print("No more entries to process")
            return False
        
        print(f"\nEntry data:")
        print(f"File: {entry_data['scheduled_file']}")
        print(f"Scheduled for: {entry_data['scheduled_date']} at {entry_data['scheduled_time']}")
        print(f"Platform: {entry_data['platform']}")
        print(f"Caption: {entry_data['caption']}")
        
        # Verify entry was deleted from CSV
        if not verify_entry_deleted(csv_path, original_entry_count):
            print("Failed to verify entry deletion")
            return False
        
        print("\nSuccessfully processed entry!")
        return True
        
    except Exception as e:
        print(f"Error processing entry: {str(e)}")
        return False

def main():
    """Main function to process all entries in the schedule CSV"""
    # Check if CSV file path is provided
    if len(sys.argv) != 2:
        print("Usage: python3 capcut_scheduling.py <schedule_csv_path>")
        print("Example: python3 capcut_scheduling.py temporary_files/schedules/schedule.csv")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    
    # Create temporary files folder and processing subfolder
    temp_dir, processing_dir = create_temp_folder()
    
    try:
        # Process entries sequentially, always using index 0
        while True:
            # Read CSV to check if we have more entries
            df = pd.read_csv(csv_path)
            if len(df) == 0:
                print("\nAll entries processed!")
                break
            
            print(f"\nProcessing next entry. {len(df)} entries remaining.")
            
            # Always process index 0
            success = process_single_entry(csv_path, temp_dir, processing_dir, 0)
            
            if not success:
                print("\nFailed to process entry")
                # Ask user if they want to continue
                response = input("\nWould you like to continue with the next entry? (y/n): ")
                if response.lower() != 'y':
                    print("Stopping script as requested")
                    break
                # If user wants to continue, remove the failed entry
                if not remove_first_entry(csv_path):
                    print("No more entries to process")
                    break
    
        print("\nScript completed!")
        
    except Exception as e:
        print(f"Error in main loop: {str(e)}")
        sys.exit(1)
    finally:
        # Clean up temporary files
        clear_temp_folder(temp_dir, processing_dir)

if __name__ == "__main__":
    main() 