"""
Transcript fetching module for handling video transcript extraction and processing.
"""
import subprocess
import os
from typing import Optional
import sys

# Define the default output directory
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'temporary_files')

def fetch_transcript(video_url: str, output_dir: Optional[str] = None) -> str:
    """
    Download English auto-generated captions from a YouTube video using yt-dlp.
    
    Args:
        video_url (str): The URL of the YouTube video
        output_dir (str, optional): Directory to save the transcript. Defaults to temporary_files directory.
    
    Returns:
        str: Path to the downloaded transcript file
    """
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Define paths
    final_transcript_path = os.path.join(output_dir, "transcript.srt")
    
    # Remove existing transcript if it exists
    if os.path.exists(final_transcript_path):
        os.remove(final_transcript_path)
    
    # Change to the output directory
    original_dir = os.getcwd()
    os.chdir(output_dir)
    
    try:
        # Construct the yt-dlp command
        command = [
            'yt-dlp',
            '--write-auto-subs',
            '--sub-lang', 'en',
            '--skip-download',
            '--convert-subs', 'srt',
            '-o', 'transcript',
            video_url
        ]
        
        # Run the command
        subprocess.run(command, check=True, capture_output=True, text=True)
        
        # Change back to original directory
        os.chdir(original_dir)
        return final_transcript_path
            
    except subprocess.CalledProcessError as e:
        # Change back to original directory
        os.chdir(original_dir)
        raise Exception(f"Failed to download transcript: {e.stderr}")
    except Exception as e:
        # Change back to original directory
        os.chdir(original_dir)
        raise Exception(f"Error processing video: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python transcript_fetch.py <YouTube_URL>")
        sys.exit(1)
    video_url = sys.argv[1]
    try:
        transcript_path = fetch_transcript(video_url)
        print(f"Transcript downloaded successfully to: {transcript_path}")
    except Exception as e:
        print(f"Error: {str(e)}") 