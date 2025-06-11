"""
Script to download and trim specific segments from YouTube videos.
"""
import os
import sys
import subprocess
from typing import Tuple
import re

def convert_timestamp_to_seconds(timestamp: str) -> float:
    """
    Convert timestamp from HH:MM:SS,mmm format to seconds.
    
    Args:
        timestamp (str): Timestamp in HH:MM:SS,mmm format
        
    Returns:
        float: Timestamp in seconds
    """
    # Replace comma with dot for milliseconds
    timestamp = timestamp.replace(',', '.')
    
    # Split into hours, minutes, seconds
    parts = timestamp.split(':')
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return float(hours) * 3600 + float(minutes) * 60 + float(seconds)
    elif len(parts) == 2:
        minutes, seconds = parts
        return float(minutes) * 60 + float(seconds)
    else:
        raise ValueError(f"Invalid timestamp format: {timestamp}")

def download_clip(url: str, start_time: str, end_time: str) -> str:
    """
    Download and trim a specific segment from a YouTube video.
    
    Args:
        url (str): YouTube video URL
        start_time (str): Start timestamp in HH:MM:SS,mmm format
        end_time (str): End timestamp in HH:MM:SS,mmm format
        
    Returns:
        str: Path to the downloaded clip
    """
    try:
        # Strict URL validation
        if url != "https://www.youtube.com/watch?v=RywoFvefNOE":
            raise ValueError("This script only supports downloading clips from https://www.youtube.com/watch?v=RywoFvefNOE")
            
        # Convert timestamps to seconds
        start_seconds = convert_timestamp_to_seconds(start_time)
        end_seconds = convert_timestamp_to_seconds(end_time)
        
        # Calculate duration
        duration = end_seconds - start_seconds
        
        # Create output filename
        output_dir = os.path.join(os.path.dirname(__file__), 'temporary_files')
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"clip_RywoFvefNOE_{start_time.replace(',', '_')}_{end_time.replace(',', '_')}.mp4")
        
        # Download and trim the video with minimal parameters
        command = [
            'yt-dlp',
            '--no-cache-dir',
            '--no-warnings',
            '-f', 'best[height<=720]',  # Limit to 720p
            '--downloader', 'ffmpeg',
            '--downloader-args', f'ffmpeg_i:-ss {start_seconds} -t {duration}',
            '-o', output_file,
            url
        ]
        
        print(f"Downloading clip from {start_time} to {end_time}...")
        subprocess.run(command, check=True)
        
        if os.path.exists(output_file):
            print(f"Clip saved to: {output_file}")
            return output_file
        else:
            raise Exception("Failed to create output file")
            
    except subprocess.CalledProcessError as e:
        raise Exception(f"Error downloading video: {str(e)}")
    except Exception as e:
        raise Exception(f"Error processing video: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python clip_fetch.py <youtube_url> <start_time> <end_time>")
        print("Example: python clip_fetch.py https://www.youtube.com/watch?v=VIDEO_ID 00:00:00,160 00:01:20,000")
        sys.exit(1)
    
    url = sys.argv[1]
    start_time = sys.argv[2]
    end_time = sys.argv[3]
    
    try:
        output_file = download_clip(url, start_time, end_time)
        print(f"Successfully downloaded clip to: {output_file}")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1) 