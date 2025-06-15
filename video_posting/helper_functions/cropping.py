"""
Script to crop videos to specific dimensions and ratios.
"""
import cv2
import os
import sys
from typing import Tuple
import subprocess

def crop_video(input_path: str, output_path: str, x: int, y: int, width: int, height: int) -> str:
    """
    Crop a video to specific dimensions starting from given coordinates.
    
    Args:
        input_path (str): Path to input video file
        x (int): X coordinate of top-left corner
        y (int): Y coordinate of top-left corner
        width (int): Width of crop
        height (int): Height of crop
        
    Returns:
        str: Path to the cropped video file
    """
    try:
        # Open the video file
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise Exception(f"Could not open video file: {input_path}")
            
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create video writer with H.264 codec
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"Processing video: {input_path}")
        print(f"Cropping to: {width}x{height} from position ({x}, {y})")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Crop the frame
            cropped_frame = frame[y:y+height, x:x+width]
            
            # Write the cropped frame
            out.write(cropped_frame)
            
            # Update progress
            frame_count += 1
            if frame_count % 30 == 0:  # Show progress every 30 frames
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}%")
        
        # Release resources
        cap.release()
        out.release()
        
        # Use ffmpeg to copy audio from original video
        temp_output = output_path + '.temp.mp4'
        os.rename(output_path, temp_output)
        
        ffmpeg_cmd = [
            'ffmpeg',
            '-i', temp_output,  # Input video (no audio)
            '-i', input_path,   # Original video (with audio)
            '-c:v', 'copy',     # Copy video stream
            '-c:a', 'aac',      # Use AAC audio codec
            '-map', '0:v:0',    # Use video from first input
            '-map', '1:a:0',    # Use audio from second input
            '-shortest',        # Match duration to shortest stream
            output_path         # Output file
        ]
        
        subprocess.run(ffmpeg_cmd, check=True)
        os.remove(temp_output)  # Clean up temporary file
        
        print(f"Cropped video saved to: {output_path}")
        return output_path
        
    except Exception as e:
        raise Exception(f"Error processing video: {str(e)}")

def calculate_vertical_crop_position(video_path: str, position_percent: float) -> Tuple[int, int, int, int]:
    """
    Calculate crop dimensions for 9:16 vertical video from 16:9 source.
    Position is specified as a percentage (0-100) of the horizontal space.
    
    Args:
        video_path (str): Path to input video file
        position_percent (float): Position percentage (0-100) from left to right
        
    Returns:
        Tuple[int, int, int, int]: (x, y, width, height) for cropping
    """
    try:
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f"Could not open video file: {video_path}")
            
        # Get video dimensions
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        # Calculate 9:16 crop dimensions
        crop_width = int(height * (9/16))  # Width for 9:16 ratio
        
        # Calculate x position based on percentage
        # Convert percentage to position (0 = leftmost, 100 = rightmost)
        max_x = width - crop_width  # Maximum x position
        x = int((position_percent / 100) * max_x)
        
        # Ensure x is within bounds
        x = max(0, min(x, max_x))
        
        return x, 0, crop_width, height
            
    except Exception as e:
        raise Exception(f"Error calculating crop dimensions: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python cropping.py <input_video_path> <position_percent>")
        print("Example: python cropping.py input.mp4 50")
        print("Position percentage: 0 = leftmost, 50 = center, 100 = rightmost")
        sys.exit(1)
    
    input_path = sys.argv[1]
    position_percent = float(sys.argv[2])
    
    if not 0 <= position_percent <= 100:
        print("Error: Position percentage must be between 0 and 100")
        sys.exit(1)
    
    try:
        # Calculate crop dimensions for 9:16 vertical video
        x, y, width, height = calculate_vertical_crop_position(input_path, position_percent)
        
        # Create output path in clips directory
        output_dir = os.path.join(os.path.dirname(__file__), 'clips')
        os.makedirs(output_dir, exist_ok=True)
        
        # Get original filename and append _cropped
        original_filename = os.path.basename(input_path)
        filename_without_ext, ext = os.path.splitext(original_filename)
        output_filename = f"{filename_without_ext}_cropped{ext}"
        output_path = os.path.join(output_dir, output_filename)
        
        # Crop the video
        output_path = crop_video(input_path, output_path, x, y, width, height)
        print(f"Successfully cropped video to: {output_path}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1) 