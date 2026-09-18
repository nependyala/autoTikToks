"""
Script for generating videos using fal.ai's Veo 3 model.
This script handles the generation of video clips based on text prompts.
"""

import os
import logging
import time
from pathlib import Path
import requests
from dotenv import load_dotenv
import fal_client

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from secrets/.env
env_path = Path(__file__).parent.parent.parent / "secrets" / ".env"
load_dotenv(env_path)

# Map project env name to fal_client's expected FAL_KEY when present.
if "FAL_AI_VEO3_API_KEY" in os.environ and not os.environ.get("FAL_KEY"):
    os.environ["FAL_KEY"] = os.environ["FAL_AI_VEO3_API_KEY"]

# Constants
TEMP_DIR = Path(__file__).parent / "temporary_files"
VIDEO_CLIPS_DIR = TEMP_DIR / "video_clips"
VIDEO_CLIPS_DIR.mkdir(exist_ok=True, parents=True)


def _ensure_fal_credentials() -> None:
    if os.environ.get("FAL_KEY") or os.environ.get("FAL_AI_VEO3_API_KEY"):
        if not os.environ.get("FAL_KEY"):
            os.environ["FAL_KEY"] = os.environ["FAL_AI_VEO3_API_KEY"]
        return
    raise ValueError(
        "Set FAL_AI_VEO3_API_KEY (or FAL_KEY) in secrets/.env before generating videos."
    )

def download_video(url: str, output_path: str) -> bool:
    """
    Download a video from a URL to the specified path.
    
    Args:
        url (str): URL of the video to download
        output_path (str): Path where the video should be saved
    
    Returns:
        bool: True if download was successful, False otherwise
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return True
    except Exception as e:
        logger.error(f"Error downloading video: {str(e)}")
        return False

def generate_video(prompt: str, aspect_ratio: str = "9:16", duration: str = "8s", 
                  enhance_prompt: bool = True, generate_audio: bool = True) -> str:
    """
    Generate a video using fal.ai's Veo 3 model.
    
    Args:
        prompt (str): Text prompt describing the video to generate
        aspect_ratio (str): Aspect ratio of the video ("16:9", "9:16", or "1:1")
        duration (str): Duration of the video (currently only "8s" is supported)
        enhance_prompt (bool): Whether to enhance the prompt
        generate_audio (bool): Whether to generate audio for the video
    
    Returns:
        str: Path to the generated video file, or None if generation failed
    """
    try:
        _ensure_fal_credentials()
        logger.info(f"Generating video with prompt: {prompt}")
        
        # Set up progress callback
        def on_queue_update(update):
            if isinstance(update, fal_client.InProgress):
                for log in update.logs:
                    logger.info(log["message"])
        
        # Submit the generation request
        result = fal_client.subscribe(
            "fal-ai/veo3",
            arguments={
                "prompt": prompt,
                "aspect_ratio": aspect_ratio,
                "duration": duration,
                "enhance_prompt": enhance_prompt,
                "generate_audio": generate_audio
            },
            with_logs=True,
            on_queue_update=on_queue_update
        )
        
        if not result or "video" not in result:
            logger.error("No video URL in the response")
            return None
            
        video_url = result["video"]["url"]
        logger.info(f"Video generated successfully. URL: {video_url}")
        
        # Generate output filename
        timestamp = int(time.time())
        output_filename = f"video_{timestamp}.mp4"
        output_path = str(VIDEO_CLIPS_DIR / output_filename)
        
        # Download the video
        logger.info(f"Downloading video to: {output_path}")
        if download_video(video_url, output_path):
            logger.info("Video downloaded successfully")
            return output_path
        else:
            logger.error("Failed to download video")
            return None
            
    except Exception as e:
        logger.error(f"Error generating video: {str(e)}")
        return None

def get_next_video_path() -> str:
    """
    Get the path for the next video file.
    
    Returns:
        str: Path for the next video file
    """
    existing_files = list(VIDEO_CLIPS_DIR.glob("video_*.mp4"))
    next_num = len(existing_files) + 1
    return str(VIDEO_CLIPS_DIR / f"video_{next_num}.mp4")

if __name__ == "__main__":
    # Example usage
    test_prompt = "A cinematic, photorealistic close-up of a amethyst translucent glass pear being sliced in an ASMR-style video.\n\nThe shot begins with a knife blade already positioned just above the fruit — no motion or approach. The knife makes a single clean, vertical cut through the center of the fruit in smooth, deliberate slow motion, continuing all the way down until it touches the cutting board. The knife then lifts up and exits the frame as the freshly cut slice falls.\n\nThe fruit must have a realistic shape and a fully translucent glass appearance, both inside and out. The slicing motion should dominate the duration of the clip.\n\nOnly one slice is removed, and it must cleanly fall away in the same orientation as the knife cut. The fall lasts no longer than one second and ends with a clear, satisfying, resonant clink as the slice hits the wooden cutting board. The slice and remaining fruit then remain completely still.\n\nThere are no hands, fingers, or knife handles visible — only the knife blade in contact with the fruit is shown. The fruit is untouched and rests naturally on the board. No transitions, fades, or additional effects are used.\n\nThe audio must enhance the ASMR quality: a soft, shimmering, glass-like slicing sound as the knife cuts through, followed by a crisp, crystal-clear clink as the slice lands."
    
    output_path = generate_video(
        prompt=test_prompt,
        aspect_ratio="9:16",
        duration="8s",
        enhance_prompt=True,
        generate_audio=True
    )
    
    if output_path:
        print(f"Video generated successfully: {output_path}")
    else:
        print("Failed to generate video") 