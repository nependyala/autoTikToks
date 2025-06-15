"""
Script to generate AI videos using Google's Veo 3.0 model.
This module handles the creation of AI-generated video content with audio.
"""
import os
import re
import time
from dotenv import load_dotenv
from google.oauth2 import service_account
from google.auth.transport.requests import AuthorizedSession
from google.cloud import storage

# Constants
PROJECT_ID = "autotiktoks"
LOCATION = "us-central1"
SERVICE_ACCOUNT_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'secrets', 'tiktokgeneration-key.json')
DOWNLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'temporary_files')
API_URL = f"https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{LOCATION}/generativeModels/video-generator-001:generate"
SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]

# Video generation settings
MAX_POLLING_ATTEMPTS = 60  # 5 minutes at 5-second intervals
POLLING_INTERVAL = 5  # seconds

# Ensure directory exists
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Load .env (optional, for other config)
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'secrets', '.env')
load_dotenv(env_path)

def get_next_clip_path() -> str:
    """Auto-increment filename like clip_001.mp4."""
    existing = [f for f in os.listdir(DOWNLOAD_DIR) if f.startswith("clip_") and f.endswith(".mp4")]
    nums = []
    for f in existing:
        try:
            nums.append(int(f.replace("clip_", "").replace(".mp4", "")))
        except ValueError:
            continue
    next_num = max(nums, default=0) + 1
    return os.path.join(DOWNLOAD_DIR, f"clip_{next_num:03d}.mp4")

def download_from_gcs(gcs_uri: str, destination_path: str, credentials):
    """Download a file from GCS to local path."""
    match = re.match(r'gs://([^/]+)/(.+)', gcs_uri)
    if not match:
        raise ValueError(f"Invalid GCS URI: {gcs_uri}")
    
    bucket_name, blob_name = match.groups()
    storage_client = storage.Client(credentials=credentials, project=PROJECT_ID)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    print(f"⬇️ Downloading to {destination_path}...")
    blob.download_to_filename(destination_path)
    print("✅ Downloaded.")

def generate_text_to_video(prompt: str) -> str:
    """
    Generate a video using Google's Veo 3.0 model via REST API and save it to temporary files.
    The video will be exactly 5 seconds long with audio.
    
    Args:
        prompt: The exact prompt to use for video generation
    
    Returns:
        str: Path to the saved video file
    
    Raises:
        Exception: If video generation or download fails
        TimeoutError: If video generation takes too long
    """
    try:
        # Authenticate
        credentials = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=SCOPES
        )
        authed_session = AuthorizedSession(credentials)
        
        # Prepare the request payload
        payload = {
            "prompt": {
                "text": prompt
            },
            "videoConfig": {
                "duration": "5s",
                "aspectRatio": {
                    "width": 16,
                    "height": 9
                },
                "enableAudio": True
            }
        }

        print("📤 Starting video generation...")
        response = authed_session.post(API_URL, json=payload)
        response.raise_for_status()
        response_json = response.json()
        print("🎬 Video generation started:", response_json)

        # Poll for completion
        operation_name = response_json["name"]
        operation_url = f"https://{LOCATION}-aiplatform.googleapis.com/v1/{operation_name}"
        
        attempts = 0
        while attempts < MAX_POLLING_ATTEMPTS:
            op_response = authed_session.get(operation_url)
            op_response.raise_for_status()
            op_json = op_response.json()
            
            if op_json.get("done"):
                print("Full operation response:", op_json)  # Debug the response structure
                
                if "error" in op_json:
                    error_msg = op_json["error"].get("message", "Unknown error")
                    raise Exception(f"Video generation failed: {error_msg}")
                
                # Try different possible locations for the video URI
                video_uri = None
                if "response" in op_json:
                    response_data = op_json["response"]
                    if isinstance(response_data, dict):
                        video_uri = response_data.get("uri")
                    elif isinstance(response_data, str):
                        video_uri = response_data
                
                if video_uri:
                    print(f"✅ Video ready at: {video_uri}")
                    break
                else:
                    raise Exception("Video generation completed but no URI found in response")
            else:
                print(f"⏳ Waiting for video generation to complete... (attempt {attempts + 1}/{MAX_POLLING_ATTEMPTS})")
                time.sleep(POLLING_INTERVAL)
                attempts += 1
        
        if attempts >= MAX_POLLING_ATTEMPTS:
            raise TimeoutError(f"Video generation timed out after {MAX_POLLING_ATTEMPTS * POLLING_INTERVAL} seconds")

        # Download the video
        local_path = get_next_clip_path()
        download_from_gcs(video_uri, local_path, credentials)
        print(f"📁 Saved to: {local_path}")
        
        return local_path
        
    except Exception as e:
        print(f"Error generating video: {str(e)}")
        raise

if __name__ == "__main__":
    # Example prompt
    prompt = (
        "A cinematic, medium wide 16:9 shot of a realistic hand slicing a translucent glass apple "
        "on a wooden cutting board. The cutting board is centered with ample empty wooden table space "
        "on the left and right sides, leaving room for cropping. Sharp, crisp sound of the glass cracking "
        "as the knife cuts through, followed by a satisfying, soft tap when the apple slice lands on the board. "
        "Realistic textures, slow-motion, high detail. Emphasize the sound design: glass crack and gentle impact."
    )
    generate_text_to_video(prompt) 