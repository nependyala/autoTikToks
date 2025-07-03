import os
import sys
import subprocess
import json

# Constants
TEMP_DIR = os.path.join(os.path.dirname(__file__), 'temporary_files')


def run_subprocess(cmd, cwd=None):
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running {' '.join(cmd)}:")
        print(result.stdout)
        print(result.stderr)
        sys.exit(1)
    return result.stdout

def get_video_id(youtube_url):
    import re
    match = re.search(r'(?:v=|/)([0-9A-Za-z_-]{11})', youtube_url)
    if match:
        return match.group(1)
    raise ValueError("Could not extract video ID from URL")


def main():
    if len(sys.argv) != 2:
        print("Usage: python master.py <YouTube_URL>")
        sys.exit(1)
    
    youtube_url = sys.argv[1]
    video_id = get_video_id(youtube_url)
    transcript_file = f"transcript_{video_id}.txt"
    moments_file = f"transcript_{video_id}_moments.json"

    # Step 1: Fetch transcript
    print("\n=== Step 1: Fetching transcript ===")
    run_subprocess([sys.executable, 'transcript_fetch.py', youtube_url], cwd=os.path.dirname(__file__))
    transcript_path = os.path.join(TEMP_DIR, transcript_file)
    if not os.path.exists(transcript_path):
        print(f"Transcript file not found: {transcript_path}")
        sys.exit(1)

    # Step 2: Analyze transcript for viral moments
    print("\n=== Step 2: Analyzing transcript for viral moments ===")
    run_subprocess([sys.executable, 'decide_clip.py', transcript_file], cwd=os.path.dirname(__file__))
    moments_path = os.path.join(TEMP_DIR, moments_file)
    if not os.path.exists(moments_path):
        print(f"Moments file not found: {moments_path}")
        sys.exit(1)

    # Step 3: Download only the first viral clip
    print("\n=== Step 3: Downloading the first viral clip ===")
    with open(moments_path, 'r', encoding='utf-8') as f:
        moments = json.load(f)
    if not moments:
        print("No viral moments found.")
        sys.exit(0)
    # Only process the first moment
    moment = moments[0]
    start = moment['start']
    end = moment['end']
    title = moment.get('title', 'clip_1')
    print(f"\n--- Downloading first clip: {title} ---")
    run_subprocess([
        sys.executable, 'clip_fetch.py', youtube_url, start, end
    ], cwd=os.path.dirname(__file__))
    print("\nFirst viral clip downloaded!")

    # Step 4: Edit the viral clip with maximum settings
    print("\n=== Step 4: Editing viral clip with maximum settings ===")
    print("Running edit.py with best quality, best frame detection, and facial movement check every 3 frames...")
    
    # Find the downloaded clip file
    clip_files = [f for f in os.listdir(TEMP_DIR) if f.startswith('clip_') and f.endswith('.mp4')]
    if not clip_files:
        print("No clip files found in temporary directory")
        sys.exit(1)
    
    # Use the most recent clip file (should be the one we just downloaded)
    input_clip = os.path.join(TEMP_DIR, clip_files[-1])
    output_clip = os.path.join(TEMP_DIR, f"edited_{clip_files[-1]}")
    
    print(f"Input clip: {input_clip}")
    print(f"Output clip: {output_clip}")
    
    run_subprocess([
        sys.executable, 'edit.py', 
        input_clip, output_clip,
        '--quality', 'high',
        '--face-sample-interval', '3',
        '--frame-perfect',
        '--dynamic-face-crop',
        '--progress'
    ], cwd=os.path.dirname(__file__))
    print("\nViral clip editing completed with maximum settings!")

if __name__ == "__main__":
    main() 