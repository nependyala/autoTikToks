"""
Transcript fetching module for handling video transcript extraction and processing.
AWS Lambda function that downloads YouTube video transcripts.
"""
import subprocess
import os
import json
import tempfile
from typing import Optional, Dict, Any

def fetch_transcript(video_url: str, output_dir: Optional[str] = None) -> str:
    """
    Download English auto-generated captions from a YouTube video using yt-dlp.
    
    Args:
        video_url (str): The URL of the YouTube video
        output_dir (str, optional): Directory to save the transcript. Defaults to /tmp for Lambda.
    
    Returns:
        str: Path to the downloaded transcript file
    """
    if output_dir is None:
        output_dir = '/tmp'  # Lambda's writable directory
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Define paths
    final_transcript_path = os.path.join(output_dir, "transcript.srt")
    
    # Remove existing transcript if it exists
    if os.path.exists(final_transcript_path):
        os.remove(final_transcript_path)
    
    try:
        # Construct the yt-dlp command
        command = [
            './yt-dlp',
            '--write-auto-subs',
            '--sub-lang', 'en',
            '--skip-download',
            '--convert-subs', 'srt',
            '-o', os.path.join(output_dir, 'transcript'),
            video_url
        ]
        
        # Run the command
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        
        if os.path.exists(final_transcript_path):
            # Read the transcript content
            with open(final_transcript_path, 'r', encoding='utf-8') as f:
                transcript_content = f.read()
            
            # Clean up the file
            os.remove(final_transcript_path)
            
            return transcript_content
        else:
            raise FileNotFoundError(f"Transcript file not found at {final_transcript_path}")
            
    except subprocess.CalledProcessError as e:
        raise Exception(f"Failed to download transcript: {e.stderr}")
    except Exception as e:
        raise Exception(f"Error processing video: {str(e)}")

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    AWS Lambda handler function.
    
    Args:
        event (dict): The event data passed to the Lambda function.
            Expected format: {"url": "https://www.youtube.com/watch?v=..."}
        context (object): The context object provided by AWS Lambda.
    
    Returns:
        dict: A response object with statusCode and body.
    """
    try:
        # Extract video URL from the event
        video_url = event.get('url')
        if not video_url:
            return {
                'statusCode': 400,
                'body': json.dumps({
                    'error': 'Missing url in the event data'
                })
            }
        
        # Fetch the transcript
        transcript_content = fetch_transcript(video_url)
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'transcript': transcript_content
            })
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e)
            })
        }

# For local testing
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python transcript_fetch.py <YouTube_URL>")
        sys.exit(1)
    video_url = sys.argv[1]
    try:
        transcript_content = fetch_transcript(video_url)
        print(f"Transcript content:\n{transcript_content}")
    except Exception as e:
        print(f"Error: {str(e)}") 