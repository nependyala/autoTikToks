"""
Script to analyze video transcripts using GPT-4 and identify viral-worthy moments.
"""
import os
import json
import sys
from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv

# Debug: Print current file location
print(f"Current file: {__file__}")
print(f"Current directory: {os.path.dirname(__file__)}")
print(f"Parent directory: {os.path.dirname(os.path.dirname(__file__))}")

# Construct and verify .env path
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'secrets', '.env')
print(f"Looking for .env file at: {env_path}")
print(f"File exists: {os.path.exists(env_path)}")

# Load environment variables from .env file
load_dotenv(env_path)

# Debug: Print environment variables
print(f"OPENAI_API_KEY exists: {bool(os.getenv('OPENAI_API_KEY'))}")

def load_credentials():
    """Load credentials from environment variables."""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise Exception("OpenAI API key not found in environment variables")
    return {'OPENAI_API_KEY': api_key}

def read_transcript(transcript_name: str) -> str:
    """
    Read transcript from the temporary files directory.
    
    Args:
        transcript_name (str): Name of the transcript file
        
    Returns:
        str: Content of the transcript file
    """
    transcript_path = os.path.join(os.path.dirname(__file__), 'temporary_files', transcript_name)
    if not os.path.exists(transcript_path):
        raise FileNotFoundError(f"Transcript file not found: {transcript_name}")
    
    with open(transcript_path, 'r', encoding='utf-8') as f:
        return f.read()

def analyze_transcript(transcript_content: str) -> List[Dict]:
    """
    Analyze transcript using GPT-4 to identify viral-worthy moments.
    
    Args:
        transcript_content (str): Content of the transcript
        
    Returns:
        List[Dict]: List of identified moments with start and end timestamps
    """
    prompt = """You are an assistant that analyzes video transcripts and extracts the funniest, most amusing, or shocking moments that could go viral online.

Input transcript format:
Each segment has an index number, a start and end timestamp in "HH:MM:SS,mmm --> HH:MM:SS,mmm" format, followed by the spoken text.

Your task:
- Identify and extract segments or contiguous groups of segments that contain funny, outlandish, amusing, or shocking content.
- Each extracted moment MUST be:
  - At least 15 seconds long
  - Never longer than 1 minute and 20 seconds (1:20)
  - Ideally between 20-45 seconds for maximum impact
- Return a JSON array of objects, each with:
  - "start": the start timestamp of the moment (string, format HH:MM:SS,mmm)
  - "end": the end timestamp of the moment (string, format HH:MM:SS,mmm)
  - "title": a brief description of what Charlie is talking about in this moment (string, in second person, maximum 7 words)

IMPORTANT: 
- Your response must be a valid JSON array starting with [ and ending with ].
- Do not include any other text.
- Always refer to the speaker as "Charlie" in the title.
- Write titles in second person (e.g., "Charlie talks about...", "Charlie explains...", "Charlie reacts to...")
- Ensure each clip is at least 15 seconds and never exceeds 1:20 in length
- Keep titles concise and never longer than 7 words

Example response:
[
  {
    "start": "00:00:00,160",
    "end": "00:00:35,200",
    "title": "Charlie explains Elon and Trump's falling out"
  },
  {
    "start": "00:01:20,000",
    "end": "00:01:55,000",
    "title": "Charlie reacts to the debate's biggest moment"
  }
]"""

    try:
        credentials = load_credentials()
        # Initialize OpenAI client
        client = OpenAI(api_key=credentials.get('OPENAI_API_KEY'))
        model = os.getenv('OPENAI_MODEL', 'gpt-4.1-mini-2025-04-14')
        
        print("Analyzing transcript...")
        # Get response from OpenAI
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that analyzes video transcripts to find viral-worthy moments. You must respond with only a valid JSON array. Always refer to the speaker as Charlie in the titles. Each clip must be at least 15 seconds and never longer than 1:20. Keep titles under 7 words."},
                {"role": "user", "content": f"Transcript text: {transcript_content}\n\nPrompt: {prompt}"}
            ]
        )

        # Parse JSON response
        try:
            content = response.choices[0].message.content.replace('```json', '').replace('```', '').strip()
            
            # Clean up the response string
            if not content.startswith('['):
                content = '[' + content
            if not content.endswith(']'):
                content = content + ']'
                
            return json.loads(content)
            
        except json.JSONDecodeError as e:
            print(f"Error parsing response: {str(e)}")
            return []
            
    except Exception as e:
        print(f"Error: {str(e)}")
        raise Exception(f"Error analyzing transcript: {str(e)}")

def save_moments(moments: List[Dict], transcript_name: str) -> str:
    """
    Save identified moments to a JSON file.
    
    Args:
        moments (List[Dict]): List of identified moments
        transcript_name (str): Name of the original transcript file
        
    Returns:
        str: Path to the saved moments file
    """
    # Create moments filename based on transcript name
    base_name = os.path.splitext(transcript_name)[0]
    moments_file = f"{base_name}_moments.json"
    moments_path = os.path.join(os.path.dirname(__file__), 'temporary_files', moments_file)
    
    # Save moments to file
    with open(moments_path, 'w', encoding='utf-8') as f:
        json.dump(moments, f, indent=2)
    
    return moments_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python decide_clip.py <transcript_filename>")
        print("Example: python decide_clip.py transcript_RywoFvefNOE.txt")
        sys.exit(1)
    
    transcript_name = sys.argv[1]
    
    try:
        # Read transcript
        transcript_content = read_transcript(transcript_name)
        
        # Analyze transcript
        moments = analyze_transcript(transcript_content)
        
        # Save moments
        moments_path = save_moments(moments, transcript_name)
        
        print(f"Found {len(moments)} viral-worthy moments")
        print(f"Moments saved to: {moments_path}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1) 