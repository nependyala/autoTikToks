"""
Script to analyze video transcripts using Gemini and identify viral-worthy moments.
"""
import os
import json
import sys
import requests
from typing import List, Dict
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'secrets', '.env')
load_dotenv(env_path)

def load_credentials():
    """Load credentials from environment variables."""
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise Exception("GEMINI_API_KEY not found in environment variables")
    return {'GEMINI_API_KEY': api_key}

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
    Analyze transcript using Gemini to identify viral-worthy moments.
    
    Args:
        transcript_content (str): Content of the transcript
        
    Returns:
        List[Dict]: List of identified moments with start and end timestamps
    """
    prompt = """You are an expert assistant that analyzes video transcripts to extract complete, self-contained moments that could go viral online.

Input transcript format:
Each segment has an index number, a start and end timestamp in "HH:MM:SS,mmm --> HH:MM:SS,mmm" format, followed by the spoken text.

CRITICAL INSTRUCTION - COMPLETE THOUGHT BOUNDARIES:
You MUST identify where one complete thought ends and a new thought begins. Look for these natural boundary markers:

THOUGHT COMPLETION SIGNALS (good ending points):
- Conclusive statements: "And I want to keep it that way", "That's why...", "So basically..."
- Summary phrases: "The point is...", "What I'm saying is...", "In other words..."
- Transition phrases before topic changes: "That being said...", "Now...", "Moving on..."
- Natural pauses before new topics or examples
- Rhetorical questions followed by answers that conclude the point

AVOID CUTTING AT:
- Mid-sentence or mid-clause
- In the middle of explanations or examples
- Before the speaker finishes their main point
- During lists or enumerations that aren't complete

PROCESS:
1. Read through the entire transcript first
2. Identify where Charlie introduces a topic or idea
3. Follow that idea through its development and explanation
4. Find where he naturally concludes that thought BEFORE moving to a new topic
5. Extract from the introduction to the natural conclusion

Your task:
- Extract segments that represent ONE complete thought or narrative
- Each clip should be 30 seconds to 1 minute 30 seconds
- Prioritize funny, shocking, or engaging content
- Ensure the ending feels natural and complete, not abrupt

Return a JSON array of objects with:
- "start": start timestamp (HH:MM:SS,mmm)
- "end": end timestamp (HH:MM:SS,mmm)
- "title": what Charlie discusses (second person, max 7 words)

EXAMPLE OF GOOD BOUNDARY DETECTION:
If Charlie says: "I've been happier because I go outside more. And I want to keep it that way. That being said, there's something new..." 
- END the clip at "keep it that way" (complete thought)
- DON'T include "That being said..." (starts new thought)

Your response must be valid JSON only, no other text.

[
  {
    "start": "00:00:00,080",
    "end": "00:01:11,520",
    "title": "Charlie explains his secret to happiness"
  }
]
"""

    try:
        credentials = load_credentials()
        api_key = credentials.get('GEMINI_API_KEY')
        
        # Gemini API endpoint
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
        
        # Prepare the request payload
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": f"Transcript text: {transcript_content}\n\nPrompt: {prompt}"
                        }
                    ]
                }
            ]
        }
        
        headers = {
            'Content-Type': 'application/json'
        }
        
        print("Analyzing transcript with Gemini...")
        
        # Make the API request
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Parse the response
        response_data = response.json()
        
        # Extract the text from Gemini response
        if 'candidates' in response_data and len(response_data['candidates']) > 0:
            content = response_data['candidates'][0]['content']['parts'][0]['text'].strip()
        else:
            raise Exception("No content found in Gemini response")
        
        print(f"Raw Gemini response: {content}")
        
        # Clean up the response string - remove markdown code blocks
        content = content.replace('```json', '').replace('```', '').strip()
        
        # Clean up the response string
        if not content.startswith('['):
            content = '[' + content
        if not content.endswith(']'):
            content = content + ']'
            
        return json.loads(content)
        
    except json.JSONDecodeError as e:
        print(f"Error parsing response: {str(e)}")
        return []
    except requests.exceptions.RequestException as e:
        print(f"Error making API request: {str(e)}")
        raise Exception(f"Error analyzing transcript: {str(e)}")
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