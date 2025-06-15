"""
Script for generating Veo3 video prompts using OpenAI API.
This script generates highly specific, photorealistic prompts for ASMR-style glass fruit slicing videos.
"""

import os
import json
import logging
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

print("OpenAI class is from:", OpenAI.__module__)
def generate_prompt() -> str:
    """
    Generate a Veo3-ready prompt using OpenAI API.
    
    Returns:
        str: Generated prompt for Veo3 video generation
    """
    try:
        # No need to pass api_key, OpenAI() will use env var
        client = OpenAI()
        model = os.getenv('OPENAI_MODEL', 'gpt-4.1-mini-2025-04-14')
        
        logger.info("Generating Veo3 prompt...")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a video prompt generator for Veo3. Your task is to write highly specific, photorealistic video prompts of a translucent glass fruit being sliced in an ASMR-style video. Follow these rules:\n\n1. Choose a random fruit (e.g. apple, pear, plum, peach, orange, fig).\n2. Choose a random color (e.g. green, red, blue, pink, purple, amber).\n3. The fruit must be described as **realistic** and **translucent glass**, with cinematic lighting.\n4. The camera should be close-up and **still** (no movement).\n5. The **knife should slice cleanly**, with **one distinct slice separating and falling away**.\n6. After the slice falls, there must be a **visible missing section** in the fruit.\n7. The slice must **stay intact and shaped like a natural wedge**.\n8. The rest of the fruit remains motionless and visibly incomplete.\n9. Emphasize **object continuity** and **no shape warping**.\n10. The visuals must be **high-resolution, clean, and photorealistic**.\n11. The audio must be **ASMR-style**: a soft, shimmering glassy tone as the knife cuts, followed by a gentle, satisfying clink when the slice hits the wooden cutting board.\n\nOutput only the final prompt. Do not include any notes, preamble, or quotes. The prompt should be Veo3-ready."}
            ]
        )
        
        prompt = response.choices[0].message.content.strip()
        logger.info("Successfully generated prompt")
        return prompt
        
    except Exception as e:
        logger.error(f"Error generating prompt: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        prompt = generate_prompt()
        print("\nGenerated Prompt:")
        print("-" * 80)
        print(prompt)
        print("-" * 80)
    except Exception as e:
        print(f"Failed to generate prompt: {str(e)}") 