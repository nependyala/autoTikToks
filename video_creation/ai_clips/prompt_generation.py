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

def get_random_real_fruit_and_color():
    """
    Use OpenAI API to get a random real fruit and color (not made up).
    """
    try:
        client = OpenAI()
        model = os.getenv('OPENAI_MODEL', 'gpt-4o')

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Pick one random real fruit (e.g. mango, kiwi, fig, etc.) and one real color (e.g. teal, amber, magenta, etc.). "
                        "Return only valid, real-world examples. Output only a JSON object with 'fruit' and 'color'. "
                        "No explanation or extra text."
                    )
                }
            ]
        )
        content = response.choices[0].message.content.strip()
        logger.info(f"API response: {content}")

        data = json.loads(content)
        return data["fruit"], data["color"]

    except Exception as e:
        logger.error(f"Error getting fruit and color: {str(e)}")
        raise

def generate_prompt(fruit, color):
    """
    Build a consistent Veo3-style slicing video prompt using the given fruit and color.
    """
    return f"""
A highly detailed, photorealistic close-up of a {color} translucent glass {fruit} being sliced in a cinematic ASMR video. 
The camera is perfectly still. A sharp knife makes one clean slice through the fruit. 
The slice separates and falls away naturally, leaving a visible wedge missing. 
The sliced piece remains fully intact and shaped like a natural wedge. 
The rest of the fruit stays motionless and visibly incomplete. 
The lighting is cinematic and high-resolution, emphasizing the realistic, glassy texture of the fruit. 
The audio features a soft shimmering glassy tone as the knife cuts, followed by a gentle, satisfying clink 
when the slice hits the wooden cutting board.
""".strip()

if __name__ == "__main__":
    try:
        fruit, color = get_random_real_fruit_and_color()
        prompt = generate_prompt(fruit, color)
        print("\nGenerated Prompt:")
        print("-" * 80)
        print(prompt)
        print("-" * 80)
    except Exception as e:
        print(f"Failed to generate prompt: {str(e)}") 