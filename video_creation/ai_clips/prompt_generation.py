"""
Script for generating Veo3 video prompts using OpenAI API.
This script generates highly specific, photorealistic prompts for ASMR-style glass fruit slicing videos.
"""

import os
import json
import logging
import re
from openai import OpenAI
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'secrets', '.env')
load_dotenv(env_path)

def get_random_real_fruit_and_color():
    """
    Use OpenAI API to get a random real fruit and color (not made up).
    """
    try:
        # Get API key from environment
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        # Clear any proxy-related environment variables that might interfere
        proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'NO_PROXY', 'no_proxy']
        for var in proxy_vars:
            if var in os.environ:
                logger.info(f"Clearing proxy environment variable: {var}")
                del os.environ[var]
        
        # Initialize client with explicit API key and no proxies
        client = OpenAI(
            api_key=api_key,
            # Explicitly set no proxies
            http_client=None
        )
        model = os.getenv('OPENAI_MODEL', 'gpt-4o')

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Pick one random real fruit (e.g. mango, kiwi, fig, etc.) and one real color (e.g. teal, amber, magenta, etc.). Do not suggest a pineapple or banana."
                        "Return only valid, real-world examples. Output only a JSON object with 'fruit' and 'color'. "
                        "No explanation or extra text."
                    )
                }
            ]
        )
        raw_content = response.choices[0].message.content.strip()
        logger.info("Raw API response: " + raw_content)

        # Strip any markdown (e.g. ```json) so that json.loads() receives a valid JSON string.
        cleaned_content = re.sub(r"```(?:json)?\s*", "", raw_content, flags=re.IGNORECASE).strip()
        logger.info("Cleaned API response: " + cleaned_content)

        data = json.loads(cleaned_content)
        return data["fruit"], data["color"]

    except Exception as e:
        logger.error(f"Error getting fruit and color: {str(e)}")
        raise

def generate_prompt(fruit, color):
    return f"""
A cinematic, photorealistic close-up of a {color} translucent glass {fruit} being sliced in an ASMR-style video.

The shot begins with a knife blade already positioned just above the fruit — no motion or approach. The knife makes a single, smooth, vertical cut through the center of the fruit, continuing all the way down until it touches the cutting board. This slicing motion should be in slow motion and occupy nearly five full seconds of the video, emphasizing the texture and resistance of the glassy material.

Immediately after the slice is fully separated, the knife lifts up and exits the frame as the freshly cut slice falls.

The fruit must have a realistic shape and a fully translucent glass appearance, both inside and out. Only one slice is removed, and it must cleanly fall away in the same orientation as the knife cut. The fall lasts no longer than one second and ends with a clear, satisfying, resonant clink as the slice hits the wooden cutting board. The slice and remaining fruit then remain completely still.

There are no hands, fingers, or knife handles visible — only the knife blade in contact with the fruit is shown. The fruit is untouched and rests naturally on the board. No transitions, fades, or additional effects are used.

The audio must enhance the ASMR quality: a soft, shimmering, glass-like slicing sound during the cut, followed by a crisp, crystal-clear clink as the slice lands.
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