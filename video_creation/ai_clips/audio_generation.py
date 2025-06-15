"""
Script for generating AI audio using Hugging Face's AudioGen model locally.
This script uses the audiocraft library to generate audio from text prompts.
"""

import os
import logging
from pathlib import Path
import torch
from audiocraft.models import AudioGen
from audiocraft.data.audio import audio_write
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DOWNLOAD_DIR = Path(__file__).parent / "temporary_files"
DOWNLOAD_DIR.mkdir(exist_ok=True)

def get_next_audio_path():
    """Get the next available audio file path."""
    existing_files = list(DOWNLOAD_DIR.glob("generated_audio_*.wav"))
    if not existing_files:
        return DOWNLOAD_DIR / "generated_audio_1.wav"
    # Use a regex to extract the number (e.g. "generated_audio_1" yields 1) and ignore non-numeric suffixes.
    filtered_files = (f for f in existing_files if re.search(r"generated_audio_(\d+)", f.stem) is not None)
    last_num = max((int(re.search(r"generated_audio_(\d+)", f.stem).group(1)) for f in filtered_files), default=0)
    return DOWNLOAD_DIR / f"generated_audio_{last_num + 1}.wav"

def generate_audio(prompt, duration=5.0):
    """
    Generate audio based on a prompt using the local AudioGen model.
    
    Args:
        prompt (str): Text description of the sound to generate
        duration (float): Duration of the audio in seconds (default: 5.0)
    
    Returns:
        str: Path to the generated audio file, or None if generation failed
    """
    try:
        logger.info("Loading AudioGen model...")
        model = AudioGen.get_pretrained('facebook/audiogen-medium')
        
        # Log device info but don't try to move the model
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        logger.info(f"Generating audio for prompt: {prompt}")
        logger.info("This may take a few minutes...")
        
        # Generate audio
        wav = model.generate(
            descriptions=[prompt],
            progress=True
        )
        
        # Save the generated audio
        output_path = get_next_audio_path()
        logger.info(f"Saving audio to: {output_path}")
        
        # Save with loudness normalization
        audio_write(
            output_path,
            wav[0].cpu(),
            model.sample_rate,
            strategy="loudness",
            loudness_compressor=True
        )
        
        logger.info("Audio generation completed successfully!")
        return str(output_path)
        
    except Exception as e:
        logger.error(f"Error generating audio: {str(e)}")
        return None

if __name__ == "__main__":
    output_file = generate_audio("An ultra-satisfying ASMR sound of a razor-sharp blade gliding through a smooth, dense object — like polished fruit or gel. The slice is clean, with a silky shimmer, a soft metallic sheen, and a gentle swish. No crunch, no crackle — just a serene, fluid motion. Emphasize a studio-recorded, whisper-quiet environment with rich, polished acoustics")
    if output_file:
        print(f"Generated audio saved to: {output_file}")
    else:
        print("Failed to generate audio") 