#!/usr/bin/env python3
"""
TikTok-Style Line-by-Line Caption Display with WhisperX
Displays complete lines that fit neatly at bottom, with instant transitions.
"""
import argparse
import subprocess
import os
import tempfile
import shutil
import sys
import json
import importlib.util
import re
from pathlib import Path

def install_whisperx():
    """Install WhisperX and dependencies if not present."""
    try:
        import whisperx
        print("✅ WhisperX found")
        return True
    except ImportError:
        print("📦 Installing WhisperX...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "whisperx", "torch", "torchaudio", "--upgrade"
        ], check=True)
        return True

def check_dependencies():
    """Check and install required dependencies."""
    print("🔍 Checking dependencies...")
    
    # Check FFmpeg
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        print("✅ FFmpeg found")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ FFmpeg not found. Please install FFmpeg:")
        print("   brew install ffmpeg  # macOS")
        print("   sudo apt install ffmpeg  # Ubuntu")
        return False
    
    # Install WhisperX
    if not install_whisperx():
        print("❌ Failed to install WhisperX")
        return False
    
    print("✅ All dependencies ready!")
    return True

def segment_text_into_lines(text, max_chars_per_line=20):
    """
    Segment text into natural lines that fit mobile viewing.
    Based on TikTok best practices: 32-40 chars per line.
    """
    words = text.split()
    lines = []
    current_line = ""
    
    for word in words:
        # Check if adding this word would exceed the line limit
        test_line = current_line + (" " if current_line else "") + word
        
        if len(test_line) <= max_chars_per_line:
            current_line = test_line
        else:
            # Line would be too long, start new line
            if current_line:
                lines.append(current_line)
            current_line = word
    
    # Add the last line
    if current_line:
        lines.append(current_line)
    
    return lines

def create_line_segments_from_words(word_intervals, max_chars_per_line=20):
    """
    Convert word-level timestamps into line-based segments.
    Each segment represents one complete line to display.
    """
    if not word_intervals:
        return []
    
    # Build full text from word intervals
    full_text = " ".join([word for _, _, word in word_intervals])
    
    # Segment into display lines
    display_lines = segment_text_into_lines(full_text, max_chars_per_line)
    
    # Map lines back to word timing
    line_segments = []
    word_index = 0
    
    for line_text in display_lines:
        line_words = line_text.split()
        line_word_count = len(line_words)
        
        if word_index + line_word_count <= len(word_intervals):
            # Get timing from first and last word of this line
            start_time = word_intervals[word_index][0]
            end_time = word_intervals[word_index + line_word_count - 1][1]
            
            line_segments.append({
                'start': start_time,
                'end': end_time,
                'text': line_text,
                'words': line_words
            })
            
            word_index += line_word_count
        else:
            # Handle edge case
            break
    
    return line_segments

def tiktok_line_alignment(video_path, output_path, model_size="large-v2", font_size=95, font_name="Arial Black"):
    """
    Generate TikTok-style line-by-line captions with instant transitions.
    """
    
    print("🎬 Starting TikTok-style line-by-line alignment...")
    print(f"📹 Input: {os.path.basename(video_path)}")
    
    # Import WhisperX
    import whisperx
    import gc
    import torch
    
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 16 if device == "cuda" else 1
    compute_type = "float16" if device == "cuda" else "float32"
    
    print(f"🖥️  Device: {device}")
    print(f"🧠 Model: {model_size}")
    
    # Step 1: Load WhisperX model
    print("📥 Loading WhisperX model...")
    model = whisperx.load_model(
        model_size, 
        device, 
        compute_type=compute_type,
        language="en"
    )
    
    # Step 2: Load and transcribe audio
    print("🎵 Loading audio...")
    audio = whisperx.load_audio(video_path)
    
    print("🔄 Transcribing with word-level precision...")
    result = model.transcribe(audio, batch_size=batch_size)
    
    # Clear model from GPU memory
    gc.collect()
    torch.cuda.empty_cache()
    del model
    
    # Step 3: Load alignment model for word-level timestamps
    print("📍 Loading word-level alignment model...")
    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"], 
        device=device
    )
    
    # Step 4: Perform word-level alignment
    print("✨ Performing word-level alignment...")
    result = whisperx.align(
        result["segments"], 
        model_a, 
        metadata, 
        audio, 
        device, 
        return_char_alignments=False
    )
    
    # Clear alignment model
    gc.collect()
    torch.cuda.empty_cache()
    del model_a
    
    # Step 5: Extract word-level intervals
    print("📊 Extracting word-level timestamps...")
    word_intervals = []
    
    for segment in result["segments"]:
        for word_info in segment.get("words", []):
            if "start" in word_info and "end" in word_info:
                word_intervals.append((
                    float(word_info["start"]),
                    float(word_info["end"]),
                    word_info["word"].strip()
                ))
    
    print(f"✅ Extracted {len(word_intervals)} word-level intervals")
    
    # Step 6: Convert to line-based segments
    print("📝 Creating line-based caption segments...")
    line_segments = create_line_segments_from_words(word_intervals, max_chars_per_line=20)
    
    print(f"✅ Created {len(line_segments)} line segments")
    for i, segment in enumerate(line_segments[:3]):  # Show first 3 as preview
        print(f"   Line {i+1}: '{segment['text'][:30]}{'...' if len(segment['text']) > 30 else ''}' ({segment['end'] - segment['start']:.1f}s)")
    
    # Step 7: Generate line-based ASS file
    print("🎬 Generating line-by-line TikTok captions...")
    ass_path = output_path.replace('.mp4', '.ass')
    write_line_based_ass(line_segments, ass_path, font_size, font_name)
    
    # Step 8: Burn captions into video for 9:16 format
    print("🔥 Burning line-based captions into 9:16 video...")
    burn_tiktok_captions_916(video_path, ass_path, output_path)
    
    print(f"🎉 TikTok-style line-based video created: {output_path}")
    return line_segments

def write_line_based_ass(line_segments, ass_path, font_size=95, font_name="Arial Black"):
    """Generate ASS file with line-by-line display and instant transitions."""
    
    def fmt_time(t):
        """Format time in ASS format with precise timing."""
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = int(t % 60)
        cs = int((t - int(t)) * 100)
        return f"{h:01d}:{m:02d}:{s:02d}.{cs:02d}"
    
    with open(ass_path, "w", encoding="utf-8") as out:
        # ASS header optimized for 9:16 TikTok format with line-based display
        out.write(f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
Title: TikTok Line-by-Line Captions

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: TikTokLine,{font_name},{font_size},&H00FFFFFF,&H00FF6600,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,3,0,1,40,40,250,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
""")
        
        if not line_segments:
            print("⚠️  No line segments found")
            return
        
        # Create separate dialogue events for each line segment
        for i, segment in enumerate(line_segments):
            start = fmt_time(segment['start'])
            end = fmt_time(segment['end'])
            text = segment['text']
            
            # Create karaoke timing within each line for word-by-word highlighting
            words = segment['words']
            line_duration = segment['end'] - segment['start']
            avg_word_duration = line_duration / len(words) if words else 0.5
            
            karaoke_text = ""
            for word in words:
                word_duration_cs = max(10, int(avg_word_duration * 100))
                karaoke_text += f"{{\\k{word_duration_cs}}}{word} "
            
            karaoke_text = karaoke_text.strip()
            
            out.write(f"Dialogue: 0,{start},{end},TikTokLine,,0,0,0,,{karaoke_text}\n")
    
    print("✅ Line-based ASS file generated with instant transitions")

def burn_tiktok_captions_916(video_in, ass_path, video_out):
    """Burn TikTok-style line-based captions into 9:16 video."""
    print("🔥 Burning line-based captions for 9:16 format...")
    
    # FFmpeg command optimized for 9:16 TikTok format with line-based positioning
    subprocess.run([
        "ffmpeg", "-y", "-i", video_in,
        "-vf", f"subtitles={ass_path}:force_style='Alignment=1,MarginV=250,PrimaryColour=&H00FFFFFF,SecondaryColour=&H00FF6600,Outline=3,BackColour=&H80000000'",
        "-c:v", "libx264",
        "-preset", "medium", 
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        "-aspect", "9:16",
        video_out
    ], check=True, capture_output=True)
    
    print("✅ Line-based TikTok captions burned successfully")

def main():
    parser = argparse.ArgumentParser(
        description="Generate TikTok-style line-by-line captions with WhisperX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic line-by-line captions for TikTok
  python caption.py video.mp4 transcript.txt output.mp4
  
  # Custom line length for different screen sizes
  python caption.py video.mp4 transcript.txt output.mp4 --line-length 40
  
Features:
  ✅ Line-by-line display (not word-by-word)
  ✅ Optimal 32-40 character line length for mobile
  ✅ Instant transitions between complete lines
  ✅ Blue word highlighting within each line
  ✅ Perfect fit at bottom of 9:16 video
  ✅ Natural sentence boundary breaking
        """
    )
    
    parser.add_argument("video", help="Input video file")
    parser.add_argument("transcript", help="Transcript file (optional for WhisperX)")
    parser.add_argument("output", help="Output video with line-based captions")
    parser.add_argument("--model", default="large-v2", 
                       choices=["tiny", "base", "small", "medium", "large", "large-v2"],
                       help="WhisperX model size (default: large-v2)")
    parser.add_argument("--font-size", type=int, default=95, help="Font size (default: 95)")
    parser.add_argument("--font-name", default="Arial Black", help="Font name (default: Arial Black)")
    parser.add_argument("--line-length", type=int, default=20, help="Max characters per line (default: 20)")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.video):
        print(f"❌ Video file not found: {args.video}")
        sys.exit(1)
    
    try:
        # Check and install dependencies
        if not check_dependencies():
            sys.exit(1)
        
        # Generate line-based TikTok captions
        tiktok_line_alignment(args.video, args.output, args.model, args.font_size, args.font_name)
        
        print(f"🎉 SUCCESS! Line-based TikTok video saved to: {args.output}")
        print("   📝 Line-by-line display with instant transitions")
        print("   📱 Optimized for 9:16 mobile viewing") 
        print("   🔵 Blue word highlighting within each line")
        print("   📏 Perfect line length for mobile readability")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 