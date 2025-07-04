#!/usr/bin/env python3
"""
TikTok-Style Line-by-Line Caption Display with WhisperX
Simplified font handling - install once, use always.
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
import textwrap
from pathlib import Path
from PIL import ImageFont

# Simple font path - install here once
FONT_PATH = "/Library/Fonts/Arial_Black.ttf"

def _text_pixel_width(text, font_path, pts):
    """Return the pixel width of a text string in the given font + point size."""
    font = ImageFont.truetype(font_path, pts)
    return font.getlength(text)          # Pillow ≥9.2

def setup_font():
    """
    Simple font setup: copy to /Library/Fonts/ if needed.
    """
    if os.path.exists(FONT_PATH):
        print(f"✅ Font ready at: {FONT_PATH}")
        return FONT_PATH
    
    # Try to copy from system location
    source_font = "/System/Library/Fonts/Supplemental/Arial Black.ttf"
    if os.path.exists(source_font):
        try:
            print(f"📦 Installing font to: {FONT_PATH}")
            shutil.copy2(source_font, FONT_PATH)
            print("✅ Font installed successfully")
            return FONT_PATH
        except Exception as e:
            print(f"❌ Failed to install font: {e}")
            print("Using system font instead")
            return None
    
    print("⚠️  Arial Black not found, using system font")
    return None

def test_font_with_ffmpeg(font_path):
    """
    Test if a font works with FFmpeg by running a simple test command.
    """
    try:
        test_cmd = [
            'ffmpeg', '-f', 'lavfi', '-i', 'color=red:size=100x100:duration=1',
            '-vf', f"drawtext=fontfile='{font_path}':text='test':x=10:y=10",
            '-f', 'null', '-'
        ]
        
        result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except:
        return False

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

def wrap_title_text(title, max_chars_per_line=25):
    """
    Wrap title text into multiple lines for 9:16 video format.
    Optimized for mobile viewing with proper line breaks.
    """
    if not title:
        return []
    
    # Use Python's textwrap for intelligent word wrapping
    wrapped_lines = textwrap.fill(title, width=max_chars_per_line).split('\n')
    
    # Clean up any extra whitespace
    wrapped_lines = [line.strip() for line in wrapped_lines if line.strip()]
    
    return wrapped_lines

def create_title_overlay_filter(title, font_path=None,
                                font_size=95, line_wrap=16,
                                top_margin=80, pad=40):
    """
    Draw every wrapped-title line with its own centred blue box.
    Each box width is driven by drawtext's real text_w value,
    so it is always perfectly centred on a 9:16 (1080-pixel-wide) frame.
    """
    if not title:
        return ""

    title_font_size = font_size - 20                       # smaller title
    wrapped         = textwrap.fill(title, width=line_wrap).split("\n")
    line_h          = title_font_size + pad
    filters         = []

    for i, txt in enumerate(wrapped):
        y_pos = top_margin + i * line_h
        esc   = txt.replace("'", r"\'").replace(":", r"\:")

        font_clause = (f"fontfile='{font_path}'"
                       if font_path and os.path.exists(font_path)
                       else "font='Arial Black'")

        # One drawtext per line: its own blue box, centred with (w-text_w)/2
        filters.append(
            "drawtext="
            f"{font_clause}:"
            f"fontsize={title_font_size}:"
            "fontcolor=white:"
            "shadowcolor=black:shadowx=0:shadowy=0:"
            "box=1:"                               # blue background ON
            "boxcolor=#2F6EF6@1:"
            "boxborderw=10:"                       # thickness / padding
            "x=(w-text_w)/2:"                      # centre on 1080-wide frame
            f"y={y_pos}:"
            f"text='{esc}'"
        )

    return ",".join(filters)

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

def tiktok_line_alignment(video_path, output_path, model_size="large-v2",
    font_size=95, font_name="Arial Black", title=None, line_length=20):
    """
    Generate TikTok-style line-by-line captions with instant transitions.
    """
    
    print("🎬 Starting TikTok-style line-by-line alignment...")
    print(f"📹 Input: {os.path.basename(video_path)}")
    
    # Setup font once
    font_path = setup_font()
    print(f"🔤 Using font: {font_path if font_path else 'System font'}")
    
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
    
    # Load WhisperX model
    print("📥 Loading WhisperX model...")
    model = whisperx.load_model(model_size, device, compute_type=compute_type, language="en")
    
    # Load and transcribe audio
    print("🎵 Loading audio...")
    audio = whisperx.load_audio(video_path)
    
    print("🔄 Transcribing with word-level precision...")
    result = model.transcribe(audio, batch_size=batch_size)
    
    # Clear model from GPU memory
    gc.collect()
    torch.cuda.empty_cache()
    del model
    
    # Load alignment model for word-level timestamps
    print("📍 Loading word-level alignment model...")
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    
    # Perform word-level alignment
    print("✨ Performing word-level alignment...")
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    
    # Clear alignment model
    gc.collect()
    torch.cuda.empty_cache()
    del model_a
    
    # Extract word-level intervals
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
    
    # Convert to line-based segments
    print("📝 Creating line-based caption segments...")
    line_segments = create_line_segments_from_words(word_intervals, max_chars_per_line=20)
    
    print(f"✅ Created {len(line_segments)} line segments")
    for i, segment in enumerate(line_segments[:3]):
        print(f"   Line {i+1}: '{segment['text'][:30]}{'...' if len(segment['text']) > 30 else ''}' ({segment['end'] - segment['start']:.1f}s)")
    
    # Generate line-based ASS file
    print("🎬 Generating line-by-line TikTok captions...")
    ass_path = output_path.replace('.mp4', '.ass')
    write_line_based_ass(line_segments, ass_path, font_size, font_name)
    
    # Burn captions into video
    print("🔥 Burning line-based captions into 9:16 video...")
    burn_tiktok_captions_916_enhanced(video_path, ass_path, output_path, title, font_size, font_name, line_length, font_path)
    
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
Style: TikTokLine,{font_name},{font_size},&H00FFFFFF,&H00FF6600,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,3,0,1,40,40,350,1

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

def burn_tiktok_captions_916_enhanced(video_in, ass_path, video_out, title=None, font_size=95, font_name="Arial Black", line_length=20, font_path=None):
    """Burn TikTok-style captions with enhanced title handling for 9:16 format."""
    print("🔥 Burning enhanced captions for 9:16 format...")
    
    filters = []
    
    # Add title overlay if provided
    if title:
        print(f"📝 Adding multi-line title: '{title}'")
        title_filter = create_title_overlay_filter(
            title, font_path, font_size, line_length, 80, 40
        )
        if title_filter:
            filters.append(title_filter)
    
    # Add subtitle filter
    subtitle_filter = (
        f"subtitles={ass_path}:"
        f"force_style='Alignment=1,MarginV=350,PrimaryColour=&H00FFFFFF,"
        f"SecondaryColour=&H00FF6600,Outline=3,BackColour=&H80000000'"
    )
    filters.append(subtitle_filter)
    
    # Combine all filters
    filter_chain = ",".join(filters)
    
    # FFmpeg command
    cmd = [
        "ffmpeg", "-y", "-i", video_in,
        "-vf", filter_chain,
        "-c:v", "libx264", "-preset", "medium", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k", "-aspect", "9:16",
        video_out
    ]
    
    print("🎬 Running FFmpeg with corrected parameters...")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Enhanced TikTok captions burned successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg error: {e}")
        print("Error output:", e.stderr)
        raise

def main():
    parser = argparse.ArgumentParser(description="Generate TikTok-style captions with simplified font handling")
    
    parser.add_argument("video", help="Input video file")
    parser.add_argument("transcript", help="Transcript file (optional for WhisperX)")
    parser.add_argument("output", help="Output video with enhanced captions")
    parser.add_argument("--model", default="large-v2", 
                       choices=["tiny", "base", "small", "medium", "large", "large-v2"],
                       help="WhisperX model size (default: large-v2)")
    parser.add_argument("--font-size", type=int, default=95, help="Caption font size (default: 95)")
    parser.add_argument("--font-name", default="Arial Black", help="Font name (default: Arial Black)")
    parser.add_argument("--line-length", type=int, default=20, help="Max characters per caption line (default: 20)")
    parser.add_argument("--title", help="Title with automatic line wrapping and white background")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.video):
        print(f"❌ Video file not found: {args.video}")
        sys.exit(1)
    
    try:
        # Check dependencies
        if not check_dependencies():
            sys.exit(1)
        
        # Generate enhanced TikTok captions
        tiktok_line_alignment(args.video, args.output, args.model, args.font_size, args.font_name, args.title, args.line_length)
        
        print(f"🎉 SUCCESS! Enhanced TikTok video saved to: {args.output}")
        print("   📝 Line-by-line captions with instant transitions")
        print("   📱 Optimized for 9:16 mobile viewing")
        print("   🔵 Blue word highlighting within each line")
        print("   🔤 Simple font handling - installed once, works always")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 