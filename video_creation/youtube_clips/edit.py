"""
Video editing script with facial recognition, scene detection, and video processing capabilities.
"""
import cv2
import face_recognition
import scenedetect
from scenedetect import detect, ContentDetector
import numpy as np
from moviepy import VideoFileClip, concatenate_videoclips, ColorClip, CompositeVideoClip
import os
import sys
from skimage.metrics import structural_similarity as ssim

class MoistCritikalVideoProcessor:
    def __init__(self, video_path, output_path, face_detection_confidence=0.6, scene_threshold=30.0):
        """
        Initialize the video processor.
        Args:
            video_path (str): Path to the input video file.
            output_path (str): Path to save the processed video.
            face_detection_confidence (float, optional): Confidence threshold for face detection. Default is 0.6.
            scene_threshold (float, optional): Threshold for scene detection. Default is 30.0.
        """
        self.video_path = video_path
        self.output_path = output_path
        self.face_detection_confidence = face_detection_confidence
        self.scene_threshold = scene_threshold
        self.detected_segments = []  # To store detected scene segments
        self.face_locations = []     # To store detected face locations
        self.scene_boundaries = []   # To store scene boundaries (start, end timestamps)
        self.static_segments = []    # To store detected static/freeze frame segments

    def detect_scene_changes(self):
        """
        Detect scene changes using PySceneDetect's AdaptiveDetector.
        Returns:
            list: List of (start_time, end_time) tuples for each scene boundary.
        """
        from scenedetect import VideoManager, SceneManager
        from scenedetect.detectors import AdaptiveDetector
        import datetime

        print(f"Detecting scene changes in: {self.video_path}")
        if not os.path.exists(self.video_path):
            print(f"Error: Video file not found: {self.video_path}")
            return []
        try:
            video_manager = VideoManager([self.video_path])
            scene_manager = SceneManager()
            scene_manager.add_detector(AdaptiveDetector())

            # Start video manager and perform scene detection
            video_manager.set_downscale_factor()
            video_manager.start()
            print("Processing video for scene detection...")
            scene_manager.detect_scenes(frame_source=video_manager)
            scene_list = scene_manager.get_scene_list()
            print(f"Detected {len(scene_list)} scenes.")

            # Convert scene boundaries to timestamps
            self.scene_boundaries = []
            for i, (start, end) in enumerate(scene_list):
                start_time = str(datetime.timedelta(seconds=int(start.get_seconds())))
                end_time = str(datetime.timedelta(seconds=int(end.get_seconds())))
                self.scene_boundaries.append((start_time, end_time))
                print(f"Scene {i+1}: {start_time} --> {end_time}")

            video_manager.release()
            return self.scene_boundaries
        except Exception as e:
            print(f"Error during scene detection: {e}")
            return []

    def detect_faces_in_video(self, sample_interval=5):
        """
        Detect faces in the video using face_recognition, sampling frames at regular intervals.
        Args:
            sample_interval (int): Number of frames to skip between samples (default: 5).
        Returns:
            dict: Mapping of frame numbers to list of face bounding boxes [(top, right, bottom, left)].
        """
        import face_recognition
        import cv2
        
        print(f"Detecting faces in video: {self.video_path}")
        if not os.path.exists(self.video_path):
            print(f"Error: Video file not found: {self.video_path}")
            return {}
        
        video_capture = cv2.VideoCapture(self.video_path)
        if not video_capture.isOpened():
            print(f"Error: Could not open video file {self.video_path}")
            return {}
        
        frame_number = 0
        faces_by_frame = {}
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        print(f"Total frames: {total_frames}, FPS: {fps}")
        
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            if frame_number % sample_interval == 0:
                rgb_frame = frame[:, :, ::-1]  # Convert BGR to RGB
                face_locations = face_recognition.face_locations(rgb_frame, model='hog')
                filtered_faces = face_locations  # Just use the bounding boxes
                if filtered_faces:
                    faces_by_frame[frame_number] = filtered_faces
                
                # Show progress as percentage
                progress = (frame_number / total_frames) * 100
                print(f"\rFace detection progress: {progress:.1f}%", end="", flush=True)
            frame_number += 1
        
        print()  # New line after progress
        video_capture.release()
        if not faces_by_frame:
            print("No faces detected in any sampled frames.")
        else:
            print(f"Detected faces in {len(faces_by_frame)} frames.")
        return faces_by_frame

    def detect_static_frames(self, similarity_threshold=0.95, min_static_duration=1.0):
        """
        Detect freeze frame/static segments in the video using structural similarity.
        Args:
            similarity_threshold (float): SSIM threshold to consider frames as static (default: 0.95).
            min_static_duration (float): Minimum duration (in seconds) to consider a segment as static (default: 1.0).
        Returns:
            list: List of dicts with 'start_time', 'end_time', and 'confidence' for each static segment.
        """
        import cv2
        import datetime

        print(f"Detecting static (freeze) frames in video: {self.video_path}")
        if not os.path.exists(self.video_path):
            print(f"Error: Video file not found: {self.video_path}")
            return []
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {self.video_path}")
            return []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Total frames: {total_frames}, FPS: {fps}")
        
        prev_gray = None
        static_start = None
        static_segments = []
        static_scores = []
        frame_number = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                score, _ = ssim(prev_gray, gray, full=True)
                if score >= similarity_threshold:
                    if static_start is None:
                        static_start = frame_number - 1  # Start at previous frame
                        static_scores = [score]
                    else:
                        static_scores.append(score)
                else:
                    if static_start is not None:
                        static_end = frame_number - 1
                        duration = (static_end - static_start + 1) / fps
                        if duration >= min_static_duration:
                            start_time = str(datetime.timedelta(seconds=int(static_start / fps)))
                            end_time = str(datetime.timedelta(seconds=int(static_end / fps)))
                            confidence = float(np.mean(static_scores))
                            static_segments.append({
                                'start_time': start_time,
                                'end_time': end_time,
                                'confidence': confidence
                            })
                            print(f"Static segment: {start_time} --> {end_time} (confidence: {confidence:.3f})")
                        static_start = None
                        static_scores = []
            prev_gray = gray
            frame_number += 1
        # Handle static segment at end of video
        if static_start is not None:
            static_end = frame_number - 1
            duration = (static_end - static_start + 1) / fps
            if duration >= min_static_duration:
                start_time = str(datetime.timedelta(seconds=int(static_start / fps)))
                end_time = str(datetime.timedelta(seconds=int(static_end / fps)))
                confidence = float(np.mean(static_scores))
                static_segments.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'confidence': confidence
                })
                print(f"Static segment: {start_time} --> {end_time} (confidence: {confidence:.3f})")
        cap.release()
        if not static_segments:
            print("No static segments detected.")
        else:
            print(f"Detected {len(static_segments)} static segments.")
        return static_segments 

    def classify_video_segments(self):
        """
        Classify each video segment as 'talking_head', 'freeze_frame', 'video_clip', or 'transition'.
        Combines scene detection, face detection, and static frame analysis.
        Returns:
            list: List of dicts with segment metadata and crop strategy recommendations.
        """
        print("\nClassifying video segments...")
        # Run scene detection if not already done
        if not hasattr(self, 'scene_boundaries') or not self.scene_boundaries:
            self.detect_scene_changes()
        # Run face detection if not already done
        if not hasattr(self, 'face_locations') or not self.face_locations:
            self.face_locations = self.detect_faces_in_video()
        # Run static frame detection if not already done
        if not hasattr(self, 'static_segments') or not self.static_segments:
            self.static_segments = self.detect_static_frames()

        segments = []
        # Convert static segments to frame indices for quick lookup
        static_ranges = []
        for seg in self.static_segments:
            static_ranges.append((seg['start_time'], seg['end_time']))
        
        # Helper to check if a time is within a static segment
        def is_static(time_str):
            for start, end in static_ranges:
                if start <= time_str <= end:
                    return True
            return False

        # For each scene, classify
        for i, (start_time, end_time) in enumerate(self.scene_boundaries):
            # Find frames in this segment with faces
            frames_with_faces = [f for f in self.face_locations if start_time <= self._frame_to_time(f) <= end_time]
            # Check if this segment is a freeze frame
            static = is_static(start_time) and is_static(end_time)
            # Heuristic: classify
            if static:
                segment_type = 'freeze_frame'
                crop_strategy = 'center_crop'
            elif frames_with_faces:
                segment_type = 'talking_head'
                crop_strategy = 'face_crop'
            else:
                segment_type = 'video_clip'
                crop_strategy = 'full_frame'
            # Optionally, mark transitions (very short scenes)
            duration = self._time_to_seconds(end_time) - self._time_to_seconds(start_time)
            if duration < 1.0:
                segment_type = 'transition'
                crop_strategy = 'full_frame'
            segments.append({
                'index': i+1,
                'start_time': start_time,
                'end_time': end_time,
                'duration': duration,
                'type': segment_type,
                'crop_strategy': crop_strategy,
                'frames_with_faces': frames_with_faces
            })
            print(f"Segment {i+1}: {start_time} --> {end_time} | {segment_type} | Crop: {crop_strategy}")
        return segments

    def _frame_to_time(self, frame_number):
        # Helper to convert frame number to HH:MM:SS string
        import datetime
        fps = self._get_fps()
        seconds = int(frame_number / fps)
        return str(datetime.timedelta(seconds=seconds))

    def _time_to_seconds(self, time_str):
        # Helper to convert HH:MM:SS string to seconds
        h, m, s = [int(x) for x in time_str.split(":")]
        return h * 3600 + m * 60 + s

    def _get_fps(self):
        import cv2
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return fps 

    def analyze_motion_levels(self):
        """
        Analyze motion intensity for each video segment using optical flow (Farneback) to help distinguish between static screenshots and dynamic video clips.
        Stores motion scores for each segment.
        Returns:
            list: List of dicts with segment index, start_time, end_time, and average motion score.
        """
        import cv2
        import numpy as np
        import datetime

        print("\nAnalyzing motion levels for each segment...")
        if not hasattr(self, 'scene_boundaries') or not self.scene_boundaries:
            self.detect_scene_changes()
        motion_scores = []
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {self.video_path}")
            return []
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Total frames: {total_frames}, FPS: {fps}")

        # Build a list of (start_frame, end_frame) for each segment
        segment_frames = []
        for start_time, end_time in self.scene_boundaries:
            start_sec = self._time_to_seconds(start_time)
            end_sec = self._time_to_seconds(end_time)
            start_frame = int(start_sec * fps)
            end_frame = int(end_sec * fps)
            segment_frames.append((start_frame, end_frame))

        # For each segment, calculate average motion
        for idx, (start_frame, end_frame) in enumerate(segment_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            prev_gray = None
            frame_count = 0
            motion_sum = 0.0
            for f in range(start_frame, min(end_frame, total_frames)):
                ret, frame = cap.read()
                if not ret:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if prev_gray is not None:
                    # Use Farneback optical flow
                    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    motion = np.mean(mag)
                    motion_sum += motion
                    frame_count += 1
                prev_gray = gray
            avg_motion = motion_sum / frame_count if frame_count > 0 else 0.0
            motion_scores.append({
                'index': idx+1,
                'start_time': self.scene_boundaries[idx][0],
                'end_time': self.scene_boundaries[idx][1],
                'avg_motion': avg_motion
            })
            print(f"Segment {idx+1}: {self.scene_boundaries[idx][0]} --> {self.scene_boundaries[idx][1]} | Avg motion: {avg_motion:.4f}")
        cap.release()
        self.motion_scores = motion_scores
        return motion_scores 

    def determine_crop_strategy(self, classified_segments):
        """
        Assign cropping strategies and parameters for each segment.
        Args:
            classified_segments (list): List of segment dicts from classify_video_segments.
        Returns:
            list: List of dicts with cropping parameters for each segment (x, y, zoom, and strategy).
        """
        crop_results = []
        for seg in classified_segments:
            crop = {'index': seg['index'], 'start_time': seg['start_time'], 'end_time': seg['end_time'], 'type': seg['type']}
            if seg['type'] == 'talking_head' and seg['frames_with_faces']:
                # Face-centered crop: center on the average face location
                face_boxes = []
                for frame in seg['frames_with_faces']:
                    for box in self.face_locations.get(frame, []):
                        face_boxes.append(box)
                if face_boxes:
                    # Average face box
                    avg_top = int(np.mean([b[0] for b in face_boxes]))
                    avg_right = int(np.mean([b[1] for b in face_boxes]))
                    avg_bottom = int(np.mean([b[2] for b in face_boxes]))
                    avg_left = int(np.mean([b[3] for b in face_boxes]))
                    # Center and size
                    x_center = (avg_left + avg_right) // 2
                    y_center = (avg_top + avg_bottom) // 2
                    box_width = avg_right - avg_left
                    box_height = avg_bottom - avg_top
                    zoom = max(1.0, 1.5 * 224 / max(box_width, box_height))  # Heuristic zoom
                    crop.update({'strategy': 'face_centered', 'x': x_center, 'y': y_center, 'zoom': zoom})
                else:
                    crop.update({'strategy': 'face_centered', 'x': None, 'y': None, 'zoom': 1.0})
            elif seg['type'] == 'freeze_frame':
                # For freeze frames: preserve aspect ratio and add black bars
                crop.update({'strategy': 'center_crop', 'x': None, 'y': None, 'zoom': 1.0})
            elif seg['type'] == 'video_clip':
                # Smart crop: analyze content focus (placeholder: full frame)
                crop.update({'strategy': 'smart_crop', 'x': None, 'y': None, 'zoom': 1.0})
            else:
                # For transitions or unknown, use full frame
                crop.update({'strategy': 'full_frame', 'x': None, 'y': None, 'zoom': 1.0})
            crop_results.append(crop)
            print(f"Segment {seg['index']}: {seg['type']} | Crop: {crop['strategy']} | x: {crop['x']} y: {crop['y']} zoom: {crop['zoom']}")
        return crop_results 

    def process_to_vertical_format(self, crop_strategies, output_resolution=(1080, 1920)):
        """
        Process the original video into a 9:16 vertical format using the provided cropping strategies.
        Args:
            crop_strategies (list): List of cropping parameter dicts for each segment.
            output_resolution (tuple): (width, height) for the output video (default: 1080x1920).
        Returns:
            str: Path to the final processed video.
        """
        from moviepy import VideoFileClip, concatenate_videoclips
        import numpy as np
        import os

        print("\nProcessing video to vertical 9:16 format...")
        vertical_width, vertical_height = output_resolution
        clips = []
        video = VideoFileClip(self.video_path)
        
        for i, crop in enumerate(crop_strategies):
            start = self._time_to_seconds(crop['start_time'])
            end = self._time_to_seconds(crop['end_time'])
            seg_clip = video.subclipped(start, end)
            
            # Single crop to 9:16 format based on strategy
            if crop['strategy'] == 'face_centered' and crop['x'] is not None and crop['y'] is not None:
                # Face-centered crop: center on the detected face
                x_center, y_center = crop['x'], crop['y']
                # Calculate crop area to fit 9:16 aspect ratio
                aspect_ratio = vertical_width / vertical_height  # 9:16 = 0.5625
                
                # Calculate crop dimensions
                if seg_clip.w / seg_clip.h > aspect_ratio:
                    # Video is wider than 9:16, crop width
                    crop_height = seg_clip.h
                    crop_width = int(crop_height * aspect_ratio)
                else:
                    # Video is taller than 9:16, crop height
                    crop_width = seg_clip.w
                    crop_height = int(crop_width / aspect_ratio)
                
                # Center crop around face
                x1 = max(0, x_center - crop_width // 2)
                y1 = max(0, y_center - crop_height // 2)
                x2 = min(seg_clip.w, x1 + crop_width)
                y2 = min(seg_clip.h, y1 + crop_height)
                
                # Adjust if crop goes outside bounds
                if x2 > seg_clip.w:
                    x1 = seg_clip.w - crop_width
                    x2 = seg_clip.w
                if y2 > seg_clip.h:
                    y1 = seg_clip.h - crop_height
                    y2 = seg_clip.h
                
                seg_clip = seg_clip.cropped(x1=x1, y1=y1, x2=x2, y2=y2)
                
            elif crop['strategy'] == 'center_crop':
                # For freeze frames: preserve aspect ratio and add black bars
                # Calculate scaling to fit within 9:16 while preserving aspect ratio
                input_aspect = seg_clip.w / seg_clip.h
                target_aspect = vertical_width / vertical_height  # 9:16 = 0.5625
                
                if input_aspect > target_aspect:
                    # Image is wider than target, fit by width
                    new_width = vertical_width
                    new_height = int(vertical_width / input_aspect)
                else:
                    # Image is taller than target, fit by height  
                    new_height = vertical_height
                    new_width = int(vertical_height * input_aspect)
                
                # Resize while preserving aspect ratio
                seg_clip = seg_clip.resized(new_size=(new_width, new_height))
                
                # Create black background and composite the resized clip centered
                from moviepy import ColorClip
                black_bg = ColorClip(size=(vertical_width, vertical_height), 
                                    color=(0,0,0), duration=seg_clip.duration)
                
                # Center the resized clip on the black background
                seg_clip = seg_clip.with_position('center')
                seg_clip = CompositeVideoClip([black_bg, seg_clip])
            else:
                # For other types (transitions, etc.): simple center crop
                aspect_ratio = vertical_width / vertical_height
                if seg_clip.w / seg_clip.h > aspect_ratio:
                    crop_height = seg_clip.h
                    crop_width = int(crop_height * aspect_ratio)
                else:
                    crop_width = seg_clip.w
                    crop_height = int(crop_width / aspect_ratio)
                
                x1 = (seg_clip.w - crop_width) // 2
                y1 = (seg_clip.h - crop_height) // 2
                x2 = x1 + crop_width
                y2 = y1 + crop_height
                
                seg_clip = seg_clip.cropped(x1=x1, y1=y1, x2=x2, y2=y2)
            
            # Final resize to exact output dimensions
            seg_clip = seg_clip.resized(new_size=(vertical_width, vertical_height))
            clips.append(seg_clip)
        
        # Concatenate clips
        final_video = concatenate_videoclips(clips, method="compose")
        
        # Maintain audio sync
        final_video = final_video.with_audio(video.audio)
        
        # Output path
        output_path = self.output_path
        if not output_path.endswith('.mp4'):
            output_path += '.mp4'
        print(f"Writing final vertical video to: {output_path}")
        final_video.write_videofile(output_path, fps=30, codec='libx264', audio_codec='aac', threads=4, preset='medium')
        print("Processing complete!")
        return output_path

    def finalize_video_output(self, input_video_path=None, quality_preset='medium', custom_bitrate=None, 
                            target_platform='tiktok', add_metadata=True, progress_callback=None):
        """
        Finalize video output with encoding optimization, quality settings, and metadata for social media.
        
        Args:
            input_video_path (str): Path to input video (if None, uses self.output_path)
            quality_preset (str): Quality preset - 'high', 'medium', 'low', or 'custom'
            custom_bitrate (int): Custom bitrate in kbps (used when quality_preset='custom')
            target_platform (str): Target platform - 'tiktok', 'instagram', 'youtube', 'twitter'
            add_metadata (bool): Whether to add metadata to the video
            progress_callback (callable): Optional callback function for progress updates
            
        Returns:
            str: Path to the finalized video file
        """
        from moviepy import VideoFileClip
        import subprocess
        import json
        import os
        from datetime import datetime
        
        # Determine input and output paths
        if input_video_path is None:
            input_video_path = self.output_path
            if not input_video_path.endswith('.mp4'):
                input_video_path += '.mp4'
        
        # Create output filename with quality indicator
        base_name = os.path.splitext(input_video_path)[0]
        output_path = f"{base_name}_finalized_{quality_preset}.mp4"
        
        print(f"\nFinalizing video output...")
        print(f"Input: {input_video_path}")
        print(f"Output: {output_path}")
        print(f"Quality: {quality_preset}")
        print(f"Platform: {target_platform}")
        
        # Quality presets for different platforms
        quality_settings = {
            'tiktok': {
                'high': {'video_bitrate': 8000, 'audio_bitrate': 192, 'fps': 30, 'preset': 'slow'},
                'medium': {'video_bitrate': 5000, 'audio_bitrate': 128, 'fps': 30, 'preset': 'medium'},
                'low': {'video_bitrate': 3000, 'audio_bitrate': 96, 'fps': 30, 'preset': 'fast'}
            },
            'instagram': {
                'high': {'video_bitrate': 10000, 'audio_bitrate': 256, 'fps': 30, 'preset': 'slow'},
                'medium': {'video_bitrate': 6000, 'audio_bitrate': 128, 'fps': 30, 'preset': 'medium'},
                'low': {'video_bitrate': 4000, 'audio_bitrate': 96, 'fps': 30, 'preset': 'fast'}
            },
            'youtube': {
                'high': {'video_bitrate': 12000, 'audio_bitrate': 256, 'fps': 30, 'preset': 'slow'},
                'medium': {'video_bitrate': 8000, 'audio_bitrate': 192, 'fps': 30, 'preset': 'medium'},
                'low': {'video_bitrate': 5000, 'audio_bitrate': 128, 'fps': 30, 'preset': 'fast'}
            },
            'twitter': {
                'high': {'video_bitrate': 6000, 'audio_bitrate': 192, 'fps': 30, 'preset': 'medium'},
                'medium': {'video_bitrate': 4000, 'audio_bitrate': 128, 'fps': 30, 'preset': 'medium'},
                'low': {'video_bitrate': 2500, 'audio_bitrate': 96, 'fps': 30, 'preset': 'fast'}
            }
        }
        
        # Get settings for target platform and quality
        if quality_preset == 'custom' and custom_bitrate:
            settings = {
                'video_bitrate': custom_bitrate,
                'audio_bitrate': min(custom_bitrate // 10, 256),
                'fps': 30,
                'preset': 'medium'
            }
        else:
            settings = quality_settings.get(target_platform, quality_settings['tiktok'])[quality_preset]
        
        # Load video to get properties
        try:
            video = VideoFileClip(input_video_path)
            duration = video.duration
            original_fps = video.fps
            video.close()
        except Exception as e:
            print(f"Error loading video: {e}")
            return None
        
        # Progress callback wrapper
        def update_progress(progress, message):
            if progress_callback:
                progress_callback(progress, message)
            else:
                print(f"[{progress:.1f}%] {message}")
        
        update_progress(0, "Starting video finalization...")
        
        # Build FFmpeg command for optimal encoding
        ffmpeg_cmd = [
            'ffmpeg',
            '-i', input_video_path,
            '-c:v', 'libx264',
            '-preset', settings['preset'],
            '-crf', '18' if quality_preset == 'high' else ('23' if quality_preset == 'medium' else '28'),
            '-maxrate', f"{settings['video_bitrate']}k",
            '-bufsize', f"{settings['video_bitrate'] * 2}k",
            '-c:a', 'aac',
            '-b:a', f"{settings['audio_bitrate']}k",
            '-ar', '48000',
            '-movflags', '+faststart',
            '-pix_fmt', 'yuv420p',
            '-y'  # Overwrite output file
        ]
        
        # Add metadata if requested
        if add_metadata:
            metadata = {
                'title': f'Processed Video - {quality_preset} quality',
                'artist': 'AutoTikToks Processor',
                'date': datetime.now().strftime('%Y-%m-%d'),
                'comment': f'Processed for {target_platform} with {quality_preset} quality preset'
            }
            
            for key, value in metadata.items():
                ffmpeg_cmd.extend(['-metadata', f'{key}={value}'])
        
        # Add output path
        ffmpeg_cmd.append(output_path)
        
        update_progress(10, "Starting FFmpeg encoding...")
        
        try:
            # Run FFmpeg with progress monitoring
            process = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # Monitor progress
            last_progress = 10
            while True:
                output = process.stderr.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    # Parse FFmpeg progress output
                    if 'time=' in output:
                        try:
                            time_str = output.split('time=')[1].split()[0]
                            # Convert time to seconds
                            time_parts = time_str.split(':')
                            if len(time_parts) == 3:
                                hours, minutes, seconds = map(float, time_parts)
                                current_time = hours * 3600 + minutes * 60 + seconds
                                progress = min(90, 10 + (current_time / duration) * 80)
                                if progress - last_progress >= 5:  # Update every 5%
                                    update_progress(progress, f"Encoding... {current_time:.1f}s / {duration:.1f}s")
                                    last_progress = progress
                        except:
                            pass
            
            # Wait for process to complete
            return_code = process.wait()
            
            if return_code == 0:
                update_progress(95, "Encoding completed successfully!")
                
                # Verify output file
                if os.path.exists(output_path):
                    file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
                    update_progress(100, f"Finalization complete! File size: {file_size:.1f} MB")
                    
                    # Optional: Run additional optimization for high quality
                    if quality_preset == 'high':
                        update_progress(100, "Running additional quality optimization...")
                        self._optimize_for_high_quality(output_path)
                    
                    return output_path
                else:
                    print("Error: Output file was not created")
                    return None
            else:
                print(f"FFmpeg encoding failed with return code: {return_code}")
                return None
                
        except Exception as e:
            print(f"Error during video finalization: {e}")
            return None
    
    def _optimize_for_high_quality(self, video_path):
        """
        Additional optimization steps for high quality videos.
        """
        try:
            # Run additional optimization with FFmpeg
            optimize_cmd = [
                'ffmpeg',
                '-i', video_path,
                '-c:v', 'libx264',
                '-preset', 'veryslow',
                '-crf', '16',
                '-c:a', 'copy',
                '-y',
                video_path.replace('.mp4', '_optimized.mp4')
            ]
            
            subprocess.run(optimize_cmd, check=True, capture_output=True)
            
            # Replace original with optimized version
            os.replace(video_path.replace('.mp4', '_optimized.mp4'), video_path)
            print("High quality optimization completed!")
            
        except Exception as e:
            print(f"High quality optimization failed: {e}")
    
    def get_video_info(self, video_path=None):
        """
        Get detailed information about a video file.
        
        Args:
            video_path (str): Path to video file (if None, uses self.output_path)
            
        Returns:
            dict: Video information including resolution, bitrate, duration, etc.
        """
        if video_path is None:
            video_path = self.output_path
        
        try:
            # Use FFprobe to get detailed video information
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            info = json.loads(result.stdout)
            
            # Extract relevant information
            video_info = {
                'filename': os.path.basename(video_path),
                'file_size_mb': os.path.getsize(video_path) / (1024 * 1024),
                'duration': float(info['format']['duration']),
                'bitrate': int(info['format']['bit_rate']),
                'format': info['format']['format_name']
            }
            
            # Video stream info
            for stream in info['streams']:
                if stream['codec_type'] == 'video':
                    video_info.update({
                        'width': int(stream['width']),
                        'height': int(stream['height']),
                        'fps': eval(stream['r_frame_rate']),
                        'video_codec': stream['codec_name'],
                        'video_bitrate': int(stream.get('bit_rate', 0))
                    })
                elif stream['codec_type'] == 'audio':
                    video_info.update({
                        'audio_codec': stream['codec_name'],
                        'audio_bitrate': int(stream.get('bit_rate', 0)),
                        'audio_channels': int(stream['channels']),
                        'audio_sample_rate': int(stream['sample_rate'])
                    })
            
            return video_info
            
        except Exception as e:
            print(f"Error getting video info: {e}")
            return None 

def main():
    """
    Command-line interface for the MoistCritikalVideoProcessor.
    """
    import argparse
    import sys
    import os
    from pathlib import Path
    
    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Video processing tool with facial recognition, scene detection, and social media optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic video processing with default settings
  python edit.py input_video.mp4 output_video.mp4

  # Process with high quality for Instagram
  python edit.py input.mp4 output.mp4 --quality high --platform instagram

  # Custom bitrate processing
  python edit.py input.mp4 output.mp4 --quality custom --bitrate 8000

  # Scene detection and classification only
  python edit.py input.mp4 --detect-scenes --classify-segments

  # Face detection with custom confidence
  python edit.py input.mp4 --detect-faces --face-confidence 0.8

  # Motion analysis
  python edit.py input.mp4 --analyze-motion

  # Full pipeline with progress callback
  python edit.py input.mp4 output.mp4 --full-pipeline --progress
        """
    )
    
    # Required arguments
    parser.add_argument(
        'input_video',
        help='Path to the input video file'
    )
    
    parser.add_argument(
        'output_video',
        nargs='?',
        help='Path for the output video file (optional for analysis-only modes)'
    )
    
    # Processing options
    parser.add_argument(
        '--quality', '-q',
        choices=['high', 'medium', 'low', 'custom'],
        default='medium',
        help='Video quality preset (default: medium)'
    )
    
    parser.add_argument(
        '--platform', '-p',
        choices=['tiktok', 'instagram', 'youtube', 'twitter'],
        default='tiktok',
        help='Target platform for optimization (default: tiktok)'
    )
    
    parser.add_argument(
        '--bitrate', '-b',
        type=int,
        help='Custom bitrate in kbps (required when quality=custom)'
    )
    
    parser.add_argument(
        '--face-confidence', '-fc',
        type=float,
        default=0.6,
        help='Face detection confidence threshold (0.0-1.0, default: 0.6)'
    )
    
    parser.add_argument(
        '--scene-threshold', '-st',
        type=float,
        default=30.0,
        help='Scene detection threshold (default: 30.0)'
    )
    
    # Analysis modes
    parser.add_argument(
        '--detect-scenes',
        action='store_true',
        help='Detect and display scene boundaries'
    )
    
    parser.add_argument(
        '--detect-faces',
        action='store_true',
        help='Detect faces in the video'
    )
    
    parser.add_argument(
        '--detect-static',
        action='store_true',
        help='Detect static/freeze frame segments'
    )
    
    parser.add_argument(
        '--classify-segments',
        action='store_true',
        help='Classify video segments by type'
    )
    
    parser.add_argument(
        '--analyze-motion',
        action='store_true',
        help='Analyze motion levels in video segments'
    )
    
    parser.add_argument(
        '--get-info',
        action='store_true',
        help='Display detailed video information'
    )
    
    # Processing modes
    parser.add_argument(
        '--full-pipeline',
        action='store_true',
        help='Run complete processing pipeline (scene detection, face detection, classification, cropping, and finalization)'
    )
    
    parser.add_argument(
        '--vertical-format',
        action='store_true',
        help='Convert to 9:16 vertical format with smart cropping'
    )
    
    parser.add_argument(
        '--finalize-only',
        action='store_true',
        help='Only run video finalization (encoding optimization)'
    )
    
    # Output options
    parser.add_argument(
        '--no-metadata',
        action='store_true',
        help='Skip adding metadata to output video'
    )
    
    parser.add_argument(
        '--progress',
        action='store_true',
        help='Show detailed progress information'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.input_video):
        print(f"Error: Input video file not found: {args.input_video}")
        sys.exit(1)
    
    if args.quality == 'custom' and not args.bitrate:
        print("Error: --bitrate is required when --quality=custom")
        sys.exit(1)
    
    if args.bitrate and args.quality != 'custom':
        print("Warning: --bitrate is ignored when --quality is not 'custom'")
    
    if not args.output_video and not any([
        args.detect_scenes, args.detect_faces, args.detect_static,
        args.classify_segments, args.analyze_motion, args.get_info
    ]):
        print("Error: Output video path is required unless running analysis-only modes")
        sys.exit(1)
    
    # Set output path if not provided
    if not args.output_video:
        input_path = Path(args.input_video)
        args.output_video = str(input_path.parent / f"{input_path.stem}_processed{input_path.suffix}")
    
    # Progress callback function
    def progress_callback(progress, message):
        if args.progress:
            print(f"[{progress:.1f}%] {message}")
    
    try:
        # Initialize processor
        print(f"Initializing video processor...")
        print(f"Input: {args.input_video}")
        print(f"Output: {args.output_video}")
        print(f"Quality: {args.quality}")
        print(f"Platform: {args.platform}")
        
        processor = MoistCritikalVideoProcessor(
            video_path=args.input_video,
            output_path=args.output_video,
            face_detection_confidence=args.face_confidence,
            scene_threshold=args.scene_threshold
        )
        
        # Run requested operations
        if args.get_info:
            print("\n=== Video Information ===")
            info = processor.get_video_info(args.input_video)
            if info:
                print(f"Filename: {info['filename']}")
                print(f"File size: {info['file_size_mb']:.1f} MB")
                print(f"Duration: {info['duration']:.1f} seconds")
                print(f"Resolution: {info['width']}x{info['height']}")
                print(f"FPS: {info['fps']:.1f}")
                print(f"Video codec: {info['video_codec']}")
                print(f"Audio codec: {info['audio_codec']}")
                print(f"Total bitrate: {info['bitrate']} bps")
            else:
                print("Failed to get video information")
        
        if args.detect_scenes:
            print("\n=== Scene Detection ===")
            scenes = processor.detect_scene_changes()
            if scenes:
                print(f"Detected {len(scenes)} scenes:")
                for i, (start, end) in enumerate(scenes, 1):
                    print(f"  Scene {i}: {start} --> {end}")
            else:
                print("No scenes detected or error occurred")
        
        if args.detect_faces:
            print("\n=== Face Detection ===")
            faces = processor.detect_faces_in_video()
            if faces:
                total_faces = sum(len(face_list) for face_list in faces.values())
                print(f"Detected faces in {len(faces)} frames (total: {total_faces} faces)")
                for frame, face_list in faces.items():
                    print(f"  Frame {frame}: {len(face_list)} face(s)")
            else:
                print("No faces detected")
        
        if args.detect_static:
            print("\n=== Static Frame Detection ===")
            static_segments = processor.detect_static_frames()
            if static_segments:
                print(f"Detected {len(static_segments)} static segments:")
                for seg in static_segments:
                    print(f"  {seg['start_time']} --> {seg['end_time']} (confidence: {seg['confidence']:.3f})")
            else:
                print("No static segments detected")
        
        if args.classify_segments:
            print("\n=== Segment Classification ===")
            segments = processor.classify_video_segments()
            if segments:
                for seg in segments:
                    print(f"  Segment {seg['index']}: {seg['start_time']} --> {seg['end_time']} | {seg['type']} | Crop: {seg['crop_strategy']}")
            else:
                print("No segments classified")
        
        if args.analyze_motion:
            print("\n=== Motion Analysis ===")
            motion_scores = processor.analyze_motion_levels()
            if motion_scores:
                for score in motion_scores:
                    print(f"  Segment {score['index']}: {score['start_time']} --> {score['end_time']} | Motion: {score['avg_motion']:.4f}")
            else:
                print("No motion analysis results")
        
        # Full pipeline processing
        if args.full_pipeline:
            print("\n=== Running Full Pipeline ===")
            
            # Step 1: Scene detection
            print("Step 1: Detecting scenes...")
            processor.detect_scene_changes()
            
            # Step 2: Face detection
            print("Step 2: Detecting faces...")
            processor.face_locations = processor.detect_faces_in_video()
            
            # Step 3: Static frame detection
            print("Step 3: Detecting static frames...")
            processor.static_segments = processor.detect_static_frames()
            
            # Step 4: Segment classification
            print("Step 4: Classifying segments...")
            segments = processor.classify_video_segments()
            
            # Step 5: Determine crop strategies
            print("Step 5: Determining crop strategies...")
            crop_strategies = processor.determine_crop_strategy(segments)
            
            # Step 6: Process to vertical format
            print("Step 6: Processing to vertical format...")
            processed_path = processor.process_to_vertical_format(crop_strategies)
            
            # Step 7: Finalize video
            print("Step 7: Finalizing video...")
            finalized_path = processor.finalize_video_output(
                input_video_path=processed_path,
                quality_preset=args.quality,
                custom_bitrate=args.bitrate,
                target_platform=args.platform,
                add_metadata=not args.no_metadata,
                progress_callback=progress_callback
            )
            
            if finalized_path:
                print(f"\n✅ Full pipeline completed successfully!")
                print(f"Final output: {finalized_path}")
                
                # Show final video info
                final_info = processor.get_video_info(finalized_path)
                if final_info:
                    print(f"Final file size: {final_info['file_size_mb']:.1f} MB")
                    print(f"Final resolution: {final_info['width']}x{final_info['height']}")
            else:
                print("❌ Full pipeline failed")
                sys.exit(1)
        
        elif args.vertical_format:
            print("\n=== Vertical Format Processing ===")
            
            # Run classification and cropping
            segments = processor.classify_video_segments()
            crop_strategies = processor.determine_crop_strategy(segments)
            
            # Process to vertical format
            processed_path = processor.process_to_vertical_format(crop_strategies)
            
            if processed_path:
                print(f"✅ Vertical format processing completed: {processed_path}")
            else:
                print("❌ Vertical format processing failed")
                sys.exit(1)
        
        elif args.finalize_only:
            print("\n=== Video Finalization ===")
            finalized_path = processor.finalize_video_output(
                input_video_path=args.input_video,
                quality_preset=args.quality,
                custom_bitrate=args.bitrate,
                target_platform=args.platform,
                add_metadata=not args.no_metadata,
                progress_callback=progress_callback
            )
            
            if finalized_path:
                print(f"✅ Video finalization completed: {finalized_path}")
            else:
                print("❌ Video finalization failed")
                sys.exit(1)
        
        # If no specific processing mode was selected, just run finalization
        elif not any([
            args.detect_scenes, args.detect_faces, args.detect_static,
            args.classify_segments, args.analyze_motion, args.get_info
        ]):
            print("\n=== Basic Video Finalization ===")
            finalized_path = processor.finalize_video_output(
                quality_preset=args.quality,
                custom_bitrate=args.bitrate,
                target_platform=args.platform,
                add_metadata=not args.no_metadata,
                progress_callback=progress_callback
            )
            
            if finalized_path:
                print(f"✅ Video processing completed: {finalized_path}")
            else:
                print("❌ Video processing failed")
                sys.exit(1)
        
        print("\n🎉 Processing completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 