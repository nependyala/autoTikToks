"""
Video editing script with facial recognition, scene detection, and video processing capabilities.
"""
import cv2
import face_recognition
import scenedetect
from scenedetect import detect, ContentDetector
import numpy as np
from moviepy import VideoFileClip, concatenate_videoclips, ColorClip, CompositeVideoClip, VideoClip
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
        
        # Optimal face tracking configuration
        self.face_sample_interval = 10  # Check every 10 frames for smoother tracking
        self.smoothing_window = 5       # Increased smoothing for frequent updates
        self.dynamic_face_crop_enabled = True  # Enable dynamic face cropping by default

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

        # Get video properties for frame calculations
        fps = self._get_fps()
        total_frames = self._get_total_frames()
        
        # First pass: classify segments and collect freeze frame adjustments
        segment_data = []
        freeze_frame_adjustments = {}
        
        for i, (start_time, end_time) in enumerate(self.scene_boundaries):
            # Convert time boundaries to frame numbers
            start_frame = int(self._time_to_seconds(start_time) * fps)
            end_frame = int(self._time_to_seconds(end_time) * fps)
            
            # Find frames in this segment with faces
            frames_with_faces = [f for f in self.face_locations if start_frame <= f <= end_frame]
            # Check if this segment is a freeze frame
            static = is_static(start_time) and is_static(end_time)
            # Heuristic: classify
            if static:
                segment_type = 'freeze_frame'
                crop_strategy = 'center_crop'
                
                # Add buffer to freeze frame segments for tighter crop transitions (in frames)
                # Only add buffer to start_time, not subtract from end_time
                buffer_seconds = 0.2
                buffer_frames = int(buffer_seconds * fps)
                
                # Add buffer to start frame (but don't go below 0)
                buffered_start_frame = max(0, start_frame + buffer_frames)
                # Keep end frame unchanged for immediate transition back to talking head
                buffered_end_frame = end_frame
                
                # Convert back to time strings
                buffered_start_time = self._frame_to_time(buffered_start_frame)
                buffered_end_time = self._frame_to_time(buffered_end_frame)
                
                # Store adjustment for later use
                freeze_frame_adjustments[i] = {
                    'original_start': start_time,
                    'original_end': end_time,
                    'adjusted_start': buffered_start_time,
                    'adjusted_end': buffered_end_time
                }
                
                start_time = buffered_start_time
                end_time = buffered_end_time
                
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
            
            segment_data.append({
                'index': i+1,
                'start_time': start_time,
                'end_time': end_time,
                'duration': duration,
                'type': segment_type,
                'crop_strategy': crop_strategy,
                'frames_with_faces': frames_with_faces,
                'is_freeze_frame': static
            })
        
        # Second pass: adjust adjacent segment boundaries
        for i, segment in enumerate(segment_data):
            if segment['is_freeze_frame'] and i in freeze_frame_adjustments:
                adjustment = freeze_frame_adjustments[i]
                
                # Adjust previous segment's end_time to match freeze frame start_time
                if i > 0:
                    prev_segment = segment_data[i-1]
                    prev_segment['end_time'] = adjustment['adjusted_start']
                    prev_segment['duration'] = self._time_to_seconds(prev_segment['end_time']) - self._time_to_seconds(prev_segment['start_time'])
                
                # Adjust next segment's start_time to match freeze frame end_time
                if i < len(segment_data) - 1:
                    next_segment = segment_data[i+1]
                    next_segment['start_time'] = adjustment['adjusted_end']
                    next_segment['duration'] = self._time_to_seconds(next_segment['end_time']) - self._time_to_seconds(next_segment['start_time'])
        
        # Convert to final segments list
        for segment in segment_data:
            segments.append({
                'index': segment['index'],
                'start_time': segment['start_time'],
                'end_time': segment['end_time'],
                'duration': segment['duration'],
                'type': segment['type'],
                'crop_strategy': segment['crop_strategy'],
                'frames_with_faces': segment['frames_with_faces']
            })
            print(f"Segment {segment['index']}: {segment['start_time']} --> {segment['end_time']} | {segment['type']} | Crop: {segment['crop_strategy']}")
        
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

    def _get_video_duration(self):
        """Get the total duration of the video in seconds."""
        import cv2
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return total_frames / fps if fps > 0 else 0

    def _get_total_frames(self):
        """Get the total number of frames in the video."""
        import cv2
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return total_frames

    def _seconds_to_time(self, seconds):
        """Convert seconds to HH:MM:SS string format."""
        import datetime
        return str(datetime.timedelta(seconds=int(seconds)))

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

    def process_to_vertical_format(self, crop_strategies=None, output_resolution=(1080, 1920), use_frame_perfect=True, cut_frames=None):
        """
        Process the original video into a 9:16 vertical format using frame-perfect crop transitions.
        Args:
            crop_strategies (list): List of cropping parameter dicts for each segment (optional if use_frame_perfect=True)
            output_resolution (tuple): (width, height) for the output video (default: 1080x1920)
            use_frame_perfect (bool): Whether to use frame-perfect cut detection (default: True)
            cut_frames (list): Pre-detected cut frames (optional)
        Returns:
            str: Path to the temporary processed video file
        """
        from moviepy import VideoFileClip, concatenate_videoclips, ColorClip, CompositeVideoClip
        import numpy as np
        import os
        import tempfile

        print("\nProcessing video to vertical 9:16 format...")
        vertical_width, vertical_height = output_resolution
        
        if use_frame_perfect:
            # Frame-perfect approach: detect cuts, split, label, and crop individually
            print("Using frame-perfect crop transitions...")
            
            # Step 1: Detect hard cuts (if not provided)
            if cut_frames is None:
                cut_frames = self.detect_hard_cuts()
            
            # Step 2: Split video at cuts
            subclips = self.split_video_at_cuts(cut_frames)
            
            # Step 3: Label subclips with content type
            labeled_subclips = self.label_subclips(subclips)
            
            # Step 4: Process each subclip individually
            processed_clips = []
            for i, subclip_data in enumerate(labeled_subclips):
                print(f"Processing subclip {i+1}/{len(labeled_subclips)}: {subclip_data['content_type']}")
                
                # Apply crop strategy based on content type
                processed_clip = self._apply_crop_strategy_to_subclip(
                    subclip_data['clip'], 
                    subclip_data['crop_strategy'], 
                    subclip_data['content_type'],
                    vertical_width, 
                    vertical_height
                )
                
                processed_clips.append(processed_clip)
            
            # Step 5: Concatenate all processed clips
            final_video = concatenate_videoclips(processed_clips, method="compose")
            
            # Close all subclips to prevent file handle leaks
            for subclip_data in labeled_subclips:
                if subclip_data['clip'] is not None:
                    subclip_data['clip'].close()
            
        else:
            # Legacy approach: use provided crop strategies
            print("Using legacy crop strategy approach...")
            clips = []
            video = VideoFileClip(self.video_path)
            
            for i, crop in enumerate(crop_strategies):
                start = self._time_to_seconds(crop['start_time'])
                end = self._time_to_seconds(crop['end_time'])
                seg_clip = video.subclipped(start, end)
                
                # Apply single crop strategy for the entire segment
                seg_clip = self._apply_single_crop_strategy(seg_clip, crop, vertical_width, vertical_height)
                clips.append(seg_clip)
            
            # Concatenate clips
            final_video = concatenate_videoclips(clips, method="compose")
            video.close()
        
        # Maintain audio sync
        original_video = VideoFileClip(self.video_path)
        has_audio = original_video.audio is not None
        original_video.close()
        
        if has_audio:
            # Load audio separately to keep its reader independent
            from moviepy.audio.io.AudioFileClip import AudioFileClip
            audio_clip = AudioFileClip(self.video_path)
            final_video = final_video.with_audio(audio_clip)
        else:
            final_video = final_video.without_audio()
        
        # Create temporary output file
        temp_dir = tempfile.gettempdir()
        temp_filename = f"temp_vertical_processed_{os.path.basename(self.video_path)}"
        temp_output_path = os.path.join(temp_dir, temp_filename)
        
        print(f"Writing temporary vertical video to: {temp_output_path}")
        final_video.write_videofile(
            temp_output_path,
            fps=30,
            codec='libx264',
            audio=False if not has_audio else True,
            audio_codec='aac' if has_audio else None,
            threads=4,
            preset='medium'
        )
        print("Processing complete!")
        return temp_output_path

    def _apply_single_crop_strategy(self, seg_clip, crop, vertical_width, vertical_height):
        """Apply a single crop strategy to a segment."""
        if crop['strategy'] == 'face_centered' and crop['x'] is not None and crop['y'] is not None:
            # Face-centered crop: center on the detected face
            x_center, y_center = crop['x'], crop['y']
            # Calculate crop area to fit 9:16 aspect ratio
            aspect_ratio = vertical_width / vertical_height
            
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
            # Center crop for freeze frames and other content
            subclip = self._apply_center_crop(seg_clip, vertical_width, vertical_height)
            
        elif crop['strategy'] == 'smart_crop':
            # Smart crop for video clips - analyze content focus
            # For now, use center crop as fallback
            subclip = self._apply_center_crop(seg_clip, vertical_width, vertical_height)
            
        else:
            # Default to center crop
            subclip = self._apply_center_crop(seg_clip, vertical_width, vertical_height)
        
        # No final resize - cropping and padding already deliver exact dimensions
        return subclip
    
    def _apply_center_crop(self, clip, vertical_width, vertical_height):
        """
        Apply center crop with pure 9:16 crop window using full original height.
        """
        # Original frame dimensions
        h, w = clip.h, clip.w

        # Compute horizontal crop width to match 9:16 using original height
        crop_w = int(h * 9 / 16)  # 9:16 aspect, using original height

        # Compute centered x-offset
        x1 = max(0, (w - crop_w) // 2)
        x2 = x1 + crop_w

        # Crop horizontally only, preserve full height
        cropped = clip.cropped(x1=x1, y1=0, x2=x2, y2=h)
        
        # Add resize to ensure the crop fills the output
        cropped = cropped.resized(new_size=(vertical_width, vertical_height))
        
        return cropped

    def _apply_dynamic_crop_switching(self, seg_clip, start_frame, end_frame, static_frames, face_frames, crop, vertical_width, vertical_height, fps):
        """Apply dynamic crop switching based on frame content."""
        # For now, use the single crop strategy as a fallback
        # In a full implementation, this would process frame by frame
        # and switch crop strategies based on content detection
        print(f"Dynamic crop switching not fully implemented, using fallback strategy")
        return self._apply_single_crop_strategy(seg_clip, crop, vertical_width, vertical_height)

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
        
        # Create final output filename with quality indicator
        base_name = os.path.splitext(self.output_path)[0]
        if not self.output_path.endswith('.mp4'):
            base_name = self.output_path
        final_output_path = f"{base_name}_finalized_{quality_preset}.mp4"
        
        print(f"\nFinalizing video output...")
        print(f"Input: {input_video_path}")
        print(f"Final output: {final_output_path}")
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
        ffmpeg_cmd.append(final_output_path)
        
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
                if os.path.exists(final_output_path):
                    file_size = os.path.getsize(final_output_path) / (1024 * 1024)  # MB
                    update_progress(100, f"Finalization complete! File size: {file_size:.1f} MB")
                    
                    # Optional: Run additional optimization for high quality
                    if quality_preset == 'high':
                        update_progress(100, "Running additional quality optimization...")
                        self._optimize_for_high_quality(final_output_path)
                    
                    # Clean up temporary input file if it's different from self.output_path
                    if (input_video_path != self.output_path and 
                        input_video_path != self.video_path and 
                        os.path.exists(input_video_path)):
                        try:
                            os.remove(input_video_path)
                            print(f"Cleaned up temporary file: {input_video_path}")
                        except Exception as e:
                            print(f"Warning: Could not delete temporary file {input_video_path}: {e}")
                    
                    # Clean up any other temporary files in temp directory
                    import tempfile
                    temp_dir = tempfile.gettempdir()
                    temp_pattern = f"temp_vertical_processed_{os.path.basename(self.video_path)}"
                    for temp_file in os.listdir(temp_dir):
                        if temp_file.startswith("temp_vertical_processed_") and temp_file.endswith(".mp4"):
                            temp_file_path = os.path.join(temp_dir, temp_file)
                            try:
                                os.remove(temp_file_path)
                                print(f"Cleaned up additional temporary file: {temp_file_path}")
                            except Exception as e:
                                print(f"Warning: Could not delete temporary file {temp_file_path}: {e}")
                    
                    return final_output_path
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

    def detect_hard_cuts(self, ssim_threshold=0.7, pixel_diff_threshold=40):
        """
        Detect exact frame numbers where hard visual cuts occur.
        Args:
            ssim_threshold (float): SSIM threshold for cut detection (default: 0.7)
            pixel_diff_threshold (int): Pixel difference threshold (default: 40)
        Returns:
            list: Sorted list of frame numbers where cuts occur
        """
        import cv2
        import numpy as np
        from skimage.metrics import structural_similarity as ssim
        
        print(f"Detecting hard cuts in video: {self.video_path}")
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {self.video_path}")
            return []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Use MoviePy's duration-based frame count for consistency
        from moviepy import VideoFileClip
        temp_video = VideoFileClip(self.video_path)
        total_frames = int(temp_video.duration * fps)
        temp_video.close()
        
        print(f"Total frames: {total_frames}, FPS: {fps}")
        
        cut_frames = []
        prev_frame = None
        frame_number = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if prev_frame is not None:
                    # Convert frames to grayscale for comparison
                    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                    curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    # Calculate SSIM
                    ssim_score, _ = ssim(prev_gray, curr_gray, full=True)
                    
                    # Calculate pixel difference
                    diff = cv2.absdiff(prev_gray, curr_gray)
                    pixel_diff = np.mean(diff)
                    
                    # Detect cut if either threshold is exceeded
                    if ssim_score < ssim_threshold or pixel_diff > pixel_diff_threshold:
                        cut_frames.append(frame_number)
                        print(f"Cut detected at frame {frame_number} (SSIM: {ssim_score:.3f}, Pixel diff: {pixel_diff:.1f})")
                
                prev_frame = frame.copy()
                frame_number += 1
                
                # Show progress
                if frame_number % 100 == 0:
                    progress = (frame_number / total_frames) * 100
                    print(f"\rHard cut detection progress: {progress:.1f}%", end="", flush=True)
        finally:
            cap.release()
        
        print()  # New line after progress
        
        # Always include frame 0 as the first cut
        if 0 not in cut_frames:
            cut_frames.insert(0, 0)
        
        # Always include the last frame as the final cut
        if (total_frames - 1) not in cut_frames:
            cut_frames.append(total_frames - 1)
        
        # Sort and remove duplicates
        cut_frames = sorted(list(set(cut_frames)))
        
        print(f"Detected {len(cut_frames)} hard cuts: {cut_frames}")
        return cut_frames

    def split_video_at_cuts(self, cut_frames):
        """
        Split the video into subclips at the exact cut frames.
        Args:
            cut_frames (list): List of frame numbers where cuts occur
        Returns:
            list: List of VideoFileClip subclips
        """
        from moviepy import VideoFileClip
        
        print(f"Splitting video at {len(cut_frames)} cut points...")
        
        # Get video properties without keeping the video open
        temp_video = VideoFileClip(self.video_path)
        fps = temp_video.fps
        total_frames = int(temp_video.duration * fps)
        temp_video.close()
        
        subclips = []
        
        for i in range(len(cut_frames) - 1):
            start_frame = cut_frames[i]
            end_frame = cut_frames[i + 1]
            
            # Ensure we don't exceed video duration
            if end_frame > total_frames:
                end_frame = total_frames
            
            # Convert frame numbers to time
            start_time = start_frame / fps
            end_time = end_frame / fps
            
            # Ensure end_time doesn't exceed video duration
            if end_time > (total_frames / fps):
                end_time = total_frames / fps
            
            # Skip if start_time >= end_time
            if start_time >= end_time:
                print(f"Warning: Skipping subclip {i+1} (start_time >= end_time)")
                continue
            
            # Create independent subclip with its own reader
            subclip = VideoFileClip(self.video_path).subclipped(start_time, end_time)
            subclips.append({
                'index': i,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'start_time': start_time,
                'end_time': end_time,
                'duration': end_time - start_time,
                'clip': subclip
            })
            
            print(f"Subclip {i+1}: frames {start_frame}-{end_frame} ({start_time:.2f}s - {end_time:.2f}s)")
        
        print(f"Created {len(subclips)} subclips")
        return subclips

    def label_subclips(self, subclips, sample_frames=3):
        """
        Label each subclip with content type and crop strategy.
        Args:
            subclips (list): List of subclip dictionaries from split_video_at_cuts
            sample_frames (int): Number of frames to sample for analysis
        Returns:
            list: List of labeled subclips with content type and crop strategy
        """
        import cv2
        import numpy as np
        from skimage.metrics import structural_similarity as ssim
        
        print(f"Labeling {len(subclips)} subclips...")
        
        labeled_subclips = []
        
        for i, subclip_data in enumerate(subclips):
            subclip = subclip_data['clip']
            duration = subclip_data['duration']
            
            # Sanity check for None subclips
            if subclip is None:
                raise ValueError(f"Subclip {i+1} is None, check split_video_at_cuts logic")
            
            # Sample frames from the subclip
            sample_times = np.linspace(0, duration, sample_frames + 2)[1:-1]  # Skip start and end
            frames = []
            
            for t in sample_times:
                frame = subclip.get_frame(t)
                frames.append(frame)
            
            # Analyze for static content (freeze frame)
            is_static = True
            if len(frames) > 1:
                for j in range(1, len(frames)):
                    # Convert to grayscale
                    prev_gray = cv2.cvtColor(frames[j-1], cv2.COLOR_RGB2GRAY)
                    curr_gray = cv2.cvtColor(frames[j], cv2.COLOR_RGB2GRAY)
                    
                    # Calculate SSIM
                    ssim_score, _ = ssim(prev_gray, curr_gray, full=True)
                    
                    if ssim_score < 0.95:  # High threshold for static detection
                        is_static = False
                        break
            
            # Analyze for face content
            has_faces = False
            for frame in frames:
                # Convert BGR to RGB for face_recognition
                rgb_frame = frame[:, :, ::-1]
                face_locations = face_recognition.face_locations(rgb_frame, model='hog')
                if face_locations:
                    has_faces = True
                    break
            
            # Determine content type and crop strategy
            if is_static:
                content_type = 'freeze_frame'
                crop_strategy = 'center_crop'
            elif has_faces:
                content_type = 'talking_head'
                crop_strategy = 'face_crop'
            else:
                content_type = 'video_clip'
                crop_strategy = 'smart_crop'
            
            # Add labels to subclip data
            labeled_subclip = subclip_data.copy()
            labeled_subclip.update({
                'content_type': content_type,
                'crop_strategy': crop_strategy,
                'is_static': is_static,
                'has_faces': has_faces
            })
            
            labeled_subclips.append(labeled_subclip)
            
            print(f"Subclip {i+1}: {content_type} | Crop: {crop_strategy} | Duration: {duration:.2f}s")
        
        return labeled_subclips

    def _apply_crop_strategy_to_subclip(self, subclip, crop_strategy, content_type, vertical_width, vertical_height):
        """
        Apply crop strategy to a single subclip based on its content type.
        Args:
            subclip: VideoFileClip subclip
            crop_strategy (str): Crop strategy to apply
            content_type (str): Type of content in the subclip
            vertical_width (int): Target width
            vertical_height (int): Target height
        Returns:
            VideoFileClip: Processed subclip
        """
        if crop_strategy == 'face_crop':
            # Dynamic face crop: crop to 9:16 window, follow face, NO black bars
            if content_type == 'talking_head' and hasattr(self, 'dynamic_face_crop_enabled') and self.dynamic_face_crop_enabled:
                # Use dynamic face cropping for smooth pan transitions
                sample_interval = getattr(self, 'face_sample_interval', 10)
                max_speed = getattr(self, 'max_pan_speed', 300)
                return self.apply_dynamic_face_cropping(subclip, vertical_width, vertical_height, sample_interval, max_speed)
            else:
                # Fallback to static face crop for other content types or when disabled
                sample_frame = subclip.get_frame(subclip.duration / 2)
                rgb_frame = sample_frame[:, :, ::-1]  # Convert BGR to RGB
                face_locations = face_recognition.face_locations(rgb_frame, model='hog')
                
                if face_locations:
                    # Use the first detected face
                    top, right, bottom, left = face_locations[0]
                    x_center = (left + right) // 2
                    y_center = (top + bottom) // 2
                    
                    # Calculate crop area to fit 9:16 aspect ratio
                    aspect_ratio = vertical_width / vertical_height
                    
                    if subclip.w / subclip.h > aspect_ratio:
                        # Video is wider than 9:16, crop width
                        crop_height = subclip.h
                        crop_width = int(crop_height * aspect_ratio)
                    else:
                        # Video is taller than 9:16, crop height
                        crop_width = subclip.w
                        crop_height = int(crop_width / aspect_ratio)
                    
                    # Center crop around face
                    x1 = max(0, x_center - crop_width // 2)
                    y1 = max(0, y_center - crop_height // 2)
                    x2 = min(subclip.w, x1 + crop_width)
                    y2 = min(subclip.h, y1 + crop_height)
                    
                    # Adjust if crop goes outside bounds
                    if x2 > subclip.w:
                        x1 = subclip.w - crop_width
                        x2 = subclip.w
                    if y2 > subclip.h:
                        y1 = subclip.h - crop_height
                        y2 = subclip.h
                    
                    subclip = subclip.cropped(x1=x1, y1=y1, x2=x2, y2=y2)
                    # Add resize to ensure the crop fills the output
                    subclip = subclip.resized(new_size=(vertical_width, vertical_height))
                else:
                    # No face detected, use center crop
                    subclip = self._apply_center_crop(subclip, vertical_width, vertical_height)
                
        elif crop_strategy == 'center_crop':
            # Freeze frame: show full frame, add black bars above/below to fit 9:16
            if content_type == 'freeze_frame':
                return self._apply_freeze_frame_full_frame(subclip, vertical_width, vertical_height)
            else:
                # For other content types, use center crop
                subclip = self._apply_center_crop(subclip, vertical_width, vertical_height)
            
        elif crop_strategy == 'smart_crop':
            # Smart crop for video clips - analyze content focus
            # For now, use center crop as fallback
            subclip = self._apply_center_crop(subclip, vertical_width, vertical_height)
            
        else:
            # Default to center crop
            subclip = self._apply_center_crop(subclip, vertical_width, vertical_height)
        
        # No final resize - cropping and padding already deliver exact dimensions
        return subclip
    
    def manual_cut_correction(self, cut_frames, corrected_frames=None):
        """
        Allow manual correction of detected cut frames for edge cases.
        Args:
            cut_frames (list): Original detected cut frames
            corrected_frames (list): Manually corrected cut frames (if None, will prompt user)
        Returns:
            list: Corrected cut frames
        """
        print(f"\nDetected cut frames: {cut_frames}")
        
        if corrected_frames is None:
            print("\nManual cut frame correction:")
            print("Enter corrected frame numbers separated by commas, or press Enter to use detected frames")
            print("Example: 0, 45, 120, 180, 240")
            
            try:
                user_input = input("Corrected frames: ").strip()
                if user_input:
                    corrected_frames = [int(x.strip()) for x in user_input.split(',')]
                    corrected_frames = sorted(list(set(corrected_frames)))
                else:
                    corrected_frames = cut_frames
            except (ValueError, KeyboardInterrupt):
                print("Invalid input or interrupted, using detected frames")
                corrected_frames = cut_frames
        
        print(f"Using cut frames: {corrected_frames}")
        return corrected_frames

    def export_cut_frames(self, cut_frames, output_file=None):
        """
        Export detected cut frames to a file for manual review.
        Args:
            cut_frames (list): List of cut frame numbers
            output_file (str): Output file path (if None, uses default name)
        Returns:
            str: Path to the exported file
        """
        import json
        import os
        
        if output_file is None:
            base_name = os.path.splitext(os.path.basename(self.video_path))[0]
            output_file = f"{base_name}_cut_frames.json"
        
        # Get video properties for context
        fps = self._get_fps()
        total_frames = self._get_total_frames()
        duration = total_frames / fps
        
        # Create export data
        export_data = {
            'video_path': self.video_path,
            'total_frames': total_frames,
            'fps': fps,
            'duration_seconds': duration,
            'cut_frames': cut_frames,
            'cut_times': [frame / fps for frame in cut_frames],
            'cut_times_formatted': [self._seconds_to_time(frame / fps) for frame in cut_frames]
        }
        
        # Write to file
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Cut frames exported to: {output_file}")
        print(f"Total cuts: {len(cut_frames)}")
        print(f"Cut times: {export_data['cut_times_formatted']}")
        
        return output_file

    def detect_face_centers(self, video_path, sample_interval=10):
        """
        Detect face centers at frequent intervals for smooth tracking.
        Args:
            video_path (str): Path to the video file
            sample_interval (int): Frame sampling interval (default: 10 for smoother tracking)
        Returns:
            tuple: (times, xs, ys) arrays of detected face centers
        """
        import cv2
        import face_recognition
        import numpy as np
        
        print(f"Detecting face centers with sample interval: {sample_interval}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return None, None, None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        centers = []
        frame_idx = 0
        
        # Ensure we always sample first and last frames
        critical_frames = {0, total_frames - 1}
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            should_sample = (frame_idx % sample_interval == 0) or (frame_idx in critical_frames)
            
            if should_sample:
                rgb = frame[:, :, ::-1]  # BGR→RGB
                faces = face_recognition.face_locations(rgb, model="hog")
                if faces:
                    # Pick the largest face
                    top, right, bottom, left = max(faces, key=lambda b: (b[2]-b[0])*(b[1]-b[3]))
                    cx = (left + right) / 2.0  # Ensure float
                    cy = (top + bottom) / 2.0  # Ensure float
                    time_stamp = frame_idx / fps  # Ensure float
                    centers.append((time_stamp, cx, cy))
                
                # Show progress
                progress = (frame_idx / total_frames) * 100
                print(f"\rFace center detection progress: {progress:.1f}%", end="", flush=True)
            frame_idx += 1
        
        print()  # New line after progress
        cap.release()
        
        if not centers:
            print("Warning: No faces detected; cannot track.")
            return None, None, None
        
        # Ensure we return numpy arrays of scalars
        times, xs, ys = zip(*centers)
        return np.array(times, dtype=float), np.array(xs, dtype=float), np.array(ys, dtype=float)

    def make_smooth_interpolators(self, times, xs, ys, kind="cubic", smoothing_window=5, frame_w=None):
        """
        Create smooth interpolators for face center trajectories using cubic interpolation.
        Args:
            times (np.array): Time points
            xs (np.array): X coordinates
            ys (np.array): Y coordinates
            kind (str): Interpolation kind ('cubic', 'quadratic', 'linear')
            smoothing_window (int): Window size for smoothing
            frame_w (int): Frame width for center initialization
        Returns:
            tuple: (Xfunc, Yfunc) interpolator functions
        """
        import numpy as np
        from scipy.interpolate import interp1d
        
        # Ensure arrays are properly flattened and converted to scalars
        times = np.asarray(times).flatten()
        xs = np.asarray(xs).flatten()
        ys = np.asarray(ys).flatten()
        
        if len(times) < 2:
            # If only one point, create constant functions
            def constant_func(t):
                return np.full_like(t, float(xs[0]) if len(xs) > 0 else 0)
            return constant_func, constant_func
        
        # Apply sliding-window mean to smooth the trajectory
        if smoothing_window > 1 and len(xs) >= smoothing_window:
            xs_smooth = np.convolve(xs, np.ones(smoothing_window)/smoothing_window, mode='same')
            ys_smooth = np.convolve(ys, np.ones(smoothing_window)/smoothing_window, mode='same')
        else:
            xs_smooth = xs
            ys_smooth = ys
        
        # Create smooth interpolator functions with cubic interpolation
        # FIX: Use first detected position instead of frame center for fill_value
        first_x = float(xs_smooth[0]) if len(xs_smooth) > 0 else (frame_w / 2.0 if frame_w else 0)
        last_x = float(xs_smooth[-1]) if len(xs_smooth) > 0 else first_x
        
        Xfunc = interp1d(times, xs_smooth, kind=kind,
                         fill_value=(first_x, last_x), bounds_error=False)
        Yfunc = interp1d(times, ys_smooth, kind=kind,
                         fill_value=(float(ys_smooth[0]), float(ys_smooth[-1])), bounds_error=False)
        
        return Xfunc, Yfunc

    def capped_interpolator(self, Xfunc, times, max_speed=300):
        """
        Create a speed-capped interpolator to avoid sudden large jumps.
        Args:
            Xfunc: Original interpolator function
            times: Time array for the clip
            max_speed: Maximum pan speed in pixels per second
        Returns:
            function: Speed-capped interpolator function
        """
        import numpy as np
        
        # Precompute the full trajectory with speed capping
        xs = Xfunc(times)
        
        # Compute frame-to-frame deltas
        dt = np.diff(times, prepend=times[0])
        dx = np.diff(xs, prepend=xs[0])
        
        # Calculate maximum allowed movement per frame
        max_dx = max_speed * dt
        
        # Clamp each dx to respect maximum speed
        dx_clamped = np.sign(dx) * np.minimum(np.abs(dx), max_dx)
        
        # Reconstruct a new, smoothed trajectory
        xs_smooth = np.cumsum(dx_clamped)
        xs_smooth += xs[0] - xs_smooth[0]  # Preserve starting position
        
        # CRITICAL FIX: Preserve the original trajectory mean to prevent drift
        original_mean = np.mean(xs)
        current_mean = np.mean(xs_smooth)
        xs_smooth += (original_mean - current_mean)  # Correct for systematic drift
        
        # Create a new interpolator from the capped trajectory
        from scipy.interpolate import interp1d
        capped_Xfunc = interp1d(times, xs_smooth, kind='linear',
                               fill_value=(float(xs_smooth[0]), float(xs_smooth[-1])), 
                               bounds_error=False)
        
        return capped_Xfunc

    def dynamic_face_crop(self, clip, Xfunc, Yfunc, target_w, target_h, max_speed=300):
        """
        Apply dynamic face-centered cropping with smooth continuous interpolation.
        Args:
            clip: VideoFileClip to process
            Xfunc: X-coordinate interpolator function
            Yfunc: Y-coordinate interpolator function (not used in width-only crop)
            target_w (int): Target crop width (9 in 9:16 ratio)
            target_h (int): Target crop height (16 in 9:16 ratio)
            max_speed (int): Maximum pan speed in pixels per second
        Returns:
            VideoFileClip: Dynamically cropped clip
        """
        import numpy as np
        from moviepy import VideoClip

        frame_w, frame_h = clip.size
        # Use exact target aspect ratio to prevent sub-pixel positioning errors
        target_aspect = target_w / target_h
        crop_w = int(frame_h * target_aspect)

        # Build speed-capped interpolator
        fps = clip.fps
        times = np.linspace(0, clip.duration, int(clip.duration * fps))
        capped_Xfunc = self.capped_interpolator(Xfunc, times, max_speed)

        def frame_function(t):
            cx = float(capped_Xfunc(t))
            half = crop_w / 2.0

            # 1) Ideal, unclamped origin
            orig_x1 = int(cx - half)

            # 2) Determine overlap with frame
            content_x1 = max(0, orig_x1)
            content_x2 = min(frame_w, orig_x1 + crop_w)
            content_width = content_x2 - content_x1

            # 3) Compute padding needed to restore full width
            pad_left  = max(0, -orig_x1)
            pad_right = max(0, (orig_x1 + crop_w) - frame_w)

            # 4) Extract only the overlapping slice
            frame = clip.get_frame(t)
            crop = frame[:, content_x1:content_x2, :]

            # 5) Pad back to exactly crop_w
            if pad_left or pad_right:
                h, w, c = crop.shape
                assert w == content_width
                full = np.zeros((h, crop_w, c), dtype=crop.dtype)
                full[:, pad_left:pad_left + w, :] = crop
                crop = full

            return crop

        new_clip = VideoClip().with_updated_frame_function(frame_function)
        new_clip = new_clip.with_duration(clip.duration)
        new_clip.size = (crop_w, frame_h)
        return new_clip.resized(new_size=(target_w, target_h))

    def apply_dynamic_face_cropping(self, subclip, target_width, target_height, sample_interval=10, max_speed=300):
        """
        Apply dynamic face-centered cropping to a subclip for smooth pan transitions.
        Args:
            subclip: VideoFileClip subclip to process
            target_width (int): Target crop width
            target_height (int): Target crop height
            sample_interval (int): Frame sampling interval for face detection (default: 10 for smoother tracking)
            max_speed (int): Maximum pan speed in pixels per second (default: 300)
        Returns:
            VideoFileClip: Dynamically cropped subclip
        """
        print(f"Applying dynamic face cropping to subclip...")
        
        # Save subclip to temporary file for face detection
        import tempfile
        import os
        
        temp_dir = tempfile.gettempdir()
        temp_filename = f"temp_subclip_{os.path.basename(self.video_path)}"
        temp_path = os.path.join(temp_dir, temp_filename)
        
        # Write subclip to temp file
        subclip.write_videofile(temp_path, fps=30, codec='libx264', audio=False, logger=None)
        
        try:
            # Detect face centers
            times, xs, ys = self.detect_face_centers(temp_path, sample_interval)
            
            if times is None:
                print("No faces detected, using center crop fallback")
                return self._apply_center_crop(subclip, target_width, target_height)
            
            # Create smooth interpolators with cubic interpolation
            Xfunc, Yfunc = self.make_smooth_interpolators(times, xs, ys, kind="cubic", smoothing_window=5, frame_w=subclip.w)
            
            # Apply dynamic cropping with smooth continuous interpolation
            cropped_clip = self.dynamic_face_crop(subclip, Xfunc, Yfunc, target_width, target_height, max_speed)
            
            print("Dynamic face cropping applied successfully with smooth interpolation")
            return cropped_clip
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def _apply_freeze_frame_full_frame(self, clip, output_width, output_height):
        """
        Apply freeze frame cropping: show full frame with black bars above/below to fit 9:16.
        """
        # Scale original so width matches output width
        scale = output_width / clip.w
        new_height = int(clip.h * scale)
        resized = clip.resized(width=output_width)
        pad_top = (output_height - new_height) // 2
        pad_bottom = output_height - new_height - pad_top
        
        from moviepy import ColorClip, CompositeVideoClip
        black_top = ColorClip((output_width, pad_top), color=(0,0,0), duration=clip.duration)
        black_bottom = ColorClip((output_width, pad_bottom), color=(0,0,0), duration=clip.duration)
        
        final = CompositeVideoClip([
            black_top.with_position(("center", "top")),
            resized.with_position(("center", pad_top)),
            black_bottom.with_position(("center", pad_top + new_height))
        ], size=(output_width, output_height))
        
        return final

    def build_face_trajectory(self, times, xs, total_duration, fps, frame_width):
        """
        Build a per-frame array of crop positions that holds the last known face position.
        Args:
            times: list of detection times
            xs: list of x positions (face centers)
            total_duration: clip duration in seconds
            fps: frames per second
            frame_width: width of the video frame
        Returns:
            list: crop_xs - per-frame crop center positions
        """
        # Ensure times and xs are numpy arrays and flatten if needed
        import numpy as np
        times = np.asarray(times).flatten()
        xs = np.asarray(xs).flatten()
        
        # Apply low-pass filter to smooth face positions
        if len(xs) > 0:
            smoothed_xs = self.apply_low_pass_filter(xs, alpha=0.15)
        else:
            smoothed_xs = xs
        
        # Build a per-frame array of crop positions
        num_frames = int(total_duration * fps)
        crop_xs = []
        last_x = frame_width // 2  # default to center

        detection_idx = 0
        for i in range(num_frames):
            t = i / fps
            # Fix: Use proper array indexing and scalar comparison
            while (detection_idx < len(times) and 
                   detection_idx < len(smoothed_xs) and
                   float(times[detection_idx]) <= t):  # Convert to scalar
                last_x = float(smoothed_xs[detection_idx])  # Convert to scalar
                detection_idx += 1
            crop_xs.append(last_x)
        
        return crop_xs

    def apply_low_pass_filter(self, positions, alpha=0.15):
        """
        Apply low-pass filter to smooth face position data.
        Args:
            positions: list or np.array of x coordinates
            alpha: smoothing factor (0-1, lower = smoother)
        Returns:
            list: smoothed positions
        """
        # Guard against empty or missing positions
        if positions is None or len(positions) == 0:
            return [] if positions is None else list(positions)
        
        smoothed = [positions[0]]  # First position unchanged
        for i in range(1, len(positions)):
            # Low-pass filter: filtered = alpha * raw + (1-alpha) * previous_filtered
            smoothed_pos = alpha * positions[i] + (1 - alpha) * smoothed[i-1]
            smoothed.append(smoothed_pos)
        return smoothed

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
        description="Video processing tool with facial recognition, scene detection, and social media optimization. Features optimal face tracking with 10-frame sampling and 5-frame smoothing by default. DEFAULT: Runs optimal TikTok pipeline with frame-perfect processing and high quality output.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic video processing with default settings
  python edit.py input_video.mp4 output_video.mp4

  # Frame-perfect crop transitions (recommended)
  python edit.py input.mp4 output.mp4 --frame-perfect

  # Frame-perfect with dynamic face cropping (smooth pan transitions) - OPTIMAL CONFIGURATION
  python edit.py input.mp4 output.mp4 --frame-perfect --dynamic-face-crop

  # Dynamic face cropping with custom settings (for advanced users)
  python edit.py input.mp4 output.mp4 --frame-perfect --dynamic-face-crop --face-sample-interval 5 --smoothing-window 7 --max-pan-speed 200

  # Ultra-smooth face tracking (slower pan speed)
  python edit.py input.mp4 output.mp4 --frame-perfect --dynamic-face-crop --max-pan-speed 150 --face-sample-interval 5

  # Frame-perfect with manual cut correction
  python edit.py input.mp4 output.mp4 --frame-perfect --manual-cuts

  # Frame-perfect with specific cut frames
  python edit.py input.mp4 output.mp4 --frame-perfect --cut-frames "0,45,120,180,240"

  # Process with high quality for Instagram
  python edit.py input.mp4 output.mp4 --quality high --platform instagram --frame-perfect

  # Custom bitrate processing
  python edit.py input.mp4 output.mp4 --quality custom --bitrate 8000

  # Scene detection and classification only
  python edit.py input.mp4 --detect-scenes --classify-segments

  # Face detection with custom confidence
  python edit.py input.mp4 --detect-faces --face-confidence 0.8

  # Motion analysis
  python edit.py input.mp4 --analyze-motion

  # Full pipeline with frame-perfect processing and dynamic face cropping
  python edit.py input.mp4 output.mp4 --full-pipeline --frame-perfect --dynamic-face-crop --progress

  # Legacy segment-based processing (old method)
  python edit.py input.mp4 output.mp4 --legacy-mode

  # Static face cropping (faster but less smooth)
  python edit.py input.mp4 output.mp4 --frame-perfect --static-face-crop
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
        '--frame-perfect',
        action='store_true',
        help='Use frame-perfect crop transitions (detect hard cuts and apply crops per subclip)'
    )
    
    parser.add_argument(
        '--legacy-mode',
        action='store_true',
        help='Use legacy segment-based processing instead of frame-perfect cuts'
    )
    
    parser.add_argument(
        '--manual-cuts',
        action='store_true',
        help='Allow manual correction of detected cut frames'
    )
    
    parser.add_argument(
        '--cut-frames',
        type=str,
        help='Comma-separated list of manual cut frame numbers (e.g., "0,45,120,180")'
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
    
    parser.add_argument(
        '--export-cuts',
        action='store_true',
        help='Export detected cut frames to JSON file for manual review'
    )
    
    parser.add_argument(
        '--dynamic-face-crop',
        action='store_true',
        help='Use dynamic face-centered cropping for smooth pan transitions (default: enabled for talking head segments)'
    )
    
    parser.add_argument(
        '--face-sample-interval',
        type=int,
        default=10,
        help='Frame sampling interval for dynamic face detection (default: 10 for smoother tracking)'
    )
    
    parser.add_argument(
        '--smoothing-window',
        type=int,
        default=5,
        help='Smoothing window size for face trajectory (default: 5 for optimal smoothing)'
    )
    
    parser.add_argument(
        '--static-face-crop',
        action='store_true',
        help='Use static face cropping instead of dynamic (faster but less smooth)'
    )
    
    parser.add_argument(
        '--max-pan-speed',
        type=int,
        default=300,
        help='Maximum pan speed in pixels per second for smooth face tracking (default: 300)'
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
        args.classify_segments, args.analyze_motion, args.get_info, args.export_cuts
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
        
        # Store dynamic face cropping options (use optimal defaults unless overridden)
        processor.dynamic_face_crop_enabled = args.dynamic_face_crop or not args.static_face_crop
        processor.face_sample_interval = args.face_sample_interval  # Default is now 10
        processor.smoothing_window = args.smoothing_window  # Default is now 5
        processor.max_pan_speed = args.max_pan_speed  # Default is now 300
        
        # Run requested operations
        if args.export_cuts:
            print("\n=== Exporting Cut Frames ===")
            cut_frames = processor.detect_hard_cuts()
            export_file = processor.export_cut_frames(cut_frames)
            print(f"✅ Cut frames exported to: {export_file}")
        
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
            
            if args.frame_perfect:
                # Frame-perfect approach
                print("Using frame-perfect crop transitions...")
                
                # Step 1: Detect hard cuts
                print("Step 1: Detecting hard cuts...")
                cut_frames = processor.detect_hard_cuts()
                
                # Step 2: Manual correction if requested
                if args.manual_cuts or args.cut_frames:
                    if args.cut_frames:
                        corrected_frames = [int(x.strip()) for x in args.cut_frames.split(',')]
                        corrected_frames = sorted(list(set(corrected_frames)))
                    else:
                        corrected_frames = None
                    cut_frames = processor.manual_cut_correction(cut_frames, corrected_frames)
                
                # Step 3: Process to vertical format with frame-perfect approach
                print("Step 2: Processing to vertical format...")
                processed_path = processor.process_to_vertical_format(use_frame_perfect=True, cut_frames=cut_frames)
                
            else:
                # Legacy approach
                print("Using legacy segment-based processing...")
                
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
                processed_path = processor.process_to_vertical_format(crop_strategies, use_frame_perfect=False)
            
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
            
            if args.frame_perfect:
                # Frame-perfect approach
                print("Using frame-perfect crop transitions...")
                processed_path = processor.process_to_vertical_format(use_frame_perfect=True)
            else:
                # Legacy approach
                print("Using legacy segment-based processing...")
                segments = processor.classify_video_segments()
                crop_strategies = processor.determine_crop_strategy(segments)
                processed_path = processor.process_to_vertical_format(crop_strategies, use_frame_perfect=False)
            
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
        
        # If no specific processing mode was selected, run optimal TikTok pipeline by default
        elif not any([
            args.detect_scenes, args.detect_faces, args.detect_static,
            args.classify_segments, args.analyze_motion, args.get_info
        ]):
            print("\n=== Running Optimal TikTok Pipeline (Default) ===")
            print("Using frame-perfect crop transitions with dynamic face cropping...")
            
            # Step 1: Detect hard cuts
            print("Step 1: Detecting hard cuts...")
            cut_frames = processor.detect_hard_cuts()
            
            # Step 2: Process to vertical format with frame-perfect approach
            print("Step 2: Processing to vertical format...")
            processed_path = processor.process_to_vertical_format(use_frame_perfect=True, cut_frames=cut_frames)
            
            # Step 3: Finalize video with high quality TikTok settings
            print("Step 3: Finalizing video with high quality TikTok settings...")
            finalized_path = processor.finalize_video_output(
                input_video_path=processed_path,
                quality_preset='high',
                target_platform='tiktok',
                add_metadata=True,
                progress_callback=progress_callback
            )
            
            if finalized_path:
                print(f"\n✅ Optimal TikTok pipeline completed successfully!")
                print(f"Final output: {finalized_path}")
                
                # Show final video info
                final_info = processor.get_video_info(finalized_path)
                if final_info:
                    print(f"Final file size: {final_info['file_size_mb']:.1f} MB")
                    print(f"Final resolution: {final_info['width']}x{final_info['height']}")
            else:
                print("❌ Optimal TikTok pipeline failed")
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