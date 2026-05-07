import cv2
import mediapipe as mp
import numpy as np

class Watchdog:
    def __init__(self):
        print("🚀 Initializing Google MediaPipe Pose...")
        
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,       # False = optimized for video stream
            model_complexity=1,            # 0=Fast, 1=Balanced, 2=Accurate
            smooth_landmarks=True,         # Jitter reduction
            enable_segmentation=False,     # We don't need background removal
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Drawing utilities
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def process_frame(self, frame):
        """
        Input: Raw OpenCV frame (BGR)
        Output: Annotated Frame, Foul_Detected (Boolean)
        """
        # 1. Convert BGR to RGB (MediaPipe needs RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 2. Inference
        results = self.pose.process(frame_rgb)
        
        annotated_frame = frame.copy()
        foul_detected = False 

        # 3. Draw Skeletons
        if results.pose_landmarks:
            # MediaPipe provides a built-in function to draw the skeleton stick figure
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
            # --- OPTIONAL: Simple Logic to "Detect" Action ---
            # Example: If hands are above nose (Blocking/Cheering), flag it.
            landmarks = results.pose_landmarks.landmark
            nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
            left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
            
            # Y-coordinates are normalized (0.0 is top, 1.0 is bottom)
            # So if wrist.y < nose.y, the hand is HIGHER than the nose.
            if left_wrist.y < nose.y or right_wrist.y < nose.y:
                foul_detected = True

        return annotated_frame, foul_detected