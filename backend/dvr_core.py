import cv2
import os
import time
from collections import deque

class SmartDVR:
    def __init__(self, buffer_size=150, temp_dir="temp_clips"):
        # Keep last ~150 frames (approx 5-7 seconds)
        self.frame_buffer = deque(maxlen=buffer_size)
        self.temp_dir = temp_dir
        
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)

    def write_frame(self, frame):
        """Add frame to rolling buffer"""
        if frame is not None:
            self.frame_buffer.append(frame)

    def save_last_clip(self):
        """Saves buffer to MP4"""
        if len(self.frame_buffer) < 10:
            return None
            
        timestamp = int(time.time())
        filename = f"{self.temp_dir}/clip_{timestamp}.mp4"
        
        # Get dimensions from first frame
        height, width, _ = self.frame_buffer[0].shape
        
        # mp4v is the safest codec for headless environments
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, 20.0, (width, height))
        
        for frame in list(self.frame_buffer):
            out.write(frame)
            
        out.release()
        print(f"📼 DVR Saved: {filename}")
        return filename

    def release(self):
        self.frame_buffer.clear()