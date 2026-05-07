import cv2
import os

# USE THE EXACT PATH YOU PASTED IN MAIN.PY
video_path = r"/Users/yashwanth/Documents/Projects/ref-0-hackathon/mock_data/test_match.mp4"

print(f"📂 Verifying: {video_path}")

if not os.path.exists(video_path):
    print("❌ FILE NOT FOUND on disk.")
    print("   -> Check spelling or folder name.")
else:
    print("✅ File found on disk.")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ OpenCV FAILED to open file.")
        print("   -> The codec might be unsupported or file is corrupt.")
    else:
        ret, frame = cap.read()
        if ret:
            print(f"✅ Success! Video is readable. Size: {frame.shape}")
        else:
            print("❌ File exists but has NO frames (Empty video).")