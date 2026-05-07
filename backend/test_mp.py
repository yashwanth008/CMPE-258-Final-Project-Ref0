import mediapipe as mp
try:
    print(f"✅ MediaPipe Location: {mp.__file__}")
    pose = mp.solutions.pose
    print("✅ SUCCESS: mp.solutions.pose is found!")
except AttributeError as e:
    print(f"❌ FAIL: Still broken. Error: {e}")
except Exception as e:
    print(f"❌ FAIL: Other error: {e}")