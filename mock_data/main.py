import asyncio
import base64
import os
import shutil
import subprocess
import urllib.request
import json
import time
import numpy as np
import cv2  # Headless (Safe)
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from dvr_core import SmartDVR

# --- CONFIG ---
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    import google.generativeai as genai
    genai.configure(api_key=api_key)
else:
    print("⚠️ GEMINI_API_KEY not found")

# ─────────────────────────────────────────────────────────────
# CHANGE 1 — YOLO11 Foul Detection Model
# ─────────────────────────────────────────────────────────────
from ultralytics import YOLO

def load_foul_model():
    """
    Load YOLO11 foul detection model.
    Falls back to yolo11n-pose.pt (already in repo) if no trained
    checkpoint exists yet — so the app works before training is done.
    """
    checkpoint_file = "runs/best_model.txt"
    if os.path.exists(checkpoint_file):
        path = open(checkpoint_file).read().strip()
        if os.path.exists(path):
            print(f"✅ Loaded trained foul model: {path}")
            return YOLO(path)
    # Fallback: use the pose model already in the repo
    fallback = "yolo11n-pose.pt"
    print(f"⚠️  No trained checkpoint found — using fallback: {fallback}")
    return YOLO(fallback)

foul_model = load_foul_model()
FOUL_CLASSES = {"foul", "contact"}   # class names that trigger DVR + Gemini review
# ─────────────────────────────────────────────────────────────

BUFFER_DIR = "temp_buffer"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
MODEL_PATH = "pose_landmarker.task"

active_websockets = []
executor = ThreadPoolExecutor(max_workers=1)

# --- GLOBAL STATE ---
VISION_ACTIVE = True
global_dvr = SmartDVR(temp_dir=BUFFER_DIR)

# --- 1. AI SETUP ---
if not os.path.exists(MODEL_PATH):
    print("⬇️ Downloading MediaPipe Model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

landmarker = None
def get_landmarker():
    global landmarker
    if landmarker is None and os.path.exists(MODEL_PATH):
        try:
            base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_poses=4,
                min_pose_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            landmarker = vision.PoseLandmarker.create_from_options(options)
            print("🚀 AI Engine Ready (VIDEO MODE)!")
        except Exception as e:
            print(f"❌ AI Init Failed: {e}")
    return landmarker

# --- 2. DRAWER ---
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28)
]

def draw_landmarks(image, detection_result):
    if not detection_result.pose_landmarks: return image
    h, w, _ = image.shape
    for landmarks in detection_result.pose_landmarks:
        for p1_idx, p2_idx in POSE_CONNECTIONS:
            p1 = landmarks[p1_idx]
            p2 = landmarks[p2_idx]
            cv2.line(image, (int(p1.x*w), int(p1.y*h)), (int(p2.x*w), int(p2.y*h)), (0, 255, 0), 2)
    return image

# --- 3. FASTAPI ---
async def cleanup_loop():
    while True:
        try:
            await asyncio.sleep(600)
            if os.path.exists(BUFFER_DIR):
                shutil.rmtree(BUFFER_DIR)
                os.makedirs(BUFFER_DIR)
        except: break

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.path.exists(BUFFER_DIR): os.makedirs(BUFFER_DIR)
    task = asyncio.create_task(cleanup_loop())
    yield
    task.cancel()
    executor.shutdown()

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# --- 4. ENDPOINTS ---
async def analyze_clip(clip_path):
    print(f"⚖️ Reviewing Clip: {clip_path}")
    for ws in active_websockets:
        try: await ws.send_json({"type": "score_update", "data": "VAR CHECKING..."})
        except: pass

    try:
        import google.generativeai as genai

        video_file = genai.upload_file(path=clip_path)
        while video_file.state.name == "PROCESSING":
            await asyncio.sleep(0.5)
            video_file = genai.get_file(video_file.name)

        if video_file.state.name == "FAILED":
            raise ValueError("Video processing failed.")

        model_name = "models/gemini-2.0-flash"
        print(f"🔄 Using Model: {model_name}")
        model = genai.GenerativeModel(model_name)

        prompt = """
        You are an expert Sports Referee using Video Assistant Referee (VAR) technology.
        Analyze this video clip frame-by-frame.

        1. Identify the Sport.
        2. Break down the key action (e.g., "Player A drives to basket", "Defender B makes contact").
        3. Identify any Rules Violated with specific terminology (e.g., "NBA Rule 12B - Blocking Foul", "FIFA Law 12 - Trip").
        4. Deliver a final Verdict: CLEAN, FOUL, or VIOLATION.

        RETURN ONLY RAW JSON. Do not use Markdown formatting.
        Structure:
        {
          "sport": "Basketball/Soccer/etc",
          "action_breakdown": "Clear description of what happened...",
          "rule_violated": "Specific Rule Name or 'None'",
          "verdict": "FOUL/CLEAN/VIOLATION",
          "explanation": "A concise explanation of why this ruling was made.",
          "confidence": 95
        }
        """
        response = model.generate_content([video_file, prompt])
        print(f"🤖 Verdict: {response.text}")

        for ws in active_websockets:
            try: await ws.send_json({"type": "verdict", "data": response.text})
            except: pass

        try:
            video_file.delete()
            os.remove(clip_path)
        except: pass

    except Exception as e:
        print(f"Gemini Error: {e}")
        for ws in active_websockets:
            try: await ws.send_json({"type": "score_update", "data": "REVIEW FAILED"})
            except: pass

@app.post("/api/trigger_review")
async def trigger_review(background_tasks: BackgroundTasks):
    print(f"🚨 Requesting Review. Buffer Size: {len(global_dvr.frame_buffer)}")
    clip_path = global_dvr.save_last_clip()
    if clip_path:
        background_tasks.add_task(analyze_clip, clip_path)
        return {"status": "Review Started", "clip": clip_path}
    return {"status": "Buffer Empty"}

@app.post("/api/toggle_vision")
async def toggle_vision():
    global VISION_ACTIVE
    VISION_ACTIVE = not VISION_ACTIVE
    return {"status": "ON" if VISION_ACTIVE else "OFF", "enabled": VISION_ACTIVE}

# ─────────────────────────────────────────────────────────────
# CHANGE 3 — POST /api/retrain endpoint
# ─────────────────────────────────────────────────────────────
@app.post("/api/retrain")
async def trigger_retrain(background_tasks: BackgroundTasks):
    """
    Kick off YOLO11 fine-tuning as a background task.
    After training completes, hot-swaps foul_model to the new checkpoint
    without requiring a server restart.
    """
    def run_training():
        global foul_model
        print("🏋️  Starting background retraining...")
        result = subprocess.run(
            ["python", "train.py", "--model-size", "nano"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(__file__)  # run from backend/ directory
        )
        if result.returncode == 0:
            foul_model = load_foul_model()  # hot-swap to new checkpoint
            print("✅ Retrain complete — new foul model loaded")
        else:
            print(f"❌ Retrain failed:\n{result.stderr}")

    background_tasks.add_task(run_training)
    return {
        "status": "Retraining started",
        "model_size": "nano",
        "message": "Training runs in background. Model will auto-reload when done."
    }

@app.get("/api/model-status")
async def model_status():
    """Return which checkpoint is currently loaded."""
    checkpoint_file = "runs/best_model.txt"
    if os.path.exists(checkpoint_file):
        path = open(checkpoint_file).read().strip()
        return {"checkpoint": path, "trained": True}
    return {"checkpoint": "yolo11n-pose.pt (fallback)", "trained": False}
# ─────────────────────────────────────────────────────────────

# --- 5. WEBSOCKET HANDLER ---
def process_frame_sync(frame_data, timestamp_ms):
    try:
        np_arr = np.frombuffer(base64.b64decode(frame_data), np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None: return None, None, None

        global_dvr.write_frame(frame)

        action = None
        if VISION_ACTIVE:
            # ── MediaPipe skeleton overlay (unchanged) ────────────
            ai = get_landmarker()
            if ai:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
                result = ai.detect_for_video(mp_image, timestamp_ms)
                frame = draw_landmarks(frame, result)

            # ─────────────────────────────────────────────────────
            # CHANGE 2 — YOLO11 foul detection
            # Replaces the old wrist-above-nose heuristic:
            #   if pose[15].y < pose[0].y or pose[16].y < pose[0].y:
            #       action = "ACTION DETECTED"
            # ─────────────────────────────────────────────────────
            yolo_results = foul_model(frame, conf=0.5, verbose=False)
            for r in yolo_results:
                for cls_id in r.boxes.cls:
                    cls_name = foul_model.names[int(cls_id)]
                    if cls_name in FOUL_CLASSES:
                        action = f"FOUL DETECTED ({cls_name.upper()})"
                        # Draw YOLO bounding boxes on frame
                        for box in r.boxes.xyxy:
                            x1, y1, x2, y2 = map(int, box)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(frame, cls_name.upper(),
                                       (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                                       0.6, (0, 0, 255), 2)
            # ─────────────────────────────────────────────────────

        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        b64 = base64.b64encode(buffer).decode('utf-8')
        return b64, action, None

    except Exception as e:
        print(f"Error: {e}")
        return None, None, None

@app.websocket("/ws/stream")
async def video_stream(websocket: WebSocket):
    await websocket.accept()
    active_websockets.append(websocket)
    print("🟢 Frontend Connected")
    loop = asyncio.get_running_loop()
    start_time = int(time.time() * 1000)

    try:
        while True:
            data = await websocket.receive_text()
            if "data:image" in data:
                _, encoded = data.split(",", 1)
                current_timestamp = int(time.time() * 1000) - start_time

                b64, action, _ = await loop.run_in_executor(
                    executor, process_frame_sync, encoded, current_timestamp
                )

                if b64:
                    await websocket.send_json({"type": "video_frame", "data": b64})
                    if action and current_timestamp % 10 == 0:
                        await websocket.send_json({"type": "score_update", "data": action})
    except WebSocketDisconnect:
        if websocket in active_websockets: active_websockets.remove(websocket)
        print("🔴 Disconnected")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
