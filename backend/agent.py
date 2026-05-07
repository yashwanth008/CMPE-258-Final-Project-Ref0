import asyncio
import os
import cv2
import json
import easyocr
import time
from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import JobContext, WorkerOptions, cli
import google.generativeai as genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if api_key: genai.configure(api_key=api_key)

# 1. REMOVED GLOBAL READER INITIALIZATION HERE (Crucial Fix)

async def entrypoint(ctx: JobContext):
    await ctx.connect()
    print(f"🤖 Agent Connected: {ctx.room.name}")
    
    # 2. INITIALIZE READER HERE INSTEAD
    print("📚 Loading OCR Model (this happens once)...")
    reader = easyocr.Reader(['en'], gpu=False)
    print("✅ OCR Ready")

    @ctx.room.on("track_subscribed")
    def on_track_subscribed(track, publication, participant):
        if track.kind == rtc.TrackKind.KIND_VIDEO:
            print(f"🎥 Found Screen Share from {participant.identity}")
            # Pass reader to the processing function
            asyncio.create_task(process_video_track(ctx, track, reader))

async def process_video_track(ctx: JobContext, track: rtc.VideoTrack, reader):
    video_stream = rtc.VideoStream(track)
    print("🚀 Analysis Pipeline Started")
    
    frame_count = 0
    last_analysis_time = 0
    
    async for frame in video_stream:
        # Convert LiveKit YUV -> OpenCV BGR
        img = frame.buffer.to_numpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        h, w = img.shape[:2]
        if h > 480: img = cv2.resize(img, (int(w*(480/h)), 480))

        # 1. OCR (Fast - Every 0.5s)
        if frame_count % 15 == 0:
            score = detect_score(img, reader) # Pass reader instance
            if score:
                payload = json.dumps({"score": score}).encode('utf-8')
                await ctx.room.local_participant.publish_data(
                    payload,
                    topic="ai_analysis"
                )

        # 2. GEMINI REFEREE (Every 5 seconds)
        current_time = time.time()
        if current_time - last_analysis_time > 5:
            last_analysis_time = current_time
            asyncio.create_task(run_gemini_analysis(ctx, img))
        
        frame_count += 1

def detect_score(frame, reader):
    try:
        # Scans top 15% of screen
        roi = frame[0:int(frame.shape[0]*0.15), :]
        res = reader.readtext(roi, detail=0)
        return " | ".join([x for x in res if any(c.isdigit() for c in x)])
    except: return None

async def run_gemini_analysis(ctx, img_frame):
    try:
        temp_path = f"temp_{int(time.time())}.jpg"
        cv2.imwrite(temp_path, img_frame)
        
        myfile = genai.upload_file(temp_path)
        while myfile.state.name == "PROCESSING":
            await asyncio.sleep(0.5)
            myfile = genai.get_file(myfile.name)
        
        # Using Gemini 2.0 Flash
        model = genai.GenerativeModel("models/gemini-2.0-flash-exp")
        
        result = model.generate_content([
            myfile, 
            "You are an NBA Ref. Analyze this image. Return JSON ONLY: {\"event\": \"description\", \"foul\": boolean, \"rule_citation\": \"rule name\"}"
        ])
        
        if result.text:
            clean_json = result.text.replace("```json", "").replace("```", "")
            await ctx.room.local_participant.publish_data(
                clean_json.encode('utf-8'),
                reliable=True,
                topic="ai_verdict"
            )
            
        try: 
            myfile.delete()
            os.remove(temp_path)
        except: pass
            
    except Exception as e:
        print(f"Gemini Error: {e}")

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))