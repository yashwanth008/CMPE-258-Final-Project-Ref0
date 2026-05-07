import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ No API Key found in .env")
    exit()

genai.configure(api_key=api_key)

print(f"🔑 Checking models for API Key: {api_key[:5]}...")

try:
    print("\n--- AVAILABLE MODELS ---")
    found_flash = False
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"✅ {m.name}")
            if "flash" in m.name:
                found_flash = True

    if not found_flash:
        print("\n⚠️ WARNING: No 'Flash' model found. Your API Key might be old or restricted.")
        print("👉 Solution: Go to https://aistudio.google.com/app/apikey and create a NEW key.")
    else:
        print("\n🚀 SUCCESS: Flash model is available! We will use that name.")

except Exception as e:
    print(f"\n❌ ERROR: Could not list models.\n{e}")