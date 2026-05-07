import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from livekit import api

app = FastAPI()

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# DEFAULT DEV KEYS (From 'livekit-server --dev')
LIVEKIT_URL = "ws://localhost:7880"
API_KEY = "devkey"
API_SECRET = "secret"

@app.get("/api/token")
async def get_token():
    # Generate a token for a user
    token = api.AccessToken(API_KEY, API_SECRET) \
        .with_identity("Player") \
        .with_name("Player") \
        .with_grants(api.VideoGrants(
            room_join=True,
            room="ref-room",
            can_publish=True,
            can_subscribe=True
        ))
    
    return {"token": token.to_jwt(), "url": LIVEKIT_URL}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)