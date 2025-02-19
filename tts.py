from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import requests
import os

app = FastAPI()

DG_API_KEY = os.getenv("DEEPGRAM_API_KEY")
MODEL_NAME = "aura-asteria-en"

@app.get("/speak/")
async def speak(text: str):
    DEEPGRAM_URL = f"https://api.deepgram.com/v1/speak?model={MODEL_NAME}&encoding=linear16&sample_rate=24000"
    headers = {
        "Authorization": f"Token {DG_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {"text": text}

    response = requests.post(DEEPGRAM_URL, stream=True, headers=headers, json=payload)
    
    if response.status_code != 200:
        return {"error": f"Deepgram API Error {response.status_code}"}
    
    return StreamingResponse(response.iter_content(chunk_size=1024), media_type="audio/wav")
