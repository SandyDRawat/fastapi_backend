import os
import asyncio
import json
import requests
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from deepgram import Deepgram
from fastapi.responses import StreamingResponse

app = FastAPI()

DG_API_KEY = os.getenv("DEEPGRAM_API_KEY")
MODEL_NAME = "aura-asteria-en"  # Change as needed

# Initialize Deepgram STT client
deepgram = Deepgram(DG_API_KEY)

class ConversationManager:
    def __init__(self):
        self.llm = LanguageModelProcessor()

    async def process_transcript(self, transcript: str):
        """Process the transcript with LLM and generate response."""
        if "goodbye" in transcript.lower():
            return "Goodbye!", None  # End conversation

        whiteboard_content = "Extracted whiteboard content"  # Placeholder for actual extraction
        llm_response_instructions, llm_response_voice = self.llm.process(transcript, whiteboard_content)

        audio_response = await self.text_to_speech(llm_response_voice)
        return llm_response_instructions, audio_response

    async def text_to_speech(self, text: str):
        """Convert text to speech using Deepgram TTS."""
        url = f"https://api.deepgram.com/v1/speak?model={MODEL_NAME}&encoding=linear16&sample_rate=24000"
        headers = {"Authorization": f"Token {DG_API_KEY}", "Content-Type": "application/json"}
        response = requests.post(url, json={"text": text}, headers=headers, stream=True)

        if response.status_code != 200:
            return None
        
        return response.content  # Return audio data

conversation_manager = ConversationManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    try:
        while True:
            audio_data = await websocket.receive_bytes()  # Receive audio input
            transcript = await transcribe_audio(audio_data)  # Convert audio to text
            response_text, audio_output = await conversation_manager.process_transcript(transcript)

            # Send text response
            await websocket.send_text(json.dumps({"text": response_text}))

            # Send audio response
            if audio_output:
                await websocket.send_bytes(audio_output)

    except WebSocketDisconnect:
        print("Client disconnected")

async def transcribe_audio(audio_data):
    """Send audio to Deepgram STT and get transcript."""
    deepgram_connection = deepgram.transcription.live({"punctuate": True, "language": "en-US"})
    transcript = ""

    def callback(response):
        nonlocal transcript
        transcript = response["channel"]["alternatives"][0]["transcript"]

    deepgram_connection.on("transcript", callback)
    deepgram_connection.send(audio_data)
    await asyncio.sleep(2)  # Wait for response
    return transcript
