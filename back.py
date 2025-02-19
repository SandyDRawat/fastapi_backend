from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain.chains import LLMChain
import json
import requests

load_dotenv()

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load System Prompt
with open('system_prompt.txt', 'r') as file:
    system_prompt = file.read().strip()

# LLM Processor
class LanguageModelProcessor:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model='gemini-1.5-pro',
            temperature=0,
            google_api_key=os.getenv("GEMINI_API_KEY")
        )
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{text}")
        ])
        self.conversation = LLMChain(llm=self.llm, prompt=self.prompt, memory=self.memory)

    def process_input(self, text: str):
        response = self.conversation.run({"text": text})
        return response

# Initialize LLM Processor
llm_processor = LanguageModelProcessor()

# Define TTS API Wrapper
class TTSProcessor:
    def __init__(self):
        self.tts_api_url = os.getenv("TTS_API_URL")  # External TTS service URL
        self.tts_api_key = os.getenv("TTS_API_KEY")

    def generate_speech(self, text: str):
        headers = {"Authorization": f"Bearer {self.tts_api_key}", "Content-Type": "application/json"}
        payload = {"text": text, "voice": "en-US-Wavenet-D"}
        
        try:
            response = requests.post(self.tts_api_url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json().get("audio_url")  # Assuming API returns an audio URL
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=500, detail=f"TTS API error: {str(e)}")

# Initialize TTS Processor
tts_processor = TTSProcessor()

# Request Model
class QueryRequest(BaseModel):
    text: str
    tts: bool = False  # Flag to indicate whether TTS output is required

# Endpoint for LLM Processing
@app.post("/llm")
async def process_llm(request: QueryRequest):
    try:
        response = llm_processor.process_input(request.text)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint for TTS Processing
@app.post("/tts")
async def process_tts(request: QueryRequest):
    try:
        audio_url = tts_processor.generate_speech(request.text)
        return {"audio_url": audio_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Combined Endpoint for LLM and TTS
@app.post("/process")
async def process_request(request: QueryRequest):
    try:
        ai_response = llm_processor.process_input(request.text)
        audio_url = tts_processor.generate_speech(ai_response) if request.tts else None
        return {"response": ai_response, "audio_url": audio_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
