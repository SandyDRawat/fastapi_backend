import asyncio
from dotenv import load_dotenv
import shutil
import subprocess
import requests
import time
import os

from langchain_core.prompts import ChatPromptTemplate

from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain

from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
    Microphone,
)
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()

import asyncio
from dotenv import load_dotenv
import shutil
import subprocess
import requests
import time
import os
import re

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
import json
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain

from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
    Microphone,
)


load_dotenv()
import asyncio
API_URL = "http://127.0.0.1:8000"
def call_tts(text):
    url = f"{API_URL}/speak/"
    params = {"text": text}
    response = requests.get(url, params=params, stream=True)
    if response.status_code == 200:
        output_file = "output.wav"
        with open(output_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        os.system(f"ffplay -nodisp -autoexit {output_file}")
    else:
        print(f"Error: {response.status_code}, {response.text}")

def call_ai(question, latex_text, command, chat_history):
    url = f"{API_URL}/whiteboard"
    payload = {
        "question": question,
        "whiteboard_content_latex_text": latex_text,
        "command": command,
        "chat_history": chat_history
    }
    response = requests.post(url, json=payload)
    return response.json()
class TranscriptCollector:
    def __init__(self):
        self.reset()

    def reset(self):
        self.transcript_parts = []

    def add_part(self, part):
        self.transcript_parts.append(part)

    def get_full_transcript(self):
        return ' '.join(self.transcript_parts)

transcript_collector = TranscriptCollector()

async def get_transcript(callback):
    transcription_complete = asyncio.Event()  # Event to signal transcription completion

    try:
        # example of setting up a client config. logging values: WARNING, VERBOSE, DEBUG, SPAM
        config = DeepgramClientOptions(options={"keepalive": "true"})
        deepgram: DeepgramClient = DeepgramClient("", config)

        dg_connection = deepgram.listen.asynclive.v("1")
        print ("Listening...")

        async def on_message(self, result, **kwargs):
            sentence = result.channel.alternatives[0].transcript
            
            if not result.speech_final:
                transcript_collector.add_part(sentence)
            else:
                # This is the final part of the current sentence
                transcript_collector.add_part(sentence)
                full_sentence = transcript_collector.get_full_transcript()
                # Check if the full_sentence is not empty before printing
                if len(full_sentence.strip()) > 0:
                    full_sentence = full_sentence.strip()
                    print(f"Human: {full_sentence}")
                    callback(full_sentence)  # Call the callback with the full_sentence
                    transcript_collector.reset()
                    transcription_complete.set()  # Signal to stop transcription and exit

        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)

        options = LiveOptions(
            model="nova-2",
            punctuate=True,
            language="en-US",
            encoding="linear16",
            channels=1,
            sample_rate=16000,
            endpointing=300,
            smart_format=True,
        )

        await dg_connection.start(options)

        # Open a microphone stream on the default input device
        microphone = Microphone(dg_connection.send)
        microphone.start()

        await transcription_complete.wait()  # Wait for the transcription to complete instead of looping indefinitely

        # Wait for the microphone to close
        microphone.finish()

        # Indicate that we've finished
        await dg_connection.finish()

    except Exception as e:
        print(f"Could not open socket: {e}")
        return

import asyncio
import asyncio
# ...existing code...

class ConversationManager:
    def __init__(self):
        self.transcription_response = ""
        self.exit_flag = False

    async def fetch_whiteboard_content(self):
        await asyncio.sleep(2)
        return "Extracted whiteboard content"

    async def main(self, question, chat_history):
        def handle_full_sentence(full_sentence):
            self.transcription_response = full_sentence
            if "goodbye" in full_sentence.lower():
                self.exit_flag = True

        while not self.exit_flag:
            whiteboard_task = asyncio.create_task(self.fetch_whiteboard_content())
            await get_transcript(handle_full_sentence)

            if self.exit_flag:
                break

            whiteboard_content = await whiteboard_task  
            response = call_ai(question, whiteboard_content, self.transcription_response, chat_history)
            print(response['instructions'])
            call_tts(response['voice_instructions'])
            chat_history = response['chat_history']

            self.transcription_response = ""

if __name__ == "__main__":
    question = "If the tangent to the curve y = x3 at the point P(t, t3) meets the curve again at Q, then the ordinate of the point which divides PQ internally in the ratio 1 ∶ 2 is:"
    chat_history = ""
    loop = asyncio.get_event_loop()
    loop.run_until_complete(ConversationManager().main(question, chat_history))
    
    print("Conversation ended. Exiting...")
    loop.close()