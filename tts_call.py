import asyncio
import requests
import os
from dotenv import load_dotenv
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
    Microphone,
)

load_dotenv()

API_URL = "http://127.0.0.1:8000"

def call_whiteboard(question, latex_text, command, chat_history):
    url = f"{API_URL}/whiteboard"
    payload = {
        "question": question,
        "whiteboard_content_latex_text": latex_text,
        "command": command,
        "chat_history": chat_history
    }
    response = requests.post(url, json=payload)
    return response.json()

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

class TranscriptCollector:
    def __init__(self):
        self.reset()

    def reset(self):
        self.transcript = ""

    def add_part(self, part):
        if part:
            self.transcript = part  # Store only the latest sentence

    def get_full_transcript(self):
        return self.transcript.strip()

transcript_collector = TranscriptCollector()

async def get_transcript(callback):
    transcription_complete = asyncio.Event()

    try:
        config = DeepgramClientOptions(options={"keepalive": "true"})
        deepgram = DeepgramClient("", config)
        dg_connection = deepgram.listen.asyncwebsocket.v("1")  # Updated method
        print("Listening...")

        async def on_message(self, result, **kwargs):
            sentence = result.channel.alternatives[0].transcript

            # Process only final transcriptions
            if result.speech_final:
                transcript_collector.add_part(sentence)
                full_sentence = transcript_collector.get_full_transcript()

                if full_sentence:
                    print(f"Human: {full_sentence}")
                    transcript_collector.reset()
                    transcription_complete.set()  # Signal completion
                    callback(full_sentence)

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
        microphone = Microphone(dg_connection.send)
        microphone.start()

        await transcription_complete.wait()  # Wait until a full sentence is received

        microphone.finish()
        await dg_connection.finish()

    except Exception as e:
        print(f"Could not open socket: {e}")
        return

class ConversationManager:
    def __init__(self):
        self.chat_history = []
        self.exit_flag = False

    async def fetch_whiteboard_content(self):
        await asyncio.sleep(2)
        return "Extracted whiteboard content"

    async def main(self):
        def handle_full_sentence(full_sentence):
            if "goodbye" in full_sentence.lower():
                self.exit_flag = True
            else:
                # Get whiteboard content
                whiteboard_task = asyncio.create_task(self.fetch_whiteboard_content())
                whiteboard_content = asyncio.run(whiteboard_task)

                # Call whiteboard API
                response = call_whiteboard(full_sentence, whiteboard_content, full_sentence, self.chat_history)
                self.chat_history.append({"user": full_sentence, "response": response.get("voice_instructions", "")})

                print("Instructions:", response.get("instructions", ""))
                print("Voice Instructions:", response.get("voice_instructions", ""))

                # Convert response to speech
                call_tts(response.get("voice_instructions", ""))

        while not self.exit_flag:
            await get_transcript(handle_full_sentence)
            if self.exit_flag:
                break

if __name__ == "__main__":
    asyncio.run(ConversationManager().main())  # Fixed event loop issue
    print("Conversation ended. Exiting...")
