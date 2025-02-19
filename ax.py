
import asyncio
from dotenv import load_dotenv
import os

from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
    Microphone,
)

load_dotenv()

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

async def get_transcript(callback, transcription_complete):
    try:
        config = DeepgramClientOptions(options={"keepalive": "true"})
        deepgram: DeepgramClient = DeepgramClient("", config)

        dg_connection = deepgram.listen.asyncwebsocket.v("1") #Corrected line
        print("Listening...")

        async def on_message(self, result, **kwargs):
            try:
                sentence = result.channel.alternatives[0].transcript

                if not result.speech_final:
                    transcript_collector.add_part(sentence)
                else:
                    transcript_collector.add_part(sentence)
                    full_sentence = transcript_collector.get_full_transcript()
                    if len(full_sentence.strip()) > 0:
                        full_sentence = full_sentence.strip()
                        print(f"Human: {full_sentence}")
                        callback(full_sentence)
                        transcript_collector.reset()
                        transcription_complete.set()
            except Exception as e:
                print(f"Error in on_message: {e}")

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

        await transcription_complete.wait()

        microphone.finish()
        await dg_connection.finish()

    except Exception as e:
        print(f"Could not open socket: {e}")
        return

class ConversationManager:
    def __init__(self):
        self.transcription_response = ""

    async def main(self):
        def handle_full_sentence(full_sentence):
            self.transcription_response = full_sentence

        while True:
            transcription_complete = asyncio.Event()
            await get_transcript(handle_full_sentence, transcription_complete)

            if "goodbye" in self.transcription_response.lower():
                break

            self.transcription_response = ""

if __name__ == "__main__":
    manager = ConversationManager()
    asyncio.run(manager.main())
"""

import asyncio
from dotenv import load_dotenv
import os
import threading
import httpx

from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
)

load_dotenv()

class ConversationManager:
    def __init__(self):
        self.transcription_response = ""
        self.exit_flag = False
        self.exit_lock = threading.Lock()

    def handle_full_sentence(self, sentence):
        self.transcription_response = sentence
        print(f"Human: {sentence}")

    async def get_transcript(self, dg_connection):
        def on_message(self, result, **kwargs):
            sentence = result.channel.alternatives[0].transcript
            if len(sentence) > 0:
                self.handle_full_sentence(sentence)

        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)

        options = LiveOptions(model="nova-2", punctuate=True, language="en-US")

        if not dg_connection.start(options):
            print("Failed to start connection")
            return

        def http_stream():
            try:
                with httpx.stream("GET", "http://stream.live.vc.bbcmedia.co.uk/bbc_world_service") as r:
                    for data in r.iter_bytes():
                        self.exit_lock.acquire()
                        if self.exit_flag:
                            break
                        self.exit_lock.release()
                        dg_connection.send(data)
            except httpx.RequestError as e:
                print(f"HTTP request error: {e}")
            except Exception as e:
                print(f"Error in http_stream: {e}")
            finally:
                dg_connection.finish()

        http_thread = threading.Thread(target=http_stream)
        http_thread.start()

        while True:
            self.exit_lock.acquire()
            if self.exit_flag:
                self.exit_lock.release()
                break
            self.exit_lock.release()
            await asyncio.sleep(0.1)

        http_thread.join()

    async def main(self):
        try:
            deepgram: DeepgramClient = DeepgramClient()
            dg_connection = deepgram.listen.websocket.v("1")

            await self.get_transcript(dg_connection)

        except Exception as e:
            print(f"Could not open socket: {e}")

    def stop(self):
        self.exit_lock.acquire()
        self.exit_flag = True
        self.exit_lock.release()

if __name__ == "__main__":
    manager = ConversationManager()
    loop = asyncio.get_event_loop()
    task = loop.create_task(manager.main())

    try:
        loop.run_until_complete(task)
    except KeyboardInterrupt:
        print("Stopping...")
        manager.stop()
        loop.run_until_complete(asyncio.sleep(1)) #Give time to the thread to close
        if not task.cancelled():
            task.cancel()
        loop.run_until_complete(task)
        print("Stopped.")
    finally:
        loop.close()
import asyncio
import time
import numpy as np
from deepgram import DeepgramClient, LiveOptions, DeepgramClientOptions
import pyaudio

DEEPGRAM_API_KEY = ""  # Replace with your API key

config = DeepgramClientOptions(options={"keepalive": "true"})
deepgram = DeepgramClient(DEEPGRAM_API_KEY, config)
dg_connection = deepgram.listen.asyncwebsocket.v("1")

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
SILENCE_THRESHOLD = 50  # Adjust as needed for RMS
SILENCE_TIMEOUT = 5
LISTEN_TIMEOUT = 60

audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
last_speech_time = time.time()


def calculate_rms(audio_chunk):
    Calculates RMS of an audio chunk.
    audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
    rms = np.sqrt(np.mean(np.square(audio_data)))
    return rms


async def on_message(result, **kwargs):
    if not result.speech_final:
        return
    transcript = result.channel.alternatives[0].transcript.strip()
    if transcript:
        print(f"Human: {transcript}")
        global last_speech_time
        last_speech_time = time.time()


async def start_transcription():
    options = LiveOptions(
        model="nova-2",
        punctuate=True,
        language="en-US",
        encoding="linear16",
        channels=1,
        sample_rate=16000,
        endpointing=5000,
        smart_format=True,
    )
    try:
        await dg_connection.start(options)
        while True:
            audio_chunk = stream.read(CHUNK, exception_on_overflow=False)
            await dg_connection.send(audio_chunk)
            if time.time() - last_speech_time > SILENCE_TIMEOUT:
                print("\n[Silence detected] Stopping transcription...\n")
                try:
                    await dg_connection.finish()
                except Exception as finish_error:
                    print(f"Error during dg_connection.finish(): {finish_error}")
                break
            await asyncio.sleep(0.1)
    except Exception as e:
        print(f"Deepgram Error: {e}")

async def listen_loop():
    global last_speech_time
    while True:
        print("\n[Waiting for speech...]\n")
        last_speech_time = time.time()
        while time.time() - last_speech_time < LISTEN_TIMEOUT:
            audio_chunk = stream.read(CHUNK, exception_on_overflow=False)
            volume = calculate_rms(audio_chunk)
            if volume > SILENCE_THRESHOLD:
                print("\n[Speech detected] Starting transcription...\n")
                await start_transcription()
                break
            await asyncio.sleep(0.1)
        print("\n[Still waiting for speech...]\n")

async def main():
    try:
        await listen_loop()
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if 'stream' in locals() and stream.is_active():
            stream.stop_stream()
            stream.close()
        if 'audio' in locals():
            audio.terminate()
        if 'dg_connection' in locals() and dg_connection.started:
            try:
                await dg_connection.finish()
            except Exception as final_error:
                print(f"Error during final dg_connection.finish():{final_error}")

if __name__ == "__main__":
    asyncio.run(main())
import asyncio
import time
import numpy as np
from deepgram import DeepgramClient, LiveOptions, DeepgramClientOptions, LiveTranscriptionEvents, Microphone
import pyaudio
from dotenv import load_dotenv
import os

load_dotenv()

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

config = DeepgramClientOptions(options={"keepalive": "true"})
deepgram = DeepgramClient(DEEPGRAM_API_KEY, config)
dg_connection = deepgram.listen.asyncwebsocket.v("1")

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 512 #reduced from 1024
SILENCE_THRESHOLD = 30 #reduced from 50
SILENCE_TIMEOUT = 4
IDLE_TIMEOUT = 60

audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

def calculate_rms(audio_chunk):
    audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
    squared_mean = np.mean(np.square(audio_data))
    if squared_mean >= 0:
        rms = np.sqrt(squared_mean)
    else:
        rms = 0
    return rms

class TranscriptHandler:
    def __init__(self):
        self.transcript_parts = []
        self.last_transcript = ""

    def reset(self):
        self.transcript_parts = []

    def add_part(self, part):
        self.transcript_parts.append(part)

    def get_full_transcript(self):
        return ' '.join(self.transcript_parts).strip()

    async def on_message(self, result, **kwargs):
        try:
            sentence = result.channel.alternatives[0].transcript

            if not result.speech_final:
                self.add_part(sentence)
            else:
                self.add_part(sentence)
                full_sentence = self.get_full_transcript()
                if full_sentence and full_sentence != self.last_transcript: #duplicate check
                    print(f"Human: {full_sentence}")
                    self.last_transcript = full_sentence #update last transcript.
                    self.reset()
        except Exception as e:
            print(f"Error in on_message: {e}")

transcript_handler = TranscriptHandler()

async def transcribe():
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
    try:
        await dg_connection.start(options)
        dg_connection.on(LiveTranscriptionEvents.Transcript, lambda *args, **kwargs: transcript_handler.on_message(**kwargs))
        microphone = Microphone(dg_connection.send)
        microphone.start()

        last_speech_time = time.time()
        while True:
            audio_chunk = stream.read(CHUNK, exception_on_overflow=False)
            volume = calculate_rms(audio_chunk)

            if volume > SILENCE_THRESHOLD:
                last_speech_time = time.time()

            if time.time() - last_speech_time > SILENCE_TIMEOUT:
                microphone.finish()
                await dg_connection.finish()
                print("\n[Silence detected] Transcription complete.\n")
                break
            await asyncio.sleep(0.025) #reduced sleep time

    except Exception as e:
        print(f"Deepgram Error: {e}")

async def listen_and_transcribe():
    while True:
        print("\n[Waiting for speech...]\n")
        last_idle = time.time()
        while time.time() - last_idle < IDLE_TIMEOUT:
            audio_chunk = stream.read(CHUNK, exception_on_overflow=False)
            volume = calculate_rms(audio_chunk)

            if volume > SILENCE_THRESHOLD:
                print("\n[Speech detected] Starting transcription...\n")
                await transcribe()
                break
            await asyncio.sleep(0.025) #reduced sleep time
        if time.time() - last_idle >= IDLE_TIMEOUT:
            print("\n[Idle Timeout] Resetting")

async def main():
    try:
        await listen_and_transcribe()
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if 'stream' in locals() and stream.is_active():
            stream.stop_stream()
            stream.close()
        if 'audio' in locals():
            audio.terminate()
        if 'dg_connection' in locals() and dg_connection.started:
            try:
                await dg_connection.finish()
            except Exception as final_error:
                print(f"Error during final dg_connection.finish():{final_error}")

if __name__ == "__main__":
    asyncio.run(main())
    """