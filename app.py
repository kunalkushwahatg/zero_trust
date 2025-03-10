from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import base64
import io
from openai import OpenAI
from fastapi.middleware.cors import CORSMiddleware
from api import api, serpapi
import wave
from SearchVerification import FactChecker
from concurrent.futures import ThreadPoolExecutor
import asyncio
from audio_processor import AudioProcessor
from yt_live_fetch import LiveExtraction

print("Starting server...")

# Create a thread pool executor with 4 threads
executor = ThreadPoolExecutor(max_workers=4)
app = FastAPI()

# Initialize the FactChecker class with the API keys
fact_checker = FactChecker(api, serpapi)

# for live youtube stream
live_extraction = LiveExtraction()

# Add CORS middleware to allow cross-origin requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the OpenAI client
client = OpenAI(api_key=api)

def get_transcription(audio_data):
    with io.BytesIO(audio_data) as wav_buffer:
        wav_buffer.name = "audio.wav"  
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=wav_buffer  
        )
    return response

@app.get("/")
async def get():
    with open("index.html") as f:
        return HTMLResponse(f.read())


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    processor = AudioProcessor()
    text_buffer = [] # buffer to store text data for one session of fact checking
    
    async def process_chunk(chunk):
        with io.BytesIO() as wav_buffer:
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                wav_file.writeframes(chunk)
            
            wav_buffer.seek(0)
            try:
                transcription = await asyncio.get_event_loop().run_in_executor(
                    executor, 
                    lambda: get_transcription(wav_buffer.read())
                )
                return transcription.text
            except Exception as e:
                print(f"Transcription error: {e}")
                return None

    async def handle_fact_check(text):
        response = await asyncio.get_event_loop().run_in_executor(
            executor,
            lambda: fact_checker.fact_check_with_openai(text)
        )
        sentiment, verification = FactChecker.extract_json(response)
        return sentiment, verification

    try:
        while True:
            data = await websocket.receive_text() # receive audio data in base64 format
            pcm_data = base64.b64decode(data) # decode the base64 data
            processor.add_audio(pcm_data) # add the audio data to the processor
            
            for chunk in processor.get_speech_chunks():
                text = await process_chunk(chunk) # process the chunk of audio data
                if text:
                    print("Transcription:", text)
                    await websocket.send_text(text)
                    text_buffer.append(text)
                    
                    if len(' '.join(text_buffer).split()) >= 20:
                        '''
                        after 90 words, we will send the text to the fact checker and complete the session
                        '''
                        current_text = ' '.join(text_buffer)
                        text_buffer.clear()
                        #sentiment, verification = await handle_fact_check(current_text)
                        await websocket.send_text(f"Sentiment: {current_text}")
                        #await websocket.send_text(f"Verification: {verification}")
                    
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Process remaining audio if it meets minimum length
        if len(processor.speech_buffer) >= int(0.1 * processor.sample_rate * 2):
            text = await process_chunk(bytes(processor.speech_buffer))
            if text:
                await websocket.send_text(text)
            

@app.websocket("/live")
async def live_endpoint(websocket: WebSocket):
    await websocket.accept()
    link = await websocket.receive_text()
    print("link ",link)
    if "https://www.youtube.com" in link:
        stream_url = live_extraction.get_live_stream_url(link)
        while True:
            waveform = live_extraction.extract_audio_clip_as_waveform(stream_url, duration=10)
            print("shape of waveform ",waveform.shape)
            await websocket.send_text(str(waveform.shape))

#to run /live in curl use the following command
#curl -X GET "http://0.0.0.0:8000/live" -H  "accept: application/json" -d "https://www.youtube.com/watch?v=YDvsBbKfLPA"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

#run the server with the following command
#uvicorn app:app --reload

    