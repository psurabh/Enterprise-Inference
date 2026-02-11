import sys
import os
import shutil
import uuid
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import PlainTextResponse
import uvicorn

# Ensure the current directory is in sys.path so we can import 'main'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from main import AudioToText
except ImportError:
    # Fallback or error handling if main.py is missing or broken
    print("Error: Could not import AudioToText from main.py")
    AudioToText = None

app = FastAPI(title="OpenAI Compatible ASR API")

# Global model instance
asr_model: Optional[AudioToText] = None

@app.on_event("startup")
def load_model():
    global asr_model
    if AudioToText is None:
        print("AudioToText class not found.")
        return

    print("Loading ASR model...")
    try:
        # Initializing the model with defaults as per main.py
        # Assuming run from /home/ubuntu/ASR/ASR/Wav2Vec2
        asr_model = AudioToText(
            model_path="hindi.pt",
            warmup_iterations=1,
            sample_audio_path="./samples/hindi.wav"
        )
        print("ASR model loaded successfully.")
    except Exception as e:
        print(f"Failed to load ASR model: {e}")

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": asr_model is not None}

@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
):
    """
    OpenAI-compatible audio transcription endpoint.
    """
    if asr_model is None:
        raise HTTPException(status_code=503, detail="ASR model is not loaded.")

    # Generate a temporary file path
    # We use .wav extension because the model (librosa) likely supports it well
    # The uploaded file might be mp3 etc, librosa usually handles it via ffmpeg offering
    temp_file_path = f"acc_{uuid.uuid4()}.wav"
    
    try:
        # Save the uploaded file
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Transcribe
        # asr_model.transcribe returns (transcriptions: List[str], time_taken: float)
        transcriptions, _ = asr_model.transcribe(temp_file_path)
        
        # Join transcriptions
        text_output = " ".join(transcriptions)
        
        # Handle response formats
        if response_format == "text":
            return PlainTextResponse(text_output)
        elif response_format == "verbose_json":
            return {
                "text": text_output,
                "language": "hindi", 
                "duration": 0.0, 
                "segments": [] 
            }
        else:
            # Default to json
            return {"text": text_output}

    except Exception as e:
        # Log error
        print(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except:
                pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
