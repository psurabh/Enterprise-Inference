"""
FastAPI Server for Svara TTS
Integrates VLLM token generation with SNAC audio decoding
"""
import logging
import asyncio
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import httpx

from tts_engine.codec import SNACCodec
from tts_engine.voice_config import get_voice, get_all_voices

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global instances
snac_codec: Optional[SNACCodec] = None
vllm_client: Optional[httpx.AsyncClient] = None

# Configuration
VLLM_BASE_URL = "http://10.233.104.79:2080"  # K8s pod IP
VLLM_MODEL = "kenpath/svara-tts-v1"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global snac_codec, vllm_client
    
    # Startup
    logger.info("Initializing Svara TTS API server...")
    
    # Initialize SNAC codec
    logger.info("Loading SNAC codec...")
    snac_codec = SNACCodec(device='cpu')
    logger.info("SNAC codec loaded successfully")
    
    # Initialize voice configuration
    logger.info("Loading voice configuration...")
    voices = get_all_voices()
    logger.info(f"Loaded {len(voices)} voices")
    
    # Initialize VLLM HTTP client
    vllm_client = httpx.AsyncClient(
        base_url=VLLM_BASE_URL,
        timeout=httpx.Timeout(60.0)
    )
    logger.info(f"Connected to VLLM at {VLLM_BASE_URL}")
    
    logger.info("Server initialization complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down server...")
    if vllm_client:
        await vllm_client.aclose()
    logger.info("Server shutdown complete")


app = FastAPI(
    title="Svara TTS API",
    description="Text-to-Speech API using VLLM + SNAC",
    version="1.0.0",
    lifespan=lifespan
)


# Request/Response Models
class TTSRequest(BaseModel):
    text: str = Field(..., description="Text to convert to speech", min_length=1, max_length=1000)
    voice_id: str = Field(default="en-US-male-1", description="Voice ID to use")
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Speech speed multiplier")
    format: str = Field(default="wav", description="Audio format (wav, mp3, pcm)")


class VoiceInfo(BaseModel):
    id: str
    name: str
    language: str
    language_code: str
    gender: str
    description: str


class HealthResponse(BaseModel):
    status: str
    snac_loaded: bool
    vllm_connected: bool
    voices_loaded: int


# Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    vllm_ok = False
    
    try:
        if vllm_client:
            response = await vllm_client.get("/health", timeout=5.0)
            vllm_ok = response.status_code == 200
    except Exception as e:
        logger.warning(f"VLLM health check failed: {e}")
    
    return HealthResponse(
        status="healthy" if (snac_codec and vllm_ok) else "degraded",
        snac_loaded=snac_codec is not None,
        vllm_connected=vllm_ok,
        voices_loaded=len(get_all_voices())
    )


@app.get("/v1/voices", response_model=list[VoiceInfo])
async def get_voices():
    """List all available voices"""
    return [
        VoiceInfo(
            id=voice.voice_id,
            name=voice.name,
            language=voice.languages[0] if voice.languages else "en",
            language_code=voice.languages[0] if voice.languages else "en",
            gender=voice.gender or "unknown",
            description=voice.description or ""
        )
        for voice in get_all_voices()
    ]


@app.get("/v1/voices/{voice_id}", response_model=VoiceInfo)
async def get_voice_info(voice_id: str):
    """Get information about a specific voice"""
    voice = get_voice(voice_id)
    if not voice:
        raise HTTPException(status_code=404, detail=f"Voice '{voice_id}' not found")
    
    return VoiceInfo(
        id=voice.voice_id,
        name=voice.name,
        language=voice.languages[0] if voice.languages else "en",
        language_code=voice.languages[0] if voice.languages else "en",
        gender=voice.gender or "unknown",
        description=voice.description or ""
    )


async def generate_tokens_from_vllm(prompt: str, max_tokens: int = 2048) -> list[int]:
    """
    Call VLLM to generate audio tokens from text prompt
    """
    try:
        payload = {
            "model": VLLM_MODEL,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "stop": ["<|im_end|>"],
        }
        
        logger.info(f"Sending request to VLLM: {len(prompt)} chars")
        response = await vllm_client.post("/v1/completions", json=payload)
        response.raise_for_status()
        
        result = response.json()
        generated_text = result["choices"][0]["text"]
        
        # TODO: Parse tokens from generated text
        # For now, return empty list - needs proper token parsing
        logger.warning("Token parsing not yet implemented - returning empty list")
        return []
        
    except httpx.HTTPError as e:
        logger.error(f"VLLM request failed: {e}")
        raise HTTPException(status_code=502, detail=f"VLLM service error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in token generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def format_tts_prompt(text: str, voice: Any) -> str:
    """
    Format text into VLLM prompt following Svara-TTS format
    """
    lang = voice.languages[0] if voice.languages else "en"
    # Based on official repo's prompt format
    prompt = f"""<|im_start|>system
You are a text-to-speech system. Generate natural speech for the given text.<|im_end|>
<|im_start|>user
Voice: {voice.name} ({lang})
Text: {text}<|im_end|>
<|im_start|>assistant
<|audio|>"""
    
    return prompt


@app.post("/v1/text-to-speech")
async def text_to_speech(request: TTSRequest):
    """
    Convert text to speech audio
    
    Returns streaming audio in the requested format
    """
    # Validate voice
    voice = get_voice(request.voice_id)
    if not voice:
        raise HTTPException(status_code=404, detail=f"Voice '{request.voice_id}' not found")
    
    # Check codec availability
    if not snac_codec:
        raise HTTPException(status_code=503, detail="SNAC codec not available")
    
    logger.info(f"TTS request: text='{request.text[:50]}...', voice={request.voice_id}")
    
    try:
        # Format prompt for VLLM
        prompt = format_tts_prompt(request.text, voice)
        
        # Generate tokens from VLLM
        tokens = await generate_tokens_from_vllm(prompt)
        
        if not tokens:
            raise HTTPException(
                status_code=500, 
                detail="No audio tokens generated - token parsing not yet implemented"
            )
        
        # Decode tokens to audio using SNAC
        # Group tokens into 7-token windows for SNAC decoding
        audio_chunks = []
        for i in range(0, len(tokens), 7):
            window = tokens[i:i+7]
            if len(window) == 7:  # Only decode complete windows
                audio_bytes = snac_codec.decode_window(window)
                audio_chunks.append(audio_bytes)
        
        if not audio_chunks:
            raise HTTPException(status_code=500, detail="Failed to decode audio")
        
        # Combine all audio chunks
        full_audio = b''.join(audio_chunks)
        
        # Return appropriate format
        if request.format == "wav":
            # Add WAV header
            wav_data = create_wav_header(len(full_audio)) + full_audio
            return Response(
                content=wav_data,
                media_type="audio/wav",
                headers={
                    "Content-Disposition": "attachment; filename=speech.wav"
                }
            )
        elif request.format == "pcm":
            return Response(
                content=full_audio,
                media_type="audio/pcm",
                headers={
                    "Content-Disposition": "attachment; filename=speech.pcm"
                }
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {request.format}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def create_wav_header(data_size: int, sample_rate: int = 24000, channels: int = 1, bits_per_sample: int = 16) -> bytes:
    """
    Create WAV file header
    """
    import struct
    
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    
    header = b'RIFF'
    header += struct.pack('<I', data_size + 36)  # File size - 8
    header += b'WAVE'
    header += b'fmt '
    header += struct.pack('<I', 16)  # Subchunk1 size
    header += struct.pack('<H', 1)   # Audio format (PCM)
    header += struct.pack('<H', channels)
    header += struct.pack('<I', sample_rate)
    header += struct.pack('<I', byte_rate)
    header += struct.pack('<H', block_align)
    header += struct.pack('<H', bits_per_sample)
    header += b'data'
    header += struct.pack('<I', data_size)
    
    return header


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Svara TTS API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "voices": "/v1/voices",
            "tts": "/v1/text-to-speech"
        },
        "documentation": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
