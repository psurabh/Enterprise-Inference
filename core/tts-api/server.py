"""
FastAPI TTS Server - Integrates VLLM + SNAC for Text-to-Speech
Streaming implementation based on Kenpath/svara-tts-inference
"""
import asyncio
import concurrent.futures
import os
import sys
import time
from typing import Optional, List, AsyncIterator
import logging

import httpx
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

# Add tts_engine to path
sys.path.insert(0, '/app')
from tts_engine.codec import SNACCodec
from tts_engine.constants import SAMPLE_RATE, BIT_DEPTH
from tts_engine.mapper import SvaraMapper, extract_custom_token_numbers
from tts_engine.transports import VLLMCompletionsTransportAsync
from tts_engine.buffers import AudioBuffer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
VLLM_BASE_URL = os.getenv("VLLM_URL", "http://10.233.104.79:2080")
VLLM_MODEL = "kenpath/svara-tts-v1"
API_PORT = int(os.getenv("API_PORT", "8000"))

# Initialize FastAPI
app = FastAPI(
    title="Svara TTS API",
    description="Text-to-Speech API powered by VLLM and SNAC",
    version="1.0.0"
)

# Global instances
snac_codec = None
vllm_transport = None


# Request/Response Models
class TTSRequest(BaseModel):
    text: str = Field(..., description="Text to convert to speech", min_length=1, max_length=500)
    voice: str = Field(default="English (Male)", description="Voice ID to use (e.g., 'English (Male)', 'Hindi (Female)')")
    speed: float = Field(default=1.0, description="Speech speed multiplier", ge=0.5, le=2.0)
    
class VoiceInfo(BaseModel):
    id: str
    name: str
    language: str
    gender: str

class HealthResponse(BaseModel):
    status: str
    vllm_status: str
    snac_loaded: bool
    timestamp: float


# Startup/Shutdown Events
@app.on_event("startup")
async def startup_event():
    """Initialize SNAC codec and VLLM transport on startup"""
    global snac_codec, vllm_transport
    logger.info("🚀 Starting TTS API Server...")
    logger.info(f"📍 VLLM URL: {VLLM_BASE_URL}")
    logger.info(f"🎤 Loading SNAC codec...")
    
    try:
        snac_codec = SNACCodec(device='cpu')
        logger.info(f"✅ SNAC codec loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load SNAC: {e}")
        raise
    
    # Initialize VLLM transport
    try:
        # Ensure URL ends with /v1 for completions endpoint
        base_url = VLLM_BASE_URL.rstrip('/').removesuffix('/v1') + '/v1'
        vllm_transport = VLLMCompletionsTransportAsync(base_url, VLLM_MODEL)
        logger.info(f"✅ VLLM transport initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize VLLM transport: {e}")
        raise
    
    # Test VLLM connection
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            health_url = VLLM_BASE_URL.rstrip('/').removesuffix('/v1') + '/health'
            response = await client.get(health_url)
            if response.status_code == 200:
                logger.info(f"✅ VLLM connection verified")
            else:
                logger.warning(f"⚠️ VLLM returned status {response.status_code}")
    except Exception as e:
        logger.warning(f"⚠️ Could not verify VLLM connection: {e}")
    
    logger.info("✅ TTS API Server ready!")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 Shutting down TTS API Server...")


# API Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    vllm_status = "unknown"
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            health_url = VLLM_BASE_URL.rstrip('/').removesuffix('/v1') + '/health'
            response = await client.get(health_url)
            vllm_status = "healthy" if response.status_code == 200 else "unhealthy"
    except:
        vllm_status = "unreachable"
    
    return HealthResponse(
        status="healthy" if snac_codec is not None else "degraded",
        vllm_status=vllm_status,
        snac_loaded=snac_codec is not None,
        timestamp=time.time()
    )


@app.get("/v1/voices", response_model=List[VoiceInfo])
async def list_voices():
    """List available voices"""
    voices = [
        VoiceInfo(id="English (Male)", name="English (Male)", language="en", gender="male"),
        VoiceInfo(id="English (Female)", name="English (Female)", language="en", gender="female"),
        VoiceInfo(id="Hindi (Male)", name="Hindi (Male)", language="hi", gender="male"),
        VoiceInfo(id="Hindi (Female)", name="Hindi (Female)", language="hi", gender="female"),
    ]
    return voices


@app.post("/v1/text-to-speech")
async def text_to_speech(request: TTSRequest):
    """
    Convert text to speech with streaming
    
    Returns: PCM16 audio data at 24kHz sample rate
    """
    if snac_codec is None or vllm_transport is None:
        raise HTTPException(status_code=503, detail="TTS service not initialized")
    
    try:
        # Create async generator for streaming audio
        audio_stream = stream_tts_audio(request.text, request.voice)
        
        return StreamingResponse(
            audio_stream,
            media_type="audio/pcm",
            headers={
                "Content-Type": "audio/pcm",
                "X-Sample-Rate": str(SAMPLE_RATE),
                "X-Bit-Depth": str(BIT_DEPTH),
                "X-Channels": "1",
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")


# Core TTS Pipeline
async def stream_tts_audio(text: str, speaker_id: str) -> AsyncIterator[bytes]:
    """
    Stream audio generation using VLLM + Mapper + SNAC pipeline
    
    Flow:
    1. Format prompt for VLLM
    2. Stream text chunks from VLLM
    3. Extract custom token numbers from text
    4. Feed to mapper to get 28-code windows
    5. Decode windows to audio with SNAC
    6. Yield audio bytes
    """
    # Format prompt
    prompt = format_tts_prompt(text, speaker_id)
    logger.info(f"📝 Prompt length: {len(prompt)} chars")
    
    # Initialize mapper and audio buffer
    mapper = SvaraMapper()
    prebuffer_samples = int(SAMPLE_RATE * 0.5)  # 0.5 second prebuffer
    audio_buf = AudioBuffer(prebuffer_samples)
    
    # Use thread pool for concurrent SNAC decoding
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    loop = asyncio.get_running_loop()
    pending: List[asyncio.Task] = []
    
    def decode_window(window: List[int]) -> bytes:
        """Decode a window using SNAC codec"""
        return snac_codec.decode_window(window)
    
    async def submit_decode(window: List[int]) -> bytes:
        """Submit decode task to executor"""
        return await loop.run_in_executor(executor, decode_window, window)
    
    try:
        # Stream tokens from VLLM
        async for token_text in vllm_transport.astream(prompt):
            # Extract custom token numbers from text
            for token_num in extract_custom_token_numbers(token_text):
                # Feed to mapper
                window = mapper.feed_raw(token_num)
                
                if window is not None:
                    # Submit decode task
                    pending.append(asyncio.create_task(submit_decode(window)))
                    
                    # Yield when we have enough pending tasks
                    while len(pending) > 2:
                        audio_chunk = await pending.pop(0)
                        result = audio_buf.process(audio_chunk)
                        if result:
                            yield result
        
        # Flush remaining tasks
        for task in pending:
            audio_chunk = await task
            result = audio_buf.process(audio_chunk)
            if result:
                yield result
        
        logger.info("✅ Audio generation complete")
        
    except Exception as e:
        logger.error(f"Error in audio streaming: {e}", exc_info=True)
        raise
    finally:
        executor.shutdown(wait=True)


def format_tts_prompt(text: str, speaker_id: str) -> str:
    """
    Format text into VLLM prompt following Svara-TTS structure
    
    Format: <|begin_of_text|>
            <custom_token_3>
            <|audio|>
            {speaker_id}: {text}
            <custom_token_4>
            <|eot_id|>
            <custom_token_5>
            <custom_token_1>
    """
    from tts_engine.constants import (
        BOS_TOKEN_STR, AUDIO_TOKEN_STR, END_OF_TURN_STR,
        START_OF_HUMAN_STR, END_OF_HUMAN_STR,
        START_OF_AI_STR, START_OF_SPEECH_STR
    )
    
    # Build prompt following the exact structure
    prompt = f"{BOS_TOKEN_STR}"
    prompt += f"{START_OF_HUMAN_STR}"
    prompt += f"{AUDIO_TOKEN_STR}"
    prompt += f" {speaker_id}: {text}"
    prompt += f"{END_OF_HUMAN_STR}"
    prompt += f"{END_OF_TURN_STR}"
    prompt += f"{START_OF_AI_STR}"
    prompt += f"{START_OF_SPEECH_STR}"
    
    return prompt


# Development endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Svara TTS API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "voices": "/v1/voices",
            "tts": "/v1/text-to-speech"
        }
    }


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=API_PORT,
        log_level="info"
    )
