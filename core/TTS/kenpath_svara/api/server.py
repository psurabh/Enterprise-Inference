"""
FastAPI Server for Svara TTS
Integrates VLLM token generation with SNAC audio decoding
"""
import logging
import asyncio
import os
import time
from typing import Any, Dict, Literal, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import httpx

from tts_engine.codec import SNACCodec
from tts_engine.voice_config import get_voice, get_all_voices, get_speaker_id
from tts_engine.mapper import SvaraMapper
from tts_engine.constants import (
    BOS_TOKEN_STR,
    START_OF_HUMAN_STR,
    END_OF_HUMAN_STR,
    START_OF_AI_STR,
    END_OF_AI_STR,
    START_OF_SPEECH_STR,
    AUDIO_TOKEN_STR,
    END_OF_TURN_STR
)

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
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:2080")
VLLM_MODEL = os.getenv("VLLM_MODEL", "kenpath/svara-tts-v1")

# ---------------------------------------------------------------------------
# OpenAI voice alias → Svara voice ID mapping
# Allows drop-in replacement for clients built against the OpenAI TTS API.
# ---------------------------------------------------------------------------
OPENAI_VOICE_MAP: Dict[str, str] = {
    "alloy":   "en_male",
    "echo":    "en_male",
    "fable":   "en_female",
    "onyx":    "hi_male",
    "nova":    "en_female",
    "shimmer": "hi_female",
}


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
        timeout=httpx.Timeout(600.0)
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
    description=(
        "OpenAI-compatible Text-to-Speech API powered by VLLM + SNAC.\n\n"
        "Drop-in replacement for `POST /v1/audio/speech` and `GET /v1/models`.\n"
        "Native Svara endpoints (`/v1/voices`, `/v1/text-to-speech`) are also available."
    ),
    version="1.0.0",
    lifespan=lifespan,
    openapi_tags=[
        {"name": "Audio",   "description": "OpenAI-compatible speech synthesis"},
        {"name": "Models",  "description": "Available TTS models"},
        {"name": "Voices",  "description": "Svara native voice management"},
        {"name": "System",  "description": "Health and service metadata"},
    ],
)


# Request/Response Models
class TTSRequest(BaseModel):
    text: str = Field(..., description="Text to convert to speech", min_length=1, max_length=1000)
    voice_id: str = Field(default="en_male", description="Voice ID to use", alias="voice")
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


# ---------------------------------------------------------------------------
# OpenAI-compatible models
# Reference: https://platform.openai.com/docs/api-reference/audio/createSpeech
# ---------------------------------------------------------------------------

OPENAI_VOICES = Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
OPENAI_FORMATS = Literal["wav", "mp3", "opus", "aac", "flac", "pcm"]
OPENAI_MODELS = Literal["tts-1", "tts-1-hd"]


class OpenAITTSRequest(BaseModel):
    model: OPENAI_MODELS = Field(default="tts-1", description="TTS model ID")
    input: str = Field(..., min_length=1, max_length=4096, description="Text to synthesize")
    voice: OPENAI_VOICES = Field(default="alloy", description="OpenAI voice alias")
    response_format: OPENAI_FORMATS = Field(default="wav", description="Output audio format")
    speed: float = Field(default=1.0, ge=0.25, le=4.0, description="Speech speed multiplier")


class OpenAIModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str


class OpenAIModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[OpenAIModelCard]


# Endpoints
@app.get("/health", response_model=HealthResponse, tags=["System"])
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


@app.get("/v1/voices", response_model=list[VoiceInfo], tags=["Voices"])
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


@app.get("/v1/voices/{voice_id}", response_model=VoiceInfo, tags=["Voices"])
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


async def generate_text_from_vllm(prompt: str, max_tokens: int = 2048) -> str:
    """
    Call VLLM to generate raw text containing tokens from prompt
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
        
        return generated_text
        
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
    # Construct speaker ID (e.g., "Hindi (Male)")
    try:
        speaker_id = get_speaker_id(voice.voice_id)
    except Exception as e:
        logger.warning(f"Could not get speaker ID for {voice.voice_id}, falling back to name: {e}")
        speaker_id = voice.name
    
    # Construct prompt using the specific token sequence expected by the model
    # Format: BOS + <human_turn> + <ai_start>
    # <human_turn>: START_HUMAN + AUDIO_TOKEN + "Speaker: Text" + END_HUMAN + END_TURN
    # <ai_start>: START_AI + START_SPEECH
    
    prompt = (
        f"{BOS_TOKEN_STR}"
        f"{START_OF_HUMAN_STR}{AUDIO_TOKEN_STR}"
        f"{speaker_id}: {text}"
        f"{END_OF_HUMAN_STR}{END_OF_TURN_STR}"
        f"{START_OF_AI_STR}{START_OF_SPEECH_STR}"
    )
    
    return prompt


async def _synthesize(text: str, voice_id: str, fmt: str, speed: float) -> Response:
    """
    Shared synthesis coroutine used by both the native and OpenAI-compatible endpoints.

    Args:
        text:     Input text to synthesize.
        voice_id: Svara voice ID (e.g. "en_male", "hi_female").
        fmt:      Output format — "wav" or "pcm".  Other formats return 501.
        speed:    Speed multiplier (reserved; pipeline does not alter pitch/speed yet).

    Returns:
        FastAPI ``Response`` containing the encoded audio bytes.

    Raises:
        HTTPException 404  – voice not found
        HTTPException 501  – unsupported audio format
        HTTPException 502  – upstream VLLM error
        HTTPException 503  – SNAC codec not initialised
        HTTPException 500  – unexpected synthesis failure
    """
    voice = get_voice(voice_id)
    if not voice:
        raise HTTPException(status_code=404, detail=f"Voice '{voice_id}' not found")

    if not snac_codec:
        raise HTTPException(status_code=503, detail="SNAC codec not available")

    if fmt not in ("wav", "pcm"):
        raise HTTPException(
            status_code=501,
            detail=(
                f"Audio format '{fmt}' is not yet supported. "
                "Supported formats: wav, pcm."
            ),
        )

    logger.info("Synthesis request | voice=%s format=%s text_len=%d", voice_id, fmt, len(text))

    try:
        prompt = format_tts_prompt(text, voice)
        generated_text = await generate_text_from_vllm(prompt)

        if not generated_text:
            raise HTTPException(
                status_code=500,
                detail="VLLM returned an empty response",
            )

        logger.info("VLLM response received | chars=%d", len(generated_text))

        mapper = SvaraMapper()
        token_windows = mapper.feed_text(generated_text)

        if not token_windows:
            raise HTTPException(
                status_code=500,
                detail="No audio tokens found in the generated output",
            )

        logger.info("Audio token windows parsed | windows=%d", len(token_windows))

        audio_chunks: list[bytes] = []
        for window in token_windows:
            audio_bytes = snac_codec.decode_window(window)
            if audio_bytes:
                audio_chunks.append(audio_bytes)

        if not audio_chunks:
            raise HTTPException(status_code=500, detail="SNAC decode produced no audio output")

        pcm_audio = b"".join(audio_chunks)

        if fmt == "wav":
            audio_data = create_wav_header(len(pcm_audio)) + pcm_audio
            media_type = "audio/wav"
            filename = "speech.wav"
        else:  # pcm
            audio_data = pcm_audio
            media_type = "audio/pcm"
            filename = "speech.pcm"

        return Response(
            content=audio_data,
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Synthesis pipeline failure | voice=%s error=%s", voice_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Native Svara endpoint  (backwards-compatible)
# ---------------------------------------------------------------------------

@app.post(
    "/v1/text-to-speech",
    tags=["Audio"],
    summary="Synthesize speech (Svara native)",
    response_description="Audio file in the requested format",
)
async def text_to_speech(request: TTSRequest):
    """
    Convert text to speech using a Svara native voice ID.

    Kept for backwards compatibility. New integrations should prefer
    the OpenAI-compatible ``POST /v1/audio/speech`` endpoint.
    """
    return await _synthesize(
        text=request.text,
        voice_id=request.voice_id,
        fmt=request.format,
        speed=request.speed,
    )


# ---------------------------------------------------------------------------
# OpenAI-compatible endpoint
# Reference: https://platform.openai.com/docs/api-reference/audio/createSpeech
# ---------------------------------------------------------------------------

@app.post(
    "/v1/audio/speech",
    tags=["Audio"],
    summary="Synthesize speech (OpenAI-compatible)",
    response_description="Audio file in the requested format",
    responses={
        200: {"content": {"audio/wav": {}, "audio/pcm": {}}, "description": "Synthesized audio"},
        404: {"description": "Voice not found"},
        501: {"description": "Audio format not supported"},
        502: {"description": "VLLM upstream error"},
        503: {"description": "SNAC codec not available"},
    },
)
async def openai_text_to_speech(request: OpenAITTSRequest):
    """
    OpenAI-compatible speech synthesis endpoint.

    Accepts the same request body as the OpenAI ``POST /v1/audio/speech`` API,
    mapping OpenAI voice aliases to the corresponding Svara voices:

    | OpenAI voice | Svara voice  |
    |--------------|--------------|
    | alloy        | en\_male     |
    | echo         | en\_male     |
    | fable        | en\_female   |
    | onyx         | hi\_male     |
    | nova         | en\_female   |
    | shimmer      | hi\_female   |
    """
    svara_voice_id = OPENAI_VOICE_MAP[request.voice]
    logger.info(
        "OpenAI TTS request | model=%s openai_voice=%s svara_voice=%s",
        request.model, request.voice, svara_voice_id,
    )
    return await _synthesize(
        text=request.input,
        voice_id=svara_voice_id,
        fmt=request.response_format,
        speed=request.speed,
    )


# ---------------------------------------------------------------------------
# OpenAI-compatible models listing
# Reference: https://platform.openai.com/docs/api-reference/models/list
# ---------------------------------------------------------------------------

_SVARA_MODELS: list[OpenAIModelCard] = [
    OpenAIModelCard(id="tts-1",    created=1677610602, owned_by="svara"),
    OpenAIModelCard(id="tts-1-hd", created=1677610602, owned_by="svara"),
]


@app.get(
    "/v1/models",
    response_model=OpenAIModelList,
    tags=["Models"],
    summary="List available TTS models",
)
async def list_models() -> OpenAIModelList:
    """Return the list of available TTS models in OpenAI format."""
    return OpenAIModelList(data=_SVARA_MODELS)


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


@app.get("/", tags=["System"])
async def root():
    """Service discovery — returns endpoint map and compatibility information."""
    return {
        "service": "Svara TTS API",
        "version": "1.0.0",
        "openai_compatible": True,
        "endpoints": {
            # OpenAI-compatible
            "speech":          "POST /v1/audio/speech",
            "models":          "GET  /v1/models",
            # Svara native
            "voices":          "GET  /v1/voices",
            "voice_detail":    "GET  /v1/voices/{voice_id}",
            "tts_native":      "POST /v1/text-to-speech",
            # System
            "health":          "GET  /health",
            "documentation":   "GET  /docs",
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
