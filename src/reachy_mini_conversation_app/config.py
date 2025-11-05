import os
import logging
from pathlib import Path

from dotenv import load_dotenv


logger = logging.getLogger(__name__)

# Check if .env file exists
env_file = Path(".env")
if not env_file.exists():
    raise RuntimeError(
        ".env file not found. Please create one based on .env.example:\n"
        "  cp .env.example .env\n"
        "Then add your OPENAI_API_KEY to the .env file.",
    )

# Load .env and verify it was loaded successfully
if not load_dotenv():
    raise RuntimeError(
        "Failed to load .env file. Please ensure the file is readable and properly formatted.",
    )

logger.info("Configuration loaded from .env file")


class Config:
    """Configuration class for the conversation app."""

    # ===== Provider Selection =====
    PROVIDER = os.getenv("PROVIDER", "openai")  # openai|anthropic|elevenlabs|local
    
    # ===== OpenAI =====
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    # Fix this line - check the string value, not the variable:
    if OPENAI_API_KEY is None and PROVIDER == "openai":  # ✅ This is correct
        raise RuntimeError(
            "OPENAI_API_KEY is not set in .env file. Please add it:\n"
            "  OPENAI_API_KEY=your_api_key_here",
        )
    if OPENAI_API_KEY and not OPENAI_API_KEY.strip():
        raise RuntimeError(
            "OPENAI_API_KEY is empty in .env file. Please provide a valid API key.",
        )
    
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-realtime")
    
    # ===== Anthropic =====
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4.5-20250929")
    
    # ===== ElevenLabs =====
    ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
    ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
    
    # ===== Local Models =====
    HF_HOME = os.getenv("HF_HOME", "./cache")
    LOCAL_VISION_MODEL = os.getenv("LOCAL_VISION_MODEL", "HuggingFaceTB/SmolVLM2-2.2B-Instruct")
    LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
    LOCAL_TTS_MODEL = os.getenv("LOCAL_TTS_MODEL", "parler-tts/parler-tts-mini-v1")
    HF_TOKEN = os.getenv("HF_TOKEN")  # Optional
    
    logger.debug(f"Provider: {PROVIDER}, Model: {MODEL_NAME}")


config = Config()
