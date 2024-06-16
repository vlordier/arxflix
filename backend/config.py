import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env file
load_dotenv()

# ElevenLabs configuration
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", default="lxYfHSkYm1EzQzGhdbfc")
ELEVENLABS_STABILITY = float(os.getenv("ELEVENLABS_STABILITY", default=0.35))
ELEVENLABS_SIMILARITY_BOOST = float(
    os.getenv("ELEVENLABS_SIMILARITY_BOOST", default=0.8)
)
ELEVENLABS_MODEL = os.getenv("ELEVENLABS_MODEL", default="eleven_turbo_v2")

# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# Remotion configuration
VIDEO_FPS = int(os.getenv("VIDEO_FPS", 30))
VIDEO_HEIGHT = int(os.getenv("VIDEO_HEIGHT", 1080))
VIDEO_WIDTH = int(os.getenv("VIDEO_WIDTH", 1920))
REMOTION_ROOT_PATH = Path(os.getenv("REMOTION_ROOT_PATH", "frontend"))
REMOTION_COMPOSITION_ID = os.getenv("REMOTION_COMPOSITION_ID", "composition")
REMOTION_CONCURRENCY = int(os.getenv("REMOTION_CONCURRENCY", 1))
