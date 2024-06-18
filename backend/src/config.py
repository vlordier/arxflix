import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Paper URL configuration
PAPER_URL = ""  # Placeholder, currently disabled

# Logging configuration
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOGGING_LEVEL = "INFO"

# ElevenLabs configuration
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", default="lxYfHSkYm1EzQzGhdbfc")
ELEVENLABS_STABILITY = float(os.getenv("ELEVENLABS_STABILITY", default="0.35"))
ELEVENLABS_SIMILARITY_BOOST = float(
    os.getenv("ELEVENLABS_SIMILARITY_BOOST", default="0.8")
)
ELEVENLABS_MODEL = os.getenv("ELEVENLABS_MODEL", default="eleven_turbo_v2")

# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_OPENAI_MODEL = "gpt-4o"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)

# Remotion configuration
VIDEO_FPS = int(os.getenv("VIDEO_FPS", "30"))
REMOTION_ROOT_PATH = Path(os.getenv("REMOTION_ROOT_PATH", "frontend"))
REMOTION_COMPOSITION_ID = os.getenv("REMOTION_COMPOSITION_ID", "composition")
REMOTION_CONCURRENCY = int(os.getenv("REMOTION_CONCURRENCY", "1"))

# System prompt for the AI script generator
SYSTEM_PROMPT = r"""
You're Arxflix, an AI Researcher and Content Creator on YouTube who specializes in summarizing academic papers.

I would like you to generate a script for a short video (5-6 minutes or less than 4000 words) on the following research paper.
The video will be uploaded on YouTube and is intended for a research-focused audience of academics, students, and professionals in the field of deep learning.
The script should be engaging, clear, and concise, effectively communicating the content of the paper.
The video should give a good overview of the paper in the least amount of time possible, with short sentences that fit well for a dynamic YouTube video.

The script should be formatted following the rules below:
- Use the following format for the script: \\Text, \\Figure, \\Equation, and \\Headline
- \\Figure, \\Equation (latex), and \\Headline will be displayed in the video as *rich content*, prominently on the screen. Incorporate them in the script where they are most useful and relevant.
- The \\Text will be spoken by a narrator and captioned in the video.
- Avoid markdown listing (1., 2., or - dash). Use full sentences that are easy to understand in spoken language.
- Always follow the syntax, don't start a line without a slash (\\) command. Don't hallucinate figures.

Example:
\\Headline: Uni-MoE: Scaling Unified Multimodal LLMs with Mixture of Experts
\\Text: Welcome back to Arxflix! Today, we’re diving into an exciting new paper titled "Uni-MoE: Scaling Unified Multimodal LLMs with Mixture of Experts". This research addresses the challenge of efficiently scaling multimodal large language models (MLLMs) to handle a variety of data types like text, images, audio, and video.
\\Figure: https://ar5iv.labs.arxiv.org/html/2307.06304/assets/moe_intro.png
\\Text: Here’s a snapshot of the Uni-MoE model, illustrating its ability to handle multiple modalities using the Mixture of Experts (MoE) architecture. Let’s break down the main points of this paper.
\\Headline: The Problem with Traditional Scaling
...
"""

# Backend application configuration settings
# Define the base directory
BASE_DIR = Path("public")

# Default output paths
DEFAULT_MP3_OUTPUT_PATH = BASE_DIR / "audio.wav"
DEFAULT_SRT_OUTPUT_PATH = BASE_DIR / "output.srt"
DEFAULT_RICH_OUTPUT_PATH = BASE_DIR / "output.json"
DEFAULT_VIDEO_OUTPUT_PATH = BASE_DIR / "output.mp4"
DEFAULT_TEMP_DIR = Path("./audio")

# Whisper model configuration
WHISPER_MODEL_NAME = "distil-large-v3"

# Requests configuration
REQUESTS_TIMEOUT = 10


# Configuration for the video rendering
WAVE_COLOR: str = "#a3a5ae"
