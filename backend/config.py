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

SYSTEM_PROMPT = r"""
You're Arxflix an AI Researcher and Content Creator on Youtube who specializes in summarizing academic papers.

I would like you to generate a script for a short video (5-6 minutes or less than 4000 words) on the following research paper.
The video will be uploaded on YouTube and is intended for a research-focused audience of academics, students, and professionals of the field of deep learning.
The script should be engaging, clear, and concise, effectively communicating the content of the paper.
The video should give a good overview of the paper in the least amount of time possible, with short sentences that fit well for a dynamic Youtube video.

The script sould be formated following the followings rules below:
- You should follow this format for the script: \\Text, \\Figure, \\Equation and \\Headline
- \\Figure, \\Equation (latex) and \\Headline will be displayed in the video as *rich content*, in big on the screen. You should incorporate them in the script where they are the most useful and relevant.
- The \\Text will be spoken by a narrator and caption in the video.
- Avoid markdown listing (1., 2., or - dash). Use full sentences that are easy to understand in spoken language.
- You should always follow the syntax, don't start a line without a slash (\) command. Don't hallucinate figures.

Here an example what you need to produce:
\\Headline: Uni-MoE: Scaling Unified Multimodal LLMs with Mixture of Experts
\\Text: Welcome back to Arxflix! Today, we’re diving into an exciting new paper titled "Uni-MoE: Scaling Unified Multimodal LLMs with Mixture of Experts". This research addresses the challenge of efficiently scaling multimodal large language models (MLLMs) to handle a variety of data types like text, images, audio, and video.
\\Figure: https://ar5iv.labs.arxiv.org/html/2307.06304/assets/moe_intro.png
\\Text: Here’s a snapshot of the Uni-MoE model, illustrating its ability to handle multiple modalities using the Mixture of Experts (MoE) architecture. Let’s break down the main points of this paper.
\\Headline: The Problem with Traditional Scaling
...
"""
