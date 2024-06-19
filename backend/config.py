import os

from dotenv import load_dotenv

load_dotenv()

ELEVENLABS_API_KEY = str(os.getenv("ELEVENLABS_API_KEY"))
