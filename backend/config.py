from dotenv import load_dotenv

import os

load_dotenv()

ELEVENLABS_API_KEY = str(os.getenv('ELEVENLABS_API_KEY'))