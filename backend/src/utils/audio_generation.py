""" Module for generating audio from text using the ElevenLabs API. """

from pathlib import Path

from elevenlabs import Voice, VoiceSettings, save
from elevenlabs.client import ElevenLabs

from backend.src.config import (
    ELEVENLABS_API_KEY,
    ELEVENLABS_MODEL,
    ELEVENLABS_SIMILARITY_BOOST,
    ELEVENLABS_STABILITY,
    ELEVENLABS_VOICE_ID,
)


class AudioGenerator:
    """
    Class for generating audio from text using the ElevenLabs API.

    Attributes:
        elevenlabs_client (ElevenLabs): An instance of the ElevenLabs client.
        voice (Voice): An instance of the ElevenLabs voice.


    Methods:
        generate_audio(text: str, output_path: Path) -> Path:
            Generate audio from text using the ElevenLabs API.

    """

    def __init__(self):
        self.elevenlabs_client = ElevenLabs(
            api_key=ELEVENLABS_API_KEY or "fake_api_key"
        )
        self.voice = Voice(
            voice_id=ELEVENLABS_VOICE_ID,
            settings=VoiceSettings(
                stability=ELEVENLABS_STABILITY,
                similarity_boost=ELEVENLABS_SIMILARITY_BOOST,
                style=0.0,
                use_speaker_boost=True,
            ),
        )

    def generate_audio(self, text: str, output_path: Path):
        """Generate audio from text using the ElevenLabs API.

        Args:
            text (str): The text to generate audio from.
            output_path (Path): The path to save the audio file.

        Returns:
            Path: The path to the saved audio file.
        """
        audio = self.elevenlabs_client.generate(
            text=text, voice=self.voice, model=ELEVENLABS_MODEL
        )
        save(audio, output_path)
        return output_path


def generate_audio_and_caption(
    text: str, audio_output_path: Path, caption_output_path: Path
):
    """
    Generate audio and caption files from text.

    Args:
        text (str): The text to generate audio and captions from.
        audio_output_path (Path): The path to save the audio file.
        caption_output_path (Path): The path to save the caption file.

    Returns:
        Tuple[Path, Path]: The paths to the saved audio and caption files.
    """
    audio_generator = AudioGenerator()
    audio_path = audio_generator.generate_audio(text, audio_output_path)

    # Assuming the caption generation logic is implemented elsewhere
    caption_text = text  # Placeholder for actual caption generation logic
    caption_output_path.write_text(caption_text)

    return audio_path, caption_output_path
