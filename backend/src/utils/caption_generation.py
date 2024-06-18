""" This module contains the CaptionGenerator class, which is responsible for generating captions from audio files. """

from pathlib import Path
from typing import List

import whisper

from backend.src.models import Caption


class CaptionGenerator:
    """This class is responsible for generating captions from the audio file."""

    def __init__(self, model_name: str) -> None:
        """Initialize the caption generator.

        Args:
            model_name (str): The name of the model to use for caption generation.
        """
        self.model = whisper.load_model(model_name)

    def generate_captions(self, audio_path: Path) -> List[Caption]:
        """Generate captions from the audio file.

        Args:
            audio_path (str): The path to the audio file.

        Returns:
            List[Caption]: A list of Caption objects.
        """

        result = self.model.transcribe(audio_path, word_timestamps=True)
        captions = []
        for segment in result["segments"]:
            for word in segment["words"]:
                word_text = word["word"].lstrip()
                caption = Caption(word=word_text, start=word["start"], end=word["end"])
                captions.append(caption)
        return captions
