""" This module contains the MP3Exporter class, which is responsible for exporting the audio files to a single MP3 file. """

from pathlib import Path
from typing import List

import torch
import torchaudio

from backend.src.models import Text


class MP3Exporter:
    @staticmethod
    def export_mp3(text_contents: List[Text], output_path: Path) -> None:
        """Export the audio files to a single MP3 file.

        Args:
            text_contents (List[Text]): A list of Text objects containing the audio files.
            output_path (Path): The path to the output MP3 file.
        """
        audio_segments = []
        sample_rate = None
        for text in text_contents:
            if text.audio_path:
                audio, sample_rate = torchaudio.load(text.audio_path)
                audio_segments.append(audio)
        if sample_rate is None:
            raise ValueError("Sample rate not found in the audio files.")
        combined_audio = torch.cat([a.clone().detach() for a in audio_segments], dim=1)
        torchaudio.save(output_path, combined_audio, sample_rate)


def export_mp3(text_contents: List[Text], output_path: Path) -> None:
    MP3Exporter.export_mp3(text_contents, output_path)
