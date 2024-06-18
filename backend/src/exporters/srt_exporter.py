""" This module contains the SRTExporter class, which is responsible for exporting captions to an SRT file. """

from datetime import timedelta
from pathlib import Path
from typing import List

from backend.src.models import Text

import srt

from backend.src.models import Caption


class SRTExporter:
    @staticmethod
    def export(captions: List[Caption], output_path: Path) -> None:
        """Export captions to an SRT file.

        Args:
            captions (List[Caption]): A list of Caption objects to export.
            output_path (Path): The path to the output SRT file.
        """
        subtitles = [
            srt.Subtitle(
                index=i,
                start=timedelta(seconds=caption.start),
                end=timedelta(seconds=caption.end),
                content=caption.word,
            )
            for i, caption in enumerate(captions)
        ]
        srt_text = srt.compose(subtitles)
        with open(output_path, "w", encoding="utf-8") as file:
            file.write(srt_text)


def export_srt(text_contents: List[Text], output_path: Path) -> None:
    """Export the text contents to an SRT file.

    Args:
        text_contents (List[Text]): A list of Text objects containing the captions.
        output_path (Path): The path to the output SRT file.
    """
    with open(output_path, "w", encoding="utf-8") as file:
        for index, text in enumerate(text_contents):
            start_time = text.start
            end_time = text.end
            content = text.content

            start_time_str = format_time(start_time)
            end_time_str = format_time(end_time)

            file.write(f"{index + 1}\n")
            file.write(f"{start_time_str} --> {end_time_str}\n")
            file.write(f"{content}\n\n")


def format_time(seconds: float) -> str:
    """Format time in seconds to SRT time format.

    Args:
        seconds (float): Time in seconds.

    Returns:
        str: Time in SRT format (HH:MM:SS,MS).
    """
    milliseconds = int((seconds % 1) * 1000)
    seconds = int(seconds)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"
