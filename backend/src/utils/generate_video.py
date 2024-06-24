""" Module for generating a video using Remotion. """

import json
import logging
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, Optional

from src.settings import Settings

settings = Settings()


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CompositionProps:
    """
    Dataclass for holding the properties of a video composition.

    Attributes:
        duration_in_seconds (int): The duration of the video in seconds.
        audio_offset_in_seconds (int): The offset of the audio in seconds.
        subtitles_file_name (str): The path to the subtitles file.
        audio_file_name (str): The path to the audio file.
        rich_content_file_name (str): The path to the rich content file.
        wave_color (str): The color of the wave.
        subtitles_line_per_page (int): The number of lines per page.
        subtitles_line_height (int): The height of each line.
        subtitles_zoom_measurer_size (int): The size of the zoom measurer.
        only_display_current_sentence (bool): Whether to only display the current sentence.
        mirror_wave (bool): Whether to mirror the wave.
        wave_lines_to_display (int): The number of lines to display.
        wave_freq_range_start_index (int): The start index of the frequency range.
        wave_number_of_samples (Literal["32", "64", "128", "256", "512"]): The number of samples.
        duration_in_frames (int): The duration of the video in frames.

    Methods:
        __post_init__(): Post-initialization to calculate the duration in frames.
    """

    duration_in_seconds: int = 5
    audio_offset_in_seconds: int = 0
    subtitles_file_name: str = "public/output.srt"
    audio_file_name: str = "public/audio.wav"
    rich_content_file_name: str = "public/output.json"
    wave_color: str = "#a3a5ae"
    subtitles_line_per_page: int = 2
    subtitles_line_height: int = 98
    subtitles_zoom_measurer_size: int = 10
    only_display_current_sentence: bool = True
    mirror_wave: bool = False
    wave_lines_to_display: int = 300
    wave_freq_range_start_index: int = 5
    wave_number_of_samples: Literal["32", "64", "128", "256", "512"] = "512"
    duration_in_frames: int = field(init=False)

    def __post_init__(self) -> None:
        """
        Post-initialization to calculate the duration in frames.
        """
        self.duration_in_frames = self.duration_in_seconds * settings.VIDEO.fps


def process_video(
    output_path: Optional[Path],
    composition_props: Optional[CompositionProps],
) -> None:
    """
    Processes the video by running the Remotion render command with the specified properties.

    Args:
        output_path (Path): The path to save the output video.
        composition_props (CompositionProps): The properties of the video composition.

    Raises:
        subprocess.CalledProcessError: If the video processing fails.
        Exception: If any other unexpected error occurs.

    """

    if composition_props is None:
        composition_props = CompositionProps()

    if output_path is None:
        output_path = Path("public/output.mp4")

    try:
        command = [
            "npx",
            "remotion",
            "render",
            Path(settings.REMOTION.root_path).absolute().as_posix(),
            "--props",
            json.dumps(asdict(composition_props)),
            "--compositionId",
            settings.REMOTION.composition_id,
            "--concurrency",
            str(settings.REMOTION.concurrency),
            "--output",
            output_path.absolute().as_posix(),
        ]
        logger.info("Running command: %s", " ".join(command))
        subprocess.run(command, check=True)
        logger.info("Video processing completed successfully.")
    except subprocess.CalledProcessError as e:
        logger.error("Video processing failed: %s", e)
        raise
