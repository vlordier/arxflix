""" This module contains the dataclasses used to represent the different types of content in the application. """

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel

from backend.config import VIDEO_FPS


class ScriptInput(BaseModel):
    """
    Input parameters for generating a script.

    Attributes:
        paper (str): The paper content or the path to the paper file.
        use_path (bool, optional): Whether to treat `paper` as a file path. Defaults to False.

    """

    paper: str
    use_path: bool = False


class AssetsInput(BaseModel):
    """
    Input parameters for generating assets.

    Attributes:
        script (str): The script content or the path to the script file.
        use_path (bool, optional): Whether to treat `script` as a file path. Defaults to False.
        mp3_output (str, optional): Path to save the MP3 output file. Defaults to "public/audio.wav".
        srt_output (str, optional): Path to save the SRT output file. Defaults to "public/output.srt".
        rich_output (str, optional): Path to save the rich content JSON file. Defaults to "public/output.json".
    """

    script: str
    use_path: bool = False
    mp3_output: str = "public/audio.wav"
    srt_output: str = "public/output.srt"
    rich_output: str = "public/output.json"


@dataclass
class Caption:
    """
    Caption dataclass

    Attributes:
        word: str
        start: float
        end: float

    """

    word: str
    start: float
    end: float


@dataclass
class RichContent:
    """
    RichContent dataclass

    Attributes:
        content: str
        start: float | None
        end: float | None
        audio: bytes | None
        captions: list[Caption] | None

    """

    content: str
    start: float | None = None
    end: float | None = None
    audio: bytes | None = None
    captions: list[Caption] | None = None


@dataclass
class Figure(RichContent):
    """
    Figure dataclass
    """

    pass


@dataclass
class Text(RichContent):
    """
    Text dataclass

    Attributes:
        content: str
        audio: bytes | Iterator[bytes] | None
        audio_path: str | None
        captions: list[Caption] | None
        start: float | None
        end: float | None

    """

    content: str
    audio: bytes | None = None
    audio_path: str | None = None
    captions: list[Caption] | None = None
    start: float | None = None
    end: float | None = None


@dataclass
class Equation(RichContent):
    """
    Equation dataclass

    """

    pass


@dataclass
class Headline(RichContent):
    """
    Headline dataclass


    """

    pass


@dataclass
class Paragraph(RichContent):  # dead: disable
    """
    Paragraph dataclass


    """

    pass


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
    subtitles_file_name: str = "frontend/public/output.srt"
    audio_file_name: str = "frontend/public/audio.wav"
    rich_content_file_name: str = "frontend/public/output.json"
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
        self.duration_in_frames = self.duration_in_seconds * VIDEO_FPS
