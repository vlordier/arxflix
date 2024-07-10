""" This module contains the Pydantic models for the FastAPI application. """

from dataclasses import dataclass, field
from typing import Iterator, Literal

import tiktoken
from pydantic import BaseModel, Field, validator
from settings import settings


class ArxivPaper(BaseModel):
    """
    arxiv_id: str
    markdown: str
    path: Path
    """

    arxiv_id: str = Field(default=None)
    markdown: str = Field(default=None)
    path: str = Field(default=None)


class Prompt(BaseModel):
    """Prompt model
    Attributes:
        system_prompt: str
        user_prompt: str
    """

    system_prompt: str
    user_prompt: str

    @validator("system_prompt", "user_prompt")
    def check_token_limit(cls: type, v: str, values: dict) -> str:
        """Validates that the token count is below the maximum allowed tokens."""
        encoder = tiktoken.get_encoding("cl100k_base")  # Choose appropriate encoding

        # Tokenize the prompts
        system_tokens = (
            encoder.encode(values["system_prompt"]) if "system_prompt" in values else []
        )
        user_tokens = encoder.encode(v)

        total_tokens = len(system_tokens) + len(user_tokens)

        if total_tokens > settings.OPENAI.max_tokens:
            raise ValueError(
                f"Total token count exceeds the maximum limit of {settings.OPENAI.max_tokens}. "
                f"Found {total_tokens} tokens."
            )

        return v


class ScriptInput(BaseModel):
    """
    Input parameters for generating a script.

    Attributes:
        paper (str): The paper content or the path to the paper file.
        use_path (bool, optional): Whether to treat `paper` as a file path. Defaults to False.

    """

    paper: str
    use_path: bool = Field(default=False, json_schema_extra={"env": "USE_PATH"})


class AssetsInput(BaseModel):
    """
    Input parameters for generating assets.

    Attributes:
        script (str): The script content or the path to the script file.
        use_path (bool, optional): Whether to treat `script` as a file path. Defaults to False.
        mp3_output (str, optional): Path to save the MP3 output file. Defaults to "audio.mp3".
        srt_output (str, optional): Path to save the SRT output file. Defaults to "output.srt".
        rich_output (str, optional): Path to save the rich content JSON file. Defaults to "output.json".
    """

    script: str
    use_path: bool = Field(default=False, json_schema_extra={"env": "USE_PATH"})
    mp3_output: str = Field(
        default="audio.mp3", json_schema_extra={"env": "MP3_OUTPUT"}
    )
    srt_output: str = Field(
        default="output.srt", json_schema_extra={"env": "SRT_OUTPUT"}
    )
    rich_output: str = Field(
        default="output.json", json_schema_extra={"env": "RICH_OUTPUT"}
    )


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

    # check that the start time is less than the end time
    def __post_init__(self) -> None:
        if self.start >= self.end:
            raise ValueError("Start time must be less than end time")


@dataclass
class RichContent:
    """
    RichContent dataclass

    Attributes:
        identifier: str
        content: str
        start: float | None
        end: float | None

    """

    identifier: str = r""
    content: str = ""
    start: float | None = None
    end: float | None = None

    # check that the start time is less than the end time
    def __post_init__(self) -> None:
        if self.start is not None and self.end is not None and self.start > self.end:
            raise ValueError("Start time must be less than end time")


@dataclass
class Figure(RichContent):
    """
    Figure dataclass

    Attributes:
        identifier: str
    """

    identifier: str = r"\Figure:"


@dataclass
class Text:
    """
    Text dataclass

    Attributes:
        identifier: str
        content: str
        audio: bytes | Iterator[bytes] | None
        audio_path: str | None
        captions: list[Caption] | None
        start: float | None
        end: float | None

    """

    identifier: str = r"\Text:"
    content: str = ""
    audio: bytes | Iterator[bytes] | None = None
    audio_path: str | None = None
    captions: list[Caption] | None = None
    start: float | None = None
    end: float | None = None


@dataclass
class Equation(RichContent):
    """
    Equation dataclass

    Attributes:
        identifier: str

    """

    identifier: str = r"\Equation:"


@dataclass
class Headline(RichContent):
    """
    Headline dataclass

    Attributes:
        identifier: str

    """

    identifier: str = r"\Headline:"


@dataclass
class Paragraph(RichContent):  # dead: disable
    """
    Paragraph dataclass

    Attributes:
        identifier: str

    """

    identifier: str = r"\Paragraph"


@dataclass
class CompositionProps:
    """
    Dataclass for holding the properties of a video composition.

    Attributes:
        duration_in_seconds (int): The duration of the video in seconds.
        duration_in_frames (int): The duration of the video in frames.
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
    """

    duration_in_seconds: int = 5
    audio_offset_in_seconds: int = 0
    subtitles_file_name: str = "output.srt"
    audio_file_name: str = "audio.wav"
    rich_content_file_name: str = "output.json"
    wave_color: str = "#003366"
    subtitles_line_per_page: int = 2
    subtitles_line_height: int = 98
    subtitles_zoom_measurer_size: int = 10
    only_display_current_sentence: bool = True
    mirror_wave: bool = False
    wave_lines_to_display: int = 300
    wave_freq_range_start_index: int = 5
    wave_number_of_samples: Literal["32", "64", "128", "256", "512"] = "256"
    duration_in_frames: int = field(init=False)

    def __post_init__(self) -> None:
        """Post-initialization to calculate the duration in frames."""
        self.duration_in_frames = self.duration_in_seconds * settings.VIDEO.fps
