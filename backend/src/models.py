""" This module contains the Pydantic models for the FastAPI application. """

from dataclasses import dataclass
from typing import Iterator

import tiktoken
from pydantic import BaseModel, Field, validator
from src.settings import settings


class Prompt(BaseModel):
    """Prompt model
    Attributes:
        system_prompt: str
        user_prompt: str
    """

    system_prompt: str
    user_prompt: str

    @validator("system_prompt", "user_prompt")
    def check_token_limit(cls, v, values):
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
        mp3_output (str, optional): Path to save the MP3 output file. Defaults to "public/audio.wav".
        srt_output (str, optional): Path to save the SRT output file. Defaults to "public/output.srt".
        rich_output (str, optional): Path to save the rich content JSON file. Defaults to "public/output.json".
    """

    script: str
    use_path: bool = Field(default=False, json_schema_extra={"env": "USE_PATH"})
    mp3_output: str = Field(
        default="public/audio.wav", json_schema_extra={"env": "MP3_OUTPUT"}
    )
    srt_output: str = Field(
        default="public/output.srt", json_schema_extra={"env": "SRT_OUTPUT"}
    )
    rich_output: str = Field(
        default="public/output.json", json_schema_extra={"env": "RICH_OUTPUT"}
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
