""" This module contains the dataclasses used to represent the different types of content in the application. """

from dataclasses import dataclass
from typing import Iterator


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

    """

    content: str
    start: float | None = None
    end: float | None = None


@dataclass
class Figure(RichContent):
    """
    Figure dataclass
    """

    pass


@dataclass
class Text:
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
    audio: bytes | Iterator[bytes] | None = None
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
