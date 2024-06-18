""" Utility functions to generate audio, caption, and other assets. """

import json
import logging
import os
import tempfile
from datetime import timedelta
from pathlib import Path
from typing import List, Optional, Union

import srt
import torch
import torchaudio
import whisper
from dotenv import load_dotenv
from elevenlabs import Voice, VoiceSettings, save
from elevenlabs.client import ElevenLabs

load_dotenv()

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# Load the environment variables
from backend.config import (
    ELEVENLABS_MODEL,
    ELEVENLABS_SIMILARITY_BOOST,
    ELEVENLABS_STABILITY,
    ELEVENLABS_VOICE_ID,
    WHISPER_MODEL_NAME,
)
from backend.models import Caption, Equation, Figure, Headline, RichContent, Text

# Setup logging
logger = logging.getLogger(__name__)

ELEVENLABS_VOICE = Voice(
    voice_id=ELEVENLABS_VOICE_ID,
    settings=VoiceSettings(
        stability=ELEVENLABS_STABILITY,
        similarity_boost=ELEVENLABS_SIMILARITY_BOOST,
        style=0.0,
        use_speaker_boost=True,
    ),
)


def parse_script(
    script: str,
) -> List[Union[RichContent, Text, Figure, Equation, Headline]]:
    """Parse the script and return a list of RichContent, Text, Figure, Equation, or Headline objects.

    Parameters
    ----------
    script : str
        The script to parse as a string

    Returns
    -------
    List[Union[RichContent, Text, Figure, Equation, Headline]]
        List of RichContent, Text, Figure, Equation, or Headline objects
    """
    lines = script.split("\n")
    content: List[Union[RichContent, Text, Figure, Equation, Headline]] = []

    # Mapping prefixes to their respective classes and any required additional arguments
    content_mapping = {
        "\\Figure:": Figure,
        "\\Text:": Text,
        "\\Equation:": Equation,
        "\\Headline:": Headline,
    }

    for line in lines:
        if line.strip() == "":
            continue
        for prefix, cls in content_mapping.items():
            if line.startswith(prefix):
                content_value = line[
                    len(prefix) :
                ].strip()  # Strip any leading/trailing whitespace
                if cls in [
                    Figure,
                    Equation,
                ]:  # Add specific arguments for Figure and Equation
                    content.append(
                        cls(
                            content=content_value,
                            start=0.0,
                            end=0.5,
                            audio=None,
                            captions=None,
                        )
                    )
                else:
                    content.append(cls(content=content_value))
                break
        else:
            logger.warning(f"Unknown line: {line}")

    return content


def make_caption(result: dict) -> List[Caption]:
    """
    Create a list of Caption objects from the result of the Whisper model.

    Args:
        result (dict): Result dictionary from the Whisper model.

    Returns:
        List[Caption]: List of Caption objects.
    """
    captions = []
    for segment in result["segments"]:
        for word in segment["words"]:
            word_text = word["word"].lstrip()
            caption = Caption(word=word_text, start=word["start"], end=word["end"])
            captions.append(caption)
    return captions


def generate_audio_and_caption(
    script_contents: List[Union[RichContent, Text]],
    temp_dir: Optional[Path],
) -> List[Union[RichContent, Text]]:
    """
    Generate audio and caption for each text segment in the script.

    Args:
        script_contents (List[Union[RichContent, Text]]): List of RichContent or Text objects.
        temp_dir (Path, optional): Temporary directory to store the audio files. Defaults to Path(tempfile.gettempdir()).

    Returns:
        List[Union[RichContent, Text]]: List of RichContent or Text objects with audio and caption.
    """
    if temp_dir is None:
        temp_dir = Path(tempfile.gettempdir())

    elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY or "fake_api_key")
    if not temp_dir.exists():
        temp_dir.mkdir(parents=True)

    for index, content in enumerate(script_contents):
        if isinstance(content, Text) and content.audio is None:
            audio_path = (temp_dir / f"audio_{index}.wav").absolute().as_posix()
            if not os.path.exists(audio_path):
                logger.info("Generating audio %d at %s", index, audio_path)
                content.audio = elevenlabs_client.generate(
                    text=content.content,
                    voice=ELEVENLABS_VOICE,
                    model=ELEVENLABS_MODEL,
                )
                save(content.audio, audio_path)
            audio, sample_rate = torchaudio.load(audio_path)
            model = whisper.load_model("base.en")
            result = model.transcribe(audio_path, word_timestamps=True)
            content.captions = make_caption(result)
            content.audio_path = audio_path
            content.end = audio.shape[1] / sample_rate

    offset = 0.0
    for content in script_contents:
        if isinstance(content, Text) and content.captions:
            for caption in content.captions:
                caption.start += offset
                caption.end += offset
            content.start = offset
            content.end = content.captions[-1].end
            offset = content.end
    return script_contents


def fill_rich_content_time(
    script_contents: List[Union[RichContent, Text]]
) -> List[Union[RichContent, Text]]:
    """
    Fill the time for each rich content based on the text duration.

    Args:
        script_contents (List[Union[RichContent, Text]]): List of RichContent or Text objects.

    Returns:
        List[Union[RichContent, Text]]: List of RichContent or Text objects with time assigned.
    """
    index = 0
    while index < len(script_contents):
        current_rich_content_group = []
        while index < len(script_contents) and not isinstance(
            script_contents[index], Text
        ):
            current_rich_content_group.append(script_contents[index])
            index += 1

        if index >= len(script_contents):
            break

        next_text_group = []
        while index < len(script_contents) and isinstance(script_contents[index], Text):
            next_text_group.append(script_contents[index])
            index += 1

        if not next_text_group:
            break

        if current_rich_content_group and next_text_group:
            total_duration = (next_text_group[-1].end or 0.0) - (
                next_text_group[0].start or 0.0
            )
            duration_per_rich_content = total_duration / (
                len(current_rich_content_group) + 1
            )
            offset = next_text_group[0].start or 0.0
            for _i, rich_content in enumerate(current_rich_content_group):
                rich_content.start = offset
                rich_content.end = offset + duration_per_rich_content
                offset += duration_per_rich_content

    return script_contents


def export_mp3(text_contents: List[Text], output_path: Path) -> None:
    """
    Export the audio of the text contents to a single MP3 file.

    Args:
        text_contents (List[Text]): List of Text objects.
        output_path (Path): Path to save the MP3 file.

    Raises:
        ValueError: Sample rate not found in the audio files.
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


def export_srt(full_audio_path: Path, output_path: Path) -> None:
    """
    Export the SRT file for the full audio using the Whisper model.

    Args:
        full_audio_path (Path): Path to the full audio file.
        output_path (Path): Path to save the SRT file.
    """
    model = whisper.load_model(WHISPER_MODEL_NAME)
    result = model.transcribe(full_audio_path, word_timestamps=True)
    captions = make_caption(result)
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


def export_rich_content_json(
    rich_contents: List[RichContent], output_path: Path
) -> None:
    """
    Export the rich content to a JSON file.

    Args:
        rich_contents (List[RichContent]): List of RichContent objects.
        output_path (Path): Path to save the JSON file.
    """
    rich_content_dicts = [
        {
            "type": content.__class__.__name__.lower(),
            "content": content.content,
            "start": content.start,
            "end": content.end,
        }
        for content in rich_contents
    ]
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(rich_content_dicts, file, indent=4)
