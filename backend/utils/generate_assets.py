"""
Utility functions to generate audio, caption, and other assets.
"""

import json
import logging
import os
import tempfile
from datetime import timedelta
from pathlib import Path
from typing import List, Optional, Union

import srt  # type: ignore
import torch
import whisper_timestamped as whisper  # type: ignore
from dotenv import load_dotenv
from elevenlabs import Voice, VoiceSettings, save
from elevenlabs.client import ElevenLabs
from pydub import AudioSegment  # type: ignore

from backend.models import Caption, Equation, Figure, Headline, RichContent, Text
from backend.settings import Settings

# Load environment variables
load_dotenv()

# Load configuration
settings = Settings()

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Load ElevenLabs API key
ELEVENLABS_API_KEY = str(os.getenv("ELEVENLABS_API_KEY"))
assert ELEVENLABS_API_KEY, "ELEVENLABS_API_KEY is not set."

# Configure ElevenLabs voice
ELEVENLABS_VOICE = Voice(
    voice_id=settings.ELEVENLABS.voice_id,
    settings=VoiceSettings(
        stability=settings.ELEVENLABS.stability,
        similarity_boost=settings.ELEVENLABS.similarity_boost,
        style=0.0,
        use_speaker_boost=True,
    ),
)
logger.info("Using ElevenLabs voice: %s", settings.ELEVENLABS.voice_id)

# Load Whisper model once
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    WHISPER_MODEL = whisper.load_model(settings.WHISPER_MODEL, device=device)
    logger.info("Whisper model loaded on device: %s", device)
except Exception as e:
    logger.error("Failed to load Whisper model: %s", e)
    raise ValueError("Failed to load Whisper model.") from e


def parse_script(script: str) -> List[Union[RichContent, Text]]:
    """
    Parse the script and return a list of RichContent or Text objects.

    Args:
        script (str): The script to parse as a string.

    Returns:
        List[Union[RichContent, Text]]: List of RichContent or Text objects.
    """
    content_list: List[Union[RichContent, Text]] = []
    type_map = {
        Figure.identifier: Figure,
        Text.identifier: Text,
        Equation.identifier: Equation,
        Headline.identifier: Headline,
    }

    for line in script.split("\n"):
        for prefix, content_type in type_map.items():
            if line.startswith(prefix):
                content = line[len(prefix) :]
                content_list.append(content_type(content))
                break
        else:
            logger.warning("Unknown line format: %s", line)

    return content_list


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
            word_text = word["text"].lstrip()
            caption = Caption(word=word_text, start=word["start"], end=word["end"])
            captions.append(caption)
    return captions


def generate_audio_and_caption(
    script_contents: List[Union[RichContent, Text]],
    model: whisper.Whisper = WHISPER_MODEL,
    temp_dir: Optional[Path] = None,
) -> List[Union[RichContent, Text]]:
    """
    Generate audio and caption for each text segment in the script.

    Args:
        script_contents (List[Union[RichContent, Text]]): List of RichContent or Text objects.
        model: The Whisper model instance.
        temp_dir (Optional[Path]): Temporary directory to store the audio files. Defaults to Path(tempfile.gettempdir()).

    Returns:
        List[Union[RichContent, Text]]: List of RichContent or Text objects with audio and caption.

    Raises:
        ValueError: If the ElevenLabs API key is not set.
    """
    temp_dir = temp_dir or Path(tempfile.gettempdir())
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        logger.info("Using ElevenLabs API key: ****%s", ELEVENLABS_API_KEY[-4:])
        logger.info("Using ElevenLabs model: %s", settings.ELEVENLABS.model)
        logger.info("Using ElevenLabs voice: %s", ELEVENLABS_VOICE.name)
    except Exception as e:
        logger.error("Failed to create ElevenLabs client: %s", e)
        raise ValueError("Failed to create ElevenLabs client.") from e

    for index, content in enumerate(script_contents):
        if isinstance(content, Text) and content.audio is None:
            if content.content.strip() == "":
                logger.info("Skipping empty text segment %d", index)
                continue
            logger.info("Generating audio for text segment %d", index)
            audio_path = temp_dir / f"audio_{index}.wav"
            logger.info("Audio path: %s", audio_path)

            if not audio_path.exists():
                content.audio = elevenlabs_client.generate(
                    text=content.content,
                    voice=ELEVENLABS_VOICE,
                    model=settings.ELEVENLABS.model,
                )
                save(content.audio, str(audio_path))

            try:
                audio = whisper.load_audio(str(audio_path))
            except Exception as e:
                logger.error("Failed to load audio file %s: %s", audio_path, e)
                raise ValueError("Failed to load audio file.") from e

            logger.info("Transcribing audio %d at %s", index, audio_path)

            result = whisper.transcribe(
                model,
                audio,
                language="en",
                beam_size=5,
                best_of=5,
                vad=True,
                temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                detect_disfluencies=False,
            )

            content.captions = make_caption(result)
            content.audio_path = str(audio_path)
            content.end = len(audio) / whisper.audio.SAMPLE_RATE

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

        if next_text_group[-1].end is not None and next_text_group[0].start is not None:
            total_duration = next_text_group[-1].end - next_text_group[0].start
        else:
            total_duration = 0
        duration_per_rich_content = total_duration / len(current_rich_content_group)
        offset = next_text_group[0].start
        for i, rich_content in enumerate(current_rich_content_group):
            if offset is not None:
                rich_content.start = offset + i * duration_per_rich_content
                rich_content.end = offset + (i + 1) * duration_per_rich_content

    return script_contents


def export_mp3(text_contents: List[Text], output_path: str) -> None:
    """
    Export the audio of the text contents to a single MP3 file.

    Args:
        text_contents (List[Text]): List of Text objects.
        output_path (str): Path to save the MP3 file.

    Raises:
        ValueError: If the output path does not end with .mp3 or the directory does not exist.
    """
    logger.info("Exporting to %s", output_path)

    output_dir = Path(output_path).parent
    if not output_dir.exists():
        raise ValueError(f"Directory {output_dir} does not exist.")

    if not str(output_path).endswith(".wav"):
        raise ValueError("Output path must end with .wav")

    audio_segments = [
        AudioSegment.from_file(text.audio_path)
        for text in text_contents
        if text.audio_path
    ]

    combined_audio = AudioSegment.empty()
    for audio in audio_segments:
        combined_audio += audio

    combined_audio.export(output_path, format="mp3")


def export_srt(
    full_audio_path: str,
    output_path: str,
    model: whisper.Whisper = WHISPER_MODEL,
) -> None:
    """
    Export the SRT file for the full audio using the Whisper model.

    Args:
        full_audio_path (str): Path to the full audio file.
        output_path (str): Path to save the SRT file.
        model: The Whisper model instance.
    """
    audio = whisper.load_audio(full_audio_path)

    result = whisper.transcribe(
        model,
        audio,
        language="en",
        beam_size=5,
        best_of=5,
        vad=True,
        temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        detect_disfluencies=False,
    )

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
    rich_contents: List[RichContent], output_path: str
) -> None:
    """
    Export the rich content to a JSON file.

    Args:
        rich_contents (List[RichContent]): List of RichContent objects.
        output_path (str): Path to save the JSON file.
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
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(rich_content_dicts, f, ensure_ascii=False, indent=4)
