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
# import whisper
import whisper_timestamped as whisper

from dotenv import load_dotenv
from elevenlabs import Voice, VoiceSettings, save
from elevenlabs.client import ElevenLabs

from backend.types import Caption, Equation, Figure, Headline, RichContent, Text

# from pywhispercpp.model import Model

# Set environment variables for OpenMP for whisper
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OMP_MAX_ACTIVE_LEVELS'] = '1'

# do the same with PYTORCH_ENABLE_MPS_FALLBACK=1
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# from transformers import pipeline
# import torch
# from transformers import pipeline
# from transformers.utils import is_flash_attn_2_available


# Setup logging
logger = logging.getLogger(__name__)

# Load .env file
load_dotenv()

# Load configuration
from backend.config import ELEVENLABS_API_KEY

# Load environment variables
# ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", default="EXAVITQu4vr4xnSDxMaL")

# ELEVENLABS_VOICE_NAME = os.getenv("ELEVENLABS_VOICE_NAME")

if not ELEVENLABS_API_KEY:
    logger.error("ELEVENLABS_API_KEY environment variable is not set.")
    raise ValueError("ELEVENLABS_API_KEY environment variable is not set.")
logger.error(
    "Using ElevenLabs API key: %s",
    ELEVENLABS_API_KEY[0:4] + "***" + ELEVENLABS_API_KEY[-4:],
)

ELEVENLABS_VOICE = Voice(
    voice_id=ELEVENLABS_VOICE_ID,
    settings=VoiceSettings(
        stability=float(os.getenv("ELEVENLABS_STABILITY", default="0.35")),
        similarity_boost=float(os.getenv("ELEVENLABS_SIMILARITY_BOOST", default="0.8")),
        style=0.0,
        use_speaker_boost=True,
    ),
)
ELEVENLABS_MODEL = os.getenv("ELEVENLABS_MODEL", default="eleven_turbo_v2")


def parse_script(script: str) -> List[Union[RichContent, Text]]:
    """
    Parse the script and return a list of RichContent or Text objects.

    Args:
        script (str): The script to parse as a string.

    Returns:
        List[Union[RichContent, Text]]: List of RichContent or Text objects.
    """
    lines = script.split("\n")
    content_list: List[Union[RichContent, Text, Figure, Equation, Headline]] = []
    for line in lines:
        if line.startswith(r"\Figure: "):
            figure_content = line.replace(r"\Figure: ", "")
            figure = Figure(content=figure_content)
            content_list.append(figure)
        elif line.startswith(r"\Text: "):
            text_content = line.replace(r"\Text: ", "")
            text = Text(content=text_content)
            content_list.append(text)
        elif line.startswith(r"\Equation: "):
            equation_content = line.replace(r"\Equation: ", "")
            equation = Equation(content=equation_content)
            content_list.append(equation)
        elif line.startswith(r"\Headline: "):
            headline_content = line.replace(r"\Headline: ", "")
            headline = Headline(content=headline_content)
            content_list.append(headline)
        else:
            logger.warning("Unknown line: %s", line)
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
    print(result)
    for segment in result["segments"]:
        for word in segment["words"]:
            word_text = word["text"].lstrip()
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

    Raises:
        ValueError: If the ElevenLabs API key is not set.
    """
    if temp_dir is None:
        temp_dir = Path(tempfile.gettempdir())

    if not ELEVENLABS_API_KEY:
        logger.error("ELEVENLABS_API_KEY environment variable is not set.")
        raise ValueError("ELEVENLABS_API_KEY environment variable is not set.")
    logger.error(
        "Using ElevenLabs API key: %s...%s",
        ELEVENLABS_API_KEY[:4],
        ELEVENLABS_API_KEY[-4:],
    )
    logger.error("Using ElevenLabs voice: %s", ELEVENLABS_VOICE.name)
    logger.error("Using ElevenLabs model: %s", ELEVENLABS_MODEL)

    try:
        elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    except Exception as e:
        logger.error("Failed to create ElevenLabs client: %s", e)
        raise ValueError("Failed to create ElevenLabs client.") from e

    if not temp_dir.exists():
        temp_dir.mkdir(parents=True)

    # model = whisper.load_model("tiny.en", device="cpu")
    model = whisper.load_model("tiny.en", device="cpu")

    # model = Model('base.en', n_threads=6, )

    # set to mps for Mac devices
    # device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    # logger.info("Using device: %s", device)

    # torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # logger.info("Using torch dtype: %s", torch_dtype)

    # model_id = "distil-whisper/distil-large-v3"

    # pipe = pipeline(
    #     "automatic-speech-recognition",
    #     model=model_id,
    #     torch_dtype=torch_dtype,
    #     device=device,
    #     max_new_tokens=128,
    #     chunk_length_s=30,
    #     batch_size=16,
    #     # return_timestamps=True,
    #     return_timestamps="word"
    #     )
    #     # model_kwargs={"attn_implementation": "flash_attention_2"} if is_flash_attn_2_available() else {"attn_implementation": "sdpa"},

    if not model:
        logger.error("Failed to load Whisper model.")
        raise ValueError("Failed to load Whisper model.")

    logger.info("Whisper model loaded.")

    for index, content in enumerate(script_contents):
        if isinstance(content, Text) and content.audio is None:
            if content.content.strip() == "":
                logger.info("Skipping empty text segment %d", index)
                continue
            logger.info("Generating audio for text segment %d", index)
            logger.info("Text: %s", content.content)
            audio_path = (temp_dir / f"audio_{index}.wav").absolute().as_posix()
            logger.info("Audio path: %s", audio_path)
            if not os.path.exists(audio_path):
                logger.info("Generating audio %d at %s", index, audio_path)
                content.audio = elevenlabs_client.generate(
                    text=content.content,
                    voice=ELEVENLABS_VOICE,
                    model=ELEVENLABS_MODEL,
                )
                save(content.audio, audio_path)

            # Double check the audio file exists:
            if not os.path.exists(audio_path):
                logger.error("Audio file %s does not exist.", audio_path)
                raise ValueError("Audio file does not exist.")

            logger.info("Using audio %d at %s", index, audio_path)

            #  Load the audio file:
            # try:
            #     audio, sample_rate = torchaudio.load(audio_path)
            # except Exception as e:
            #     logger.error("Failed to load audio file %s: %s", audio_path, e)
            #     raise ValueError("Failed to load audio file.")
            # audio, sample_rate = torchaudio.load(audio_path)

            # SAMPLE_RATE = 16000
            # audio = whisper.load_audio(audio_path, sr=SAMPLE_RATE)


            audio = whisper.load_audio(audio_path)


            # result = model.transcribe(audio, word_timestamps=True)
            result = whisper.transcribe(model, audio, language="en", beam_size=5, best_of=5, vad=True, temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), detect_disfluencies=False)


            # result = pipe(audio_path, chunk_length_s=30, batch_size=24, return_timestamps=True)
            logger.info("Result: %s", result)
            content.captions = make_caption(result)
            content.audio_path = audio_path
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
            if offset is not None and duration_per_rich_content is not None:
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
    combined_audio = torch.cat(audio_segments, dim=1)
    torchaudio.save(output_path, combined_audio, sample_rate)


def export_srt(full_audio_path: str, output_path: str) -> None:
    """
    Export the SRT file for the full audio using the Whisper model.

    Args:
        full_audio_path (str): Path to the full audio file.
        output_path (str): Path to save the SRT file.
    """
    model = whisper.load_model("base.en")
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
    with open(output_path, "w") as file:
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
    # df = pd.DataFrame(rich_content_dicts)
    # df.to_json(output_path, orient="records")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(rich_content_dicts, f)
