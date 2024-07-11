"""
Utility functions to generate audio, caption, and other assets.
"""

import json
import logging
import os
import re
import tempfile
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Union

import requests
import srt
import torch
import whisper_timestamped as whisper
from dotenv import load_dotenv
from elevenlabs import Voice, VoiceSettings, save
from elevenlabs.client import ElevenLabs
from models import Caption, Equation, Figure, Headline, RichContent, Text
from PIL import Image
from pydub import AudioSegment
from settings import settings

# Load environment variables
load_dotenv()

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Load ElevenLabs API key
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
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

# Load Whisper model once
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
try:
    WHISPER_MODEL = whisper.load_model(settings.WHISPER_MODEL, device=DEVICE)
    logger.debug("Whisper model loaded on device: %s", DEVICE)
except Exception as e:
    logger.error("Failed to load Whisper model: %s", e)
    raise ValueError("Failed to load Whisper model.") from e


def parse_script(script: str) -> List[Union[RichContent, Text]]:
    """
    Parse the LaTeX-like script into respective classes.

    Args:
        script (str): The script containing different components.

    Returns:
        list: A list of class instances corresponding to the parsed components.
    """
    class_map = {
        "Figure": Figure,
        "Text": Text,
        "Equation": Equation,
        "Headline": Headline,
    }
    pattern = r"\\(Figure|Text|Equation|Headline):\s*(.*?)(?=\\(Figure|Text|Equation|Headline):|$|\n(?!\\(Figure|Text|Equation|Headline):))"

    matches = re.findall(pattern, script, re.DOTALL)

    result = [
        class_map[match[0]](
            identifier=f"\\{match[0]}", content=match[1].replace("\\n", "").strip()
        )
        for match in matches
    ]
    logger.debug("Parsed script into objects %s", result)

    return result


def make_caption(result: dict) -> List[Caption]:
    """
    Create a list of Caption objects from the result of the Whisper model.

    Args:
        result (dict): Result dictionary from the Whisper model.

    Returns:
        List[Caption]: List of Caption objects.
    """
    captions = [
        Caption(word=word["text"].lstrip(), start=word["start"], end=word["end"])
        for segment in result["segments"]
        for word in segment["words"]
    ]
    return captions


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

    if not os.path.exists(full_audio_path):
        raise ValueError(f"Full audio file does not exist. {full_audio_path}")

    # if the file already exists, we don't need to re-export it
    if os.path.exists(output_path):
        return None

    try:
        audio = whisper.load_audio(full_audio_path)
        logger.debug("Exporting SRT for %s", full_audio_path)
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
        logger.debug("SRT export completed successfully.")
    except Exception as exc:
        logger.error("Failed to export SRT: %s", exc)
        raise ValueError("Failed to export SRT.") from exc


def export_rich_content_json(
    rich_contents: List[RichContent], output_path: str, arxiv_id: str
) -> None:
    """
    Export the rich content to a JSON file.

    Args:
        rich_contents (List[RichContent]): List of RichContent objects.
        output_path (str): Path to save the JSON file.
        arxiv_id (str): arXiv ID of the article.

    Raises:
        ValueError: If the output path does not end with .json.
    """

    if Path(output_path).suffix != ".json":
        raise ValueError("Output path must end with .json.")

    # Create the parent directory if it does not exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    try:
        rich_content_dicts = [
            {
                "type": content.__class__.__name__.lower(),
                "content": content.content,
                "start": content.start,
                "end": content.end,
            }
            for content in rich_contents
        ]

        rich_content_dicts = fetch_images(rich_content_dicts, arxiv_id)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(rich_content_dicts, f, ensure_ascii=False, indent=4)
        logger.debug("Rich content JSON export completed successfully.")
    except Exception as inner_e:
        logger.error("Failed to export rich content JSON: %s", inner_e)
        raise


def replace_url_with_path(text: str, new_path: str) -> str:
    """
    Replaces the URL in the given text with the specified file path.

    Args:
        text (str): The input string containing the URL.
        new_path (str): The file path to replace the URL with.

    Returns:
        str: The modified string with the URL replaced by the new path.

    Raises:
        re.error: If there is an error in the regular expression.
        Exception: If an unexpected error occurs.

    """
    try:
        # Regular expression to find the URL in the markdown link
        url_pattern = r"\(https?://[^\)]+\)"

        # Log the original text and new path
        logger.debug("Original text: %s", text)
        logger.debug("New path: %s", new_path)

        # Replace the URL with the new path
        modified_text = re.sub(url_pattern, f"({new_path})", text)

        # Log the modified text
        logger.debug("Modified text: %s", modified_text)

        return modified_text

    except re.error as e:
        logger.error("Regex error: %s", e)
        return text

    except Exception as e:
        logger.error("An unexpected error occurred: %s", e)
        return text


def extract_image_name(text: str) -> str:
    """
    Extracts the image name from a given text.

    Args:
        text (str): The input text containing the URL.

    Returns:
        str: The extracted image name or an empty string if not found.
    """

    # clean up text by removing the markdown syntax
    text = text.replace("![", "").replace("]", "")
    text = text.replace("(", "").replace(")", "")
    text = text.replace("{", "").replace("}", "")
    text = text.replace("[", "").replace("]", "")
    text = text.replace("`", "")
    text = text.replace(" ", "")  # remove spaces
    text = text.replace("\n", "")  # remove newlines
    text = text.replace("\t", "")  # remove tabs
    text = text.replace("\r", "")  # remove carriage returns
    text = text.replace("\v", "")  # remove vertical tabs
    text = text.replace("\f", "")  # remove form feeds

    # if the text has # or ? in it, keep only the part before it
    if "#" in text:
        text = text.split("#")[0]
    if "?" in text:
        text = text.split("?")[0]

    # if text ends with a /, remove it
    if text.endswith("/"):
        text = text[:-1]

    if text.startswith("/"):
        text = text[1:]

    image_name = text.strip()

    # if text.endswith("png") or text.endswith("jpg") or text.endswith("jpeg") or text.endswith("bmp"):

    # if the text is a part of an url (but witout the https and domain), return it
    # if text.startswith("/") and (text.endswith(".png") or text.endswith(".jpg") or text.endswith(".jpeg") or text.endswith(".bmp")):
    # return text[1:]

    # if the
    # url_pattern = re.compile(r"\((https?://[^\)]+)\)")
    # url_match = url_pattern.search(text)

    # if url_match:
    # url = url_match.group(1).strip().split("?")[0].split("#")[0]
    # image_name = url.split("/")[-1]

    # Check that the image name ends with .png, .jpg, .jpeg, or .bmp
    if not re.match(r".*\.(png|jpg|jpeg|bmp)$", image_name, re.IGNORECASE):
        raise ValueError(
            f"Invalid image URL, should be a PNG, JPG, JPEG, or BMP image, got {image_name}"
        )

    # Check that the image name is not empty
    if not image_name:
        raise ValueError("Image name is empty.")

    return image_name


def fetch_images(
    rich_content_dicts: List[Dict[str, str]], arxiv_id: str
) -> List[Dict[str, str]]:
    """
    Fetch images for the rich content and update their paths.

    Args:
        rich_content_dicts (List[Dict[str, str]]): List of rich content dictionaries.
        arxiv_id (str): arXiv ID of the article.

    Returns:
        List[Dict[str, str]]: List of rich content dictionaries with updated image paths.
    """
    for index, content_dict in enumerate(rich_content_dicts):
        if content_dict.get("type") == "figure" and "content" in content_dict:
            content = content_dict["content"].strip()

            if not content:
                raise ValueError("Image URL is empty.")

            image_name = extract_image_name(content)
            logger.info("Fetching image: %s", image_name)

            image_path = fetch_image(image_name, arxiv_id)
            if not image_path:
                raise ValueError("Failed to fetch image.")

            if check_image(image_path):
                content_dict["content"] = replace_url_with_path(content, image_path)
                rich_content_dicts[index] = content_dict
                logger.info("Fetched image: %s", image_path)
            else:
                logger.warning("Failed to fetch image: %s", image_path)
                raise ValueError(f"Failed to fetch image {image_path}")

    return rich_content_dicts


def check_image(image_path: str) -> bool:
    """
    Check if the image exists and is readable.

    Args:
        image_path (str): Path to the image file.

    Returns:
        bool: True if the image exists and is readable, False otherwise.
    """
    image_file = Path(image_path)
    if image_file.exists():
        try:
            with Image.open(image_file) as img:
                img.verify()
            logger.debug("Image is valid and readable: %s", image_path)
            return True
        except Exception as e:
            logger.error("Failed to open image at %s: %s", image_path, e)
            return False
    logger.error("Image does not exist at %s", image_path)
    return False


def fetch_image(image_name: str, arxiv_id: str) -> str:
    """
    Fetch the image from the URL and save it locally.

    Args:
        image_name (str): name of the image to be fetched.
        arxiv_id (str): arXiv ID of the article.

    Returns:
        str: Path to the saved image.
    """
    if not image_name.strip():
        raise ValueError("Image name is empty.")

    # image_url = image_url.strip().split("?")[0].split("#")[0]

    # if not re.match(r"^https?://", image_url):
    #     raise ValueError("Invalid image URL.")

    # image_name = image_url.split("/")[-1]
    # if not re.match(r".*\.(png)$", image_name):
    #     raise ValueError(f"Invalid image URL, should be a PNG image, got {image_name}")

    image_path = Path(settings.TEMP_DIR) / Path(arxiv_id) / Path(image_name)

    if image_path.exists():
        logger.debug("Image already exists at %s", image_path)
        return str(image_path)

    image_path = image_path.absolute().as_posix()

    image_url = f"{settings.ARXIV_BASE_URL}/{arxiv_id}/{image_name}"

    try:
        response = requests.get(image_url, timeout=settings.REQUESTS_TIMEOUT)
        response.raise_for_status()
        # generate the necessary directories in image_path
        Path(image_path).parent.mkdir(parents=True, exist_ok=True)
        with open(image_path, "wb") as f:
            f.write(response.content)
        logger.info("Saved image to %s", image_path)
        return image_path
    except Exception as exc:
        logger.error("Failed to fetch image from URL: %s", exc)
        raise ValueError("Failed to fetch image.") from exc


def replace_url_with_path(content: str, image_path: str) -> str:
    """
    Replaces any URL in the markdown image syntax [text](url) in the content with the given image path.

    Args:
        content (str): The content containing the markdown image syntax with URL.
        image_path (str): The image path to replace the URL.

    Returns:
        str: The updated content with the URL replaced by the image path.
    """
    # Regex pattern to match the markdown image syntax [text](url)
    url_pattern = re.compile(r"\[(.*?)\]\((http[s]?://.*?)\)")

    # Function to replace the URL in the match object with the image path
    def replacement(match):
        text = match.group(1)
        new_string = f"[{text}]({image_path})"
        return new_string

    # Replace the URL with the image path using the replacement function
    new_content = url_pattern.sub(replacement, content)

    logger.debug("Original content: %s", content)
    logger.debug("Updated content: %s", new_content)

    return new_content


def create_elevenlabs_client(api_key: str) -> ElevenLabs:
    """
    Create an ElevenLabs client with the provided API key.

    Args:
        api_key (str): ElevenLabs API key.

    Returns:
        ElevenLabs client.

    Raises:
        ValueError: If the ElevenLabs client creation fails.
    """
    try:
        client = ElevenLabs(api_key=api_key)
        logger.debug("ElevenLabs client created with API key: ****%s", api_key[-4:])
        return client
    except Exception as exc:
        logger.error("Failed to create ElevenLabs client: %s", exc)
        raise ValueError("Failed to create ElevenLabs client.") from exc


def generate_audio_for_text(
    text_content: Text,
    client: ElevenLabs,
    voice: Voice,
    model: str,
    temp_dir: Path,
    index: int,
) -> str:
    """
    Generate audio for the given text content using the ElevenLabs client.

    Args:
        text_content (Text): Text content object.
        client (ElevenLabs): ElevenLabs client.
        voice (Voice): Voice object.
        model (str): Model name.
        temp_dir (Path): Temporary directory.
        index (int): Index of the text content.

    Returns:
        str: Path to the generated audio file.

    Raises:
        ValueError: If the text content is empty.
    """

    if text_content.content.strip() == "":
        logger.debug("Skipping empty text segment %d", index)
        return ""

    audio_path = temp_dir / Path(f"audio_{index}.wav")
    audio_path.parent.mkdir(parents=True, exist_ok=True)

    if not audio_path.exists():
        logger.debug("Generating audio for text segment %d", index)
        text_content.audio = client.generate(
            text=text_content.content,
            voice=voice,
            model=model,
        )
        save(text_content.audio, str(audio_path))
        logger.debug("Saved audio to %s", audio_path)
    else:
        logger.debug("Audio already exists at %s", audio_path)

    return str(audio_path)


def transcribe_audio(audio_path: str, model: whisper.Whisper) -> dict:
    """
    Transcribe the audio file using the Whisper model.

    Args:
        audio_path (str): Path to the audio file.
        model (whisper.Whisper): The Whisper model instance.

    Returns:
        dict: The transcription result.
    """

    try:
        transcript_file_path = audio_path.replace(".wav", ".json")
        if os.path.exists(transcript_file_path):
            with open(transcript_file_path, "r", encoding="utf-8") as f:
                result = json.load(f)
                logger.debug("Loaded transcription from %s", transcript_file_path)
                return result

        logger.debug("Transcribing audio file %s", audio_path)
        audio = whisper.load_audio(audio_path)
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
        with open(transcript_file_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)

        return result
    except Exception as exc:
        logger.error("Failed to transcribe audio file %s: %s", audio_path, exc)
        raise ValueError("Failed to transcribe audio file.") from exc


def update_text_content_with_captions(
    text_content: Text, captions: List[Caption], audio_length: float
) -> None:
    """
    Update the text content with the captions and audio length.

    Args:
        text_content (Text): The Text object to update.
        captions (List[Caption]): List of Caption objects.
        audio_length (float): Length of the audio in seconds.
    """
    text_content.captions = captions
    text_content.end = audio_length / whisper.audio.SAMPLE_RATE


def combine_audio_segments(audio_segments: List[AudioSegment], output_path: str) -> str:
    """
    Combine the audio segments into a single audio file.

    Args:
        audio_segments (List[AudioSegment]): List of AudioSegment objects.
        output_path (str): Path to save the combined audio file.


    """

    nb_audio_segments = len(audio_segments)
    if nb_audio_segments == 0:
        raise ValueError("No audio segments to combine.")

    logger.debug("Combining %d audio segments into %s", nb_audio_segments, output_path)

    combined_audio = AudioSegment.empty()
    for audio in audio_segments:
        combined_audio += audio
    combined_audio.export(output_path, format="mp3")
    logger.debug("Exported combined audio to %s", output_path)
    if not Path(output_path).exists():
        raise ValueError("Failed to export combined audio.")

    return output_path


def process_audio_files(text_contents: List[Text]) -> List[AudioSegment]:
    """
    Process audio files for each text segment.

    Args:
        text_contents (List[Text]): List of Text objects.

    Returns:
        List[AudioSegment]: List of AudioSegment objects.
    """
    logger.debug("Processing audio files for %d text segments", len(text_contents))
    audio_segments = []
    for text in text_contents:
        logger.debug("Processing text segment: %s", text.content)
        if text.audio_path:
            logger.debug("Processing audio file: %s", text.audio_path)
            try:
                audio_segment = AudioSegment.from_file(text.audio_path)
                audio_segments.append(audio_segment)
            except FileNotFoundError as fnf_error:
                logger.error("File not found: %s", text.audio_path)
                raise fnf_error
            except Exception as exc:
                logger.error("Error processing file %s: %s", text.audio_path, exc)
                raise exc
    return audio_segments


def generate_audio_and_caption(
    script_contents: List[Union[RichContent, Text]],
    model: whisper.Whisper = WHISPER_MODEL,
    temp_dir: Path = None,
    arxiv_id: str = None,
) -> List[Union[RichContent, Text]]:
    """
    Generate audio and caption for each text segment in the script.

    Args:
        script_contents (List[Union[RichContent, Text]]): List of RichContent or Text objects.
        model: The Whisper model instance.
        temp_dir (Optional[Path]): Temporary directory to store the audio files. Defaults to Path(tempfile.gettempdir()).
        arxiv_id (str): arXiv ID of the article.

    Returns:
        List[Union[RichContent, Text]]: List of RichContent or Text objects with audio and caption.
    """
    temp_dir = temp_dir or Path(tempfile.gettempdir())
    temp_dir = Path(temp_dir) / Path(arxiv_id) if arxiv_id else temp_dir
    temp_dir = temp_dir.resolve().absolute()
    temp_dir.mkdir(parents=True, exist_ok=True)

    elevenlabs_client = create_elevenlabs_client(ELEVENLABS_API_KEY)

    for index, content in enumerate(script_contents):
        logger.debug("Processing content: %s", content)
        if isinstance(content, Text) and content.audio is None:
            audio_path = generate_audio_for_text(
                content,
                elevenlabs_client,
                ELEVENLABS_VOICE,
                settings.ELEVENLABS.model,
                temp_dir,
                index,
            )
            if audio_path:
                result = transcribe_audio(audio_path, model)
                captions = make_caption(result)
                update_text_content_with_captions(content, captions, len(audio_path))
                content.audio_path = audio_path

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


def export_mp3(text_contents: List[Text], output_path: str) -> None:
    """
    Export the audio of the text contents to a single MP3 file.

    Args:
        text_contents (List[Text]): List of Text objects.
        output_path (str): Path to save the MP3 file.

    Raises:
        ValueError: If the output path does not end with .mp3 or the directory does not exist.
    """
    logger.debug("Exporting to %s", output_path)

    output_dir = Path(output_path).parent
    if not output_dir.exists():
        raise ValueError(f"Directory {output_dir} does not exist.")

    if not str(output_path).endswith(".mp3"):
        raise ValueError("Output path must end with .mp3")

    if len(text_contents) == 0:
        raise ValueError("No text contents to export.")

    audio_segments = process_audio_files(text_contents)
    combine_audio_segments(audio_segments, output_path)
