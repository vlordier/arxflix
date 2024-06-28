"""Module for generating a video using Remotion."""

import json
import logging
import subprocess

# from dataclasses import asdict, dataclass, field
from pathlib import Path

from settings import Settings

# Initialize settings
settings = Settings()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_total_duration(arxiv_id: str) -> float:
    """Get the total duration of the audio in seconds.

    Args:
        arxiv_id (str): arXiv ID of the paper.

    Returns:
        float: The total duration of the audio in seconds.
    """
    base_dir = Path(settings.TEMP_DIR) / Path(arxiv_id)
    rich_output_path = base_dir / Path(
        settings.COMPOSITION_PROPS.rich_content_file_name
    )

    with open(rich_output_path, "r", encoding="utf-8") as f:
        rich_content = json.load(f)

    total_duration = rich_content[-1]["end"] if rich_content else 0
    return total_duration


def process_video(
    arxiv_id: str,
) -> None:
    """
    Processes the video by running the Remotion render command with the specified properties.

    Args:
        arxiv_id (str): arXiv ID of the paper.

    Raises:
        subprocess.CalledProcessError: If the video processing fails.
        Exception: If any other unexpected error occurs.
        FileNotFoundError: If the remotion root path does not exist.
    """
    composition_props = settings.COMPOSITION_PROPS

    base_dir = Path(settings.TEMP_DIR) / Path(arxiv_id)

    # composition_props.output = (
    #     (base_dir / Path(settings.COMPOSITION_PROPS.output))
    #     .absolute()
    #     .as_posix()
    # )

    output_path = (base_dir / arxiv_id / Path("output.mp4")).absolute().as_posix()

    # output_path = composition_props.output
    # Fix paths
    composition_props.subtitles_file_name = (
        (base_dir / Path(composition_props.subtitles_file_name)).absolute().as_posix()
    )

    composition_props.audio_file_name = (
        (base_dir / Path(composition_props.audio_file_name)).absolute().as_posix()
    )

    composition_props.rich_content_file_name = (
        (base_dir / Path(composition_props.rich_content_file_name))
        .absolute()
        .as_posix()
    )

    composition_props.duration_in_seconds = get_total_duration(arxiv_id)

    composition_props.duration_in_frames = int(
        composition_props.duration_in_seconds * settings.VIDEO.fps
    )

    composition_props.wave_color = settings.COMPOSITION_PROPS.wave_color
    composition_props.wave_number_of_samples = (
        settings.COMPOSITION_PROPS.wave_number_of_samples
    )
    composition_props.wave_lines_to_display = (
        settings.COMPOSITION_PROPS.wave_lines_to_display
    )
    composition_props.wave_freq_range_start_index = (
        settings.COMPOSITION_PROPS.wave_freq_range_start_index
    )
    composition_props.subtitles_line_per_page = (
        settings.COMPOSITION_PROPS.subtitles_line_per_page
    )
    composition_props.subtitles_line_height = (
        settings.COMPOSITION_PROPS.subtitles_line_height
    )
    composition_props.subtitles_zoom_measurer_size = (
        settings.COMPOSITION_PROPS.subtitles_zoom_measurer_size
    )
    composition_props.only_display_current_sentence = (
        settings.COMPOSITION_PROPS.only_display_current_sentence
    )
    composition_props.mirror_wave = settings.COMPOSITION_PROPS.mirror_wave

    remotion_root_path = Path(settings.REMOTION.root_path).absolute()

    if not remotion_root_path.exists():
        raise FileNotFoundError(
            f"Remotion root path does not exist: {remotion_root_path}"
        )

    remotion_root_path = remotion_root_path.as_posix()
    # properties = json.dumps(asdict(composition_props))
    properties = composition_props.json()
    logger.debug("Properties: %s", properties)

    try:
        command = [
            "npx",
            "remotion",
            "render",
            remotion_root_path,
            "--composition-id=",
            settings.REMOTION.composition_id,
            # "--subtitles=",
            # composition_props.subtitles_file_name,
            # "--audio=",
            # composition_props.audio_file_name,
            # "--richContent=",
            # composition_props.rich_content_file_name,
            "--props=",
            properties,
            "--concurrency=",
            settings.REMOTION.concurrency,
            "--duration=",
            str(composition_props.duration_in_seconds),
            "--fps=",
            str(settings.VIDEO.fps),
            "--output=",
            output_path,
            "--log=verbose",
        ]
        # "--compositionId=",
        # settings.REMOTION.composition_id,

        logger.info("Running command: %s", " ".join(command))
        subprocess.run(command, check=True)
        logger.info("Video processing completed successfully.")
    except subprocess.CalledProcessError as e:
        logger.error("Video processing failed: %s", e)
        raise
    except Exception as e:
        logger.error("An unexpected error occurred: %s", e)
        raise
