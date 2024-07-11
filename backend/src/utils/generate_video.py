# """Module for generating a video using Remotion."""


import json
import logging
import subprocess
from dataclasses import asdict, dataclass, field
from math import ceil
from pathlib import Path
from typing import Literal, Optional

from settings import settings

logger = logging.getLogger(__name__)


VIDEO_FPS = 30
VIDEO_HEIGHT = 1080
VIDEO_WIDTH = 1920
REMOTION_ROOT_PATH = Path("video_render/remotion/index.ts")
REMOTION_COMPOSITION_ID = "Arxflix"
REMOTION_CONCURRENCY: str = "100%"


@dataclass
class CompositionProps:
    durationInSeconds: float = 5
    audioOffsetInSeconds: int = 0
    subtitlesFileName: str = "output.srt"
    audioFileName: str = "audio.mp3"
    richContentFileName: str = "output.json"
    waveColor: str = "#a3a5ae"
    subtitlesLinePerPage: int = 2
    subtitlesLineHeight: int = 98
    subtitlesZoomMeasurerSize: int = 10
    onlyDisplayCurrentSentence: bool = True
    mirrorWave: bool = False
    waveLinesToDisplay: int = 300
    waveFreqRangeStartIndex: int = 5
    waveNumberOfSamples: Literal["32", "64", "128", "256", "512"] = "512"
    durationInFrames: int = field(init=False)

    def __post_init__(self):
        self.durationInFrames: int = ceil(self.durationInSeconds * VIDEO_FPS)


def process_video(
    arxiv_id: str,
    composition_props: Optional[CompositionProps],
    output: Optional[Path] = Path("output.mp4"),
):

    if not composition_props:
        composition_props = CompositionProps()

    # Fix paths
    output_path = (settings.TEMP_DIR / Path(arxiv_id) / output).absolute().as_posix()

    # copy to the public folder :
    # output.srt, audio.mp3, output.json, and also all the png files in the folder

    public_folder = Path(settings.PUBLIC_DIR)
    public_folder.mkdir(parents=True, exist_ok=True)
    public_folder = public_folder.absolute().as_posix()

    # copy the srt file
    srt_output_path = (
        (
            settings.TEMP_DIR
            / Path(arxiv_id)
            / settings.COMPOSITION_PROPS.subtitlesFileName
        )
        .absolute()
        .as_posix()
    )
    subprocess.run(["cp", srt_output_path, public_folder])

    # copy the audio file
    audio_output_path = (
        (settings.TEMP_DIR / Path(arxiv_id) / settings.COMPOSITION_PROPS.audioFileName)
        .absolute()
        .as_posix()
    )
    subprocess.run(["cp", audio_output_path, public_folder])

    # copy the json file
    json_output_path = (
        (
            settings.TEMP_DIR
            / Path(arxiv_id)
            / settings.COMPOSITION_PROPS.richContentFileName
        )
        .absolute()
        .as_posix()
    )
    subprocess.run(["cp", json_output_path, public_folder])

    # copy the png files
    png_files = list((settings.TEMP_DIR / Path(arxiv_id)).glob("*.png"))
    for png_file in png_files:
        subprocess.run(["cp", png_file, public_folder.absolute().as_posix()])

    assert composition_props.durationInSeconds > 0, "Duration must be greater than 0."
    assert isinstance(
        composition_props.durationInFrames, int
    ), "Duration must be an integer."

    logger.info("Rendering video for ArXiv paper %s...", arxiv_id)
    logger.info("Composition props: %s", composition_props)
    logger.info("Output path: %s", output_path)
    logger.info("Audio file name: %s", audio_output_path)
    logger.info("Subtitles file name: %s", srt_output_path)
    logger.info("Rich content file name: %s", json_output_path)
    logger.info("durationInSeconds: %s", composition_props.durationInSeconds)
    logger.info("durationInFrames: %s", composition_props.durationInFrames)

    subprocess.run(
        [
            "npx",
            "remotion",
            "render",
            REMOTION_ROOT_PATH.absolute().as_posix(),
            "--props",
            json.dumps(asdict(composition_props)),
            "--compositionId",
            REMOTION_COMPOSITION_ID,
            "--concurrency",
            str(REMOTION_CONCURRENCY),
            "--output",
            output_path,
        ]
    )


# import json
# import logging
# import subprocess

# # from dataclasses import asdict, dataclass, field
# from pathlib import Path

# from settings import Settings

# # Initialize settings
# settings = Settings()

# # Setup logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


def get_total_duration(arxiv_id: str) -> float:
    """Get the total duration of the audio in seconds.

    Args:
        arxiv_id (str): arXiv ID of the paper.

    Returns:
        float: The total duration of the audio in seconds.
    """
    base_dir = Path(settings.TEMP_DIR) / Path(arxiv_id)
    rich_output_path = base_dir / Path(settings.COMPOSITION_PROPS.richContentFileName)

    with open(rich_output_path, "r", encoding="utf-8") as f:
        rich_content = json.load(f)

    total_duration = rich_content[-1]["end"] if rich_content else 0
    return total_duration


# def process_video(
#     arxiv_id: str,
# ) -> None:
#     """
#     Processes the video by running the Remotion render command with the specified properties.

#     Args:
#         arxiv_id (str): arXiv ID of the paper.

#     Raises:
#         subprocess.CalledProcessError: If the video processing fails.
#         Exception: If any other unexpected error occurs.
#         FileNotFoundError: If the remotion root path does not exist.
#     """
#     composition_props = settings.COMPOSITION_PROPS

#     base_dir = Path(settings.TEMP_DIR) / Path(arxiv_id)

#     # composition_props.output = (
#     #     (base_dir / Path(settings.COMPOSITION_PROPS.output))
#     #     .absolute()
#     #     .as_posix()
#     # )

#     output_path = (base_dir / Path("output.mp4")).absolute().as_posix()

#     # output_path = composition_props.output
#     # Fix paths
#     composition_props.subtitles_file_name = (
#         (base_dir / Path(composition_props.subtitles_file_name)).absolute().as_posix()
#     )

#     composition_props.audio_file_name = (
#         (base_dir / Path(composition_props.audio_file_name)).absolute().as_posix()
#     )

#     composition_props.rich_content_file_name = (
#         (base_dir / Path(composition_props.rich_content_file_name))
#         .absolute()
#         .as_posix()
#     )

#     composition_props.duration_in_seconds = get_total_duration(arxiv_id)

#     composition_props.duration_in_frames = int(
#         composition_props.duration_in_seconds * settings.VIDEO.fps
#     )

#     composition_props.wave_color = settings.COMPOSITION_PROPS.wave_color
#     composition_props.wave_number_of_samples = (
#         settings.COMPOSITION_PROPS.wave_number_of_samples
#     )
#     composition_props.wave_lines_to_display = (
#         settings.COMPOSITION_PROPS.wave_lines_to_display
#     )
#     composition_props.wave_freq_range_start_index = (
#         settings.COMPOSITION_PROPS.wave_freq_range_start_index
#     )
#     composition_props.subtitles_line_per_page = (
#         settings.COMPOSITION_PROPS.subtitles_line_per_page
#     )
#     composition_props.subtitles_line_height = (
#         settings.COMPOSITION_PROPS.subtitles_line_height
#     )
#     composition_props.subtitles_zoom_measurer_size = (
#         settings.COMPOSITION_PROPS.subtitles_zoom_measurer_size
#     )
#     composition_props.only_display_current_sentence = (
#         settings.COMPOSITION_PROPS.only_display_current_sentence
#     )
#     composition_props.mirror_wave = settings.COMPOSITION_PROPS.mirror_wave

#     remotion_root_path = Path(settings.REMOTION.root_path).absolute()

#     if not remotion_root_path.exists():
#         raise FileNotFoundError(
#             f"Remotion root path does not exist: {remotion_root_path}"
#         )

#     remotion_root_path = remotion_root_path.as_posix()
#     # properties = json.dumps(asdict(composition_props))
#     properties = composition_props.json()
#     logger.debug("Properties: %s", properties)

#     try:
#         command = [
#             "npx",
#             "remotion",
#             "render",
#             remotion_root_path,
#             "--composition-id=",
#             settings.REMOTION.composition_id,
#             "--subtitles=",
#             composition_props.subtitles_file_name,
#             "--audio=",
#             composition_props.audio_file_name,
#             "--richContent=",
#             composition_props.rich_content_file_name,
#             "--props=",
#             properties,
#             "--concurrency=",
#             settings.REMOTION.concurrency,
#             "--duration=",
#             str(composition_props.duration_in_seconds),
#             "--fps=",
#             str(settings.VIDEO.fps),
#             "--output=",
#             output_path,
#             "--log=verbose",
#         ]
#         # "--compositionId=",
#         # settings.REMOTION.composition_id,

#         logger.info("Running command: %s", " ".join(command))
#         subprocess.run(command, check=True)
#         logger.info("Video processing completed successfully.")
#     except subprocess.CalledProcessError as e:
#         logger.error("Video processing failed: %s", e)
#         raise
#     except Exception as e:
#         logger.error("An unexpected error occurred: %s", e)
#         raise
