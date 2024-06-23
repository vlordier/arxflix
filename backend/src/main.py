""" Main module for the backend application. """

import logging
from pathlib import Path
from typing import Optional

import fastapi
import typer

# from src.config import LOG_FORMAT, LOG_LEVEL, PAPER_URL, TEMP_DIR
from src.models import AssetsInput, RichContent, ScriptInput, Text
from src.settings import Settings
from src.utils.generate_assets import (
    export_mp3,
    export_rich_content_json,
    export_srt,
    fill_rich_content_time,
    generate_audio_and_caption,
    parse_script,
)
from src.utils.generate_paper import process_article
from src.utils.generate_script import process_script
from src.utils.generate_video import process_video

PAPER_URL = ""

settings = Settings()

# Setup logging
logging.basicConfig(level=settings.LOGGING.level, format=settings.LOGGING.format)
logger = logging.getLogger(__name__)

# Initialize CLI and API
cli = typer.Typer()
app = fastapi.FastAPI()


@cli.command("generate_paper")
@app.get("/generate_paper/")
def generate_paper(url: str) -> str:  # dead: disable
    """Generate a paper from a given URL.

    Args:
        url (str): The URL of the paper to process.

    Returns:
        str: The content of the processed paper.

    Raises:
        ValueError: If there is an error processing the article.
    """
    global PAPER_URL
    PAPER_URL = url  # dead: disable
    logger.info("Generating paper from URL: %s", url)
    try:
        paper_content = process_article(url)
        return paper_content
    except Exception as inner_e:
        logger.exception("Error generating paper from URL %s: %s", url, inner_e)
        raise ValueError(f"Error generating paper from URL {url}") from inner_e


@cli.command("generate_script")
def generate_script(paper: str, use_path: bool = True) -> str:
    """Generate a script from a given paper.

    Args:
        paper (str): The paper content or the path to the paper file.
        use_path (bool, optional): Whether to treat `paper` as a file path. Defaults to True.

    Returns:
        str: The generated script content.

    Raises:
        ValueError: If there is an error reading the paper or generating the script.
    """
    logger.info("Generating script from paper: %s", paper)
    if use_path:
        try:
            paper_content = Path(paper).read_text()
        except Exception as inner_e:  # Change variable name from 'e' to 'inner_e'
            logger.exception("Error reading paper from path %s: %s", paper, inner_e)
            raise ValueError(f"Error reading paper from path {paper}") from inner_e
    else:
        paper_content = paper

    try:
        script_content = process_script(paper_content, settings.PAPER_URL)
        return script_content
    except Exception as inner_e:  # Change variable name from 'e' to 'inner_e'
        logger.exception("Error generating script: %s", inner_e)
        raise ValueError("Error generating script") from inner_e


@app.post("/generate_script/")
def generate_script_api(input_data: ScriptInput) -> str:  # dead: disable
    """API endpoint to generate a script from a given paper.

    Args:
        input_data (ScriptInput): Input parameters containing the paper content or path.

    Returns:
        str: The generated script content.
    """
    logger.info("API endpoint called: generate_script")

    return generate_script(input_data.paper, input_data.use_path)


@cli.command("generate_assets")
def generate_assets(
    script: str,
    mp3_output: str = "public/audio.wav",
    srt_output: str = "public/output.srt",
    rich_output: str = "public/output.json",
) -> float:
    """Generate audio, SRT, and rich content JSON assets from a script.

    Args:
        script (str): The script content or the path to the script file.
        mp3_output (str, optional): Path to save the MP3 output file. Defaults to "public/audio.wav".
        srt_output (str, optional): Path to save the SRT output file. Defaults to "public/output.srt".
        rich_output (str, optional): Path to save the rich content JSON file. Defaults to "public/output.json".

    Returns:
        float: The total duration of the audio in seconds.

    Raises:
        ValueError: If there is an error reading the script or generating the assets.
    """
    logger.info("Generating assets from script: %s", script)

    mp3_output, srt_output, rich_output = create_directories(
        mp3_output, srt_output, rich_output
    )

    logger.info("MP3 output: %s", mp3_output)
    logger.info("SRT output: %s", srt_output)
    logger.info("Rich content output: %s", rich_output)

    try:
        script_contents: list[RichContent | Text] = parse_script(script)

        if settings.TEMP_DIR:
            logger.info("Using temporary directory: %s", settings.TEMP_DIR)
            temp_dir = settings.TEMP_DIR
        else:
            temp_dir = None

        script_contents = generate_audio_and_caption(script_contents, temp_dir=temp_dir)
        script_contents = fill_rich_content_time(script_contents)

        rich_content_list = [
            item for item in script_contents if isinstance(item, RichContent)
        ]
        text_content_list = [item for item in script_contents if isinstance(item, Text)]

        if not text_content_list:
            raise ValueError("No text content found in the script")

        logger.info("Exporting assets...")
        logger.info("Exporting MP3: %s", mp3_output)
        export_mp3(text_content_list, mp3_output)

        logger.info("Exporting SRT: %s", srt_output)
        export_srt(mp3_output, srt_output)

        logger.info("Exporting rich content JSON: %s", rich_output)
        export_rich_content_json(rich_content_list, rich_output)

        total_duration = (
            text_content_list[-1].end
            if text_content_list and text_content_list[-1].end
            else 0
        )
        return total_duration
    except Exception as inner_e:  # Change variable name from 'e' to 'inner_e'
        logger.exception("Error generating assets: %s", inner_e)
        raise ValueError("Error generating assets") from inner_e


def create_directories(mp3_output: str, srt_output: str, rich_output: str) -> tuple:
    """
    Create directories for output files and return the paths as Path objects.

    Args:
        mp3_output (str): Relative path for mp3 output directory.
        srt_output (str): Relative path for srt output directory.
        rich_output (str): Relative path for rich output directory.

    Returns:
        tuple: A tuple containing the Path objects for the mp3, srt, and rich directories.
    """
    # Base directory relative to this script
    base_dir = Path(__file__).parent.parent.parent.absolute()
    logger.info("Base directory: %s", base_dir)

    # Initialize a dictionary to hold the full paths
    paths = {}

    # Define the output paths
    output_paths = {"mp3": mp3_output, "srt": srt_output, "rich": rich_output}

    # Create directories and store the paths
    for key, output_path in output_paths.items():
        target_path = base_dir / output_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        paths[key] = target_path

    # Return the paths as a tuple
    return (paths["mp3"], paths["srt"], paths["rich"])


@app.post("/generate_assets/")
def generate_assets_api(assets_input: AssetsInput) -> float:  # dead: disable
    """API endpoint to generate assets from a script.

    Args:
        assets_input (AssetsInput): Input parameters containing the script content or path and output file paths.

    Returns:
        float: The total duration of the audio in seconds.
    """
    logger.info("API endpoint called: generate_assets")

    return generate_assets(
        script=assets_input.script,
        mp3_output=assets_input.mp3_output,
        srt_output=assets_input.srt_output,
        rich_output=assets_input.rich_output,
    )


@cli.command("generate_video")
@app.post("/generate_video/")
def generate_video(output_path: Optional[Path]) -> None:  # dead: disable
    """Generate a video from the processed script.

    Args:
        output_path (Path, optional): Path to save the output video file. Defaults to "public/output.mp4".

    Raises:
        ValueError: If there is an error generating the video.
    """
    if not output_path:
        output_path = Path("public/output.mp4")

    logger.info("Generating video to %s", output_path)
    try:
        process_video(
            output_path=output_path, composition_props=None
        )  # Add the missing argument 'composition_props'
    except Exception as inner_e:  # Change variable name from 'e' to 'inner_e'
        logger.exception("Error generating video: %s", inner_e)
        raise ValueError(f"Error generating video to {output_path}") from inner_e


if __name__ == "__main__":
    logger.info("Starting CLI...")
    try:
        cli()
    except Exception as e:
        logger.exception("Error starting CLI: %s", e)
        raise
