"""Main module for the backend application."""

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import fastapi
import typer

# Import necessary modules and functions from the project
from models import AssetsInput, RichContent, ScriptInput, Text
from settings import settings
from utils.generate_assets import (
    export_mp3,
    export_rich_content_json,
    export_srt,
    fill_rich_content_time,
    generate_audio_and_caption,
    parse_script,
)
from utils.generate_paper import get_arxiv_id_from_url, process_article
from utils.generate_script import process_script
from utils.generate_video import get_total_duration, process_video

# Setup logging configuration
logging.basicConfig(level=settings.LOGGING.level, format=settings.LOGGING.format)
logger = logging.getLogger(__name__)

# Initialize CLI and API
cli = typer.Typer()
api = fastapi.FastAPI()


@cli.command("generate_paper")
@api.get("/generate_paper/")
def generate_paper(url: str) -> str:
    """Generate a markdown from a given arxiv URL.

    Args:
        url (str): The URL of the paper to process.

    Returns:
        str: The markdown content of the paper.

    Raises:
        ValueError: If there is an error processing the article.
    """
    logger.info("Generating paper from URL: %s", url)
    try:
        paper_markdown = process_article(url)
        logger.info("Paper generated successfully.")
        return paper_markdown.markdown
    except Exception as e:
        logger.exception("Error generating paper from URL %s: %s", url, e)
        raise ValueError(f"Error generating paper from URL {url}") from e


@cli.command("generate_script")
def generate_script(paper_markdown: str, arxiv_id: Optional[str] = None) -> str:
    """Generate a video script from a given paper.

    Args:
        paper_markdown (str): The markdown content or the path to the paper file.
        arxiv_id (Optional[str], optional): arXiv ID of the paper. Defaults to None.

    Returns:
        str: The generated script content.

    Raises:
        ValueError: If there is an error reading the paper or generating the script.
    """
    logger.debug("Generating script from paper: %s", paper_markdown)
    try:
        script_content = process_script(paper_markdown, arxiv_id)
        logger.info("Script generated successfully.")
        return script_content
    except Exception as e:
        logger.exception("Error generating script: %s", e)
        raise ValueError("Error generating script") from e


@api.post("/generate_script/")
def generate_script_api(input_data: ScriptInput) -> str:
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
    script: Optional[str] = None,
    arxiv_id: Optional[str] = None,
) -> float:
    """Generate audio, SRT, and rich content JSON assets from a script.

    Args:
        script (str): The script content or the path to the script file.
        arxiv_id (str, optional): arXiv ID of the paper. Defaults to None.

    Returns:
        float: The total duration of the audio in seconds.

    Raises:
        ValueError: If there is an error reading the script or generating the assets.
    """

    if not script:
        script_default_path = Path(settings.TEMP_DIR) / Path(arxiv_id) / Path("script.txt")
        script_default_path = script_default_path.absolute()
        if not script_default_path.exists():
            raise ValueError("Script not found in %s", script_default_path.as_posix())
        script = script_default_path.read_text()


    try:
        mp3_output_path, srt_output_path, rich_output_path = create_directories(
            arxiv_id
        )

        logger.info("MP3 output: %s", mp3_output_path)
        logger.info("SRT output: %s", srt_output_path)
        logger.info("Rich content output: %s", rich_output_path)

        script_contents: list[Union[RichContent, Text]] = parse_script(script)

        temp_dir = settings.TEMP_DIR if settings.TEMP_DIR else None
        if temp_dir:
            logger.info("Using temporary directory: %s", temp_dir)

        script_contents = generate_audio_and_caption(
            script_contents=script_contents, temp_dir=temp_dir, arxiv_id=arxiv_id
        )
        script_contents = fill_rich_content_time(script_contents)

        rich_content_list = [
            item for item in script_contents if isinstance(item, RichContent)
        ]
        text_content_list = [item for item in script_contents if isinstance(item, Text)]

        if not text_content_list:
            raise ValueError("No text content found in the script")

        logger.info("Exporting assets...")
        export_mp3(text_content_list, mp3_output_path)
        export_srt(mp3_output_path, srt_output_path)
        export_rich_content_json(rich_content_list, rich_output_path, arxiv_id=arxiv_id)

        total_duration = get_total_duration(arxiv_id)
        logger.info(
            "Assets generated successfully. Total duration: %s seconds", total_duration
        )
        return total_duration
    except Exception as e:
        logger.exception("Error generating assets: %s", e)
        raise ValueError("Error generating assets") from e


def create_directories(arxiv_id: str) -> Tuple[Path, Path, Path]:
    """Create directories for output files and return the paths as Path objects.

    Args:
        arxiv_id (str): arXiv ID of the paper.

    Returns:
        Tuple[Path, Path, Path]: A tuple containing the paths for MP3, SRT, and rich content JSON output files.
    """
    base_dir = Path(settings.TEMP_DIR) / Path(arxiv_id)
    base_dir = base_dir.absolute()
    base_dir.mkdir(parents=True, exist_ok=True)

    mp3_output_path = base_dir / Path(settings.COMPOSITION_PROPS.audio_file_name)
    srt_output_path = base_dir / Path(settings.COMPOSITION_PROPS.subtitles_file_name)
    rich_output_path = base_dir / Path(
        settings.COMPOSITION_PROPS.rich_content_file_name
    )

    return mp3_output_path, srt_output_path, rich_output_path


@api.post("/generate_assets/")
def generate_assets_api(assets_input: AssetsInput) -> float:
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
@api.post("/generate_video/")
def generate_video(arxiv_id: str) -> None:
    """Generate a video from the processed script.

    Args:
        arxiv_id (str): arXiv ID of the paper.

    Raises:
        ValueError: If there is an error generating the video.
    """

    try:
        logger.info("Generating video for ArXiv paper %s...", arxiv_id)
        process_video(arxiv_id=arxiv_id)
        logger.info("Video generated successfully.")
    except Exception as e:
        logger.exception("Error generating video: %s", e)
        raise ValueError("Error generating video") from e


@cli.command("run_pipeline")
def run_pipeline(url: str) -> None:
    """Run the entire pipeline from URL to video.

    Args:
        url (str): The URL of the paper to process.

    Raises:
        ValueError: If there is an error in any step of the pipeline.
    """
    try:
        logger.info("Running pipeline for URL: %s", url)

        arxiv_id = get_arxiv_id_from_url(url)
        paper_markdown = generate_paper(url)
        logger.info("Paper content generated successfully.")

        script_content = generate_script(
            paper_markdown=paper_markdown, arxiv_id=arxiv_id
        )
        logger.info("Script content generated successfully.")

        logger.info("Generating assets...")
        generate_assets(script_content, arxiv_id=arxiv_id)
        logger.info("Assets generated successfully.")

        generate_video(arxiv_id=arxiv_id)
        logger.info("Video generated successfully.")
    except Exception as e:
        logger.exception("Error running pipeline: %s", e)
        raise ValueError(f"Error running pipeline for URL {url}") from e


if __name__ == "__main__":
    logger.info("Starting CLI...")
    try:
        cli()
    except Exception as e:
        logger.exception("Error starting CLI: %s", e)
        raise
