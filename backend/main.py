"""Main module for the backend application."""

import logging
from http import HTTPStatus
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

from backend.config import (
    DEFAULT_MP3_OUTPUT_PATH,
    DEFAULT_RICH_OUTPUT_PATH,
    DEFAULT_SRT_OUTPUT_PATH,
    DEFAULT_TEMP_DIR,
    DEFAULT_VIDEO_OUTPUT_PATH,
    LOG_FORMAT,
)
from backend.models import AssetsInput, RichContent, ScriptInput, Text
from backend.utils.generate_assets import (
    export_mp3,
    export_rich_content_json,
    export_srt,
    fill_rich_content_time,
    generate_audio_and_caption,
    parse_script,
)
from backend.utils.generate_paper import process_article
from backend.utils.generate_script import process_script
from backend.utils.generate_video import process_video

# Setup logging
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize CLI and API
cli = typer.Typer()
api = FastAPI()


@cli.command("generate_paper")
def generate_paper(url: str) -> None:  # dead: disable
    """
    CLI command to generate a paper from a given URL.

    Args:
        url (str): The URL of the paper to process.

    Raises:
        typer.Exit: If an error occurs during the process.
        Exception: If an unexpected error occurs during the process.
    """
    try:
        logger.info("Generating paper from URL: %s", url)
        paper_content = process_article(url)
        typer.echo(paper_content)
    except ValueError as ve:
        logger.error("ValueError: %s", ve)
        raise typer.Exit(code=1)
    except Exception as e:
        logger.exception("Unexpected error generating paper from URL %s: %s", url, e)
        raise typer.Exit(code=1)


@api.get("/generate_paper/", response_model=str)
def generate_paper_api(url: str) -> str:  # dead: disable
    """
    API endpoint to generate a paper from a given URL.

    Args:
        url (str): The URL of the paper to process.

    Returns:
        str: The content of the processed paper.

    Raises:
        HTTPException: If an error occurs during the process.
        Exception: If an unexpected error occurs during the process.
    """
    try:
        logger.info("Generating paper from URL: %s", url)
        return process_article(url)
    except Exception as e:
        logger.exception("Unexpected error generating paper from URL %s: %s", url, e)
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail=str(e))


@cli.command("generate_script")
def generate_script(paper: str, use_path: bool = True) -> None:
    """
    CLI command to generate a script from a given paper.

    Args:
        paper (str): The paper content or the path to the paper file.
        use_path (bool, optional): Whether to treat `paper` as a file path. Defaults to True.

    Raises:
        typer.Exit: If an error occurs during the process.
        Exception: If an unexpected error occurs during the process.
    """
    try:
        logger.info("Generating script from paper: %s", paper)
        paper_content = Path(paper).read_text() if use_path else paper
        script_content = process_script(paper_content, paper)
        typer.echo(script_content)
    except FileNotFoundError as fnfe:
        logger.error("FileNotFoundError: %s", fnfe)
        raise typer.Exit(code=1)
    except Exception as e:
        logger.exception("Unexpected error generating script: %s", e)
        raise typer.Exit(code=1)


@api.post("/generate_script/", response_model=str)
def generate_script_api(input: ScriptInput) -> str:  # dead: disable
    """
    API endpoint to generate a script from a given paper.

    Args:
        input (ScriptInput): Input parameters containing the paper content or path.

    Returns:
        str: The generated script content.

    Raises:
        HTTPException: If an error occurs during the process.
        Exception: If an unexpected error occurs during the process.
    """
    try:
        return generate_script(input.paper, input.use_path)
    except Exception as e:
        logger.exception("Unexpected error generating script: %s", e)
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail=str(e))


@cli.command("generate_assets")
def generate_assets(
    script: str,
    use_path: bool = True,
    mp3_output: Path = DEFAULT_MP3_OUTPUT_PATH,
    srt_output: Path = DEFAULT_SRT_OUTPUT_PATH,
    rich_output: Path = DEFAULT_RICH_OUTPUT_PATH,
) -> None:
    """
    CLI command to generate audio, SRT, and rich content JSON assets from a script.

    Args:
        script (str): The script content or the path to the script file.
        use_path (bool, optional): Whether to treat `script` as a file path. Defaults to True.
        mp3_output (Path, optional): Path to save the MP3 output file. Defaults to DEFAULT_MP3_OUTPUT_PATH.
        srt_output (Path, optional): Path to save the SRT output file. Defaults to DEFAULT_SRT_OUTPUT_PATH.
        rich_output (Path, optional): Path to save the rich content JSON file. Defaults to DEFAULT_RICH_OUTPUT_PATH.

    Raises:
        typer.Exit: If an error occurs during the process.
        Exception: If an unexpected error occurs during the process.
    """
    try:
        logger.info("Generating assets from script: %s", script)
        create_directories([mp3_output, srt_output, rich_output])

        script_content = Path(script).read_text() if use_path else script

        script_contents = parse_script(script_content)
        script_contents = generate_audio_and_caption(
            script_contents, temp_dir=DEFAULT_TEMP_DIR
        )
        script_contents = fill_rich_content_time(script_contents)

        rich_content_list = [
            item for item in script_contents if isinstance(item, RichContent)
        ]
        text_content_list = [item for item in script_contents if isinstance(item, Text)]

        export_mp3(text_content_list, mp3_output)
        export_srt(mp3_output, srt_output)
        export_rich_content_json(rich_content_list, rich_output)

        total_duration = (
            text_content_list[-1].end
            if text_content_list and text_content_list[-1].end
            else 0
        )
        typer.echo(f"Total duration: {total_duration} seconds")
    except FileNotFoundError as fnfe:
        logger.error("FileNotFoundError: %s", fnfe)
        raise typer.Exit(code=1)
    except Exception as e:
        logger.exception("Unexpected error generating assets: %s", e)
        raise typer.Exit(code=1)


@api.post("/generate_assets/", response_model=float)
def generate_assets_api(assets_input: AssetsInput) -> float:  # dead: disable
    """
    API endpoint to generate assets from a script.

    Args:
        assets_input (AssetsInput): Input parameters containing the script content or path and output file paths.

    Returns:
        float: The total duration of the audio in seconds.

    Raises:
        HTTPException: If an error occurs during the process.
        Exception: If an unexpected error occurs during the process.
    """
    try:
        return generate_assets(
            script=assets_input.script,
            use_path=assets_input.use_path,
            mp3_output=assets_input.mp3_output,
            srt_output=assets_input.srt_output,
            rich_output=assets_input.rich_output,
        )
    except Exception as e:
        logger.exception("Unexpected error generating assets: %s", e)
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail=str(e))


@cli.command("generate_video")
def generate_video(  # dead: disable
    output_path: Optional[Path] = DEFAULT_VIDEO_OUTPUT_PATH,
) -> None:
    """
    CLI command to generate a video from the processed script.

    Args:
        output_path (Optional[Path]): Path to save the output video file. Defaults to DEFAULT_VIDEO_OUTPUT_PATH.

    Raises:
        typer.Exit: If an error occurs during the process.
        Exception: If an unexpected error occurs during the process.
    """
    try:
        output_path = output_path if output_path else DEFAULT_VIDEO_OUTPUT_PATH
        logger.info("Generating video to %s", output_path)
        process_video(output_path=Path(output_path), composition_props=None)
        typer.echo("Video generated successfully")
    except Exception as e:
        logger.exception("Unexpected error generating video: %s", e)
        raise typer.Exit(code=1)


@api.post("/generate_video/", response_model=dict)
def generate_video_api(  # dead: disable
    output_path: Optional[Path] = DEFAULT_VIDEO_OUTPUT_PATH,
) -> dict:
    """
    API endpoint to generate a video from the processed script.

    Args:
        output_path (Optional[Path]): Path to save the output video file. Defaults to DEFAULT_VIDEO_OUTPUT_PATH.

    Returns:
        dict: A JSON response indicating success or failure.

    Raises:
        HTTPException: If an error occurs during the process.
        Exception: If an unexpected error occurs during the process.
    """
    try:
        output_path = output_path if output_path else DEFAULT_VIDEO_OUTPUT_PATH
        logger.info("Generating video to %s", output_path)
        process_video(output_path=Path(output_path), composition_props=None)
        return {"message": "Video generated successfully"}
    except Exception as e:
        logger.exception("Unexpected error generating video: %s", e)
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail=str(e))


def create_directories(paths: list[Path]) -> None:
    """
    Create directories for given paths if they don't exist.

    Args:
        paths (list[Path]): List of paths to ensure directories exist.
    """
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    """
    Main entry point for the CLI.
    """
    logger.info("Starting CLI...")
    try:
        cli()
    except Exception as e:
        logger.exception("Error starting CLI: %s", e)
        raise


if __name__ == "__main__":
    main()
