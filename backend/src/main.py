"""Main module for the backend application."""

import logging
from http import HTTPStatus
from pathlib import Path
from typing import Optional, Union

import typer
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

from backend.src.models import RichContent, Text
# from .utils.create_directories import create_directories

from backend.src.config import (
    DEFAULT_MP3_OUTPUT_PATH,
    DEFAULT_RICH_OUTPUT_PATH,
    DEFAULT_SRT_OUTPUT_PATH,
    DEFAULT_TEMP_DIR,
    DEFAULT_VIDEO_OUTPUT_PATH,
    LOG_FORMAT,
)
from .models import AssetsInput, ScriptInput
from .utils.audio_generation import generate_audio_and_caption
from .utils.content_timing import fill_rich_content_time
from .utils.generate_assets import generate_assets
from .utils.generate_paper import generate_paper
from .utils.generate_script import generate_script
from .utils.generate_video import generate_video
from .utils.json_exporter import export_rich_content_json
from .utils.mp3_exporter import export_mp3
from .utils.script_parsing import parse_script
from .utils.srt_exporter import export_srt

# Setup logging
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize CLI and API
cli = typer.Typer()
app = FastAPI()

def handle_error(e: Exception, context: str, is_cli: bool = False):
    logger.exception("Unexpected error %s: %s", context, e)
    if is_cli:
        raise typer.Exit(code=1)
    else:
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail=str(e))


@cli.command("generate_paper", help="Generate a paper from a given URL.")
@app.get("/generate_paper/", response_model=str)
def generate_paper_endpoint(url: str) -> str:
    """
    CLI command and API endpoint to generate a paper from a given URL.

    Args:
        url (str): The URL of the paper to process.

    Returns:
        str: The content of the processed paper.

    Raises:
        typer.Exit: If an error occurs during the process (CLI context).
        HTTPException: If an error occurs during the process (API context).
    """
    try:
        logger.info("Generating paper from URL: %s", url)
        paper_content = generate_paper(url)
        if typer.current_context().info_name:  # Check if CLI context
            typer.echo(paper_content)
        return paper_content
    except ValueError as ve:
        logger.error("ValueError: %s", ve)
        handle_error(ve, f"generating paper from URL {url}", is_cli=typer.current_context().info_name is not None)
    except Exception as e:
        handle_error(e, f"generating paper from URL {url}", is_cli=typer.current_context().info_name is not None)


@cli.command("generate_script", help="Generate a script from a given paper.")
@app.post("/generate_script/", response_model=str)
def generate_script(paper_input: ScriptInput, use_path: bool = True) -> str:
    """
    CLI command and API endpoint to generate a script from a given paper.

    Args:
        input (str or ScriptInput): The paper content or input parameters containing the paper content or path.
        use_path (bool, optional): Whether to treat `input` as a file path. Defaults to True.

    Returns:
        str: The generated script content.

    Raises:
        typer.Exit: If an error occurs during the process (CLI context).
        HTTPException: If an error occurs during the process (API context).
    """
    try:
        if isinstance(input, ScriptInput):
            paper_content = Path(input.paper).read_text() if input.use_path else input.paper
        else:
            paper_content = Path(input).read_text() if use_path else input

        logger.info("Generating script from paper: %s", input)
        script_content = generate_script(paper_content, paper_content)
        if typer.current_context().info_name:  # Check if CLI context
            typer.echo(script_content)
        return script_content
    except FileNotFoundError as fnfe:
        logger.error("FileNotFoundError: %s", fnfe)
        handle_error(fnfe, "generating script", is_cli=typer.current_context().info_name is not None)
    except Exception as e:
        handle_error(e, "generating script", is_cli=typer.current_context().info_name is not None)


@cli.command("generate_assets", help="Generate audio, SRT, and rich content JSON assets from a script.")
@app.post("/generate_assets/", response_model=float)
def generate_assets(
    script: Union[str, AssetsInput],
    use_path: bool = True,
    mp3_output: Path = DEFAULT_MP3_OUTPUT_PATH,
    srt_output: Path = DEFAULT_SRT_OUTPUT_PATH,
    rich_output: Path = DEFAULT_RICH_OUTPUT_PATH,
) -> float:
    """
    CLI command and API endpoint to generate audio, SRT, and rich content JSON assets from a script.

    Args:
        script (str or AssetsInput): The script content or input parameters containing the script content or path.
        use_path (bool, optional): Whether to treat `script` as a file path. Defaults to True.
        mp3_output (Path, optional): Path to save the MP3 output file. Defaults to DEFAULT_MP3_OUTPUT_PATH.
        srt_output (Path, optional): Path to save the SRT output file. Defaults to DEFAULT_SRT_OUTPUT_PATH.
        rich_output (Path, optional): Path to save the rich content JSON file. Defaults to DEFAULT_RICH_OUTPUT_PATH.

    Returns:
        float: The total duration of the audio in seconds.

    Raises:
        typer.Exit: If an error occurs during the process (CLI context).
        HTTPException: If an error occurs during the process (API context).
    """
    try:
        logger.info("Generating assets from script: %s", script)
        create_directories([mp3_output, srt_output, rich_output])

        if isinstance(script, AssetsInput):
            script_content = Path(script.script).read_text() if script.use_path else script.script
            mp3_output = script.mp3_output
            srt_output = script.srt_output
            rich_output = script.rich_output
        else:
            script_content = Path(script).read_text() if use_path else script

        script_contents = parse_script(script_content)
        script_contents = generate_audio_and_caption(script_contents, temp_dir=DEFAULT_TEMP_DIR)
        script_contents = fill_rich_content_time(script_contents)

        rich_content_list = [item for item in script_contents if isinstance(item, RichContent)]
        text_content_list = [item for item in script_contents if isinstance(item, Text)]

        export_mp3(text_content_list, mp3_output)
        export_srt(mp3_output, srt_output)
        export_rich_content_json(rich_content_list, rich_output)

        total_duration = text_content_list[-1].end if text_content_list and text_content_list[-1].end else 0
        if typer.current_context().info_name:  # Check if CLI context
            typer.echo(f"Total duration: {total_duration} seconds")
        return total_duration
    except FileNotFoundError as fnfe:
        logger.error("FileNotFoundError: %s", fnfe)
        handle_error(fnfe, "generating assets", is_cli=typer.current_context().info_name is not None)
    except Exception as e:
        handle_error(e, "generating assets", is_cli=typer.current_context().info_name is not None)


@cli.command("generate_video", help="Generate a video from the processed script.")
@app.post("/generate_video/", response_model=dict)
def generate_video(output_path: Optional[Path] = DEFAULT_VIDEO_OUTPUT_PATH) -> dict:
    """
    CLI command and API endpoint to generate a video from the processed script.

    Args:
        output_path (Optional[Path]): Path to save the output video file. Defaults to DEFAULT_VIDEO_OUTPUT_PATH.

    Returns:
        dict: A JSON response indicating success or failure.

    Raises:
        typer.Exit: If an error occurs during the process (CLI context).
        HTTPException: If an error occurs during the process (API context).
    """
    try:
        output_path = output_path if output_path else DEFAULT_VIDEO_OUTPUT_PATH
        logger.info("Generating video to %s", output_path)
        generate_video(output_path=Path(output_path), composition_props=None)
        if typer.current_context().info_name:  # Check if CLI context
            typer.echo("Video generated successfully")
        return {"message": "Video generated successfully"}
    except Exception as e:
        handle_error(e, "generating video", is_cli=typer.current_context().info_name is not None)


def create_directories(paths: list[Path]) -> None:
    """
    Create directories for given paths if they don't exist.

    Args:
        paths (list[Path]): List of paths to ensure directories exist.

    Raises:
        Exception: If an unexpected error occurs during directory creation.
    """
    for path in paths:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Directory created or already exists: %s", path.parent)
        except Exception as e:
            logger.exception("Error creating directory %s: %s", path.parent, e)
            raise


def main() -> None:
    """
    Main entry point for the CLI.
    """
    logger.info("Starting CLI...")
    try:
        cli()
    except Exception as e:
        handle_error(e, "starting CLI", is_cli=True)


if __name__ == "__main__":
    main()
