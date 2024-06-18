import json
import logging
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from backend.src.config import (
    REMOTION_COMPOSITION_ID,
    REMOTION_CONCURRENCY,
    REMOTION_ROOT_PATH,
)
from backend.src.models import CompositionProps

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_video(
    output_path: Optional[Path],
    composition_props: Optional[CompositionProps],
) -> None:
    """
    Processes the video by running the Remotion render command with the specified properties.

    Args:
        output_path (Path): The path to save the output video.
        composition_props (CompositionProps): The properties of the video composition.

    Raises:
        subprocess.CalledProcessError: If the video processing fails.
        Exception: If any other unexpected error occurs.

    """

    if composition_props is None:
        composition_props = CompositionProps()

    if output_path is None:
        output_path = Path("frontend/public/output.mp4")

    try:
        command = [
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
            output_path.absolute().as_posix(),
        ]
        logger.info("Running command: %s", " ".join(command))
        subprocess.run(command, check=True)
        logger.info("Video processing completed successfully.")
    except subprocess.CalledProcessError as e:
        logger.error("Video processing failed: %s", e)
        raise
