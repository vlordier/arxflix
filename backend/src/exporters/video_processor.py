# video_processor.py

import json
import logging
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from backend.src.config import (
    DEFAULT_VIDEO_OUTPUT_PATH,
    LOGGING_LEVEL,
    REMOTION_COMPOSITION_ID,
    REMOTION_CONCURRENCY,
    REMOTION_ROOT_PATH,
)
from backend.src.models import CompositionProps

# Setup logging
logging.basicConfig(level=LOGGING_LEVEL)
logger = logging.getLogger(__name__)


class VideoProcessor:
    """
    A class to process videos using Remotion.

    Attributes:
        root_path (Path): The path to the root directory of the Remotion project
        composition_id (str): The ID of the composition to render the video with
        concurrency (int): The number of concurrent rendering processes to use

    Methods:
        _check_configuration(): Checks the configuration of the video processor
        _check_setup(): Checks the setup of the video processor
        process_video(output_path: Path, composition_props: CompositionProps): Processes the video

    """

    def __init__(
        self,
        root_path: Path = REMOTION_ROOT_PATH,
        composition_id: str = REMOTION_COMPOSITION_ID,
        concurrency: int = REMOTION_CONCURRENCY,
    ):
        self.root_path = root_path
        self.composition_id = composition_id
        self.concurrency = concurrency
        self._check_configuration()
        self._check_setup()

    def _check_configuration(self):
        if not self.root_path.exists():
            raise ValueError(f"Remotion root path '{self.root_path}' does not exist.")
        if not self.composition_id:
            raise ValueError("Remotion composition ID is not set.")
        if self.concurrency <= 0:
            raise ValueError("Remotion concurrency must be greater than 0.")

    def _check_setup(self):
        try:
            result = subprocess.run(
                ["npx", "--version"], capture_output=True, text=True, check=True
            )
            logger.info("npx version: %s", result.stdout.strip())
        except Exception as exc:
            logger.error("npx is not installed or not found in the system PATH.")
            raise exc

        # Add any additional package checks here
        try:
            subprocess.run(
                ["npx", "remotion", "--help"],
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError:
            logger.error("Remotion is not installed or not set up correctly.")
            raise EnvironmentError("Remotion is not installed or not set up correctly.")

    def process_video(
        self, output_path: Optional[Path], composition_props: Optional[CompositionProps]
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
            output_path = DEFAULT_VIDEO_OUTPUT_PATH

        if not output_path.exists():
            output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            command = [
                "npx",
                "remotion",
                "render",
                self.root_path.absolute().as_posix(),
                "--props",
                json.dumps(asdict(composition_props)),
                "--compositionId",
                self.composition_id,
                "--concurrency",
                str(self.concurrency),
                "--output",
                output_path.absolute().as_posix(),
            ]
            logger.info("Running command: %s", " ".join(command))
            subprocess.run(command, check=True)
            logger.info("Video processing completed successfully.")
        except subprocess.CalledProcessError as e:
            logger.error("Video processing failed: %s", e)
            raise
        except Exception as e:
            logger.error("An unexpected error occurred: %s", e)
            raise
