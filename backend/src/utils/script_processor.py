""" This module contains the ScriptProcessor class, which is responsible for generating a video script for a research paper using OpenAI's GPT-4 model. """

import logging

from link_corrector import LinkCorrector
from openai_client import OpenAIClient

from backend.config import OPENAI_MODEL, SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class ScriptProcessor:
    """This class provides methods for generating a video script for a research paper using OpenAI's GPT-4 model."""

    def __init__(self):
        self.openai_client = OpenAIClient()
        self.link_corrector = LinkCorrector()

    def process_script(self, paper: str, url: str) -> str:
        """
        Generate a video script for a research paper using OpenAI's GPT-4 model.

        Args:
            paper (str): A research paper in markdown format (currently HTML).
            url (str): The URL of the paper.

        Returns:
            str: The generated video script.

        Raises:
            ValueError: If no result is returned from OpenAI.
        """
        result = self.openai_client.generate_script(
            model=OPENAI_MODEL, system_prompt=SYSTEM_PROMPT, user_content=paper
        )
        script = self.link_corrector.correct(result, url)
        return script
