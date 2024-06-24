""" Generate a video script for a research paper using OpenAI's GPT-4o model. """

import http
import logging
import os

import requests  # type: ignore

# Load settings
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
from src import prompts
from src.settings import settings

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Load settings

# # Validate settings
assert OPENAI_API_KEY, "OPENAI_API_KEY not set"
assert settings.OPENAI.model, "OPENAI_MODEL not set"

# Setup logging
logger = logging.getLogger(__name__)


def correct_result_link(script: str, url: str) -> str:
    """
    Correct generated links in a research paper script.

    Args:
        script (str): The script of a research paper.
        url (str): The base URL of the research paper (can contain "/html/").

    Returns:
        str: The corrected script with valid image links.
    """
    # if "ar5iv" not in url:
    #     tmp_url = url.split("/")
    #     url = (
    #         "https://ar5iv.labs.arxiv.org/html/" + tmp_url[-1]
    #         if tmp_url[-1] != ""
    #         else "https://ar5iv.labs.arxiv.org/html/" + tmp_url[-2]
    #     )

    # remove empty lines
    split_script = [line for line in script.split("\n") if line.strip()]

    for line_idx, line in enumerate(split_script):
        if "\\Figure: " in line and not line.startswith("https"):
            tmp_line = line.replace("\\Figure: ", "")

            # Construct the potential figure URL
            modified_base_url = url.split("/html/")[0]
            figure_url = f"{modified_base_url}/images/{tmp_line.split('/images/')[-1]}"

            try:
                figure_url = (
                    figure_url.lstrip("/") if figure_url.startswith("/") else figure_url
                )
                logger.info("Verifying image URL: %s", figure_url)
                response = requests.head(
                    figure_url, allow_redirects=True, timeout=settings.REQUESTS_TIMEOUT
                )
                if response.status_code == 200 and "image/png" in response.headers.get(
                    "Content-Type", ""
                ):
                    split_script[line_idx] = f"\\Figure: {figure_url}"
                else:
                    figure_url = figure_url.replace("ar5iv.labs.", "")
                    logger.info("Trying to verify image URL: %s", figure_url)
                    response = requests.head(
                        figure_url,
                        allow_redirects=True,
                        timeout=settings.REQUESTS_TIMEOUT,
                    )
                    if (
                        response.status_code == http.HTTPStatus.OK
                        and "image/png" in response.headers.get("Content-Type", "")
                    ):
                        split_script[line_idx] = f"\\Figure: {figure_url}"
            except requests.exceptions.RequestException as e:
                logger.error("Failed to verify image URL: %s", e)

    return "\n".join(split_script)


def process_script(paper: str, url: str) -> str:
    """
    Generate a video script for a research paper using OpenAI's GPT-4o model.

    Args:
        paper (str): A research paper in markdown format (currently HTML).
        url (str): The URL of the paper.

    Returns:
        str: The generated video script.

    Raises:
        ValueError: If no result is returned from OpenAI.
    """
    logger.info("Generating script from paper: %s", url)
    if not OPENAI_API_KEY and not isinstance(OPENAI_API_KEY, str):
        raise ValueError("OPENAI_API_KEY not set")
    if not settings.OPENAI.model and not isinstance(settings.OPENAI.model, str):
        raise ValueError("OPENAI_MODEL not set")

    logger.info("Using OpenAI model: %s", settings.OPENAI.model)
    # logger.info("Using OpenAI API key: %s", "********" + settings.OPENAI.api[-4:])

    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    response = openai_client.chat.completions.create(
        model=settings.OPENAI.model,
        messages=[
            {"role": "system", "content": prompts.prompt_summary.system_prompt},
            {"role": "user", "content": prompts.prompt_summary.format(paper)},
        ],
    )
    result = response.choices[0].message.content

    if not result:
        raise ValueError("No result returned from OpenAI.")

    script = correct_result_link(result, url)
    return script
