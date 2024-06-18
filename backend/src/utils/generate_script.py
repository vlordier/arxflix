""" Generate a video script for a research paper using OpenAI's GPT-4o model. """

import logging

import requests  # type: ignore
from openai import OpenAI

from backend.config import OPENAI_API_KEY, OPENAI_MODEL, SYSTEM_PROMPT

# Setup logging
logging.basicConfig(level=logging.INFO)
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
    if "ar5iv" not in url:
        tmp_url = url.split("/")
        url = (
            "https://ar5iv.labs.arxiv.org/html/" + tmp_url[-1]
            if tmp_url[-1] != ""
            else "https://ar5iv.labs.arxiv.org/html/" + tmp_url[-2]
        )

    split_script = script.split("\n")

    for line_idx, line in enumerate(split_script):
        if "\\Figure: " in line and not line.startswith("https"):
            tmp_line = line.replace("\\Figure: ", "")

            # Construct the potential figure URL
            if "/html/" in tmp_line:
                modified_base_url = url.split("/html/")[0]
                figure_url = f"{modified_base_url}{tmp_line}"
            else:
                figure_url = f"{url.rstrip('/')}{tmp_line.lstrip('/')}"

            try:
                response = requests.head(figure_url)
                if response.status_code == 200 and "image/png" in response.headers.get(
                    "Content-Type", ""
                ):
                    split_script[line_idx] = f"\\Figure: {figure_url}"
                else:
                    figure_url = figure_url.replace("ar5iv.labs.", "")
                    response = requests.head(figure_url)
                    if (
                        response.status_code == 200
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
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": paper},
        ],
    )
    result = response.choices[0].message.content

    if not result:
        raise ValueError("No result returned from OpenAI.")

    script = correct_result_link(result, url)
    return script
