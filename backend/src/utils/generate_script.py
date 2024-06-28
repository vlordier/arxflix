""" Generate a video script for a research paper using OpenAI's GPT-4o model. """

import logging
import os

# Load environment variables
from pathlib import Path
from typing import Optional

import prompts

# Load settings
from dotenv import load_dotenv
from openai import OpenAI
from settings import settings

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Load settings

# # Validate settings
assert OPENAI_API_KEY, "OPENAI_API_KEY not set"
assert settings.OPENAI.model, "OPENAI_MODEL not set"

# Setup logging
logger = logging.getLogger(__name__)


# def correct_result_link(script: str, arxiv_id: str) -> str:
#     """
#     Correct generated links in a research paper script.

#     Args:
#         script (str): The script of a research paper.
#         arxiv_id (str): The arXiv ID of the paper.

#     Returns:
#         str: The corrected script with valid image links.
#     """

#     # Find all .png images in the line
#     png_images = re.findall(r"\b\S+\.png\b", tmp_line)
#     for image in png_images:
#         # Construct the new figure URL
#         base_url = f"https://arxiv.org/html/{arxiv_id}/"
#         new_image_url = f"{base_url}{image}"
#         logger.debug("Modified image URL: %s", new_image_url)

#         logger.debug("Verifying image URL: %s", new_image_url)
#         response = requests.head(
#             new_image_url, allow_redirects=True, timeout=settings.REQUESTS_TIMEOUT
#         )
#         if response.status_code == 200 and "image/png" in response.headers.get(
#             "Content-Type", ""
#         ):

#             # save the file to the local directory
#             local_image_path = f"images/{arxiv_id}/{image}"
#             os.makedirs(os.path.dirname(local_image_path), exist_ok=True)
#             logger.debug("Saving image to %s", local_image_path)

#             with open(local_image_path, "wb") as f:
#                 f.write(response.content)

#             if not os.path.exists(local_image_path):
#                 logger.error("Failed to save image to %s", local_image_path)
#                 raise ValueError("Failed to save image to local directory")

#             # replace the image in the script
#             script = script.replace(image, local_image_path)
#         else:
#             logger.debug("Image URL not valid: %s", new_image_url)
#             raise ValueError("Image URL not valid %s", new_image_url)
#     return script

#     # # remove empty lines
#     # split_script = [line for line in script.split("\n") if line.strip()]

#     # for line_idx, line in enumerate(split_script):
#     #     if "\\Figure: " in line and not line.startswith("https"):
#     #         tmp_line = line.replace("\\Figure:", "").strip()

#     #         # Construct the potential figure URL
#     #         base_url = f"https://arxiv.org/html/{arxiv_id}/"
#     #         figure_url = f"{base_url}{tmp_line.split('/images/')[-1]}"
#     #         logger.debug("Modified base URL: %s", modified_base_url)
#     #         logger.debug("Figure URL: %s", figure_url)

#     #         try:
#     #             figure_url = (
#     #                 figure_url.lstrip("/") if figure_url.startswith("/") else figure_url
#     #             )
#     #             logger.debug("Verifying image URL: %s", figure_url)
#     #             response = requests.head(
#     #                 figure_url, allow_redirects=True, timeout=settings.REQUESTS_TIMEOUT
#     #             )
#     #             if response.status_code == 200 and "image/png" in response.headers.get(
#     #                 "Content-Type", ""
#     #             ):
#     #                 split_script[line_idx] = f"\\Figure: {figure_url}"
#     #             else:
#     #                 figure_url = figure_url.replace("ar5iv.labs.", "")
#     #                 logger.debug("Trying to verify image URL: %s", figure_url)
#     #                 response = requests.head(
#     #                     figure_url,
#     #                     allow_redirects=True,
#     #                     timeout=settings.REQUESTS_TIMEOUT,
#     #                 )
#     #                 if (
#     #                     response.status_code == http.HTTPStatus.OK
#     #                     and "image/png" in response.headers.get("Content-Type", "")
#     #                 ):
#     #                     split_script[line_idx] = f"\\Figure: {figure_url}"
#     #         except requests.exceptions.RequestException as e:
#     #             logger.error("Failed to verify image URL: %s", e)

#     # return "\n".join(split_script)


def process_script(paper_markdown: str, arxiv_id: Optional[str]) -> str:
    """
    Generate a video script for a research paper using OpenAI's GPT-4o model.

    Args:
        paper_markdown (str): A research paper in markdown format (currently HTML).
        arxiv_id (Optional[str]): The arXiv ID of the paper.

    Returns:
        str: The generated video script.

    Raises:
        ValueError: If no result is returned from OpenAI.
    """

    script_file = Path(settings.TEMP_DIR / arxiv_id / settings.SCRIPT_NAME)
    if script_file.exists():
        logger.debug("Script already exists: %s", script_file)
        return script_file.read_text().strip()

    else:
        # logger.debug("Generating script from paper: %s", url)
        if not OPENAI_API_KEY and not isinstance(OPENAI_API_KEY, str):
            raise ValueError("OPENAI_API_KEY not set")
        if not settings.OPENAI.model and not isinstance(settings.OPENAI.model, str):
            raise ValueError("OPENAI_MODEL not set")

        logger.debug("Using OpenAI model: %s", settings.OPENAI.model)
        # logger.debug("Using OpenAI API key: %s", "********" + settings.OPENAI.api[-4:])

        if not paper_markdown.strip():
            raise ValueError("No paper provided")

        logger.debug(
            "Prompt paper: %s",
            prompts.prompt_summary.user_prompt.format(paper=paper_markdown),
        )

        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        response = openai_client.chat.completions.create(
            model=settings.OPENAI.model,
            messages=[
                {"role": "system", "content": prompts.prompt_summary.system_prompt},
                {
                    "role": "user",
                    "content": f"{prompts.prompt_summary.user_prompt}\\n{paper_markdown}",
                },
            ],
        )
        script = str(response.choices[0].message.content).strip()

        if not script:
            raise ValueError("No result returned from OpenAI.")

        # save the script to the local directory
        script_file.parent.mkdir(parents=True, exist_ok=True)
        script_file.write_text(script)

        return script
