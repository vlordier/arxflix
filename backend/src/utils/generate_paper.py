"""
This module contains functions to generate a markdown file from a given URL.
"""

import logging
import re
from typing import Any

import requests  # type: ignore
import tldextract
from bs4 import BeautifulSoup, Tag  # type: ignore
from markdownify import MarkdownConverter  # type: ignore
from models import ArxivPaper  # type: ignore
from settings import settings  # type: ignore

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_arxiv_id_from_url(url: str) -> str:
    """
    Extract the arXiv ID from a given arXiv URL.

    Args:
        url (str): The arXiv URL.

    Returns:
        str: The arXiv ID.

    Raises:
        ValueError: If the URL is not a valid arXiv URL.
    """
    url = url.strip().split("?")[0].split("#")[0]

    if not url.startswith("https://"):
        raise ValueError("URL must start with 'https://'.")

    domain = tldextract.extract(url).domain
    if domain != "arxiv":
        raise ValueError("Only arXiv URLs are allowed.")

    pattern = re.compile(r"/(abs|pdf|html)/(\d{4}\.\d{4,5}v\d+)")
    match = pattern.search(url)

    if match:
        return match.group(2)
    else:
        logger.error("Unexpected error processing URL: %s", url)
        raise ValueError(f"Unexpected error processing URL: {url}")


def fetch_html(arxiv_id: str) -> str:
    """
    Fetch HTML content from a given arXiv ID.

    Args:
        arxiv_id (str): The arXiv ID.

    Returns:
        str: The HTML content as a string.

    Raises:
        ValueError: If there is an error fetching the HTML content.
    """
    sanitized_url = f"https://arxiv.org/html/{arxiv_id}"
    logger.debug("Fetching HTML from %s", sanitized_url)

    try:
        response = requests.get(sanitized_url, timeout=settings.REQUESTS_TIMEOUT)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        logger.error("Failed to fetch HTML from %s: %s", sanitized_url, e)
        raise ValueError(f"Failed to fetch HTML from {sanitized_url}") from e


def convert_to_markdown(soup: BeautifulSoup, **options: Any) -> str:
    """
    Convert a BeautifulSoup object to Markdown.

    Args:
        soup (BeautifulSoup): The BeautifulSoup object to convert.
        options (Any): Additional options for MarkdownConverter.

    Returns:
        str: The converted Markdown string.
    """
    return MarkdownConverter(**options).convert_soup(soup)


def replace_math_tags(soup: BeautifulSoup) -> BeautifulSoup:
    """
    Replace math tags in the BeautifulSoup object with corresponding LaTeX strings.

    Args:
        soup (BeautifulSoup): The BeautifulSoup object containing math tags.

    Returns:
        BeautifulSoup: The modified BeautifulSoup object.
    """
    for math_tag in soup.find_all("math"):
        display = math_tag.attrs.get("display")
        latex = math_tag.attrs.get("alttext")

        if latex:
            latex = f"${latex}$" if display == "inline" else f"$$ {latex} $$"
            span_tag = soup.new_tag("span")
            span_tag.string = latex
            math_tag.replace_with(span_tag)
    return soup


def remove_section_by_class(soup: BeautifulSoup, class_name: str) -> BeautifulSoup:
    """
    Remove a section from the BeautifulSoup object by its class name.

    Args:
        soup (BeautifulSoup): The BeautifulSoup object.
        class_name (str): The class name of the section to remove.

    Returns:
        BeautifulSoup: The modified BeautifulSoup object.
    """
    section = soup.find("div", class_=class_name)
    if section and isinstance(section, Tag):
        section.decompose()
    return soup


def strip_attributes(soup: BeautifulSoup) -> BeautifulSoup:
    """
    Strip all attributes from tags in a BeautifulSoup object, except for 'src' attributes.

    Args:
        soup (BeautifulSoup): The BeautifulSoup object to process.

    Returns:
        BeautifulSoup: The modified BeautifulSoup object with only 'src' attributes retained.
    """
    for tag in soup.find_all(True):
        tag.attrs = {key: value for key, value in tag.attrs.items() if key == "src"}
        if tag.name == "img" and "src" in tag.attrs:
            tag.attrs["src"] = tag.attrs["src"].strip()
    return soup


def save_markdown(content: str, path: str) -> None:
    """
    Save the given content to a markdown file.

    Args:
        content (str): The markdown content to save.
        path (str): The file path where the content should be saved.
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.debug("Saved article to %s", path)


def process_article(url: str) -> ArxivPaper:
    """
    Process an article from a given URL and save it as a markdown file.

    Args:
        url (str): The URL of the article.

    Returns:
        ArxivPaper: The processed article as an ArxivPaper object.

    Raises:
        ValueError: If no article is found in the HTML content.
    """
    arxiv_id = get_arxiv_id_from_url(url)
    arxiv_paper = ArxivPaper(arxiv_id=arxiv_id)
    temp_dir = settings.TEMP_DIR / arxiv_id
    temp_dir.mkdir(parents=True, exist_ok=True)

    arxiv_md_path = temp_dir / "article.md"
    arxiv_paper.path = arxiv_md_path

    logger.debug("Processing article with arXiv ID: %s", arxiv_id)

    if arxiv_md_path.exists():
        logger.debug("Article already processed: %s", arxiv_md_path)
        with open(arxiv_md_path, "r", encoding="utf-8") as f:
            md_content = f.read().strip()
        if md_content:
            arxiv_paper.markdown = md_content
            return arxiv_paper

    html_content = fetch_html(arxiv_id)
    soup = BeautifulSoup(html_content, "html.parser")

    soup = replace_math_tags(soup)
    soup = remove_section_by_class(soup, "ltx_bibliography")
    soup = remove_section_by_class(soup, "ltx_appendix")
    soup = remove_section_by_class(soup, "ltx_para")

    article = soup.find("article")
    if not article:
        raise ValueError("No article found in the HTML content.")

    if isinstance(article, Tag):
        article = remove_section_by_class(
            BeautifulSoup(str(article), "html.parser"), "ltx_authors"
        )
    if isinstance(article, BeautifulSoup):
        article = strip_attributes(article)
        markdown_article = (
            convert_to_markdown(article, wrap_width=True, strip=["button"])
            .replace("\n\n\n", "\n\n")
            .strip()
        )
    else:
        raise ValueError("Article is not a BeautifulSoup object.")

    save_markdown(markdown_article, arxiv_md_path)

    arxiv_paper.markdown = markdown_article
    return arxiv_paper
