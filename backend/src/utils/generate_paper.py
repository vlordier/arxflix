"""
This module contains functions to generate a markdown file from a given URL.
"""

import logging
from typing import Any

import requests  # type: ignore
import tldextract
from bs4 import BeautifulSoup, Tag  # type: ignore
from markdownify import MarkdownConverter  # type: ignore
from src.settings import settings  # type: ignore

# Setup logging
logger = logging.getLogger(__name__)


def sanitize_url(url: str) -> str:
    """
    Sanitize a URL by removing any query parameters and enforcing HTTPS.

    Args:
        url (str): The URL to sanitize.

    Returns:
        str: The sanitized URL.

    Raises:
        ValueError: If the URL is not HTTPS or contains a file:// scheme.
    """
    url = url.strip()

    domain = tldextract.extract(url).registered_domain
    if domain not in settings.ALLOWED_DOMAINS:
        raise ValueError(f"Domain {domain} is not allowed.")

    if url.startswith("file://"):
        raise ValueError("File URLs are not allowed.")

    if not url.startswith("https"):
        raise ValueError("Only HTTPS URLs are allowed.")

    url = url.split("?")[0].split("#")[0]
    return url


def fetch_html(url: str) -> str:
    """
    Fetch HTML content from a given URL.

    Args:
        url (str): The URL to fetch HTML from.

    Returns:
        str: The HTML content as a string.

    Raises:
        ValueError: If there is an error fetching the HTML content.
    """
    sanitized_url = sanitize_url(url)

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
    math_tags = soup.find_all("math")
    for math_tag in math_tags:
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
    return soup


def process_article(url: str) -> str:
    """
    Process an article from a given URL and save it as a markdown file.

    Args:
        url (str): The URL of the article.

    Returns:
        str: The processed article as a markdown string.

    Raises:
        ValueError: If no article is found in the HTML content.
    """
    html_content = fetch_html(url)
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
        markdown_article = convert_to_markdown(
            article, wrap_width=True, strip=["button"]
        )
    else:
        raise ValueError("Article is not a BeautifulSoup object.")
    markdown_article = markdown_article.replace("\n\n\n", "\n\n").replace(
        "\n\n\n", "\n\n"
    )
    return markdown_article
