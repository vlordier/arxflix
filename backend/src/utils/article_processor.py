# article_processor.py

import logging

from bs4 import BeautifulSoup, Tag

from .html_fetcher import HTMLFetcher
from .html_processor import HTMLProcessor
from .markdown_converter import MarkdownConverterUtil

logger = logging.getLogger(__name__)


class ArticleProcessor:
    """
    A class for processing articles from URLs.

    """

    @staticmethod
    def process(url: str) -> str:
        """
        Process an article from a given URL and return it as a markdown string.

        Args:
            url (str): The URL of the article.

        Returns:
            str: The processed article as a markdown string.

        Raises:
            ValueError: If no article is found in the HTML content.
        """
        html_content = HTMLFetcher.fetch(url)
        soup = BeautifulSoup(html_content, "html.parser")

        soup = HTMLProcessor.replace_math_tags(soup)
        soup = HTMLProcessor.remove_section_by_class(soup, "ltx_bibliography")
        soup = HTMLProcessor.remove_section_by_class(soup, "ltx_appendix")
        soup = HTMLProcessor.remove_section_by_class(soup, "ltx_para")

        article = soup.find("article")
        if not article:
            raise ValueError("No article found in the HTML content.")

        if isinstance(article, Tag):
            article = HTMLProcessor.remove_section_by_class(
                BeautifulSoup(str(article), "html.parser"), "ltx_authors"
            )
        if isinstance(article, BeautifulSoup):
            article = HTMLProcessor.strip_attributes(article)
            markdown_article = MarkdownConverterUtil.convert(
                article, wrap_width=True, strip=["button"]
            )
        else:
            raise ValueError("Article is not a BeautifulSoup object.")
        markdown_article = markdown_article.replace("\n\n\n", "\n\n").replace(
            "\n\n\n", "\n\n"
        )
        return markdown_article
