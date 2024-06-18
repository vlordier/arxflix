""" This module contains a class to convert a BeautifulSoup object to Markdown. """

from typing import Any

from bs4 import BeautifulSoup
from markdownify import MarkdownConverter


class CustomMarkdownConverter(MarkdownConverter):
    """Custom MarkdownConverter class to override convert_h1 method."""


class MarkdownConverterUtil:
    @staticmethod
    def convert(soup: BeautifulSoup, **options: Any) -> str:
        """
        Convert a BeautifulSoup object to Markdown.

        Args:
            soup (BeautifulSoup): The BeautifulSoup object to convert.
            options (Any): Additional options for MarkdownConverter.

        Returns:
            str: The converted Markdown string.
        """
        return CustomMarkdownConverter(**options).convert_soup(soup)
