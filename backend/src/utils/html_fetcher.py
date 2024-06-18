""" HTMLFetcher class """

import logging

import requests

from backend.src.config import REQUESTS_TIMEOUT

from .url_sanitizer import URLSanitizer

logger = logging.getLogger(__name__)


class HTMLFetcher:
    """
    A class to fetch HTML content from a given URL.

    Methods:
        fetch(url: str) -> str: Fetch HTML content from a given URL.

    """

    @staticmethod
    def fetch(url: str) -> str:
        """
        Fetch HTML content from a given URL.

        Args:
            url (str): The URL to fetch HTML from.

        Returns:
            str: The HTML content as a string.

        Raises:
            ValueError: If there is an error fetching the HTML content.
        """
        sanitized_url = URLSanitizer.sanitize(url)

        try:
            response = requests.get(sanitized_url, timeout=REQUESTS_TIMEOUT)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            logger.error("Failed to fetch HTML from %s: %s", sanitized_url, e)
            raise ValueError(f"Failed to fetch HTML from {sanitized_url}") from e
