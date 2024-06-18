""" This module contains a class for sanitizing URLs. """


class URLSanitizer:
    """
    This class provides methods for sanitizing URLs.
    """

    @staticmethod
    def sanitize(url: str) -> str:
        """
        Sanitize a URL by removing any query parameters and enforcing HTTPS.

        Args:
            url (str): The URL to sanitize.

        Returns:
            str: The sanitized URL.

        Raises:
            ValueError: If the URL is not HTTPS or contains a file:// scheme.
        """

        if not isinstance(url, str):
            raise ValueError("URL must be a string.")

        # Remove trailing and leading whitespace
        url = url.strip()
        if not url:
            raise ValueError("URL cannot be empty.")

        if url.startswith("file://"):
            raise ValueError("File URLs are not allowed.")

        if not url.startswith("https"):
            raise ValueError("Only HTTPS URLs are allowed.")

        if url.endswith("/"):
            url = url[:-1]

        # Remove query parameters and fragments
        url = url.split("?")[0].split("#")[0]
        return url
