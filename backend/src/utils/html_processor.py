""" This module contains the HTMLProcessor class, which provides methods for processing HTML content. """

from bs4 import BeautifulSoup, Tag


class HTMLProcessor:
    """
    A class that provides methods for processing HTML content.

    Attributes:
        soup (BeautifulSoup): The BeautifulSoup object containing the HTML content.

    Methods:
        replace_math_tags(soup): Replace math tags in the BeautifulSoup object with corresponding LaTeX strings.
        remove_section_by_class(soup, class_name): Remove a section from the BeautifulSoup object by its class name.
        strip_attributes(soup): Strip all attributes from tags in a BeautifulSoup object, except for 'src' attributes.

    """

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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
