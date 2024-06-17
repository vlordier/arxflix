from unittest.mock import Mock, patch

import pytest
import requests
from bs4 import BeautifulSoup
from utils.generate_paper import (
    convert_to_markdown,
    fetch_html,
    process_article,
    remove_section_by_class,
    replace_math_tags,
    sanitize_url,
    strip_attributes,
)


def test_sanitize_url():
    assert sanitize_url("https://example.com?query=123") == "https://example.com"
    assert sanitize_url("https://example.com#section") == "https://example.com"

    with pytest.raises(ValueError):
        sanitize_url("http://example.com")

    with pytest.raises(ValueError):
        sanitize_url("file://example.com")


@patch("requests.get")
def test_fetch_html(mock_get):
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.text = "<html></html>"
    mock_get.return_value = mock_response

    assert fetch_html("https://example.com") == "<html></html>"

    mock_get.side_effect = requests.exceptions.RequestException("Failed to fetch")
    with pytest.raises(ValueError):
        fetch_html("https://example.com")


def test_convert_to_markdown():
    soup = BeautifulSoup("<h1>Title</h1><p>Paragraph</p>", "html.parser")
    markdown = convert_to_markdown(soup)
    assert "# Title\n\nParagraph" in markdown


def test_replace_math_tags():
    html_content = '<math display="inline" alttext="x+y">x+y</math>'
    soup = BeautifulSoup(html_content, "html.parser")
    soup = replace_math_tags(soup)
    assert "$x+y$" in str(soup)


def test_remove_section_by_class():
    html_content = '<div class="ltx_bibliography">Bibliography</div>'
    soup = BeautifulSoup(html_content, "html.parser")
    soup = remove_section_by_class(soup, "ltx_bibliography")
    assert "Bibliography" not in str(soup)


def test_strip_attributes():
    html_content = '<img src="image.jpg" alt="image" width="500">'
    soup = BeautifulSoup(html_content, "html.parser")
    soup = strip_attributes(soup)
    assert 'src="image.jpg"' in str(soup)
    assert 'alt="image"' not in str(soup)
    assert 'width="500"' not in str(soup)


@patch("utils.generate_paper.fetch_html")
def test_process_article(mock_fetch_html):
    mock_fetch_html.return_value = "<article><h1>Title</h1><p>Content</p></article>"
    url = "https://example.com"
    result = process_article(url)
    assert "# Title\n\nContent" in result

    mock_fetch_html.return_value = "<div>No article here</div>"
    with pytest.raises(ValueError):
        process_article("https://example.com")
