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


# Mock settings
class MockSettings:
    ALLOWED_DOMAINS = ["example.com"]
    REQUESTS_TIMEOUT = 10


settings = MockSettings()


@pytest.mark.parametrize(
    "url, expected",
    [
        ("https://example.com/path?query=123#fragment", "https://example.com/path"),
        ("https://example.com/path?query=123", "https://example.com/path"),
        ("https://example.com/path#fragment", "https://example.com/path"),
        ("https://example.com/path", "https://example.com/path"),
    ],
)
def test_sanitize_url(url, expected):
    assert sanitize_url(url) == expected


@pytest.mark.parametrize(
    "url, exception_message",
    [
        ("http://example.com", "Only HTTPS URLs are allowed."),
        ("file://path/to/file", "File URLs are not allowed."),
        ("https://invalid.com", "Domain invalid.com is not allowed."),
    ],
)
def test_sanitize_url_exceptions(url, exception_message):
    with pytest.raises(ValueError, match=exception_message):
        sanitize_url(url)


@patch("utils.generate_paper.requests.get")
def test_fetch_html(mock_get):
    mock_response = Mock()
    mock_response.text = "<html></html>"
    mock_get.return_value = mock_response

    url = "https://example.com/path"
    html_content = fetch_html(url)
    assert html_content == "<html></html>"

    mock_get.assert_called_once_with(url, timeout=settings.REQUESTS_TIMEOUT)

    mock_get.side_effect = requests.exceptions.RequestException
    with pytest.raises(ValueError, match=f"Failed to fetch HTML from {url}"):
        fetch_html(url)


@pytest.mark.parametrize(
    "html, expected",
    [
        (
            '<math display="inline" alttext="x^2+y^2=z^2"></math>',
            "<span>$x^2+y^2=z^2$</span>",
        ),
        ('<math display="block" alttext="E=mc^2"></math>', "<span>$$ E=mc^2 $$</span>"),
        ('<math alttext="a^2+b^2=c^2"></math>', "<span>$$ a^2+b^2=c^2 $$</span>"),
    ],
)
def test_replace_math_tags(html, expected):
    soup = BeautifulSoup(html, "html.parser")
    soup = replace_math_tags(soup)
    assert str(soup) == expected


@pytest.mark.parametrize(
    "html, class_name, expected",
    [
        ('<div class="ltx_bibliography">Bibliography</div>', "ltx_bibliography", ""),
        ('<div class="ltx_appendix">Appendix</div>', "ltx_appendix", ""),
        ('<div class="ltx_para">Paragraph</div>', "ltx_para", ""),
        (
            '<div class="ltx_para">Paragraph</div><div>Content</div>',
            "ltx_para",
            "<div>Content</div>",
        ),
    ],
)
def test_remove_section_by_class(html, class_name, expected):
    soup = BeautifulSoup(html, "html.parser")
    soup = remove_section_by_class(soup, class_name)
    assert str(soup) == expected


@pytest.mark.parametrize(
    "html, expected",
    [
        (
            '<div class="test" src="image.jpg" id="main"></div>',
            '<div src="image.jpg"></div>',
        ),
        ('<img src="image.jpg" alt="image">', '<img src="image.jpg"/>'),
        ('<a href="link" src="image.jpg">Link</a>', '<a src="image.jpg">Link</a>'),
    ],
)
def test_strip_attributes(html, expected):
    soup = BeautifulSoup(html, "html.parser")
    soup = strip_attributes(soup)
    assert str(soup) == expected


@patch("utils.generate_paper.fetch_html")
@patch("utils.generate_paper.sanitize_url")
def test_process_article(mock_sanitize_url, mock_fetch_html):
    mock_sanitize_url.return_value = "https://example.com/path"
    mock_fetch_html.return_value = "<html><article><div class='ltx_bibliography'></div><p>Content</p></article></html>"

    url = "https://example.com/path"
    markdown = process_article(url)
    assert isinstance(markdown, str)
    assert markdown != ""

    mock_sanitize_url.assert_called_once_with(url)
    mock_fetch_html.assert_called_once_with("https://example.com/path")

    with pytest.raises(ValueError, match="No article found in the HTML content."):
        mock_fetch_html.return_value = "<html></html>"
        process_article(url)


@pytest.fixture
def soup():
    html = "<html><body><article>Test article content</article></body></html>"
    return BeautifulSoup(html, "html.parser")


@pytest.mark.parametrize(
    "html, expected",
    [
        (
            "<html><body><article>Test article content</article></body></html>",
            "Test article content",
        ),
        (
            "<html><body><article><h1>Header</h1><p>Paragraph</p></article></body></html>",
            "Header\n======\n\nParagraph\n\n",
        ),
    ],
)
def test_convert_to_markdown(html, expected):
    soup = BeautifulSoup(html, "html.parser")
    markdown = convert_to_markdown(soup)
    assert isinstance(markdown, str)
    assert expected in markdown
