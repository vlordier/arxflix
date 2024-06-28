from unittest.mock import Mock, patch

import pytest
import requests
from prompts import prompt_summary
from utils.generate_script import correct_result_link, process_script


# Mock settings
class MockSettings:
    REQUESTS_TIMEOUT = 10

    class OPENAI:
        model = "gpt-4o"


settings = MockSettings()


@pytest.mark.parametrize(
    "script, url, expected",
    [
        (
            "Some content\n\\Figure: /images/fig1.png\nMore content",
            "https://example.com/html/12345",
            "Some content\n\\Figure: https://example.com/images/fig1.png\nMore content",
        ),
        (
            "Introduction\n\\Figure: /html/images/fig2.png\nConclusion",
            "https://example.com/html/67890",
            "Introduction\n\\Figure: https://example.com/images/fig2.png\nConclusion",
        ),
        (
            "No figure\nJust text",
            "https://example.com/html/54321",
            "No figure\nJust text",
        ),
    ],
)
@patch("utils.generate_script.requests.head")
def test_correct_result_link(mock_head, script, url, expected):
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.headers = {"Content-Type": "image/png"}
    mock_head.return_value = mock_response

    corrected_script = correct_result_link(script, url)
    assert corrected_script == expected

    mock_head.side_effect = requests.exceptions.RequestException
    corrected_script = correct_result_link(script, url)
    assert corrected_script == script


@patch("utils.generate_script.correct_result_link")
@patch("utils.generate_script.OpenAI")
def test_process_script(mock_openai, mock_correct_result_link):
    mock_openai_instance = Mock()
    mock_openai.return_value = mock_openai_instance
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="Generated script"))]
    mock_openai_instance.chat.completions.create.return_value = mock_response

    mock_correct_result_link.return_value = "Corrected script"

    paper = "This is a research paper in markdown format"
    url = "https://example.com/html/12345"
    script = process_script(paper, url)
    assert script == "Corrected script"

    mock_openai_instance.chat.completions.create.assert_called_once_with(
        model=settings.OPENAI.model,
        messages=[
            {"role": "system", "content": prompt_summary.system_prompt},
            {"role": "user", "content": prompt_summary.user_prompt},
        ],
    )
    mock_correct_result_link.assert_called_once_with("Generated script", url)

    mock_response.choices[0].message.content = None
    with pytest.raises(ValueError, match="No result returned from OpenAI."):
        process_script(paper, url)


@pytest.mark.parametrize(
    "script, url",
    [
        (
            "Some content\n\\Figure: /images/fig1.png\nMore content",
            "https://example.com/html/12345",
        ),
        (
            "Introduction\n\\Figure: /html/images/fig2.png\nConclusion",
            "https://example.com/html/67890",
        ),
        ("No figure\nJust text", "https://example.com/html/54321"),
    ],
)
def test_correct_result_link_no_request_exception(script, url):
    corrected_script = correct_result_link(script, url)
    assert corrected_script == script
