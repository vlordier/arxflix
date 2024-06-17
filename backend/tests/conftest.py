import tempfile
from unittest import mock
from unittest.mock import MagicMock

import pytest
import torch


@pytest.fixture
def sample_url():
    return "https://ar5iv.labs.arxiv.org/html/1234.5678"


@pytest.fixture
def test_script():
    return """
    This is a sample script.
    \\Figure: /figures/sample1.png
    Another line of text.
    \\Figure: /figures/sample2.png
    """


@pytest.fixture
def corrected_test_script():
    return """
    This is a sample script.
    \\Figure: https://ar5iv.labs.arxiv.org/html/figures/sample1.png
    Another line of text.
    \\Figure: https://ar5iv.labs.arxiv.org/html/figures/sample2.png
    """


@pytest.fixture
def mock_openai_response() -> MagicMock:
    """Fixture to mock the OpenAI API response."""
    mock_openai_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_openai_client.chat.completions.create.return_value = mock_response
    return mock_openai_client, mock_response


@pytest.fixture
def sample_script() -> str:
    """Fixture to create a sample script."""
    return (
        "\\Figure: Sample Figure\n"
        "\\Text: Sample Text\n"
        "\\Equation: E=mc^2\n"
        "\\Headline: Sample Headline\n"
    )


@pytest.fixture
def sample_paper() -> str:
    """Fixture to create a sample paper."""
    return "This is a sample paper content."


@pytest.fixture
def mock_whisper_model() -> MagicMock:
    """Fixture to mock the Whisper model."""
    with mock.patch("whisper.load_model") as mocked_load_model:
        mocked_model = mock.Mock()
        mocked_model.transcribe.return_value = {
            "segments": [
                {
                    "words": [
                        {"word": "hello", "start": 0.0, "end": 0.5},
                        {"word": "world", "start": 0.5, "end": 1.0},
                    ]
                }
            ]
        }
        mocked_load_model.return_value = mocked_model
        yield mocked_load_model


@pytest.fixture
def mock_torchaudio_load() -> MagicMock:
    """Fixture to mock the torchaudio.load function."""
    with mock.patch("torchaudio.load") as mocked_torchaudio_load:
        mocked_torchaudio_load.return_value = (torch.zeros(1, 16000), 16000)
        yield mocked_torchaudio_load


@pytest.fixture
def temp_dir_fixture() -> str:
    """Fixture to create a temporary directory."""
    with tempfile.TemporaryDirectory() as temporary_dir:
        yield temporary_dir
