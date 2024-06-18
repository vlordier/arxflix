""" This module contains the fixtures for the tests. """

import tempfile
from typing import Generator
from unittest import mock
from unittest.mock import MagicMock

import pytest
import torch


@pytest.fixture
def sample_url() -> str:
    """Fixture to create a sample URL."""
    return "https://ar5iv.labs.arxiv.org/html/1234.5678"


@pytest.fixture
def test_script() -> str:
    """Fixture to create a test script."""
    return """
    This is a sample script.
    \\Figure: /figures/sample1.png
    Another line of text.
    \\Figure: /figures/sample2.png
    """


@pytest.fixture
def corrected_test_script() -> str:
    """Fixture to create a corrected test script."""
    return """
    This is a sample script.
    \\Figure: https://ar5iv.labs.arxiv.org/html/figures/sample1.png
    Another line of text.
    \\Figure: https://ar5iv.labs.arxiv.org/html/figures/sample2.png
    """


@pytest.fixture
def mock_openai_response() -> tuple[MagicMock, MagicMock]:
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
        r"\Figure: Sample Figure\n"
        r"\Text: Sample Text\n"
        r"\Equation: E=mc^2\n"
        r"\Headline: Sample Headline\n"
    )


@pytest.fixture
def sample_paper() -> str:
    """Fixture to create a sample paper."""
    return "This is a sample paper content."


@pytest.fixture
def mock_whisper_model() -> Generator[MagicMock, None, None]:
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
def mock_torchaudio_load() -> Generator[MagicMock, None, None]:
    """Fixture to mock the torchaudio.load function."""
    with mock.patch("torchaudio.load") as mocked_torchaudio_load:
        mocked_torchaudio_load.return_value = (torch.zeros(1, 16000), 16000)
        yield mocked_torchaudio_load


@pytest.fixture
def temp_dir_fixture() -> Generator[str, None, None]:
    """Fixture to create a temporary directory."""
    with tempfile.TemporaryDirectory() as temporary_dir:
        yield temporary_dir
