import tempfile
from unittest import mock

import pytest
import torch


@pytest.fixture
def sample_script():
    """Fixture to create a sample script."""
    return (
        "\\Figure: Sample Figure\n"
        "\\Text: Sample Text\n"
        "\\Equation: E=mc^2\n"
        "\\Headline: Sample Headline\n"
    )

@pytest.fixture
def sample_paper():
    """Fixture to create a sample paper."""
    return "This is a sample paper content."

@pytest.fixture
def mock_whisper_model():
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
def mock_torchaudio_load():
    """Fixture to mock the torchaudio.load function."""
    with mock.patch("torchaudio.load") as mocked_torchaudio_load:
        mocked_torchaudio_load.return_value = (torch.zeros(1, 16000), 16000)
        yield mocked_torchaudio_load


@pytest.fixture
def temp_dir_fixture() -> str:
    """Fixture to create a temporary directory."""
    with tempfile.TemporaryDirectory() as temporary_dir:
        yield temporary_dir
