""" This module contains the fixtures for the tests """

import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
from models import AssetsInput, ScriptInput
from settings import Settings

# Add the backend directory to the PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def script_input() -> ScriptInput:
    """This fixture returns a sample ScriptInput object"""
    return ScriptInput(paper="This is a sample paper content", use_path=False)


@pytest.fixture
def assets_input() -> AssetsInput:
    """This fixture returns a sample AssetsInput object"""
    return AssetsInput(
        script=r"\Headline Headline\n\n \Figure Figure \n\Paragraph Paragraph\n \Equation Equation \n \Text Text\n",
        use_path=False,
        mp3_output="test_audio.wav",
        srt_output="test_output.srt",
        rich_output="test_output.json",
    )


@pytest.fixture
def mock_settings() -> Mock:
    """This fixture returns a mock Settings object"""
    return Settings(
        ELEVENLABS={
            "voice_id": "test_voice_id",
            "stability": 0.5,
            "similarity_boost": 0.7,
            "model": "test_model",
        },
        WHISPER_MODEL="test_model",
    )


@pytest.fixture
def mock_whisper_model() -> Mock:
    """This fixture returns a mock Whisper model"""
    model = Mock()
    model.transcribe = Mock(
        return_value={
            "segments": [{"words": [{"text": "test", "start": 0.0, "end": 1.0}]}]
        }
    )
    return model
