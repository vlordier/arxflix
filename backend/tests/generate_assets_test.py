import os
import sys
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock

# Add the root directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from models import Equation, Figure, Headline, Text
from utils.generate_assets import (
    export_mp3,
    export_rich_content_json,
    export_srt,
    fill_rich_content_time,
    generate_audio_and_caption,
    make_caption,
    parse_script,
)


def test_parse_script(sample_script: str) -> None:
    contents = parse_script(sample_script)
    assert len(contents) == 4
    assert isinstance(contents[0], Figure)
    assert isinstance(contents[1], Text)
    assert isinstance(contents[2], Equation)
    assert isinstance(contents[3], Headline)


def test_make_caption() -> None:
    result = {
        "segments": [
            {
                "words": [
                    {"word": "hello", "start": 0.0, "end": 0.5},
                    {"word": "world", "start": 0.5, "end": 1.0},
                ]
            }
        ]
    }
    captions = make_caption(result)
    assert len(captions) == 2
    assert captions[0].word == "hello"
    assert captions[1].word == "world"


def test_generate_audio_and_caption(
    sample_script: str,
    temp_dir_fixture: str,
    mock_whisper_model: MagicMock,
    mock_torchaudio_load: MagicMock,
) -> None:
    with mock.patch("utils.generate_assets.ElevenLabs") as MockElevenLabs, \
         mock.patch("utils.generate_assets.ELEVENLABS_API_KEY", "fake_api_key"):
        mock_elevenlabs_client = MockElevenLabs.return_value
        mock_elevenlabs_client.generate.return_value = iter([b"fake_audio_data"])
        mock_elevenlabs_client.save.side_effect = lambda audio, path: Path(path).write_bytes(b"fake_audio_data")
        mock_elevenlabs_client.convert.return_value = iter([b"fake_audio_data"])
        mock_elevenlabs_client._client_wrapper.httpx_client.stream.return_value = iter([b"fake_audio_data"])

        script_contents = parse_script(sample_script)
        result = generate_audio_and_caption(script_contents, Path(temp_dir_fixture))
        assert len(result) == 4
        for content in result:
            if isinstance(content, Text):
                assert content.audio is not None
                assert content.captions is not None
                assert content.audio_path is not None
                audio_path = Path(content.audio_path)
                assert audio_path.exists()
                assert audio_path.read_bytes() == b"fake_audio_data"


def test_fill_rich_content_time() -> None:
    script_contents = [
        Figure(content="Sample Figure"),
        Text(content="Sample Text", start=0.0, end=1.0),
        Equation(content="E=mc^2"),
        Text(content="Another Text", start=1.0, end=2.0),
    ]
    result = fill_rich_content_time(script_contents)
    assert result[0].start == 0.0
    assert result[0].end == 0.5
    assert result[2].start == 1.0
    assert result[2].end == 1.5


def test_export_mp3(temp_dir_fixture: str, mock_torchaudio_load: MagicMock) -> None:
    text_contents = [
        Text(content="Sample Text", audio_path=os.path.join(temp_dir_fixture, "sample.wav"))
    ]
    output_path = os.path.join(temp_dir_fixture, "output.mp3")
    export_mp3(text_contents, output_path)
    assert os.path.exists(output_path)


def test_export_srt(
    temp_dir_fixture: str, mock_whisper_model: MagicMock, mock_torchaudio_load: MagicMock
) -> None:
    full_audio_path = os.path.join(temp_dir_fixture, "full_audio.wav")
    output_path = os.path.join(temp_dir_fixture, "output.srt")
    with open(full_audio_path, "wb") as f:
        f.write(b"fake_audio_data")
    export_srt(full_audio_path, output_path)
    assert os.path.exists(output_path)


def test_export_rich_content_json(temp_dir_fixture: str) -> None:
    rich_contents = [Figure(content="Sample Figure", start=0.0, end=1.0)]
    output_path = os.path.join(temp_dir_fixture, "output.json")
    export_rich_content_json(rich_contents, output_path)
    assert os.path.exists(output_path)
    with open(output_path, "r") as f:
        data = f.read()
    assert "Sample Figure" in data
