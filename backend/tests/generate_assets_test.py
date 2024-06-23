""" Tests for the generate_assets module """

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from elevenlabs import Voice
from elevenlabs.client import ElevenLabs
from pydub import AudioSegment
from src.models import Caption, Equation, Figure, Headline, Text
from src.utils.generate_assets import (
    combine_audio_segments,
    create_elevenlabs_client,
    export_mp3,
    export_rich_content_json,
    export_srt,
    fill_rich_content_time,
    generate_audio_and_caption,
    generate_audio_for_text,
    make_caption,
    parse_script,
    process_audio_files,
    transcribe_audio,
    update_text_content_with_captions,
)

# def test_parse_script():
#     script = r"\Figure: Test figure\n\Text: Test text\n\Equation: Test equation\n\Headline: Test headline"
#     parsed_script = parse_script(script)
#     assert len(parsed_script) == 4
#     assert isinstance(parsed_script[0], Figure)
#     assert isinstance(parsed_script[1], Text)
#     assert isinstance(parsed_script[2], Equation)
#     assert isinstance(parsed_script[3], Headline)


@pytest.mark.parametrize(
    "script, expected_classes",
    [
        (
            r"\Figure: Test figure\n\Text: Test text\n\Equation: Test equation\n\Headline: Test headline",
            [Figure, Text, Equation, Headline],
        ),
        (
            r"\Figure: Test figure\n\Text: Test text\n\Headline: Test headline",
            [Figure, Text, Headline],
        ),
        (
            r"\Figure: Test figure\n\n\Text: Test text\n\n\n\Equation: Test equation\n\n\Headline: Test headline\n",
            [Figure, Text, Equation, Headline],
        ),
        (
            r"\Headline: Test headline\n\Equation: Test equation\n\Text: Test text\n\Figure: Test figure",
            [Headline, Equation, Text, Figure],
        ),
        (r"", []),
        (
            r"\Figure: \n\Text: \n\Equation: \n\Headline: ",
            [Figure, Text, Equation, Headline],
        ),
        (
            r"\Figure: Test figure\n\Text: Test text\n\Equation: \Figure: Nested figure\n\Headline: Test headline",
            [Figure, Text, Equation, Figure, Headline],
        ),
    ],
)
def test_parse_script(script, expected_classes) -> None:
    parsed_script = parse_script(script)
    assert len(parsed_script) == len(expected_classes)
    for parsed_obj, expected_class in zip(
        parsed_script, expected_classes, strict=False
    ):
        assert isinstance(parsed_obj, expected_class)
        assert parsed_obj.identifier == f"\\{expected_class.__name__}"
        assert "\\n" not in parsed_obj.content


def test_make_caption() -> None:
    result = {"segments": [{"words": [{"text": "test", "start": 0.0, "end": 1.0}]}]}
    captions = make_caption(result)
    assert len(captions) == 1
    assert captions[0].word == "test"
    assert captions[0].start == 0.0
    assert captions[0].end == 1.0


@patch("elevenlabs.client.ElevenLabs")
@patch("src.utils.generate_assets.create_elevenlabs_client")
@patch("src.utils.generate_assets.generate_audio_for_text")
@patch("src.utils.generate_assets.transcribe_audio")
@patch("src.utils.generate_assets.make_caption")
def test_generate_audio_and_caption(
    mock_make_caption: Mock,
    mock_transcribe_audio: Mock,
    mock_generate_audio_for_text: Mock,
    mock_create_elevenlabs_client: Mock,
    mock_elevenlabs: Mock,
    mock_settings: Mock,
    mock_whisper_model: Mock,
) -> None:
    mock_elevenlabs.generate.return_value = b"audio data"
    mock_create_elevenlabs_client.return_value = mock_elevenlabs
    mock_generate_audio_for_text.return_value = "audio_path"
    mock_transcribe_audio.return_value = {
        "segments": [{"words": [{"text": "test", "start": 0.0, "end": 1.0}]}]
    }
    mock_make_caption.return_value = [Caption(word="test", start=0.0, end=1.0)]

    script_contents = [Text(content="Test text")]
    updated_script = generate_audio_and_caption(script_contents, mock_whisper_model)
    assert updated_script[0].audio_path == "audio_path"
    assert updated_script[0].captions[0].word == "test"
    assert updated_script[0].captions[0].word == "test"


def test_fill_rich_content_time() -> None:
    script_contents = [
        Figure(content="Test figure"),
        Text(content="Test text", start=0.0, end=5.0),
        Equation(content="Test equation"),
        Text(content="Another text", start=5.0, end=10.0),
    ]
    filled_script = fill_rich_content_time(script_contents)
    assert filled_script[0].start is not None
    assert filled_script[0].end is not None


@patch("src.utils.generate_assets.whisper")
def test_export_mp3(mock_whisper_model) -> None:
    mock_whisper_model.transcribe.return_value = {
        "segments": [{"words": [{"text": "test", "start": 0.0, "end": 1.0}]}]
    }

    with tempfile.TemporaryDirectory() as tempdir:
        output_path = Path(tempdir) / "output.mp3"
        audio_path = Path(tempdir) / "audio.wav"

        # Create a silent audio file for testing
        silent_audio = AudioSegment.silent(duration=1000)  # 1 second of silence
        silent_audio.export(audio_path, format="wav")

        text_contents = [Text(content="Test text", audio_path=str(audio_path))]
        export_mp3(text_contents, str(output_path))

        assert output_path.exists()
        assert output_path.suffix == ".mp3"


@patch("src.utils.generate_assets.whisper")
def test_export_srt(mock_whisper_model) -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        full_audio_path = Path(tempdir) / "full_audio.wav"
        output_path = Path(tempdir) / "output.srt"
        silent_audio = AudioSegment.silent(duration=1000)  # 1 second of silence
        silent_audio.export(full_audio_path, format="wav")
        export_srt(str(full_audio_path), str(output_path), mock_whisper_model)
        assert output_path.exists()


def test_export_rich_content_json() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        output_path = Path(tempdir) / "output.json"
        rich_contents = [Figure(content="Test figure"), Text(content="Test text")]
        export_rich_content_json(rich_contents, str(output_path))
        assert output_path.exists()


def test_create_elevenlabs_client() -> None:
    client = create_elevenlabs_client("test_api_key")
    assert isinstance(client, ElevenLabs)


@pytest.fixture
def mock_elevenlabs() -> ElevenLabs:
    """Mock ElevenLabs client for testing."""
    return Mock(spec=ElevenLabs)


@patch("src.utils.generate_assets.save")
def test_generate_audio_for_text(mock_save, mock_elevenlabs) -> None:
    mock_elevenlabs.generate.return_value = b"audio data"
    with tempfile.TemporaryDirectory() as tempdir:
        text_content = Text(content="Test text")
        audio_path = generate_audio_for_text(
            text_content,
            mock_elevenlabs,
            Voice(voice_id="test_voice_id"),
            "test_model",
            Path(tempdir),
            0,
        )
        assert audio_path.endswith(".wav")


@patch("src.utils.generate_assets.whisper")
def test_transcribe_audio(mock_whisper_model) -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        audio_path = Path(tempdir) / "audio.wav"
        silent_audio = AudioSegment.silent(duration=1000)  # 1 second of silence
        silent_audio.export(audio_path, format="wav")
        mock_whisper_model.transcribe.return_value = {
            "segments": [{"words": [{"text": "test", "start": 0.0, "end": 1.0}]}]
        }
        result = transcribe_audio(str(audio_path), mock_whisper_model)
        assert "segments" in result


def test_update_text_content_with_captions() -> None:
    text_content = Text(content="Test text")
    captions = [Caption(word="test", start=0.0, end=1.0)]
    update_text_content_with_captions(text_content, captions, 1.0)
    assert text_content.captions == captions


def test_combine_audio_segments() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        output_path = Path(tempdir) / "output.mp3"
        segment = AudioSegment.silent(duration=1000)
        combine_audio_segments([segment, segment], str(output_path))
        assert output_path.exists()


def test_process_audio_files() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        audio_path = Path(tempdir) / "audio.wav"
        silent_audio = AudioSegment.silent(duration=1000)  # 1 second of silence
        silent_audio.export(audio_path, format="wav")
        text_contents = [Text(content="Test text", audio_path=str(audio_path))]
        segments = process_audio_files(text_contents)
        assert len(segments) == 1
