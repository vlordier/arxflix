""" Tests for the video_generator module """

import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import pytest
from settings import RemotionSettings
from utils.generate_video import CompositionProps, process_video

REMOTION_ROOT_PATH = (
    Path(__file__).parent.parent.parent / "frontend" / "remotion" / "index.ts"
)


# Mock settings
class MockSettings:
    VIDEO_FPS = 30
    REMOTION = RemotionSettings(
        root_path=REMOTION_ROOT_PATH,
        composition_id="Arxflix",
        concurrency=1,
    )


settings = MockSettings()


def test_composition_props_default() -> None:
    """Test default values of CompositionProps."""
    props = CompositionProps()
    assert props.duration_in_seconds == 5
    assert props.duration_in_frames == 5 * settings.VIDEO_FPS
    assert props.audio_offset_in_seconds == 0
    assert props.subtitles_file_name == "output.srt"
    assert props.audio_file_name == "audio.wav"
    assert props.rich_content_file_name == "output.json"
    assert props.wave_color == "#a3a5ae"
    assert props.subtitles_line_per_page == 2
    assert props.subtitles_line_height == 98
    assert props.subtitles_zoom_measurer_size == 10
    assert props.only_display_current_sentence is True
    assert props.mirror_wave is False
    assert props.wave_lines_to_display == 300
    assert props.wave_freq_range_start_index == 5
    assert props.wave_number_of_samples == "256"


@pytest.mark.parametrize(
    "duration_in_seconds, expected_frames",
    [
        (5, 5 * settings.VIDEO_FPS),
        (10, 10 * settings.VIDEO_FPS),
        (1, 1 * settings.VIDEO_FPS),
    ],
)
def test_composition_props_frames(duration_in_seconds, expected_frames):
    """Test frame calculation based on duration in seconds.

    Args:
        duration_in_seconds (int): Duration of the video in seconds.
        expected_frames (int): Expected number of frames for the given duration.
    """
    props = CompositionProps(duration_in_seconds=duration_in_seconds)
    assert props.duration_in_frames == expected_frames


@patch("utils.generate_video.subprocess.run")
def test_process_video_defaults(mock_subprocess_run):
    """Test processing video with default settings.

    Args:
        mock_subprocess_run (Mock): Mock for subprocess.run.
    """
    mock_subprocess_run.return_value = None
    process_video(None, None)

    expected_command = [
        "npx",
        "remotion",
        "render",
        Path(settings.REMOTION.root_path).absolute().as_posix(),
        "--props",
        json.dumps(asdict(CompositionProps())),
        "--compositionId",
        settings.REMOTION.composition_id,
        "--concurrency",
        str(settings.REMOTION.concurrency),
        "--output",
        Path("output.mp4").absolute().as_posix(),
    ]

    mock_subprocess_run.assert_called_once_with(expected_command, check=True)


@patch("utils.generate_video.subprocess.run")
def test_process_video_with_custom_props(mock_subprocess_run):
    """Test processing video with custom properties.

    Args:
        mock_subprocess_run (Mock): Mock for subprocess.run.
    """
    mock_subprocess_run.return_value = None
    custom_props = CompositionProps(duration_in_seconds=10, wave_color="#FF0000")
    output_path = Path("/custom/output.mp4")

    process_video(output_path, custom_props)

    expected_command = [
        "npx",
        "remotion",
        "render",
        Path(settings.REMOTION.root_path).absolute().as_posix(),
        "--props",
        json.dumps(asdict(custom_props)),
        "--compositionId",
        settings.REMOTION.composition_id,
        "--concurrency",
        str(settings.REMOTION.concurrency),
        "--output",
        output_path.absolute().as_posix(),
    ]

    mock_subprocess_run.assert_called_once_with(expected_command, check=True)


@patch("utils.generate_video.subprocess.run")
def test_process_video_failure(mock_subprocess_run):
    """Test processing video failure scenario.

    Args:
        mock_subprocess_run (Mock): Mock for subprocess.run.
    """
    mock_subprocess_run.side_effect = subprocess.CalledProcessError(
        1, "npx remotion render"
    )

    with pytest.raises(subprocess.CalledProcessError):
        process_video(None, None)

    expected_command = [
        "npx",
        "remotion",
        "render",
        Path(settings.REMOTION.root_path).absolute().as_posix(),
        "--props",
        json.dumps(asdict(CompositionProps())),
        "--compositionId",
        settings.REMOTION.composition_id,
        "--concurrency",
        str(settings.REMOTION.concurrency),
        "--output",
        Path("output.mp4").absolute().as_posix(),
    ]

    mock_subprocess_run.assert_called_once_with(expected_command, check=True)
