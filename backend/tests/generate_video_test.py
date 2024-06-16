import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from utils.generate_video import CompositionProps, process_video

# Constants
VIDEO_FPS = 30
REMOTION_ROOT_PATH = Path("frontend/remotion/index.ts")
REMOTION_COMPOSITION_ID = "Arxflix"
REMOTION_CONCURRENCY = 1


def test_composition_props_default() -> None:
    """
    Test the default values and post-initialization of CompositionProps.
    """
    props = CompositionProps()
    assert props.duration_in_seconds == 5
    assert props.audio_offset_in_seconds == 0
    assert props.subtitles_file_name == "frontend/public/output.srt"
    assert props.audio_file_name == "frontend/public/audio.wav"
    assert props.rich_content_file_name == "frontend/public/output.json"
    assert props.wave_color == "#a3a5ae"
    assert props.subtitles_line_per_page == 2
    assert props.subtitles_line_height == 98
    assert props.subtitles_zoom_measurer_size == 10
    assert props.only_display_current_sentence is True
    assert props.mirror_wave is False
    assert props.wave_lines_to_display == 300
    assert props.wave_freq_range_start_index == 5
    assert props.wave_number_of_samples == "512"
    assert props.duration_in_frames == 5 * VIDEO_FPS


@patch("subprocess.run")
def test_process_video_default(mock_subprocess_run: MagicMock) -> None:
    """
    Test the process_video function with default parameters.

    Args:
        mock_subprocess_run (MagicMock): Mock object for subprocess.run.

    """
    mock_subprocess_run.return_value = MagicMock()

    output_path = Path("frontend/public/output.mp4")
    composition_props = CompositionProps()

    process_video(output_path, composition_props)

    expected_command = [
        "npx",
        "remotion",
        "render",
        REMOTION_ROOT_PATH.absolute().as_posix(),
        "--props",
        json.dumps(asdict(composition_props)),
        "--compositionId",
        REMOTION_COMPOSITION_ID,
        "--concurrency",
        str(REMOTION_CONCURRENCY),
        "--output",
        output_path.absolute().as_posix(),
    ]

    mock_subprocess_run.assert_called_once_with(expected_command, check=True)


@patch("subprocess.run")
def test_process_video_custom_path(mock_subprocess_run: MagicMock) -> None:
    """
    Test the process_video function with a custom output path.

    Args:
        mock_subprocess_run (MagicMock): Mock object for subprocess.run.

    """
    mock_subprocess_run.return_value = MagicMock()

    output_path = Path("/custom/path/output.mp4")
    composition_props = CompositionProps()

    process_video(output_path, composition_props)

    expected_command = [
        "npx",
        "remotion",
        "render",
        REMOTION_ROOT_PATH.absolute().as_posix(),
        "--props",
        json.dumps(asdict(composition_props)),
        "--compositionId",
        REMOTION_COMPOSITION_ID,
        "--concurrency",
        str(REMOTION_CONCURRENCY),
        "--output",
        output_path.absolute().as_posix(),
    ]

    mock_subprocess_run.assert_called_once_with(expected_command, check=True)


@patch("subprocess.run")
def test_process_video_exception(mock_subprocess_run: MagicMock) -> None:
    """
    Test the process_video function when subprocess.run raises CalledProcessError.

    Args:
        mock_subprocess_run (MagicMock): Mock object for subprocess.run.

    Raises:
        subprocess.CalledProcessError: If subprocess.run raises CalledProcessError.

    """
    mock_subprocess_run.side_effect = subprocess.CalledProcessError(1, "cmd")

    output_path = Path("frontend/public/output.mp4")
    composition_props = CompositionProps()

    with pytest.raises(subprocess.CalledProcessError):
        process_video(output_path, composition_props)

    mock_subprocess_run.assert_called_once()
