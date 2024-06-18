import os
import tempfile
from pathlib import Path
from typing import List, Optional, Union

import torchaudio

from backend.src.config import (
    DEFAULT_MP3_OUTPUT_PATH,
    DEFAULT_RICH_OUTPUT_PATH,
    DEFAULT_SRT_OUTPUT_PATH,
    WHISPER_MODEL_NAME,
)
from backend.src.models import RichContent, Text
from backend.src.utils.audio_generation import AudioGenerator
from backend.src.utils.caption_generation import CaptionGenerator
from backend.src.utils.json_exporter import export_rich_content_json
from backend.src.utils.mp3_exporter import export_mp3
from backend.src.utils.script_parser import ScriptParser
from backend.src.utils.srt_exporter import export_srt
from backend.src.utils.timer import ContentTimer


def generate_assets(
    script: str,
    use_path: bool = True,
    mp3_output: Path = DEFAULT_MP3_OUTPUT_PATH,
    srt_output: Path = DEFAULT_SRT_OUTPUT_PATH,
    rich_output: Path = DEFAULT_RICH_OUTPUT_PATH,
    temp_dir: Optional[Path] = None,
) -> float:
    if temp_dir is None:
        temp_dir = Path(tempfile.gettempdir())
    if not temp_dir.exists():
        temp_dir.mkdir(parents=True)

    parser = ScriptParser(script)
    audio_generator = AudioGenerator()
    caption_generator = CaptionGenerator(WHISPER_MODEL_NAME)
    timer = ContentTimer()

    contents = parser.parse()
    contents = generate_audio_and_captions(
        contents, temp_dir, audio_generator, caption_generator
    )
    contents = timer.fill_time(contents)

    rich_content_list = [item for item in contents if isinstance(item, RichContent)]
    text_content_list = [item for item in contents if isinstance(item, Text)]

    export_mp3(text_content_list, mp3_output)
    export_srt(mp3_output, srt_output)
    export_rich_content_json(rich_content_list, rich_output)

    total_duration = (
        text_content_list[-1].end
        if text_content_list and text_content_list[-1].end
        else 0
    )
    return total_duration


def generate_audio_and_captions(
    script_contents: List[Union[RichContent, Text]],
    temp_dir: Path,
    audio_generator: AudioGenerator,
    caption_generator: CaptionGenerator,
) -> List[Union[RichContent, Text]]:
    for index, content in enumerate(script_contents):
        if isinstance(content, Text) and content.audio is None:
            audio_path = (temp_dir / f"audio_{index}.wav").absolute().as_posix()
            if not os.path.exists(audio_path):
                content.audio_path = audio_generator.generate_audio(
                    content.content, Path(audio_path)
                )
            content.captions = caption_generator.generate_captions(content.audio_path)
            audio, sample_rate = torchaudio.load(content.audio_path)
            content.end = audio.shape[1] / sample_rate
    return script_contents
