"""Configuration file for the backend."""

from pathlib import Path
from typing import List, Literal, Optional

from pydantic import Field, validator
from pydantic_settings import BaseSettings

# check that the .env file exists

# current file directory
parent_dir = Path(__file__).resolve().parent.parent

# .env file path in the current directory
env_file_path = parent_dir / ".env"

# check if the .env file exists
if not env_file_path.exists():
    raise FileNotFoundError(
        ".env file not found, please create one and add your API keys."
    )


class ElevenLabsSettings(BaseSettings):
    """Settings for ElevenLabs service."""

    api_key: Optional[str] = Field(
        default=None, json_schema_extra={"env": "ELEVENLABS_API_KEY"}
    )
    model: str = Field(
        default="eleven_turbo_v2", json_schema_extra={"env": "ELEVENLABS_MODEL"}
    )
    voice_id: str = Field(
        default="nkeeZ3vWAgNFBIYCA5JO", json_schema_extra={"env": "ELEVENLABS_VOICE_ID"}
    )
    stability: float = Field(
        default=0.35, json_schema_extra={"env": "ELEVENLABS_STABILITY"}
    )
    similarity_boost: float = Field(
        default=0.8, json_schema_extra={"env": "ELEVENLABS_SIMILARITY_BOOST"}
    )


class OpenAISettings(BaseSettings):
    """Settings for OpenAI service."""

    api_key: Optional[str] = Field(
        default=None, json_schema_extra={"env": "OPENAI_API_KEY"}
    )
    model: str = Field(default="gpt-4o", json_schema_extra={"env": "OPENAI_MODEL"})
    max_tokens: int = Field(
        default=8192, json_schema_extra={"env": "OPENAI_MAX_TOKENS"}
    )


class RemotionSettings(BaseSettings):
    """Settings for Remotion project."""

    root_path: Path = Field(
        default=parent_dir.parent / Path("video_render/remotion/index.ts"),
        json_schema_extra={"env": "REMOTION_ROOT_PATH"},
    )
    composition_id: str = Field(
        default="Arxflix", json_schema_extra={"env": "REMOTION_COMPOSITION_ID"}
    )
    concurrency: str = Field(
        default="80%", json_schema_extra={"env": "REMOTION_CONCURRENCY"}
    )

    @validator("root_path")
    def check_root_path(cls, v: Path) -> Path:
        """Validate that the root_path exists and is a file.

        Args:
            cls (RemotionSettings): The class instance.
            v (Path): The path to validate.

        Raises:
            ValueError: If the path does not exist or is not a file.

        Returns:
            Path: The validated path.
        """
        if not v.is_file():
            raise ValueError(f"root_path: {v} does not exist or is not a file")
        return v


class LoggingSettings(BaseSettings):
    """Settings for logging configuration."""

    format: str = Field(
        default="%(asctime)s - %(name)s - %(levellevel)s - %(message)s",
        json_schema_extra={"env": "LOG_FORMAT"},
    )
    level: str = Field(default="INFO", json_schema_extra={"env": "LOG_LEVEL"})


class VideoSettings(BaseSettings):
    """Settings for video processing."""

    fps: int = Field(default=30, json_schema_extra={"env": "VIDEO_FPS"})
    height: int = Field(default=1080, json_schema_extra={"env": "VIDEO_HEIGHT"})
    width: int = Field(default=1920, json_schema_extra={"env": "VIDEO_WIDTH"})


class AudioSettings(BaseSettings):
    """Settings for audio processing."""

    sample_rate: int = Field(
        default=44100, json_schema_extra={"env": "AUDIO_SAMPLE_RATE"}
    )
    channels: int = Field(default=2, json_schema_extra={"env": "AUDIO_CHANNELS"})
    format: str = Field(default="wav", json_schema_extra={"env": "AUDIO_FORMAT"})


class TimeoutSettings(BaseSettings):
    """Settings for timeout."""

    whisper: int = Field(default=300, json_schema_extra={"env": "WHISPER_TIMEOUT"})
    elevenlabs: int = Field(default=60, json_schema_extra={"env": "ELEVENLABS_TIMEOUT"})


class ImageSettings(BaseSettings):
    """Settings for image processing."""

    width: int = Field(default=1920, json_schema_extra={"env": "IMAGE_WIDTH"})
    height: int = Field(default=1080, json_schema_extra={"env": "IMAGE_HEIGHT"})
    format_type: str = Field(default="png", json_schema_extra={"env": "IMAGE_FORMAT"})
    quality: int = Field(default=95, json_schema_extra={"env": "IMAGE_QUALITY"})
    root_path: Path = Field(
        default=Path("./images"), json_schema_extra={"env": "IMAGE_ROOT_PATH"}
    )


class CompositionPropsSettings(BaseSettings):
    """
    Settings for composition props.

    Attributes:
        duration_in_seconds (int): The duration of the composition in seconds.
        audio_offset_in_seconds (int): The audio offset in seconds.
        subtitles_file_name (str): The subtitles file name.
        audio_file_name (str): The audio file name.
        rich_content_file_name (str): The rich content file name.
        wave_color (str): The wave color.
        subtitles_line_per_page (int): The number of lines per page for subtitles.
        subtitles_line_height (int): The height of each line for subtitles.
        subtitles_zoom_measurer_size (int): The size of the zoom measurer for subtitles.
        only_display_current_sentence (bool): Whether to only display the current sentence.
        mirror_wave (bool): Whether to mirror the wave.
        wave_lines_to_display (int): The number of lines to display for the wave.
        wave_freq_range_start_index (int): The start index for the frequency range for the wave.
        wave_number_of_samples (str): The number of samples for the wave.
        duration_in_frames (int): The duration of the composition in frames.
    """

    duration_in_seconds: int = Field(
        default=5, json_schema_extra={"env": "COMPOSITION_DURATION_IN_SECONDS"}
    )
    audio_offset_in_seconds: int = Field(
        default=0, json_schema_extra={"env": "COMPOSITION_AUDIO_OFFSET_IN_SECONDS"}
    )
    subtitles_file_name: str = Field(
        default="output.srt",
        json_schema_extra={"env": "COMPOSITION_SUBTITLES_FILE_NAME"},
    )
    audio_file_name: str = Field(
        default="audio.mp3",
        json_schema_extra={"env": "COMPOSITION_AUDIO_FILE_NAME"},
    )
    rich_content_file_name: str = Field(
        default="output.json",
        json_schema_extra={"env": "COMPOSITION_RICH_CONTENT_FILE_NAME"},
    )
    wave_color: str = Field(
        default="#0059b3", json_schema_extra={"env": "COMPOSITION_WAVE_COLOR"}
    )
    subtitles_line_per_page: int = Field(
        default=2, json_schema_extra={"env": "COMPOSITION_SUBTITLES_LINE_PER_PAGE"}
    )
    subtitles_line_height: int = Field(
        default=98, json_schema_extra={"env": "COMPOSITION_SUBTITLES_LINE_HEIGHT"}
    )
    subtitles_zoom_measurer_size: int = Field(
        default=10,
        json_schema_extra={"env": "COMPOSITION_SUBTITLES_ZOOM_MEASURER_SIZE"},
    )
    only_display_current_sentence: bool = Field(
        default=True,
        json_schema_extra={"env": "COMPOSITION_ONLY_DISPLAY_CURRENT_SENTENCE"},
    )
    mirror_wave: bool = Field(
        default=False, json_schema_extra={"env": "COMPOSITION_MIRROR_WAVE"}
    )
    wave_lines_to_display: int = Field(
        default=300, json_schema_extra={"env": "COMPOSITION_WAVE_LINES_TO_DISPLAY"}
    )
    wave_freq_range_start_index: int = Field(
        default=5, json_schema_extra={"env": "COMPOSITION_WAVE_FREQ_RANGE_START_INDEX"}
    )
    wave_number_of_samples: Literal["32", "64", "128", "256", "512"] = Field(
        default="256", json_schema_extra={"env": "COMPOSITION_WAVE_NUMBER_OF_SAMPLES"}
    )
    duration_in_frames: int = Field(
        default=1, json_schema_extra={"env": "COMPOSITION_DURATION_IN_FRAMES"}
    )


class Settings(BaseSettings):
    """Configuration settings for the backend."""

    APP_NAME: str = Field(default="Arxflix", json_schema_extra={"env": "APP_NAME"})
    APP_VERSION: str = Field(default="0.1.0", json_schema_extra={"env": "APP_VERSION"})
    APP_DESCRIPTION: str = Field(
        default="Arxflix is an ArXiv research paper summarizer",
        json_schema_extra={"env": "APP_DESCRIPTION"},
    )
    WHISPER_MODEL: str = Field(
        default="tiny.en", json_schema_extra={"env": "WHISPER_MODEL"}
    )
    ELEVENLABS: ElevenLabsSettings = ElevenLabsSettings()
    OPENAI: OpenAISettings = OpenAISettings()
    LOGGING: LoggingSettings = LoggingSettings()
    TEMP_DIR: Path = Field(default=Path("temp"), json_schema_extra={"env": "TEMP_DIR"})
    # PAPER_URL: str = Field(default="", json_schema_extra={"env": "PAPER_URL"})
    REQUESTS_TIMEOUT: int = Field(
        default=10, json_schema_extra={"env": "REQUESTS_TIMEOUT"}
    )
    VIDEO: VideoSettings = VideoSettings()
    REMOTION: RemotionSettings = RemotionSettings()
    AUDIO: AudioSettings = AudioSettings()
    TIMEOUTS: TimeoutSettings = TimeoutSettings()
    COMPOSITION_PROPS: CompositionPropsSettings = CompositionPropsSettings()
    ALLOWED_DOMAINS: List[str] = Field(
        default_factory=lambda: ["arxiv.org", "ar5iv.labs.arxiv.org", "ar5iv.org"],
        json_schema_extra={"env": "ALLOWED_DOMAINS"},
    )

    IMAGE_ROOT: Path = Field(
        default=Path("images"), json_schema_extra={"env": "IMAGE_ROOT"}
    )
    IMAGE_WIDTH: int = Field(default=1920, json_schema_extra={"env": "IMAGE_WIDTH"})
    IMAGE_HEIGHT: int = Field(default=1080, json_schema_extra={"env": "IMAGE_HEIGHT"})
    SCRIPT_NAME: str = Field(
        default="script.txt", json_schema_extra={"env": "SCRIPT_NAME"}
    )
    ARXIV_BASE_URL: str = Field(
        default="https://arxiv.org/html",
        json_schema_extra={"env": "ARXIV_BASE_URL"},
    )



settings = Settings()
