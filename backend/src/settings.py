"""Configuration file for the backend."""

from pathlib import Path
from typing import List, Literal, Optional

from pydantic import Field
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
    """
    Settings for ElevenLabs service.

    Attributes:
        api_key (str): The API key for the ElevenLabs service.
        model (str): The model name for the ElevenLabs service.
        voice_id (str): The voice ID for the ElevenLabs service.
        stability (float): The stability parameter for the ElevenLabs service.
        similarity_boost (float): The similarity boost parameter for the ElevenLabs service.
    """

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
    """
    Settings for OpenAI service.

    Attributes:
        api_key (str): The API key for the OpenAI service.
        model (str): The model name for the OpenAI service.
    """

    api_key: Optional[str] = Field(
        default=None, json_schema_extra={"env": "OPENAI_API_KEY"}
    )
    model: str = Field(default="gpt-4o", json_schema_extra={"env": "OPENAI_MODEL"})

    max_tokens: int = Field(
        default=8192, json_schema_extra={"env": "OPENAI_MAX_TOKENS"}
    )


class RemotionSettings(BaseSettings):
    """
    Settings for Remotion project.

    Attributes:
        root_path (Path): The root path for the Remotion project.
        composition_id (str): The composition ID for the Remotion project.
        concurrency (int): The concurrency for the Remotion project.
    """

    root_path: Path = Field(
        # Current file directory / frontend/remotion/index.ts
        default=parent_dir.parent / "frontend/remotion/index.ts",
        json_schema_extra={"env": "REMOTION_ROOT_PATH"},
    )

    composition_id: str = Field(
        default="Arxflix", json_schema_extra={"env": "REMOTION_COMPOSITION_ID"}
    )
    concurrency: int = Field(
        default=1, json_schema_extra={"env": "REMOTION_CONCURRENCY"}
    )


class LoggingSettings(BaseSettings):
    """
    Settings for logging configuration.

    Attributes:
        format (str): The format for logging messages.
        level (str): The level for logging messages.
    """

    format: str = Field(
        default="%(asctime)s - %(name)s - %(levellevel)s - %(message)s",
        json_schema_extra={"env": "LOG_FORMAT"},
    )
    level: str = Field(default="INFO", json_schema_extra={"env": "LOG_LEVEL"})


class VideoSettings(BaseSettings):
    """
    Settings for video processing.

    Attributes:
        fps (int): The frames per second for video processing.
        height (int): The height of the video.
        width (int): The width of the video.
    """

    fps: int = Field(default=30, json_schema_extra={"env": "VIDEO_FPS"})
    height: int = Field(default=1080, json_schema_extra={"env": "VIDEO_HEIGHT"})
    width: int = Field(default=1920, json_schema_extra={"env": "VIDEO_WIDTH"})


class AudioSettings(BaseSettings):
    """
    Settings for audio processing.

    Attributes:
        sample_rate (int): The sample rate for audio processing.
        channels (int): The number of channels for audio processing.
        format (str): The format for audio processing.
    """

    sample_rate: int = Field(
        default=44100, json_schema_extra={"env": "AUDIO_SAMPLE_RATE"}
    )
    channels: int = Field(default=2, json_schema_extra={"env": "AUDIO_CHANNELS"})
    format: str = Field(default="wav", json_schema_extra={"env": "AUDIO_FORMAT"})


class TimeoutSettings(BaseSettings):
    """
    Settings for timeout.

    Attributes:
        whisper (int): The timeout for Whisper service.
        elevenlabs (int): The timeout for ElevenLabs service.
    """

    whisper: int = Field(default=300, json_schema_extra={"env": "WHISPER_TIMEOUT"})
    elevenlabs: int = Field(default=60, json_schema_extra={"env": "ELEVENLABS_TIMEOUT"})


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
    """

    duration_in_seconds: int = Field(
        default=5, json_schema_extra={"env": "COMPOSITION_DURATION_IN_SECONDS"}
    )
    audio_offset_in_seconds: int = Field(
        default=0, json_schema_extra={"env": "COMPOSITION_AUDIO_OFFSET_IN_SECONDS"}
    )
    subtitles_file_name: str = Field(
        default="public/output.srt",
        json_schema_extra={"env": "COMPOSITION_SUBTITLES_FILE_NAME"},
    )
    audio_file_name: str = Field(
        default="public/audio.wav",
        json_schema_extra={"env": "COMPOSITION_AUDIO_FILE_NAME"},
    )
    rich_content_file_name: str = Field(
        default="public/output.json",
        json_schema_extra={"env": "COMPOSITION_RICH_CONTENT_FILE_NAME"},
    )
    wave_color: str = Field(
        default="#a3a5ae", json_schema_extra={"env": "COMPOSITION_WAVE_COLOR"}
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
        default="512", json_schema_extra={"env": "COMPOSITION_WAVE_NUMBER_OF_SAMPLES"}
    )


class Settings(BaseSettings):
    """
    Configuration settings for the backend.

    Attributes:
        APP_NAME (str): The name of the application.
        APP_VERSION (str): The version of the application.
        APP_DESCRIPTION (str): A brief description of the application.
        WHISPER_MODEL (str): The model name for the Whisper service.
        ELEVENLABS (ElevenLabsSettings): Settings for ElevenLabs service.
        OPENAI (OpenAISettings): Settings for OpenAI service.
        LOGGING (LoggingSettings): Settings for logging configuration.
        TEMP_DIR (Path): The directory for temporary files.
        PAPER_URL (str): The URL of the research paper.
        REQUESTS_TIMEOUT (int): The timeout for HTTP requests in seconds.
        VIDEO (VideoSettings): Settings for video processing.
        REMOTION (RemotionSettings): Settings for Remotion project.
        AUDIO (AudioSettings): Settings for audio processing.
        TIMEOUTS (TimeoutSettings): Timeout settings for various services.
        COMPOSITION_PROPS (CompositionPropsSettings): Settings for video composition properties.
        ALLOWED_DOMAINS (List[str]): The list of allowed domains for the application.
    """

    # Load environment variables
    # model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Application settings
    APP_NAME: str = Field(default="Arxflix", json_schema_extra={"env": "APP_NAME"})
    APP_VERSION: str = Field(default="0.1.0", json_schema_extra={"env": "APP_VERSION"})
    APP_DESCRIPTION: str = Field(
        default="A research paper summarizer",
        json_schema_extra={"env": "APP_DESCRIPTION"},
    )

    # Whisper settings
    WHISPER_MODEL: str = Field(
        default="tiny.en", json_schema_extra={"env": "WHISPER_MODEL"}
    )

    # ElevenLabs settings
    ELEVENLABS: ElevenLabsSettings = ElevenLabsSettings()

    # OpenAI settings
    OPENAI: OpenAISettings = OpenAISettings()

    # Logging settings
    LOGGING: LoggingSettings = LoggingSettings()

    # Temporary directory
    TEMP_DIR: Path = Field(
        default=Path("./audio"), json_schema_extra={"env": "TEMP_DIR"}
    )

    # Paper URL
    PAPER_URL: str = Field(default="", json_schema_extra={"env": "PAPER_URL"})

    # Requests
    REQUESTS_TIMEOUT: int = Field(
        default=10, json_schema_extra={"env": "REQUESTS_TIMEOUT"}
    )

    # Video settings
    VIDEO: VideoSettings = VideoSettings()

    # Remotion settings
    REMOTION: RemotionSettings = RemotionSettings()

    # Audio settings
    AUDIO: AudioSettings = AudioSettings()

    # Timeout settings
    TIMEOUTS: TimeoutSettings = TimeoutSettings()

    # Composition properties
    COMPOSITION_PROPS: CompositionPropsSettings = CompositionPropsSettings()

    ALLOWED_DOMAINS: List[str] = Field(
        default_factory=lambda: ["arxiv.org", "ar5iv.labs.arxiv.org", "ar5iv.org"],
        json_schema_extra={"env": "ALLOWED_DOMAINS"},
    )


settings = Settings()
