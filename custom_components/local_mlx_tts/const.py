"""Constants for Local MLX TTS."""

from __future__ import annotations

DOMAIN = "local_mlx_tts"
PLATFORM_TTS = "tts"

CONF_NAME = "name"
CONF_BASE_URL = "base_url"
CONF_MODEL = "model"
CONF_REFERENCE_ROOT = "reference_root"
CONF_REF_AUDIO = "ref_audio"
CONF_REF_TEXT = "ref_text"
CONF_LANGUAGE = "language"
CONF_RESPONSE_FORMAT = "response_format"
CONF_TIMEOUT = "timeout"

DEFAULT_NAME = "Local MLX TTS"
DEFAULT_MODEL = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"
DEFAULT_LANGUAGE = "Chinese"
DEFAULT_RESPONSE_FORMAT = "wav"
DEFAULT_TIMEOUT = 300

SUPPORTED_LANGUAGES = (
    "Chinese",
    "English",
    "Japanese",
    "Korean",
    "German",
    "French",
    "Russian",
    "Portuguese",
    "Spanish",
    "Italian",
    "auto",
)
