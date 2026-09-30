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
CONF_STREAM = "stream"
CONF_TEMPERATURE = "temperature"
CONF_TOP_P = "top_p"
CONF_TOP_K = "top_k"

DEFAULT_NAME = "Local MLX TTS"
DEFAULT_MODEL = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"
DEFAULT_LANGUAGE = "Chinese"
DEFAULT_RESPONSE_FORMAT = "wav"
DEFAULT_TIMEOUT = 300

HA_LANGUAGE_TO_MLX = {
    "zh-CN": "Chinese",
    "en-US": "English",
    "ja-JP": "Japanese",
    "ko-KR": "Korean",
    "de-DE": "German",
    "fr-FR": "French",
    "ru-RU": "Russian",
    "pt-PT": "Portuguese",
    "es-ES": "Spanish",
    "it-IT": "Italian",
    "auto": "auto",
}
MLX_LANGUAGE_TO_HA = {mlx: ha for ha, mlx in HA_LANGUAGE_TO_MLX.items()}
SUPPORTED_LANGUAGES = tuple(HA_LANGUAGE_TO_MLX)
SUPPORTED_MLX_LANGUAGES = tuple(MLX_LANGUAGE_TO_HA)
