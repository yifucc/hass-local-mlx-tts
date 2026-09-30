"""Native TTS platform for Local MLX TTS."""

from __future__ import annotations

import math
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any, Mapping

from homeassistant.components.tts import TextToSpeechEntity, TtsAudioType
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.device_registry import DeviceEntryType
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from . import LocalMlxTtsRuntimeData
from .client import (
    InvalidGenerationOptionsError,
    InvalidReferenceOptionsError,
    LocalMlxTtsClient,
    LocalMlxTtsError,
    resolve_reference_path,
)
from .const import (
    CONF_LANGUAGE,
    CONF_MODEL,
    CONF_NAME,
    CONF_REFERENCE_ROOT,
    CONF_REF_AUDIO,
    CONF_REF_TEXT,
    CONF_RESPONSE_FORMAT,
    CONF_STREAM,
    CONF_TEMPERATURE,
    CONF_TIMEOUT,
    CONF_TOP_K,
    CONF_TOP_P,
    DOMAIN,
    HA_LANGUAGE_TO_MLX,
    MLX_LANGUAGE_TO_HA,
    SUPPORTED_LANGUAGES,
)

try:
    from homeassistant.components.tts import TTSAudioRequest, TTSAudioResponse
except ImportError:  # Home Assistant before the streaming entity API.
    @dataclass
    class TTSAudioRequest:
        """Compatibility shape for a Home Assistant streaming TTS request."""

        language: str
        options: dict[str, Any]
        message_gen: AsyncGenerator[str]

    @dataclass
    class TTSAudioResponse:
        """Compatibility shape for a Home Assistant streaming TTS response."""

        extension: str
        data_gen: AsyncGenerator[bytes]

SUPPORTED_OPTIONS = [
    CONF_REF_AUDIO,
    CONF_REF_TEXT,
    CONF_MODEL,
    CONF_TIMEOUT,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONF_TOP_K,
    CONF_STREAM,
]


def resolve_voice_options(
    options: Mapping[str, Any],
    *,
    default_ref_audio: str,
    default_ref_text: str,
    reference_root: str,
) -> tuple[str, str]:
    """Resolve a complete call-specific voice pair or configured defaults."""
    has_audio = CONF_REF_AUDIO in options
    has_text = CONF_REF_TEXT in options

    if has_audio != has_text:
        raise InvalidReferenceOptionsError(
            "Reference audio and transcript must be supplied together"
        )

    if has_audio:
        ref_audio = options[CONF_REF_AUDIO]
        ref_text = options[CONF_REF_TEXT]
        if (
            not isinstance(ref_audio, str)
            or not ref_audio.strip()
            or not isinstance(ref_text, str)
            or not ref_text.strip()
        ):
            raise InvalidReferenceOptionsError(
                "Reference audio and transcript overrides cannot be blank"
            )
    else:
        ref_audio = default_ref_audio
        ref_text = default_ref_text

    return resolve_reference_path(reference_root, ref_audio), ref_text


def resolve_generation_options(
    options: Mapping[str, Any], *, default_model: str
) -> dict[str, str | float | int]:
    """Validate and resolve per-call generation options."""
    model = options.get(CONF_MODEL, default_model)
    if not isinstance(model, str) or not (model := model.strip()):
        raise InvalidGenerationOptionsError("Model must be a non-empty string")

    resolved: dict[str, str | float | int] = {CONF_MODEL: model}

    for key, minimum, inclusive in (
        (CONF_TIMEOUT, 1.0, True),
        (CONF_TEMPERATURE, 0.0, True),
        (CONF_TOP_P, 0.0, False),
    ):
        if key not in options:
            continue
        value = options[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise InvalidGenerationOptionsError(f"{key} must be a number")
        number = float(value)
        if not math.isfinite(number) or (
            number < minimum if inclusive else number <= minimum
        ):
            raise InvalidGenerationOptionsError(f"{key} is outside its valid range")
        if key == CONF_TOP_P and number > 1:
            raise InvalidGenerationOptionsError("top_p must not exceed 1")
        resolved[key] = number

    if CONF_TOP_K in options:
        top_k = options[CONF_TOP_K]
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise InvalidGenerationOptionsError("top_k must be an integer of at least 1")
        resolved[CONF_TOP_K] = top_k

    return resolved


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the TTS entity for one config entry."""
    runtime: LocalMlxTtsRuntimeData = hass.data[DOMAIN][config_entry.entry_id]
    async_add_entities(
        [LocalMlxTtsEntity(config_entry, runtime.client, runtime.settings)]
    )


class LocalMlxTtsEntity(TextToSpeechEntity):
    """Home Assistant TTS entity backed by a local MLX Audio server."""

    def __init__(
        self,
        config_entry: ConfigEntry,
        client: LocalMlxTtsClient,
        settings: Mapping[str, Any],
    ) -> None:
        """Initialize an entry-scoped TTS entity."""
        self._client = client
        self._model = settings[CONF_MODEL]
        self._reference_root = settings[CONF_REFERENCE_ROOT]
        self._default_ref_audio = settings[CONF_REF_AUDIO]
        self._default_ref_text = settings[CONF_REF_TEXT]
        self._response_format = settings[CONF_RESPONSE_FORMAT]

        self._attr_unique_id = f"{config_entry.entry_id}-tts"
        self._attr_name = settings[CONF_NAME]
        self._attr_default_language = MLX_LANGUAGE_TO_HA[settings[CONF_LANGUAGE]]
        self._attr_supported_languages = list(SUPPORTED_LANGUAGES)
        self._attr_supported_options = SUPPORTED_OPTIONS
        # Keep entry defaults internal. Home Assistant merges default_options with
        # call options, which could otherwise turn a one-sided override into a
        # mismatched audio/transcript pair.
        self._attr_default_options = {}
        self._attr_device_info = {
            "identifiers": {(DOMAIN, config_entry.entry_id)},
            "name": settings[CONF_NAME],
            "manufacturer": "MLX Audio",
            "model": "Local HTTP TTS",
            "entry_type": DeviceEntryType.SERVICE,
        }

    async def async_get_tts_audio(
        self,
        message: str,
        language: str,
        options: dict[str, Any],
    ) -> TtsAudioType:
        """Generate speech using entry defaults or a complete call override."""
        try:
            ref_audio, ref_text = resolve_voice_options(
                options or {},
                default_ref_audio=self._default_ref_audio,
                default_ref_text=self._default_ref_text,
                reference_root=self._reference_root,
            )
            generation_options = resolve_generation_options(
                options or {}, default_model=self._model
            )
            result = await self._client.async_synthesize(
                text=message,
                ref_audio=ref_audio,
                ref_text=ref_text,
                lang_code=HA_LANGUAGE_TO_MLX[
                    language or self._attr_default_language
                ],
                response_format=self._response_format,
                **generation_options,
            )
        except LocalMlxTtsError as err:
            raise HomeAssistantError(
                "Local MLX TTS speech generation failed "
                f"({type(err).__name__})"
            ) from err

        return result.format, result.audio

    async def async_stream_tts_audio(
        self, request: TTSAudioRequest
    ) -> TTSAudioResponse:
        """Stream speech chunks as MLX Audio produces them."""
        message = "".join([chunk async for chunk in request.message_gen])
        options = request.options or {}
        try:
            use_streaming = options.get(CONF_STREAM, True)
            if not isinstance(use_streaming, bool):
                raise InvalidGenerationOptionsError("stream must be a boolean")
            ref_audio, ref_text = resolve_voice_options(
                options,
                default_ref_audio=self._default_ref_audio,
                default_ref_text=self._default_ref_text,
                reference_root=self._reference_root,
            )
            generation_options = resolve_generation_options(
                options, default_model=self._model
            )
            synthesis_arguments = {
                "text": message,
                "ref_audio": ref_audio,
                "ref_text": ref_text,
                "lang_code": HA_LANGUAGE_TO_MLX[
                    request.language or self._attr_default_language
                ],
                "response_format": self._response_format,
                **generation_options,
            }
            if not use_streaming:
                result = await self._client.async_synthesize(**synthesis_arguments)

                async def complete_audio() -> AsyncGenerator[bytes]:
                    yield result.audio

                return TTSAudioResponse(result.format, complete_audio())

            audio_stream = self._client.async_stream_synthesize(
                **synthesis_arguments
            )
        except LocalMlxTtsError as err:
            raise HomeAssistantError(
                "Local MLX TTS speech generation failed "
                f"({type(err).__name__})"
            ) from err

        async def data_gen() -> AsyncGenerator[bytes]:
            try:
                async for chunk in audio_stream:
                    yield chunk
            except LocalMlxTtsError as err:
                raise HomeAssistantError(
                    "Local MLX TTS speech generation failed "
                    f"({type(err).__name__})"
                ) from err

        return TTSAudioResponse(self._response_format, data_gen())
