"""Native TTS platform for Local MLX TTS."""

from __future__ import annotations

from typing import Any, Mapping

from homeassistant.components.tts import TextToSpeechEntity, TtsAudioType
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.device_registry import DeviceEntryType
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from . import LocalMlxTtsRuntimeData
from .client import (
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
    DOMAIN,
    SUPPORTED_LANGUAGES,
)


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
        self._attr_default_language = settings[CONF_LANGUAGE]
        self._attr_supported_languages = list(SUPPORTED_LANGUAGES)
        self._attr_supported_options = [CONF_REF_AUDIO, CONF_REF_TEXT]
        self._attr_default_options = {
            CONF_REF_AUDIO: self._default_ref_audio,
            CONF_REF_TEXT: self._default_ref_text,
        }
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
            result = await self._client.async_synthesize(
                model=self._model,
                text=message,
                ref_audio=ref_audio,
                ref_text=ref_text,
                lang_code=language or self._attr_default_language,
                response_format=self._response_format,
            )
        except LocalMlxTtsError as err:
            raise HomeAssistantError(
                "Local MLX TTS speech generation failed "
                f"({type(err).__name__})"
            ) from err

        return result.format, result.audio
