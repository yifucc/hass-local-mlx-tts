"""Tests for the native Local MLX TTS entity."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from homeassistant.exceptions import HomeAssistantError
from homeassistant.util import language as language_util
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.local_mlx_tts import LocalMlxTtsRuntimeData
from custom_components.local_mlx_tts.client import (
    CannotConnectError,
    InvalidReferenceOptionsError,
    InvalidReferencePathError,
    SynthesisResult,
)
from custom_components.local_mlx_tts.const import (
    CONF_BASE_URL,
    CONF_LANGUAGE,
    CONF_MODEL,
    CONF_NAME,
    CONF_REFERENCE_ROOT,
    CONF_REF_AUDIO,
    CONF_REF_TEXT,
    CONF_RESPONSE_FORMAT,
    CONF_TIMEOUT,
    DOMAIN,
    SUPPORTED_LANGUAGES,
)
from custom_components.local_mlx_tts.tts import (
    LocalMlxTtsEntity,
    async_setup_entry,
    resolve_voice_options,
)


REFERENCE_ROOT = "/srv/mlx/reference"
ENTRY_SETTINGS = {
    CONF_NAME: "Living Room Voice",
    CONF_BASE_URL: "http://127.0.0.1:8000",
    CONF_MODEL: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
    CONF_REFERENCE_ROOT: REFERENCE_ROOT,
    CONF_REF_AUDIO: "default.m4a",
    CONF_REF_TEXT: "Default transcript",
    CONF_LANGUAGE: "Chinese",
    CONF_RESPONSE_FORMAT: "wav",
    CONF_TIMEOUT: 300,
}


@dataclass
class RecordingClient:
    """Small client fake recording the entity-to-client contract."""

    result: SynthesisResult = field(
        default_factory=lambda: SynthesisResult("wav", b"generated audio")
    )
    error: Exception | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def async_synthesize(self, **kwargs: Any) -> SynthesisResult:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.result


def _entity(
    *,
    settings: dict[str, Any] | None = None,
    client: RecordingClient | None = None,
    entry_id: str = "entry-1",
) -> tuple[LocalMlxTtsEntity, RecordingClient]:
    fake_client = client or RecordingClient()
    entry = MockConfigEntry(
        domain=DOMAIN,
        title=(settings or ENTRY_SETTINGS)[CONF_NAME],
        data=settings or ENTRY_SETTINGS,
        entry_id=entry_id,
    )
    return LocalMlxTtsEntity(entry, fake_client, settings or ENTRY_SETTINGS), fake_client


def test_voice_options_use_configured_defaults() -> None:
    assert resolve_voice_options(
        {},
        default_ref_audio="default.m4a",
        default_ref_text="Default transcript",
        reference_root=REFERENCE_ROOT,
    ) == ("/srv/mlx/reference/default.m4a", "Default transcript")


def test_voice_options_use_complete_call_override_without_changing_text() -> None:
    assert resolve_voice_options(
        {"ref_audio": "family/voice.wav", "ref_text": "  Exact transcript  "},
        default_ref_audio="default.m4a",
        default_ref_text="Default transcript",
        reference_root=REFERENCE_ROOT,
    ) == ("/srv/mlx/reference/family/voice.wav", "  Exact transcript  ")


def test_voice_options_accept_absolute_override() -> None:
    assert resolve_voice_options(
        {"ref_audio": "/Volumes/Voices/./voice.m4a", "ref_text": "Transcript"},
        default_ref_audio="default.m4a",
        default_ref_text="Default transcript",
        reference_root=REFERENCE_ROOT,
    ) == ("/Volumes/Voices/voice.m4a", "Transcript")


@pytest.mark.parametrize(
    "options",
    [
        {"ref_audio": "voice.m4a"},
        {"ref_text": "Transcript"},
        {"ref_audio": "", "ref_text": "Transcript"},
        {"ref_audio": "voice.m4a", "ref_text": ""},
        {"ref_audio": None, "ref_text": "Transcript"},
        {"ref_audio": "voice.m4a", "ref_text": None},
        {"ref_audio": "   ", "ref_text": "Transcript"},
        {"ref_audio": "voice.m4a", "ref_text": "   "},
    ],
)
def test_voice_options_reject_incomplete_or_blank_override(options) -> None:
    with pytest.raises(InvalidReferenceOptionsError):
        resolve_voice_options(
            options,
            default_ref_audio="default.m4a",
            default_ref_text="Default transcript",
            reference_root=REFERENCE_ROOT,
        )


def test_voice_options_reject_relative_parent_traversal() -> None:
    with pytest.raises(InvalidReferencePathError):
        resolve_voice_options(
            {"ref_audio": "../voice.m4a", "ref_text": "Transcript"},
            default_ref_audio="default.m4a",
            default_ref_text="Default transcript",
            reference_root=REFERENCE_ROOT,
        )


def test_entity_metadata_languages_and_options() -> None:
    entity, _ = _entity()

    assert entity.unique_id == "entry-1-tts"
    assert entity.name == "Living Room Voice"
    assert entity.default_language == "zh-CN"
    assert entity.supported_languages == list(SUPPORTED_LANGUAGES)
    assert language_util.matches(
        "zh-Hans", entity.supported_languages, country="CN"
    ) == ["zh-CN"]
    assert "Chinese" not in entity.supported_languages
    assert "auto" in entity.supported_languages
    assert entity.supported_options == [CONF_REF_AUDIO, CONF_REF_TEXT]
    # Entry defaults are resolved inside the entity. Exposing them here would let
    # Home Assistant merge a one-sided call override with the other default.
    assert entity.default_options == {}
    assert entity.device_info["identifiers"] == {(DOMAIN, "entry-1")}
    assert entity.device_info["name"] == "Living Room Voice"
    assert entity.device_info["manufacturer"] == "MLX Audio"
    assert entity.device_info["model"] == "Local HTTP TTS"


async def test_entity_uses_configured_language_and_default_voice() -> None:
    entity, client = _entity()

    result = await entity.async_get_tts_audio(
        message="欢迎回家", language="", options={}
    )

    assert result == ("wav", b"generated audio")
    assert client.calls == [
        {
            "model": ENTRY_SETTINGS[CONF_MODEL],
            "text": "欢迎回家",
            "ref_audio": "/srv/mlx/reference/default.m4a",
            "ref_text": "Default transcript",
            "lang_code": "Chinese",
            "response_format": "wav",
        }
    ]


async def test_entity_uses_explicit_language_and_voice_override() -> None:
    entity, client = _entity()

    result = await entity.async_get_tts_audio(
        message="Good morning",
        language="en-US",
        options={
            CONF_REF_AUDIO: "/Volumes/Voices/guest.m4a",
            CONF_REF_TEXT: "  Guest transcript  ",
        },
    )

    assert result == ("wav", b"generated audio")
    assert client.calls[0]["lang_code"] == "English"
    assert client.calls[0]["ref_audio"] == "/Volumes/Voices/guest.m4a"
    assert client.calls[0]["ref_text"] == "  Guest transcript  "


async def test_entity_converts_client_errors() -> None:
    entity, _ = _entity(client=RecordingClient(error=CannotConnectError("offline")))

    with pytest.raises(
        HomeAssistantError, match="speech generation failed.*CannotConnectError"
    ):
        await entity.async_get_tts_audio(
            message="Message", language="zh-CN", options={}
        )


async def test_entities_keep_default_voices_independent() -> None:
    first_settings = {
        **ENTRY_SETTINGS,
        CONF_NAME: "First Voice",
        CONF_REF_AUDIO: "first.m4a",
        CONF_REF_TEXT: "First transcript",
    }
    second_settings = {
        **ENTRY_SETTINGS,
        CONF_NAME: "Second Voice",
        CONF_REF_AUDIO: "second.m4a",
        CONF_REF_TEXT: "Second transcript",
    }
    first, first_client = _entity(
        settings=first_settings, entry_id="first-entry"
    )
    second, second_client = _entity(
        settings=second_settings, entry_id="second-entry"
    )

    await first.async_get_tts_audio("One", "zh-CN", {})
    await second.async_get_tts_audio("Two", "zh-CN", {})

    assert first_client.calls[0]["ref_audio"].endswith("/first.m4a")
    assert first_client.calls[0]["ref_text"] == "First transcript"
    assert second_client.calls[0]["ref_audio"].endswith("/second.m4a")
    assert second_client.calls[0]["ref_text"] == "Second transcript"


async def test_entity_platform_adds_runtime_entity(hass) -> None:
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Living Room Voice",
        data=ENTRY_SETTINGS,
        entry_id="platform-entry",
    )
    client = RecordingClient()
    runtime = LocalMlxTtsRuntimeData(client=client, settings=ENTRY_SETTINGS)
    hass.data[DOMAIN] = {entry.entry_id: runtime}
    added: list[LocalMlxTtsEntity] = []

    await async_setup_entry(hass, entry, lambda entities: added.extend(entities))

    assert len(added) == 1
    assert added[0].unique_id == "platform-entry-tts"
