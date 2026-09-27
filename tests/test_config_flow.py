"""Tests for the Local MLX TTS configuration flow and lifecycle."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from homeassistant import config_entries
from homeassistant.data_entry_flow import FlowResultType
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.local_mlx_tts.client import (
    CannotConnectError,
    RequestTimeoutError,
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
)


USER_INPUT = {
    CONF_NAME: "Living Room Voice",
    CONF_BASE_URL: "http://192.168.1.20:8000/",
    CONF_MODEL: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
    CONF_REFERENCE_ROOT: "/Users/ifcc/Documents/qwen3-tts/reference",
    CONF_REF_AUDIO: "family/voice.m4a",
    CONF_REF_TEXT: "这是默认参考录音的逐字稿。",
    CONF_LANGUAGE: "Chinese",
    CONF_RESPONSE_FORMAT: "wav",
    CONF_TIMEOUT: 300,
}


async def _submit_user_flow(hass, user_input: dict[str, object]):
    initial = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    assert initial["type"] is FlowResultType.FORM
    return await hass.config_entries.flow.async_configure(
        initial["flow_id"], user_input=user_input
    )


async def test_user_flow_creates_entry_with_all_fields(hass) -> None:
    with patch(
        "custom_components.local_mlx_tts.config_flow.LocalMlxTtsClient"
    ) as client_class:
        client_class.return_value.async_validate_connection = AsyncMock()
        result = await _submit_user_flow(hass, USER_INPUT)

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["title"] == "Living Room Voice"
    assert result["data"] == {
        **USER_INPUT,
        CONF_BASE_URL: "http://192.168.1.20:8000",
    }
    client_class.return_value.async_validate_connection.assert_awaited_once()


@pytest.mark.parametrize(
    ("updates", "expected_error"),
    [
        ({CONF_BASE_URL: "192.168.1.20:8000"}, "invalid_url"),
        ({CONF_REFERENCE_ROOT: "relative/voices"}, "invalid_reference_root"),
        ({CONF_REF_AUDIO: "family/../voice.m4a"}, "invalid_reference_path"),
    ],
)
async def test_user_flow_reports_local_validation_errors(
    hass, updates: dict[str, object], expected_error: str
) -> None:
    with patch(
        "custom_components.local_mlx_tts.config_flow.LocalMlxTtsClient"
    ) as client_class:
        result = await _submit_user_flow(hass, {**USER_INPUT, **updates})

    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == {"base": expected_error}
    client_class.assert_not_called()


@pytest.mark.parametrize(
    ("error", "expected_error"),
    [
        (CannotConnectError("offline"), "cannot_connect"),
        (RequestTimeoutError("slow"), "timeout"),
        (RuntimeError("unexpected"), "unknown"),
    ],
)
async def test_user_flow_reports_connection_errors(
    hass, error: Exception, expected_error: str
) -> None:
    with patch(
        "custom_components.local_mlx_tts.config_flow.LocalMlxTtsClient"
    ) as client_class:
        client_class.return_value.async_validate_connection = AsyncMock(
            side_effect=error
        )
        result = await _submit_user_flow(hass, USER_INPUT)

    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == {"base": expected_error}


async def test_user_flow_allows_multiple_entries_for_same_server(hass) -> None:
    with patch(
        "custom_components.local_mlx_tts.config_flow.LocalMlxTtsClient"
    ) as client_class:
        client_class.return_value.async_validate_connection = AsyncMock()
        first = await _submit_user_flow(hass, USER_INPUT)
        second = await _submit_user_flow(
            hass, {**USER_INPUT, CONF_NAME: "Bedroom Voice"}
        )

    assert first["type"] is FlowResultType.CREATE_ENTRY
    assert second["type"] is FlowResultType.CREATE_ENTRY
    assert len(hass.config_entries.async_entries(DOMAIN)) == 2


async def test_options_flow_uses_merged_defaults_and_saves_changes(hass) -> None:
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Living Room Voice",
        data={**USER_INPUT, CONF_BASE_URL: "http://192.168.1.20:8000"},
        options={CONF_LANGUAGE: "English", CONF_TIMEOUT: 120},
    )
    entry.add_to_hass(hass)

    with patch(
        "custom_components.local_mlx_tts.config_flow.LocalMlxTtsClient"
    ) as client_class:
        client_class.return_value.async_validate_connection = AsyncMock()
        initial = await hass.config_entries.options.async_init(entry.entry_id)
        defaults = initial["data_schema"]({})

        assert initial["type"] is FlowResultType.FORM
        assert defaults[CONF_LANGUAGE] == "English"
        assert defaults[CONF_TIMEOUT] == 120
        assert defaults[CONF_MODEL] == USER_INPUT[CONF_MODEL]

        updated = {
            **USER_INPUT,
            CONF_NAME: "Updated Voice",
            CONF_BASE_URL: "http://192.168.1.20:8000/",
            CONF_LANGUAGE: "Japanese",
            CONF_TIMEOUT: 180,
        }
        result = await hass.config_entries.options.async_configure(
            initial["flow_id"], user_input=updated
        )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert entry.options == {
        **updated,
        CONF_BASE_URL: "http://192.168.1.20:8000",
    }


async def test_lifecycle_forwards_tts_and_unloads_runtime(hass) -> None:
    from custom_components.local_mlx_tts import async_setup_entry, async_unload_entry

    first = MockConfigEntry(
        domain=DOMAIN,
        title="First Voice",
        data={**USER_INPUT, CONF_NAME: "First Voice"},
        options={CONF_LANGUAGE: "English"},
    )
    second = MockConfigEntry(
        domain=DOMAIN,
        title="Second Voice",
        data={**USER_INPUT, CONF_NAME: "Second Voice"},
        options={CONF_LANGUAGE: "Japanese"},
    )
    first.add_to_hass(hass)
    second.add_to_hass(hass)

    with (
        patch(
            "custom_components.local_mlx_tts.LocalMlxTtsClient",
            side_effect=[object(), object()],
        ) as client_class,
        patch.object(
            hass.config_entries,
            "async_forward_entry_setups",
            AsyncMock(return_value=True),
        ) as forward,
        patch.object(
            hass.config_entries,
            "async_unload_platforms",
            AsyncMock(return_value=True),
        ) as unload,
    ):
        assert await async_setup_entry(hass, first)
        assert await async_setup_entry(hass, second)

        first_runtime = hass.data[DOMAIN][first.entry_id]
        second_runtime = hass.data[DOMAIN][second.entry_id]
        assert first_runtime is not second_runtime
        assert first_runtime.settings[CONF_LANGUAGE] == "English"
        assert second_runtime.settings[CONF_LANGUAGE] == "Japanese"
        assert first_runtime.client is not second_runtime.client
        assert client_class.call_count == 2
        assert forward.await_count == 2
        assert forward.await_args_list[0].args == (first, ["tts"])

        assert await async_unload_entry(hass, first)

    unload.assert_awaited_once_with(first, ["tts"])
    assert first.entry_id not in hass.data[DOMAIN]
    assert second.entry_id in hass.data[DOMAIN]


async def test_lifecycle_reloads_entry_after_options_update(hass) -> None:
    from custom_components.local_mlx_tts import async_setup_entry

    entry = MockConfigEntry(domain=DOMAIN, title="Voice", data=USER_INPUT)
    entry.add_to_hass(hass)

    with (
        patch("custom_components.local_mlx_tts.LocalMlxTtsClient"),
        patch.object(
            hass.config_entries,
            "async_forward_entry_setups",
            AsyncMock(return_value=True),
        ),
        patch.object(
            hass.config_entries, "async_reload", AsyncMock(return_value=True)
        ) as reload_entry,
    ):
        assert await async_setup_entry(hass, entry)
        hass.config_entries.async_update_entry(
            entry, options={CONF_LANGUAGE: "Korean"}
        )
        await hass.async_block_till_done()

    reload_entry.assert_awaited_once_with(entry.entry_id)
