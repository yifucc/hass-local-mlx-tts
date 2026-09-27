"""Configuration flow for Local MLX TTS."""

from __future__ import annotations

import logging
from typing import Any

from homeassistant import config_entries
from homeassistant.core import callback
from homeassistant.helpers.aiohttp_client import async_get_clientsession
import voluptuous as vol

from .client import (
    CannotConnectError,
    InvalidConfigurationError,
    InvalidReferencePathError,
    LocalMlxTtsClient,
    RequestTimeoutError,
    normalize_base_url,
    resolve_reference_path,
)
from .const import (
    CONF_BASE_URL,
    CONF_LANGUAGE,
    CONF_MODEL,
    CONF_NAME,
    CONF_REFERENCE_ROOT,
    CONF_REF_AUDIO,
    CONF_REF_TEXT,
    CONF_RESPONSE_FORMAT,
    CONF_TIMEOUT,
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL,
    DEFAULT_NAME,
    DEFAULT_RESPONSE_FORMAT,
    DEFAULT_TIMEOUT,
    DOMAIN,
    SUPPORTED_LANGUAGES,
)

_LOGGER = logging.getLogger(__name__)


def _configuration_schema(defaults: dict[str, Any] | None = None) -> vol.Schema:
    """Build the user-facing configuration schema."""
    values = defaults or {}
    return vol.Schema(
        {
            vol.Required(
                CONF_NAME, default=values.get(CONF_NAME, DEFAULT_NAME)
            ): str,
            vol.Required(CONF_BASE_URL, default=values.get(CONF_BASE_URL, "")): str,
            vol.Required(
                CONF_MODEL, default=values.get(CONF_MODEL, DEFAULT_MODEL)
            ): str,
            vol.Required(
                CONF_REFERENCE_ROOT,
                default=values.get(CONF_REFERENCE_ROOT, ""),
            ): str,
            vol.Required(
                CONF_REF_AUDIO, default=values.get(CONF_REF_AUDIO, "")
            ): str,
            vol.Required(CONF_REF_TEXT, default=values.get(CONF_REF_TEXT, "")): str,
            vol.Required(
                CONF_LANGUAGE,
                default=values.get(CONF_LANGUAGE, DEFAULT_LANGUAGE),
            ): vol.In(SUPPORTED_LANGUAGES),
            vol.Required(
                CONF_RESPONSE_FORMAT,
                default=values.get(CONF_RESPONSE_FORMAT, DEFAULT_RESPONSE_FORMAT),
            ): vol.In(("wav", "mp3", "flac", "aac", "opus", "pcm")),
            vol.Required(
                CONF_TIMEOUT, default=values.get(CONF_TIMEOUT, DEFAULT_TIMEOUT)
            ): vol.All(vol.Coerce(float), vol.Range(min=1)),
        }
    )


class LocalMlxTtsConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle configuration for Local MLX TTS."""

    VERSION = 1

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> "LocalMlxTtsOptionsFlowHandler":
        """Return the options flow for an existing entry."""
        return LocalMlxTtsOptionsFlowHandler(config_entry)

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.ConfigFlowResult:
        """Create a Local MLX TTS config entry."""
        errors: dict[str, str] = {}

        if user_input is not None:
            try:
                normalized_url = normalize_base_url(user_input[CONF_BASE_URL])
            except InvalidConfigurationError:
                errors["base"] = "invalid_url"
            else:
                try:
                    resolve_reference_path(
                        user_input[CONF_REFERENCE_ROOT], user_input[CONF_REF_AUDIO]
                    )
                except InvalidConfigurationError:
                    errors["base"] = "invalid_reference_root"
                except InvalidReferencePathError:
                    errors["base"] = "invalid_reference_path"

            if not errors:
                client = LocalMlxTtsClient(
                    async_get_clientsession(self.hass),
                    normalized_url,
                    user_input[CONF_TIMEOUT],
                )
                try:
                    await client.async_validate_connection()
                except CannotConnectError:
                    errors["base"] = "cannot_connect"
                except RequestTimeoutError:
                    errors["base"] = "timeout"
                except Exception:  # noqa: BLE001 - flow must surface a stable error
                    _LOGGER.exception(
                        "Unexpected exception while validating MLX Audio connection"
                    )
                    errors["base"] = "unknown"

            if not errors:
                data = {**user_input, CONF_BASE_URL: normalized_url}
                return self.async_create_entry(title=data[CONF_NAME], data=data)

        return self.async_show_form(
            step_id="user",
            data_schema=_configuration_schema(user_input),
            errors=errors,
        )


class LocalMlxTtsOptionsFlowHandler(config_entries.OptionsFlow):
    """Edit a Local MLX TTS config entry."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        """Initialize with the entry being edited."""
        self.config_entry = config_entry

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.ConfigFlowResult:
        """Show and process integration options."""
        errors: dict[str, str] = {}
        merged = {**self.config_entry.data, **self.config_entry.options}

        if user_input is not None:
            try:
                normalized_url = normalize_base_url(user_input[CONF_BASE_URL])
            except InvalidConfigurationError:
                errors["base"] = "invalid_url"
            else:
                try:
                    resolve_reference_path(
                        user_input[CONF_REFERENCE_ROOT], user_input[CONF_REF_AUDIO]
                    )
                except InvalidConfigurationError:
                    errors["base"] = "invalid_reference_root"
                except InvalidReferencePathError:
                    errors["base"] = "invalid_reference_path"

            if not errors:
                client = LocalMlxTtsClient(
                    async_get_clientsession(self.hass),
                    normalized_url,
                    user_input[CONF_TIMEOUT],
                )
                try:
                    await client.async_validate_connection()
                except CannotConnectError:
                    errors["base"] = "cannot_connect"
                except RequestTimeoutError:
                    errors["base"] = "timeout"
                except Exception:  # noqa: BLE001 - stable flow error required
                    _LOGGER.exception(
                        "Unexpected exception while validating MLX Audio options"
                    )
                    errors["base"] = "unknown"

            if not errors:
                options = {**user_input, CONF_BASE_URL: normalized_url}
                return self.async_create_entry(title="", data=options)

        return self.async_show_form(
            step_id="init",
            data_schema=_configuration_schema(user_input or merged),
            errors=errors,
        )
