"""Local MLX TTS integration."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .client import LocalMlxTtsClient
from .const import CONF_BASE_URL, CONF_TIMEOUT, DOMAIN, PLATFORM_TTS


@dataclass(frozen=True, slots=True)
class LocalMlxTtsRuntimeData:
    """Entry-scoped client and immutable merged settings."""

    client: LocalMlxTtsClient
    settings: Mapping[str, Any]


async def _async_update_listener(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Reload an entry after its options change."""
    await hass.config_entries.async_reload(entry.entry_id)


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up one independent Local MLX TTS entry."""
    settings = {**entry.data, **entry.options}
    client = LocalMlxTtsClient(
        async_get_clientsession(hass),
        settings[CONF_BASE_URL],
        settings[CONF_TIMEOUT],
    )
    runtime = LocalMlxTtsRuntimeData(
        client=client,
        settings=MappingProxyType(settings),
    )
    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = runtime
    entry.async_on_unload(entry.add_update_listener(_async_update_listener))
    await hass.config_entries.async_forward_entry_setups(entry, [PLATFORM_TTS])
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a Local MLX TTS entry."""
    unloaded = await hass.config_entries.async_unload_platforms(entry, [PLATFORM_TTS])
    if unloaded:
        domain_data = hass.data.get(DOMAIN, {})
        domain_data.pop(entry.entry_id, None)
        if not domain_data:
            hass.data.pop(DOMAIN, None)
    return unloaded
