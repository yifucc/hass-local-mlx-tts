"""Release metadata and translation consistency tests."""

from __future__ import annotations

import json
import struct
import tomllib
from pathlib import Path

import pytest

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


ROOT = Path(__file__).parents[1]
INTEGRATION = ROOT / "custom_components" / DOMAIN
CONFIG_FIELDS = {
    CONF_NAME,
    CONF_BASE_URL,
    CONF_MODEL,
    CONF_REFERENCE_ROOT,
    CONF_REF_AUDIO,
    CONF_REF_TEXT,
    CONF_LANGUAGE,
    CONF_RESPONSE_FORMAT,
    CONF_TIMEOUT,
}

BRAND_IMAGES = {
    "icon.png": (256, 256),
    "icon@2x.png": (512, 512),
    "logo.png": (768, 256),
    "logo@2x.png": (1536, 512),
}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_all_json_files_parse() -> None:
    json_files = [ROOT / "hacs.json", *INTEGRATION.rglob("*.json")]
    assert json_files
    for path in json_files:
        assert isinstance(_read_json(path), dict), path


def test_manifest_release_metadata() -> None:
    manifest = _read_json(INTEGRATION / "manifest.json")
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert manifest["domain"] == DOMAIN
    assert manifest["name"] == "Local MLX TTS"
    assert manifest["version"] == "0.1.1"
    assert project["project"]["version"] == manifest["version"]
    assert manifest["config_flow"] is True
    assert manifest["integration_type"] == "service"
    assert manifest["iot_class"] == "local_polling"
    assert manifest["requirements"] == []
    assert manifest["documentation"].endswith("/hass-local-mlx-tts")


def test_hacs_metadata_and_license() -> None:
    hacs = _read_json(ROOT / "hacs.json")

    assert hacs["name"] == "Local MLX TTS"
    assert hacs["render_readme"] is True
    assert hacs["homeassistant"] == "2026.4.0"
    assert "MIT License" in (ROOT / "LICENSE").read_text(encoding="utf-8")


@pytest.mark.parametrize(("filename", "expected_size"), BRAND_IMAGES.items())
def test_brand_image_is_transparent_png(
    filename: str, expected_size: tuple[int, int]
) -> None:
    image = (INTEGRATION / "brand" / filename).read_bytes()

    assert image[:8] == b"\x89PNG\r\n\x1a\n"
    assert image[12:16] == b"IHDR"
    width, height, bit_depth, color_type = struct.unpack(">IIBB", image[16:26])
    assert (width, height) == expected_size
    assert bit_depth == 8
    assert color_type in {4, 6}


@pytest.mark.parametrize(
    "path",
    [
        INTEGRATION / "strings.json",
        INTEGRATION / "translations" / "en.json",
        INTEGRATION / "translations" / "zh-Hans.json",
    ],
)
def test_config_and_options_fields_have_strings(path: Path) -> None:
    translations = _read_json(path)

    assert set(translations["config"]["step"]["user"]["data"]) == CONFIG_FIELDS
    assert set(translations["options"]["step"]["init"]["data"]) == CONFIG_FIELDS
    assert set(translations["config"]["error"]) == {
        "invalid_url",
        "invalid_reference_root",
        "invalid_reference_path",
        "invalid_reference_options",
        "cannot_connect",
        "timeout",
        "unknown",
    }


def test_every_python_module_compiles() -> None:
    python_files = [*INTEGRATION.rglob("*.py"), *(ROOT / "tests").rglob("*.py")]
    assert python_files
    for path in python_files:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
