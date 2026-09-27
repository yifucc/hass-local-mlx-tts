"""Tests for URL normalization and server-side reference paths."""

from __future__ import annotations

import pytest

from custom_components.local_mlx_tts.client import (
    InvalidConfigurationError,
    InvalidReferencePathError,
    normalize_base_url,
    resolve_reference_path,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("http://192.168.1.20:8000/", "http://192.168.1.20:8000"),
        ("http://mlx-host.local:8000", "http://mlx-host.local:8000"),
        ("https://tts.example.test/", "https://tts.example.test"),
    ],
)
def test_normalize_base_url(value: str, expected: str) -> None:
    assert normalize_base_url(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "mlx-host.local:8000",
        "ftp://mlx-host.local",
        "http:///v1/audio/speech",
        "http://mlx-host.local?profile=voice",
        "http://mlx-host.local#server",
    ],
)
def test_invalid_base_url(value: str) -> None:
    with pytest.raises(InvalidConfigurationError):
        normalize_base_url(value)


def test_resolve_relative_reference_path() -> None:
    assert resolve_reference_path(
        "/Users/ifcc/Documents/qwen3-tts/reference", "voice.m4a"
    ) == "/Users/ifcc/Documents/qwen3-tts/reference/voice.m4a"


def test_resolve_nested_reference_path() -> None:
    assert resolve_reference_path(
        "/srv/mlx/reference", "family/voice.wav"
    ) == "/srv/mlx/reference/family/voice.wav"


def test_absolute_reference_path_is_independent_of_root() -> None:
    assert (
        resolve_reference_path("/srv/mlx/reference", "/Volumes/Voices/./voice.m4a")
        == "/Volumes/Voices/voice.m4a"
    )


@pytest.mark.parametrize("root", ["", "   ", "relative/reference"])
def test_invalid_reference_root(root: str) -> None:
    with pytest.raises(InvalidConfigurationError):
        resolve_reference_path(root, "voice.m4a")


@pytest.mark.parametrize("audio", ["", "   "])
def test_empty_reference_audio(audio: str) -> None:
    with pytest.raises(InvalidReferencePathError):
        resolve_reference_path("/srv/mlx/reference", audio)


@pytest.mark.parametrize(
    "audio",
    ["../voice.m4a", "family/../voice.m4a", "family/../../voice.m4a"],
)
def test_relative_parent_traversal_is_rejected(audio: str) -> None:
    with pytest.raises(InvalidReferencePathError):
        resolve_reference_path("/srv/mlx/reference", audio)

