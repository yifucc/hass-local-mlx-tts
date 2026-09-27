"""Client-side domain helpers for Local MLX TTS."""

from __future__ import annotations

from pathlib import PurePosixPath
from urllib.parse import urlsplit, urlunsplit


class LocalMlxTtsError(Exception):
    """Base exception for Local MLX TTS."""


class InvalidConfigurationError(LocalMlxTtsError):
    """Raised when integration configuration is invalid."""


class InvalidReferencePathError(LocalMlxTtsError):
    """Raised when a reference audio path is invalid."""


class InvalidReferenceOptionsError(LocalMlxTtsError):
    """Raised when reference audio and transcript options do not form a pair."""


def normalize_base_url(value: str) -> str:
    """Validate and normalize an MLX Audio base URL."""
    if not isinstance(value, str) or not (candidate := value.strip()):
        raise InvalidConfigurationError("The MLX Audio base URL is required")

    try:
        parsed = urlsplit(candidate)
        _ = parsed.port
    except ValueError as err:
        raise InvalidConfigurationError("The MLX Audio base URL is invalid") from err

    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise InvalidConfigurationError(
            "The MLX Audio base URL must use HTTP or HTTPS and include a host"
        )
    if parsed.query or parsed.fragment:
        raise InvalidConfigurationError(
            "The MLX Audio base URL cannot contain a query or fragment"
        )

    normalized_path = parsed.path.rstrip("/")
    return urlunsplit(
        (parsed.scheme.lower(), parsed.netloc, normalized_path, "", "")
    )


def resolve_reference_path(reference_root: str, ref_audio: str) -> str:
    """Resolve a server-side audio path without accessing the filesystem."""
    if not isinstance(reference_root, str) or not (root_value := reference_root.strip()):
        raise InvalidConfigurationError("The reference root is required")

    root = PurePosixPath(root_value)
    if not root.is_absolute():
        raise InvalidConfigurationError("The reference root must be an absolute path")

    if not isinstance(ref_audio, str) or not (audio_value := ref_audio.strip()):
        raise InvalidReferencePathError("The reference audio path is required")

    audio_path = PurePosixPath(audio_value)
    if audio_path.is_absolute():
        return str(audio_path)
    if ".." in audio_path.parts:
        raise InvalidReferencePathError(
            "Relative reference audio paths cannot contain '..'"
        )

    return str(root / audio_path)

