"""Client-side domain helpers for Local MLX TTS."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import PurePosixPath
from urllib.parse import urlsplit, urlunsplit

import aiohttp


class LocalMlxTtsError(Exception):
    """Base exception for Local MLX TTS."""


class InvalidConfigurationError(LocalMlxTtsError):
    """Raised when integration configuration is invalid."""


class InvalidReferencePathError(LocalMlxTtsError):
    """Raised when a reference audio path is invalid."""


class InvalidReferenceOptionsError(LocalMlxTtsError):
    """Raised when reference audio and transcript options do not form a pair."""


class CannotConnectError(LocalMlxTtsError):
    """Raised when the MLX Audio server cannot be reached."""


class RequestTimeoutError(LocalMlxTtsError):
    """Raised when an MLX Audio request exceeds its timeout."""


class ServerResponseError(LocalMlxTtsError):
    """Raised when the MLX Audio server returns an unsuccessful response."""


class InvalidAudioResponseError(LocalMlxTtsError):
    """Raised when a successful response does not contain valid audio."""


@dataclass(frozen=True, slots=True)
class SynthesisResult:
    """Audio bytes and the format Home Assistant should cache."""

    format: str
    audio: bytes


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


class LocalMlxTtsClient:
    """Asynchronous HTTP client for an MLX Audio server."""

    def __init__(
        self,
        session: aiohttp.ClientSession,
        base_url: str,
        timeout: float,
    ) -> None:
        """Initialize the client without performing network I/O."""
        self._session = session
        self._base_url = normalize_base_url(base_url)
        self._timeout = aiohttp.ClientTimeout(total=timeout)

    async def async_validate_connection(self) -> None:
        """Validate that the MLX Audio model endpoint is reachable."""
        try:
            async with self._session.get(
                f"{self._base_url}/v1/models", timeout=self._timeout
            ) as response:
                if not 200 <= response.status < 300:
                    raise ServerResponseError(
                        f"MLX Audio returned HTTP {response.status}"
                    )
        except asyncio.TimeoutError as err:
            raise RequestTimeoutError("MLX Audio request timed out") from err
        except aiohttp.ClientError as err:
            raise CannotConnectError("Cannot connect to MLX Audio") from err

    async def async_synthesize(
        self,
        *,
        model: str,
        text: str,
        ref_audio: str,
        ref_text: str,
        lang_code: str,
        response_format: str,
    ) -> SynthesisResult:
        """Synthesize speech with a cloned reference voice."""
        payload = {
            "model": model,
            "input": text,
            "ref_audio": ref_audio,
            "ref_text": ref_text,
            "lang_code": lang_code,
            "response_format": response_format,
        }

        try:
            async with self._session.post(
                f"{self._base_url}/v1/audio/speech",
                json=payload,
                timeout=self._timeout,
            ) as response:
                if not 200 <= response.status < 300:
                    detail_bytes = await response.content.read(512)
                    detail = detail_bytes.decode("utf-8", errors="replace").strip()
                    message = f"MLX Audio returned HTTP {response.status}"
                    if detail:
                        message = f"{message}: {detail}"
                    raise ServerResponseError(message)

                audio = await response.read()
                if not audio:
                    raise InvalidAudioResponseError(
                        "MLX Audio returned an empty audio response"
                    )

                media_type = (
                    response.headers.get("Content-Type", "")
                    .partition(";")[0]
                    .strip()
                    .lower()
                )
                if media_type in {"", "application/octet-stream"}:
                    audio_format = response_format.lower()
                elif media_type.startswith("audio/"):
                    subtype = media_type.removeprefix("audio/")
                    audio_format = {
                        "mpeg": "mp3",
                        "x-wav": "wav",
                        "wave": "wav",
                    }.get(subtype, subtype)
                else:
                    raise InvalidAudioResponseError(
                        f"MLX Audio returned unsupported content type {media_type!r}"
                    )

                return SynthesisResult(format=audio_format, audio=audio)
        except asyncio.TimeoutError as err:
            raise RequestTimeoutError("MLX Audio request timed out") from err
        except aiohttp.ClientError as err:
            raise CannotConnectError("Cannot connect to MLX Audio") from err
