"""Contract tests for the asynchronous MLX Audio client."""

from __future__ import annotations

import asyncio

import aiohttp
from aiohttp import web
import pytest

from custom_components.local_mlx_tts.client import (
    CannotConnectError,
    InvalidAudioResponseError,
    LocalMlxTtsClient,
    RequestTimeoutError,
    ServerResponseError,
    SynthesisResult,
)

pytestmark = pytest.mark.enable_socket


AUDIO_BYTES = b"RIFF\x00\x00\x00\x00WAVEfmt "
SYNTHESIS_ARGUMENTS = {
    "model": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
    "text": "Home Assistant test",
    "ref_audio": "/srv/voices/voice.m4a",
    "ref_text": "Reference transcript",
    "lang_code": "Chinese",
    "response_format": "wav",
}


async def test_validate_connection_calls_models(aiohttp_server) -> None:
    requested_paths: list[str] = []

    async def models(request: web.Request) -> web.Response:
        requested_paths.append(request.path)
        return web.json_response({"data": []})

    app = web.Application()
    app.router.add_get("/v1/models", models)
    server = await aiohttp_server(app)

    async with aiohttp.ClientSession() as session:
        client = LocalMlxTtsClient(session, str(server.make_url("")), timeout=1)
        await client.async_validate_connection()

    assert requested_paths == ["/v1/models"]


async def test_validate_connection_refused(unused_tcp_port: int) -> None:
    port = unused_tcp_port

    async with aiohttp.ClientSession() as session:
        client = LocalMlxTtsClient(session, f"http://127.0.0.1:{port}", timeout=1)
        with pytest.raises(CannotConnectError):
            await client.async_validate_connection()


async def test_validate_connection_timeout(aiohttp_server) -> None:
    async def slow_models(request: web.Request) -> web.Response:
        await asyncio.sleep(0.05)
        return web.json_response({"data": []})

    app = web.Application()
    app.router.add_get("/v1/models", slow_models)
    server = await aiohttp_server(app)

    async with aiohttp.ClientSession() as session:
        client = LocalMlxTtsClient(session, str(server.make_url("")), timeout=0.001)
        with pytest.raises(RequestTimeoutError):
            await client.async_validate_connection()


async def test_validate_connection_server_error(aiohttp_server) -> None:
    async def models(request: web.Request) -> web.Response:
        return web.Response(status=503, text="warming up")

    app = web.Application()
    app.router.add_get("/v1/models", models)
    server = await aiohttp_server(app)

    async with aiohttp.ClientSession() as session:
        client = LocalMlxTtsClient(session, str(server.make_url("")), timeout=1)
        with pytest.raises(ServerResponseError):
            await client.async_validate_connection()


@pytest.mark.parametrize("content_type", ["audio/wav", "audio/wav; charset=binary"])
async def test_synthesize_sends_contract_and_accepts_wav(
    aiohttp_server, content_type: str
) -> None:
    received: list[dict[str, object]] = []

    async def speech(request: web.Request) -> web.Response:
        received.append(await request.json())
        return web.Response(body=AUDIO_BYTES, headers={"Content-Type": content_type})

    app = web.Application()
    app.router.add_post("/v1/audio/speech", speech)
    server = await aiohttp_server(app)

    async with aiohttp.ClientSession() as session:
        client = LocalMlxTtsClient(session, str(server.make_url("")), timeout=1)
        result = await client.async_synthesize(**SYNTHESIS_ARGUMENTS)

    assert received == [
        {
            "model": SYNTHESIS_ARGUMENTS["model"],
            "input": SYNTHESIS_ARGUMENTS["text"],
            "ref_audio": SYNTHESIS_ARGUMENTS["ref_audio"],
            "ref_text": SYNTHESIS_ARGUMENTS["ref_text"],
            "lang_code": SYNTHESIS_ARGUMENTS["lang_code"],
            "response_format": SYNTHESIS_ARGUMENTS["response_format"],
        }
    ]
    assert result == SynthesisResult(format="wav", audio=AUDIO_BYTES)


@pytest.mark.parametrize("content_type", [None, "application/octet-stream"])
async def test_synthesize_uses_requested_format_for_generic_content(
    aiohttp_server, content_type: str | None
) -> None:
    async def speech(request: web.Request) -> web.StreamResponse:
        if content_type is not None:
            return web.Response(body=AUDIO_BYTES, headers={"Content-Type": content_type})
        response = web.StreamResponse(status=200)
        await response.prepare(request)
        await response.write(AUDIO_BYTES)
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_post("/v1/audio/speech", speech)
    server = await aiohttp_server(app)

    async with aiohttp.ClientSession() as session:
        client = LocalMlxTtsClient(session, str(server.make_url("")), timeout=1)
        result = await client.async_synthesize(**SYNTHESIS_ARGUMENTS)

    assert result == SynthesisResult(format="wav", audio=AUDIO_BYTES)


@pytest.mark.parametrize("content_type", ["application/json", "text/plain"])
async def test_synthesize_rejects_non_audio_success(
    aiohttp_server, content_type: str
) -> None:
    async def speech(request: web.Request) -> web.Response:
        return web.Response(body=b"not audio", headers={"Content-Type": content_type})

    app = web.Application()
    app.router.add_post("/v1/audio/speech", speech)
    server = await aiohttp_server(app)

    async with aiohttp.ClientSession() as session:
        client = LocalMlxTtsClient(session, str(server.make_url("")), timeout=1)
        with pytest.raises(InvalidAudioResponseError):
            await client.async_synthesize(**SYNTHESIS_ARGUMENTS)


async def test_synthesize_rejects_empty_body(aiohttp_server) -> None:
    async def speech(request: web.Request) -> web.Response:
        return web.Response(body=b"", headers={"Content-Type": "audio/wav"})

    app = web.Application()
    app.router.add_post("/v1/audio/speech", speech)
    server = await aiohttp_server(app)

    async with aiohttp.ClientSession() as session:
        client = LocalMlxTtsClient(session, str(server.make_url("")), timeout=1)
        with pytest.raises(InvalidAudioResponseError):
            await client.async_synthesize(**SYNTHESIS_ARGUMENTS)


async def test_synthesize_timeout(aiohttp_server) -> None:
    async def speech(request: web.Request) -> web.Response:
        await asyncio.sleep(0.05)
        return web.Response(body=AUDIO_BYTES, content_type="audio/wav")

    app = web.Application()
    app.router.add_post("/v1/audio/speech", speech)
    server = await aiohttp_server(app)

    async with aiohttp.ClientSession() as session:
        client = LocalMlxTtsClient(session, str(server.make_url("")), timeout=0.001)
        with pytest.raises(RequestTimeoutError):
            await client.async_synthesize(**SYNTHESIS_ARGUMENTS)


async def test_synthesize_connection_refused(unused_tcp_port: int) -> None:
    async with aiohttp.ClientSession() as session:
        client = LocalMlxTtsClient(
            session, f"http://127.0.0.1:{unused_tcp_port}", timeout=1
        )
        with pytest.raises(CannotConnectError):
            await client.async_synthesize(**SYNTHESIS_ARGUMENTS)


async def test_synthesize_truncates_non_utf8_server_error(aiohttp_server) -> None:
    secret_transcript = "do not leak this transcript"

    async def speech(request: web.Request) -> web.Response:
        await request.read()
        return web.Response(status=500, body=b"\xff" * 4096)

    app = web.Application()
    app.router.add_post("/v1/audio/speech", speech)
    server = await aiohttp_server(app)
    arguments = {**SYNTHESIS_ARGUMENTS, "ref_text": secret_transcript}

    async with aiohttp.ClientSession() as session:
        client = LocalMlxTtsClient(session, str(server.make_url("")), timeout=1)
        with pytest.raises(ServerResponseError) as error:
            await client.async_synthesize(**arguments)

    assert secret_transcript not in str(error.value)
    assert len(str(error.value)) < 700
