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


async def test_synthesize_sends_generation_options(aiohttp_server) -> None:
    received: list[dict[str, object]] = []

    async def speech(request: web.Request) -> web.Response:
        received.append(await request.json())
        return web.Response(body=AUDIO_BYTES, content_type="audio/wav")

    app = web.Application()
    app.router.add_post("/v1/audio/speech", speech)
    server = await aiohttp_server(app)

    async with aiohttp.ClientSession() as session:
        client = LocalMlxTtsClient(session, str(server.make_url("")), timeout=1)
        await client.async_synthesize(
            **SYNTHESIS_ARGUMENTS,
            timeout=2,
            temperature=0.6,
            top_p=0.8,
            top_k=20,
        )

    assert received[0]["temperature"] == 0.6
    assert received[0]["top_p"] == 0.8
    assert received[0]["top_k"] == 20


async def test_stream_synthesize_yields_first_chunk_before_response_finishes(
    aiohttp_server,
) -> None:
    received: list[dict[str, object]] = []
    release_final_chunk = asyncio.Event()
    first_chunk = b"first audio chunk"
    final_chunk = b"final audio chunk"

    async def speech(request: web.Request) -> web.StreamResponse:
        received.append(await request.json())
        response = web.StreamResponse(headers={"Content-Type": "audio/mpeg"})
        await response.prepare(request)
        await response.write(first_chunk)
        await release_final_chunk.wait()
        await response.write(final_chunk)
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_post("/v1/audio/speech", speech)
    server = await aiohttp_server(app)
    arguments = {**SYNTHESIS_ARGUMENTS, "response_format": "mp3"}

    async with aiohttp.ClientSession() as session:
        client = LocalMlxTtsClient(session, str(server.make_url("")), timeout=1)
        stream = client.async_stream_synthesize(**arguments)

        assert await asyncio.wait_for(anext(stream), timeout=0.2) == first_chunk
        assert not release_final_chunk.is_set()

        release_final_chunk.set()
        remaining = b"".join([chunk async for chunk in stream])

    assert remaining == final_chunk
    assert received == [
        {
            "model": SYNTHESIS_ARGUMENTS["model"],
            "input": SYNTHESIS_ARGUMENTS["text"],
            "ref_audio": SYNTHESIS_ARGUMENTS["ref_audio"],
            "ref_text": SYNTHESIS_ARGUMENTS["ref_text"],
            "lang_code": SYNTHESIS_ARGUMENTS["lang_code"],
            "response_format": "mp3",
            "stream": True,
        }
    ]


async def test_stream_synthesize_rejects_empty_body(aiohttp_server) -> None:
    async def speech(request: web.Request) -> web.Response:
        await request.read()
        return web.Response(body=b"", content_type="audio/mpeg")

    app = web.Application()
    app.router.add_post("/v1/audio/speech", speech)
    server = await aiohttp_server(app)
    arguments = {**SYNTHESIS_ARGUMENTS, "response_format": "mp3"}

    async with aiohttp.ClientSession() as session:
        client = LocalMlxTtsClient(session, str(server.make_url("")), timeout=1)
        with pytest.raises(InvalidAudioResponseError):
            _ = [
                chunk
                async for chunk in client.async_stream_synthesize(**arguments)
            ]


async def test_stream_synthesize_rejects_and_redacts_server_error(
    aiohttp_server,
) -> None:
    secret_transcript = "do not leak this transcript"

    async def speech(request: web.Request) -> web.Response:
        await request.read()
        return web.Response(
            status=500,
            body=secret_transcript.encode() + b"\xff" * 4096,
        )

    app = web.Application()
    app.router.add_post("/v1/audio/speech", speech)
    server = await aiohttp_server(app)
    arguments = {**SYNTHESIS_ARGUMENTS, "ref_text": secret_transcript}

    async with aiohttp.ClientSession() as session:
        client = LocalMlxTtsClient(session, str(server.make_url("")), timeout=1)
        with pytest.raises(ServerResponseError) as error:
            _ = [
                chunk
                async for chunk in client.async_stream_synthesize(**arguments)
            ]

    assert secret_transcript not in str(error.value)
    assert len(str(error.value)) < 700


async def test_stream_synthesize_maps_timeout(aiohttp_server) -> None:
    async def speech(request: web.Request) -> web.Response:
        await request.read()
        await asyncio.sleep(0.05)
        return web.Response(body=AUDIO_BYTES, content_type="audio/wav")

    app = web.Application()
    app.router.add_post("/v1/audio/speech", speech)
    server = await aiohttp_server(app)

    async with aiohttp.ClientSession() as session:
        client = LocalMlxTtsClient(session, str(server.make_url("")), timeout=1)
        with pytest.raises(RequestTimeoutError):
            _ = [
                chunk
                async for chunk in client.async_stream_synthesize(
                    **SYNTHESIS_ARGUMENTS,
                    timeout=0.001,
                )
            ]


async def test_synthesize_uses_per_call_timeout(aiohttp_server) -> None:
    async def speech(request: web.Request) -> web.Response:
        await asyncio.sleep(0.05)
        return web.Response(body=AUDIO_BYTES, content_type="audio/wav")

    app = web.Application()
    app.router.add_post("/v1/audio/speech", speech)
    server = await aiohttp_server(app)

    async with aiohttp.ClientSession() as session:
        client = LocalMlxTtsClient(session, str(server.make_url("")), timeout=1)
        with pytest.raises(RequestTimeoutError):
            await client.async_synthesize(**SYNTHESIS_ARGUMENTS, timeout=0.001)


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
        return web.Response(
            status=500,
            body=secret_transcript.encode() + b"\xff" * 4096,
        )

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
