# Local MLX TTS

<p align="center">
  <img src="custom_components/local_mlx_tts/brand/logo.png" alt="Local MLX TTS" width="560">
</p>

Local MLX TTS is a Home Assistant custom integration for a local
[`mlx-audio`](https://github.com/Blaizzy/mlx-audio) HTTP server. It exposes a
native Home Assistant TTS entity and supports Qwen3-TTS voice cloning with:

- one default reference recording and transcript per configuration entry;
- a different `ref_audio` and `ref_text` pair on every `tts.speak` call;
- absolute paths or paths relative to a configured server-side root;
- multiple independent TTS entries, including entries using the same server.

Reference files are never uploaded by Home Assistant. Every path is interpreted
on the Mac that runs MLX Audio.

## Requirements

- Apple Silicon Mac capable of running the selected MLX model.
- Python 3.12 for the MLX Audio environment.
- Home Assistant 2026.4.0 or newer.
- Home Assistant must be able to reach the Mac over TCP.
- `ffmpeg` on the Mac when reference recordings use M4A or another format that
  needs decoding.

The default model is:

```text
mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16
```

## Start MLX Audio on the Mac

This project does not use `uv`. Create a regular virtual environment and install
the TTS and HTTP server extras with pip:

```bash
mkdir -p ~/Documents/qwen3-tts
cd ~/Documents/qwen3-tts
/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -U "mlx-audio[tts,server]"
```

For M4A references, install FFmpeg if it is not already available:

```bash
brew install ffmpeg
ffmpeg -version
```

Start the server in the foreground. Binding to `0.0.0.0` is required when Home
Assistant runs on another machine or in a VM/container:

```bash
cd ~/Documents/qwen3-tts
source .venv/bin/activate
mlx_audio.server --host 0.0.0.0 --port 8000
```

To run it in the background with `nohup`:

```bash
cd ~/Documents/qwen3-tts
nohup .venv/bin/mlx_audio.server --host 0.0.0.0 --port 8000 > mlx-audio.log 2>&1 &
echo $! > mlx-audio.pid
```

Stop that background process with:

```bash
kill "$(cat mlx-audio.pid)"
```

Check the server locally:

```bash
curl --noproxy '*' http://127.0.0.1:8000/v1/models
```

Use the Mac's LAN address, not `127.0.0.1`, in Home Assistant when HA runs on a
different host. For example: `http://192.168.1.20:8000`. The MLX Audio endpoint
has no authentication by default, so expose it only on a trusted LAN and limit
access with the macOS firewall or network rules.

The first request may download the model and can be much slower than later
requests. The download uses the normal Hugging Face cache unless you configure a
different cache location for the MLX Audio process.

## Install the Home Assistant integration

### HACS custom repository

1. In HACS, open **Integrations**.
2. Open the menu and choose **Custom repositories**.
3. Add `https://github.com/ifcc/hass-local-mlx-tts` as an **Integration**.
4. Install **Local MLX TTS**.
5. Restart Home Assistant.

### Manual installation

Copy this directory:

```text
custom_components/local_mlx_tts
```

to this location in the Home Assistant configuration directory:

```text
/config/custom_components/local_mlx_tts
```

Restart Home Assistant afterward.

## Configure an entry

In Home Assistant, go to **Settings → Devices & services → Add integration** and
select **Local MLX TTS**.

| Field | Meaning |
| --- | --- |
| Name | Display name for this TTS entry and entity. |
| MLX Audio base URL | Server root such as `http://192.168.1.20:8000`; do not add `/v1/audio/speech`. |
| Model | Model ID sent in each request. The default is the 1.7B Base BF16 model above. |
| Reference audio root | Absolute directory on the MLX Audio Mac, such as `/Users/ifcc/Documents/qwen3-tts/reference`. |
| Default reference audio | Absolute server path or a path relative to the reference root. |
| Default reference transcript | Exact words spoken in the default recording. |
| Language | Default `lang_code` when a service call does not provide a language. |
| Response format | Requested audio format; `wav` is the default. |
| Request timeout | Maximum generation time in seconds; the default is `300`. |

The configuration form uses the language names expected by MLX Audio, such as
`Chinese` and `English`. Home Assistant service calls and Assist pipelines use
standard language tags such as `zh-CN` and `en-US`; the integration maps between
the two automatically.

An example default audio value of `family/voice.m4a` resolves to:

```text
/Users/ifcc/Documents/qwen3-tts/reference/family/voice.m4a
```

Relative paths containing a `..` component are rejected. Absolute override paths
are allowed, but still refer to the MLX Audio Mac—not the Home Assistant host.

## Call the TTS service

### Use the entry's default cloned voice

```yaml
action: tts.speak
target:
  entity_id: tts.living_room_voice
data:
  media_player_entity_id: media_player.living_room
  message: "欢迎回家，今天辛苦了。"
  language: zh-CN
```

### Override the cloned voice for one call

`ref_audio` and `ref_text` must always be provided together and both must be
nonblank:

```yaml
action: tts.speak
target:
  entity_id: tts.living_room_voice
data:
  media_player_entity_id: media_player.living_room
  message: "早上好，今天上海天气不错。"
  language: zh-CN
  cache: false
  options:
    ref_audio: family/guest.m4a
    ref_text: "早上好，欢迎来到我们的家。"
```

An absolute path works as well:

```yaml
options:
  ref_audio: /Volumes/Voices/guest.m4a
  ref_text: "早上好，欢迎来到我们的家。"
```

Home Assistant includes TTS options in its cache key. Repeating the same message,
language, and options can therefore reuse cached audio. If you replace a recording
without changing its path and need immediate regeneration, disable caching for
that call:

```yaml
action: tts.speak
target:
  entity_id: tts.living_room_voice
data:
  media_player_entity_id: media_player.living_room
  message: "请重新生成这段语音。"
  cache: false
```

## Update a configuration

Open **Settings → Devices & services → Local MLX TTS → Configure**. Saving the
options reloads only that configuration entry. Each entry keeps its own client,
default voice, language, model, response format, and timeout.

## Troubleshooting

### Cannot connect

- Verify `curl http://127.0.0.1:8000/v1/models` on the Mac.
- From the Home Assistant host, test the Mac's LAN URL.
- Bind the server to `0.0.0.0`, not `127.0.0.1`, for remote access.
- Check the macOS firewall and VLAN/container routing.
- Do not append an endpoint path, query, or URL fragment to the base URL.

### M4A fails to decode

Run `ffmpeg -version` in the same Mac environment. The M4A file must exist at the
resolved server path and be readable by the account running `mlx_audio.server`.

### First request times out

The model may still be downloading or loading. Watch `mlx-audio.log`, wait for
startup to finish, then retry. Increase the entry timeout if generation routinely
takes more than 300 seconds.

### The player cannot play the result

Start with `wav`. If the target media player has limited format support, edit the
entry and try a format supported by both MLX Audio and the player, such as `mp3`.
Also confirm that Home Assistant's TTS proxy URL is reachable by the player.

### Voice cloning sounds wrong

- Use a clean, single-speaker recording with little background noise.
- Ensure the transcript exactly matches the spoken recording.
- Avoid long silence at the beginning or end.
- Make sure the language passed to `tts.speak` matches the generated text.

## Remove the integration

1. Delete all Local MLX TTS entries in **Settings → Devices & services**.
2. Remove the integration in HACS, or delete
   `/config/custom_components/local_mlx_tts` for a manual installation.
3. Restart Home Assistant.

Removing the Home Assistant integration does not remove MLX models, reference
recordings, the Python environment, or its cache from the Mac.

## License

MIT
