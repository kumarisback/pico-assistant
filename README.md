# Pico — local voice assistant server

[![CI](https://github.com/kumarisback/pico-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/kumarisback/pico-assistant/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Fully local, no API keys. Runs on an Acer Nitro 5 (GTX 1050 Ti 4 GB, 8 GB RAM).
The ESP32-S3 is just a microphone and a speaker on a websocket — all the
intelligence lives here, so the firmware can be added last.

```
ESP32-S3 / mic_client ──ws──> server ──> VAD ──> Whisper ──> Qwen3-4B + tools ──> Piper ──> audio back
```

## Documentation

| Document | Contents |
|---|---|
| [docs/HLD.md](docs/HLD.md) | Architecture, component choices, resource budget, latency targets |
| [docs/LLD.md](docs/LLD.md) | Module internals, wire protocol, database schema, ESP32 contract |

## Layout

```
server/          assistant server
  config.py        YAML + env configuration
  vad.py           silero endpointer
  stt.py           faster-whisper
  llm.py           agent loop and tool dispatch
  tts.py           Piper streaming synthesis
  memory.py        SQLite + embeddings
  scheduler.py     reminder polling
  main.py          WebSocket server
  tools/           tool registry and implementations
clients/         chat_client (text) and mic_client (audio)
scripts/         setup, model downloads, llama-server launcher
docs/            HLD and LLD
tests/           offline unit tests
```

## Why these choices

| Decision | Reason |
|---|---|
| No PyTorch anywhere | torch costs ~2 GB RAM. CTranslate2 + onnxruntime do the same work in ~400 MB. |
| Whisper `base.en` | On clean English it matches `large-v3`, at 1/10 the size. |
| `compute_type: int8` (not `int8_float16`) | Pascal runs fp16 at **1/64** rate. fp16 on a 1050 Ti is slower than int8. |
| 10 broad tools, not 30 narrow ones | A 4B model picks reliably among a few general tools; narrow tools confuse it. |
| Search + notes as tools | The model has almost no world knowledge. Tools supply facts; the model supplies phrasing. |
| Reminders polled from SQLite | They must survive a restart. The DB is already the durable store. |

## Setup

Use **Python 3.11 or 3.12**. `onnxruntime`, `faster-whisper` and `piper-tts` do
not all ship wheels for 3.13+, and building them from source is not worth it.

```bash
cd ~/workspace/assistant
make setup          # venv + dependencies + VAD/TTS models
make models-llm     # 2.4 GB Qwen3-4B GGUF
```

`make help` lists every target. Without `make`, the equivalent scripts are
`bash scripts/setup.sh` and `bash scripts/download-models.sh --llm`.

<details>
<summary>Windows PowerShell</summary>

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
bash scripts\download-models.sh            # needs Git Bash
```
</details>

### llama.cpp with CUDA for Pascal

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61   # 61 = GTX 10-series
cmake --build build --config Release -j
export LLAMA_SERVER=$PWD/build/bin/llama-server
```

## Run

Three terminals:

```bash
make llama      # 1. LLM        :8081
make server     # 2. assistant  :8080
make chat       # 3. type to it
make mic        #    or talk to it
```

Start with `chat_client`. It skips STT and TTS entirely, so when something is
wrong you immediately know whether it is the audio path or the agent.

Health check: `curl localhost:8080/health`

## VRAM budget (4 GB card)

| | VRAM |
|---|---|
| Qwen3-4B Q4_K_M, 36 layers | ~2.4 GB |
| KV cache, 4k ctx, q8_0 | ~0.2 GB |
| Whisper base.en int8 | ~0.2 GB |
| Desktop compositor | ~0.3 GB |
| **Total** | **~3.1 GB** |

On CUDA OOM, in order: `NGL=28 bash scripts/run-llama.sh` → `NGL=20` →
set `stt.device: cpu` in config.yaml → switch to a Qwen3-1.7B GGUF.

## Tools

| Tool | Does |
|---|---|
| `now` | Current date, day, local time |
| `weather` | Open-Meteo, any city, keyless |
| `web_search` | DuckDuckGo, or SearxNG if you set `tools.search.searxng_url` |
| `fetch_page` | Full article text for summarising |
| `remember` / `recall` / `forget` | Notes and personal facts, semantic search |
| `schedule` / `list_schedule` / `cancel_schedule` | Timers, alarms, reminders |

Adding a tool: write an async handler in `server/tools/`, append a `Tool(...)` to
that module's `TOOLS` list, and add the module to `build_registry`. Nothing else.

The schemas cost ~1000 tokens on every request. That is the main reason to keep
the tool count low — it is a quarter of a 4k context window.

## Protocol

Identical for `mic_client` and the future firmware, so whatever works in Python
works on the ESP32.

**Client → server:** `{"type":"hello","token":"...","device_id":"..."}`, then
binary PCM16LE mono 16 kHz in 20 ms frames (640 bytes). Optional
`{"type":"end"}` to force endpointing, `{"type":"text","text":"..."}` to skip STT.

**Server → client:** `{"type":"state","value":"listening|thinking|speaking"}`,
`transcript`, `reply`, binary TTS frames, `audio_end`, and `push` for an
unprompted reminder.

Barge-in is handled server-side: VAD `SPEECH_START` cancels the in-flight
speaking task.

## Security

`server.auth_token` defaults to `change-me`. Before exposing this through a
Cloudflare tunnel, set a real one:

```bash
export ASSISTANT_TOKEN="$(head -c 24 /dev/urandom | base64)"
```

This endpoint carries a live microphone feed. An open websocket on a public URL
is a room bug.

## Tests

```bash
make test       # offline — no models, no network, no llama-server
make lint       # ruff
make fmt        # ruff fix + format
```

CI runs both on Python 3.11 and 3.12 for every push and pull request.

## Not done yet

- ESP32-S3 firmware (I2S → Opus → ws)
- Opus codec on the wire (currently raw PCM — 256 kbit/s, fine on LAN, heavy over a tunnel)
- Wake word (belongs on-device, via `esp-sr`)
- Cloudflare tunnel config
