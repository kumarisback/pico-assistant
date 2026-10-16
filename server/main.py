"""WebSocket voice assistant server.

Wire protocol (same for the ESP32 and the laptop mic client):

  client -> server
    text   {"type":"hello","token":"...","device_id":"..."}
    binary PCM16LE mono 16 kHz, 20 ms frames (640 bytes)
    text   {"type":"end"}          close the current utterance now
    text   {"type":"text","text":"..."}   skip STT (for testing)

  server -> client
    text   {"type":"ready"}
    text   {"type":"state","value":"listening|thinking|speaking"}
    text   {"type":"transcript","text":"..."}
    text   {"type":"reply","text":"..."}
    binary PCM16LE mono 16 kHz TTS frames
    text   {"type":"audio_end"}
    text   {"type":"push","text":"..."}    unprompted (a reminder fired)
    text   {"type":"error","message":"..."}
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from .config import Config
from .llm import Agent
from .memory import Memory
from .scheduler import ReminderService
from .stt import Transcriber
from .tools import ToolContext, build_registry
from .tts import Speaker
from .vad import Endpointer, VadEvent

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)-18s %(message)s"
)
log = logging.getLogger("assistant")

cfg = Config.load()
state: dict = {}
sessions: set[Session] = set()


async def broadcast_push(text: str) -> None:
    """Speak an unprompted message on every connected device (reminder fired)."""
    for session in list(sessions):
        try:
            await session.push(text)
        except Exception:  # noqa: BLE001 — one dead socket must not block the rest
            log.warning("push failed for %s", session.device_id)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    log.info("loading models (this takes ~20s on first run)")
    memory = Memory(cfg)
    registry = build_registry(ToolContext(cfg=cfg, memory=memory))

    state["memory"] = memory
    state["stt"] = Transcriber(cfg)
    state["tts"] = Speaker(cfg)
    state["agent"] = Agent(cfg, registry, memory)
    state["reminders"] = ReminderService(memory, broadcast_push)
    state["reminders"].start()

    if not await state["agent"].health():
        log.warning(
            "llama-server unreachable at %s — start it with scripts/run-llama.sh",
            cfg["llm"]["base_url"],
        )

    log.info("ready on %s:%s", cfg["server"]["host"], cfg["server"]["port"])
    yield

    await state["reminders"].stop()
    await state["agent"].close()


app = FastAPI(title="Pico Assistant", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "llm": await state["agent"].health(),
        "devices": [s.device_id for s in sessions],
    }


class Session:
    """One connected device. Owns its endpointer and its speaking task."""

    def __init__(self, ws: WebSocket, device_id: str) -> None:
        self.ws = ws
        self.device_id = device_id
        self.endpointer = Endpointer(cfg)
        self._speaking: asyncio.Task | None = None
        self._send_lock = asyncio.Lock()

    async def send_json(self, payload: dict) -> None:
        async with self._send_lock:
            await self.ws.send_json(payload)

    async def send_state(self, value: str) -> None:
        await self.send_json({"type": "state", "value": value})

    async def speak(self, text: str) -> None:
        await self.send_state("speaking")
        async for frame in state["tts"].stream(text):
            async with self._send_lock:
                await self.ws.send_bytes(frame)
        await self.send_json({"type": "audio_end"})
        await self.send_state("listening")

    async def stop_speaking(self) -> None:
        """Barge-in: the user started talking over the reply."""
        if self._speaking and not self._speaking.done():
            self._speaking.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._speaking
            await self.send_json({"type": "audio_end"})
        self._speaking = None

    async def handle_utterance(self, pcm: bytes) -> None:
        seconds = len(pcm) / 2 / cfg["audio"]["sample_rate"]
        log.info("[%s] utterance %.1fs", self.device_id, seconds)

        await self.send_state("thinking")
        text = await state["stt"].transcribe(pcm)
        if not text:
            log.info("[%s] empty transcript — ignoring", self.device_id)
            await self.send_state("listening")
            return

        log.info("[%s] heard: %s", self.device_id, text)
        await self.send_json({"type": "transcript", "text": text})
        await self.handle_text(text)

    async def handle_text(self, text: str) -> None:
        reply = await state["agent"].respond(text)
        log.info("[%s] reply: %s", self.device_id, reply)
        await self.send_json({"type": "reply", "text": reply})
        self._speaking = asyncio.create_task(self.speak(reply))

    async def push(self, text: str) -> None:
        await self.stop_speaking()
        await self.send_json({"type": "push", "text": text})
        self._speaking = asyncio.create_task(self.speak(text))


async def authenticate(ws: WebSocket) -> str | None:
    """Expect a hello frame first. Returns device_id, or None if rejected."""
    try:
        raw = await asyncio.wait_for(ws.receive_text(), timeout=10)
        hello = json.loads(raw)
    except (TimeoutError, json.JSONDecodeError, WebSocketDisconnect):
        return None

    if hello.get("type") != "hello":
        return None
    # compare_digest: the endpoint is reachable over a public tunnel.
    if not secrets.compare_digest(str(hello.get("token", "")), str(cfg["server"]["auth_token"])):
        log.warning("rejected connection with bad token")
        return None
    return str(hello.get("device_id", "unknown"))


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await ws.accept()

    device_id = await authenticate(ws)
    if device_id is None:
        await ws.close(code=4401, reason="unauthorized")
        return

    session = Session(ws, device_id)
    sessions.add(session)
    log.info("[%s] connected (%d device(s) online)", device_id, len(sessions))

    try:
        await session.send_json({"type": "ready"})
        await session.send_state("listening")

        while True:
            message = await ws.receive()

            if message.get("type") == "websocket.disconnect":
                break

            if (payload := message.get("bytes")) is not None:
                result = session.endpointer.push(payload)
                if result.event is VadEvent.SPEECH_START:
                    await session.stop_speaking()
                elif result.event in (VadEvent.SPEECH_END, VadEvent.TIMEOUT):
                    await session.handle_utterance(result.audio)

            elif (payload := message.get("text")) is not None:
                await handle_control(session, payload)

    except WebSocketDisconnect:
        pass
    except Exception:  # noqa: BLE001
        log.exception("[%s] session error", device_id)
    finally:
        await session.stop_speaking()
        sessions.discard(session)
        log.info("[%s] disconnected", device_id)


async def handle_control(session: Session, raw: str) -> None:
    try:
        msg = json.loads(raw)
    except json.JSONDecodeError:
        return

    kind = msg.get("type")
    if kind == "end":
        result = session.endpointer.flush()
        if result.event is VadEvent.SPEECH_END:
            await session.handle_utterance(result.audio)
    elif kind == "text":
        # Bypasses STT — used by tools/chat.py and for debugging the agent alone.
        text = (msg.get("text") or "").strip()
        if text:
            await session.stop_speaking()
            await session.send_json({"type": "transcript", "text": text})
            await session.handle_text(text)
    elif kind == "reset":
        session.endpointer.reset()
        state["memory"].clear_history()


def run() -> None:
    import uvicorn

    uvicorn.run(
        "server.main:app",
        host=cfg["server"]["host"],
        port=int(cfg["server"]["port"]),
        log_level="info",
    )


if __name__ == "__main__":
    run()
