"""Speech to text via faster-whisper (CTranslate2 — no torch)."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import numpy as np
from faster_whisper import WhisperModel

if TYPE_CHECKING:
    from .config import Config

log = logging.getLogger(__name__)


class Transcriber:
    def __init__(self, cfg: Config) -> None:
        s = cfg["stt"]
        self._beam_size = int(s["beam_size"])
        download_root = str(cfg.path("stt", "download_root"))

        log.info("loading whisper %s on %s (%s)", s["model"], s["device"], s["compute_type"])
        try:
            self._model = WhisperModel(
                s["model"],
                device=s["device"],
                compute_type=s["compute_type"],
                download_root=download_root,
            )
        except Exception as exc:  # noqa: BLE001 — CUDA absence must not be fatal
            # A missing/unsupported GPU is the common case on a dev laptop; CPU
            # int8 for base.en is ~1 s per utterance, which is still usable.
            log.warning("GPU load failed (%s) — falling back to CPU int8", exc)
            self._model = WhisperModel(
                s["model"], device="cpu", compute_type="int8", download_root=download_root
            )
        log.info("whisper ready")

    def _transcribe_sync(self, pcm: bytes) -> str:
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _ = self._model.transcribe(
            audio,
            language="en",
            beam_size=self._beam_size,
            vad_filter=False,  # the Endpointer already removed silence
            condition_on_previous_text=False,  # prevents runaway repetition loops
        )
        return " ".join(seg.text.strip() for seg in segments).strip()

    async def transcribe(self, pcm: bytes) -> str:
        """Run in a worker thread so the websocket event loop keeps draining audio."""
        return await asyncio.to_thread(self._transcribe_sync, pcm)
