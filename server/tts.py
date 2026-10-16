"""Text to speech via Piper (ONNX, CPU).

Piper voices are 22.05 kHz; the ESP32 speaker path is 16 kHz, so output is
resampled here. Synthesis is per-sentence to keep time-to-first-audio low.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import AsyncIterator, Iterator

import numpy as np

log = logging.getLogger(__name__)

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n+")


def split_sentences(text: str, min_chars: int = 25) -> list[str]:
    """Split for streaming. Short fragments are merged — synthesising three words
    at a time produces audible seams and wastes model warm-up."""
    parts = [p.strip() for p in _SENTENCE_END.split(text) if p.strip()]
    merged: list[str] = []
    for part in parts:
        if merged and len(merged[-1]) < min_chars:
            merged[-1] = f"{merged[-1]} {part}"
        else:
            merged.append(part)
    return merged


def _resample(pcm: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return pcm
    duration = pcm.shape[0] / src_rate
    dst_len = int(duration * dst_rate)
    src_idx = np.linspace(0, pcm.shape[0] - 1, num=dst_len, dtype=np.float64)
    return np.interp(src_idx, np.arange(pcm.shape[0]), pcm).astype(np.int16)


class Speaker:
    def __init__(self, cfg) -> None:
        from piper.voice import PiperVoice

        voice_path = cfg.path("tts", "voice_path")
        config_path = cfg.path("tts", "config_path")
        if not voice_path.exists():
            raise FileNotFoundError(
                f"Piper voice missing at {voice_path}. Run scripts/download-models.sh first."
            )

        log.info("loading piper voice %s", voice_path.name)
        self._voice = PiperVoice.load(str(voice_path), config_path=str(config_path))
        self._src_rate = self._voice.config.sample_rate
        self._dst_rate = int(cfg["audio"]["sample_rate"])
        self._frame_bytes = cfg.frame_bytes

        t = cfg["tts"]
        self._synth_args = {
            "length_scale": float(t["length_scale"]),
            "noise_scale": float(t["noise_scale"]),
            "noise_w": float(t["noise_w"]),
        }

    def _synth_sync(self, text: str) -> bytes:
        chunks: Iterator[bytes] = self._voice.synthesize_stream_raw(text, **self._synth_args)
        raw = b"".join(chunks)
        if not raw:
            return b""
        pcm = np.frombuffer(raw, dtype=np.int16)
        return _resample(pcm, self._src_rate, self._dst_rate).tobytes()

    async def stream(self, text: str) -> AsyncIterator[bytes]:
        """Yield fixed-size PCM16 frames, synthesising one sentence ahead.

        Pre-fetching the next sentence while the current one is still being sent
        hides most of Piper's latency behind playback.
        """
        sentences = split_sentences(text)
        if not sentences:
            return

        pending = asyncio.create_task(asyncio.to_thread(self._synth_sync, sentences[0]))
        buffer = bytearray()

        for idx, _ in enumerate(sentences):
            audio = await pending
            if idx + 1 < len(sentences):
                pending = asyncio.create_task(
                    asyncio.to_thread(self._synth_sync, sentences[idx + 1])
                )

            buffer.extend(audio)
            while len(buffer) >= self._frame_bytes:
                yield bytes(buffer[: self._frame_bytes])
                del buffer[: self._frame_bytes]

        if buffer:
            yield bytes(buffer) + b"\x00" * (self._frame_bytes - len(buffer))
