"""Voice activity detection + utterance endpointing.

Uses silero-vad v5 through onnxruntime directly, which keeps torch out of the
dependency tree. The model requires exactly 512-sample windows at 16 kHz, while
the ESP32 sends 320-sample (20 ms) frames, so windows are reassembled here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import onnxruntime as ort

if TYPE_CHECKING:
    from .config import Config

log = logging.getLogger(__name__)

WINDOW_SAMPLES = 512
CONTEXT_SAMPLES = 64
SAMPLE_RATE = 16000
WINDOW_MS = WINDOW_SAMPLES * 1000 // SAMPLE_RATE  # 32


class VadEvent(Enum):
    NONE = "none"
    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"
    TIMEOUT = "timeout"


@dataclass
class VadResult:
    event: VadEvent
    audio: bytes | None = None  # populated on SPEECH_END / TIMEOUT


class Endpointer:
    """Streaming endpointer. Feed PCM16 frames, get an event when an utterance closes.

    One instance per connection — it holds the silero RNN state and the audio buffer,
    neither of which can be shared across speakers.
    """

    def _load_session(self, model_path: Path) -> ort.InferenceSession:
        """Separated so tests can substitute a stub and skip the model file."""
        if not model_path.exists():
            raise FileNotFoundError(
                f"silero VAD model missing at {model_path}. Run scripts/download-models.sh first."
            )
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 3
        return ort.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )

    def __init__(self, cfg: Config) -> None:
        self._session = self._load_session(Path(cfg.path("vad", "model_path")))

        v = cfg["vad"]
        self._threshold = float(v["threshold"])
        self._min_speech_windows = max(1, int(v["min_speech_ms"]) // WINDOW_MS)
        self._silence_windows = max(1, int(v["silence_ms"]) // WINDOW_MS)
        self._max_windows = max(1, int(v["max_utterance_ms"]) // WINDOW_MS)
        self._preroll_bytes = int(v["preroll_ms"]) * SAMPLE_RATE // 1000 * 2

        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(CONTEXT_SAMPLES, dtype=np.float32)
        self._pending = bytearray()  # leftover samples not yet forming a window
        self._preroll = bytearray()  # rolling pre-speech audio
        self._utterance = bytearray()  # captured speech
        self._speech_windows = 0
        self._silence_run = 0
        self._total_windows = 0
        self._in_speech = False

    def reset(self) -> None:
        """Clear per-utterance state. RNN state is reset too: it is utterance-scoped."""
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(CONTEXT_SAMPLES, dtype=np.float32)
        self._pending.clear()
        self._preroll.clear()
        self._utterance.clear()
        self._speech_windows = 0
        self._silence_run = 0
        self._total_windows = 0
        self._in_speech = False

    def _probability(self, window: np.ndarray) -> float:
        # Silero v5 expects the previous 64 samples prepended to each 512-sample
        # window (576 total). Without them the model returns near-zero on clear
        # speech — it never crosses any usable threshold.
        padded = np.concatenate((self._context, window))
        out, self._state = self._session.run(
            None,
            {
                "input": padded.reshape(1, -1),
                "state": self._state,
                "sr": np.array(SAMPLE_RATE, dtype=np.int64),
            },
        )
        self._context = window[-CONTEXT_SAMPLES:]
        return float(out[0][0])

    def push(self, pcm: bytes) -> VadResult:
        """Feed one audio frame. Returns at most one event per call."""
        self._pending.extend(pcm)
        event = VadEvent.NONE

        window_bytes = WINDOW_SAMPLES * 2
        while len(self._pending) >= window_bytes:
            chunk = bytes(self._pending[:window_bytes])
            del self._pending[:window_bytes]

            samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
            prob = self._probability(samples)
            is_speech = prob >= self._threshold

            if self._in_speech:
                self._utterance.extend(chunk)
                self._total_windows += 1

                if is_speech:
                    self._silence_run = 0
                else:
                    self._silence_run += 1
                    if self._silence_run >= self._silence_windows:
                        return VadResult(VadEvent.SPEECH_END, self._take_utterance())

                if self._total_windows >= self._max_windows:
                    log.warning("utterance hit max length — closing early")
                    return VadResult(VadEvent.TIMEOUT, self._take_utterance())
            else:
                # Keep a rolling pre-roll so the first phoneme is not clipped.
                self._preroll.extend(chunk)
                if len(self._preroll) > self._preroll_bytes:
                    del self._preroll[: len(self._preroll) - self._preroll_bytes]

                if is_speech:
                    self._speech_windows += 1
                    if self._speech_windows >= self._min_speech_windows:
                        self._in_speech = True
                        self._utterance.extend(self._preroll)
                        self._preroll.clear()
                        self._silence_run = 0
                        self._total_windows = 0
                        event = VadEvent.SPEECH_START
                else:
                    self._speech_windows = max(0, self._speech_windows - 1)

        return VadResult(event)

    def flush(self) -> VadResult:
        """Close the current utterance on demand (client sent an explicit end)."""
        if self._in_speech and self._utterance:
            return VadResult(VadEvent.SPEECH_END, self._take_utterance())
        return VadResult(VadEvent.NONE)

    def _take_utterance(self) -> bytes:
        audio = bytes(self._utterance)
        self.reset()
        return audio
