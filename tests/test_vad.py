"""Endpointer tests — no model file, no onnxruntime session.

The ONNX session is stubbed, so these run anywhere. They guard the window
framing and the silero context contract, which is where the real bug was:
feeding silero v5 a bare 512-sample window (no 64-sample context) makes it
return near-zero on clear speech, and the VAD never fires.
"""

from __future__ import annotations

import numpy as np
import pytest

from server.config import Config
from server.vad import CONTEXT_SAMPLES, WINDOW_SAMPLES, Endpointer, VadEvent

SPEECH_PROB = 0.99
SILENCE_PROB = 0.01


class FakeSession:
    """Stands in for onnxruntime.InferenceSession.

    Records every input it is given so the tests can assert on the shape the
    real model would have received.
    """

    def __init__(self) -> None:
        self.inputs: list[np.ndarray] = []
        self.prob = SILENCE_PROB

    def run(self, _outputs, feeds):  # noqa: ANN001, ANN202
        self.inputs.append(feeds["input"])
        return np.array([[self.prob]], dtype=np.float32), feeds["state"]


@pytest.fixture
def endpointer(monkeypatch):
    cfg = Config(
        raw={
            "vad": {
                "model_path": "models/silero_vad.onnx",
                "threshold": 0.5,
                "min_speech_ms": 64,  # 2 windows
                "silence_ms": 128,  # 4 windows
                "max_utterance_ms": 2000,
                "preroll_ms": 64,
            }
        }
    )
    session = FakeSession()
    monkeypatch.setattr(Endpointer, "_load_session", lambda self, path: session)
    ep = Endpointer(cfg)
    return ep, session


def frames(count: int) -> bytes:
    """`count` windows worth of PCM16 bytes. Content is irrelevant — the fake
    session decides the probability."""
    return b"\x01\x00" * (WINDOW_SAMPLES * count)


def test_model_receives_context_plus_window(endpointer):
    """Silero v5 needs the previous 64 samples prepended to each 512 window."""
    ep, session = endpointer
    ep.push(frames(3))

    assert len(session.inputs) == 3
    for inp in session.inputs:
        assert inp.shape == (1, CONTEXT_SAMPLES + WINDOW_SAMPLES)


def test_context_carries_the_previous_windows_tail(endpointer):
    ep, session = endpointer
    ep.push(frames(2))

    first, second = session.inputs
    # The context of window N is the last 64 samples of window N-1.
    np.testing.assert_array_equal(second[0, :CONTEXT_SAMPLES], first[0, -CONTEXT_SAMPLES:])


def test_first_window_context_is_zeros(endpointer):
    ep, session = endpointer
    ep.push(frames(1))

    np.testing.assert_array_equal(
        session.inputs[0][0, :CONTEXT_SAMPLES], np.zeros(CONTEXT_SAMPLES, dtype=np.float32)
    )


def test_reset_clears_the_context(endpointer):
    ep, session = endpointer
    ep.push(frames(1))
    ep.reset()
    ep.push(frames(1))

    np.testing.assert_array_equal(
        session.inputs[-1][0, :CONTEXT_SAMPLES], np.zeros(CONTEXT_SAMPLES, dtype=np.float32)
    )


def test_speech_then_silence_closes_an_utterance(endpointer):
    ep, session = endpointer

    session.prob = SPEECH_PROB
    assert ep.push(frames(2)).event is VadEvent.SPEECH_START

    session.prob = SILENCE_PROB
    result = ep.push(frames(4))
    assert result.event is VadEvent.SPEECH_END
    assert result.audio


def test_silence_alone_produces_no_event(endpointer):
    ep, _ = endpointer
    assert ep.push(frames(20)).event is VadEvent.NONE


def test_partial_frames_are_buffered_until_a_full_window(endpointer):
    """The ESP32 sends 320-sample frames; windows are 512. Nothing should run
    until enough bytes have arrived."""
    ep, session = endpointer

    ep.push(b"\x00\x00" * 320)
    assert session.inputs == []

    ep.push(b"\x00\x00" * 320)
    assert len(session.inputs) == 1
