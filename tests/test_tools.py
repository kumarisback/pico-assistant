"""Offline unit tests — no network, no models, no llama-server.

Run with:  python -m pytest tests -q
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from server.config import Config
from server.tools import ToolContext
from server.tools.scheduling import _resolve_time, cancel_schedule, list_schedule, schedule
from server.tts import split_sentences


class FakeMemory:
    """Minimal in-memory stand-in for the SQLite layer."""

    def __init__(self) -> None:
        self.rows: list[dict] = []
        self._next = 1

    def add_reminder(self, fire_at: datetime, text: str, repeat: str = "") -> int:
        rid = self._next
        self._next += 1
        self.rows.append(
            {"id": rid, "fire_at": fire_at.isoformat(), "text": text,
             "repeat": repeat, "fired": 0}
        )
        return rid

    def pending_reminders(self, until: datetime | None = None) -> list[dict]:
        out = [r for r in self.rows if not r["fired"]]
        if until:
            out = [r for r in out if datetime.fromisoformat(r["fire_at"]) <= until]
        return sorted(out, key=lambda r: r["fire_at"])

    def cancel_reminder(self, reminder_id: int) -> bool:
        for r in self.rows:
            if r["id"] == reminder_id and not r["fired"]:
                r["fired"] = 1
                return True
        return False


@pytest.fixture
def ctx() -> ToolContext:
    return ToolContext(cfg=Config.load(), memory=FakeMemory())


def test_relative_offset_is_in_the_future(ctx):
    fire_at, error = _resolve_time(ctx, in_minutes=10, at=None)
    assert error == ""
    assert fire_at > datetime.now(timezone.utc)


def test_past_clock_time_rolls_to_tomorrow(ctx):
    local_now = datetime.now(_tz(ctx))
    an_hour_ago = (local_now - timedelta(hours=1)).strftime("%I:%M %p")
    fire_at, error = _resolve_time(ctx, in_minutes=None, at=an_hour_ago)
    assert error == ""
    assert fire_at > local_now


def test_negative_offset_rejected(ctx):
    fire_at, error = _resolve_time(ctx, in_minutes=-5, at=None)
    assert fire_at is None
    assert "future" in error


def test_missing_time_rejected(ctx):
    fire_at, error = _resolve_time(ctx, in_minutes=None, at=None)
    assert fire_at is None
    assert "in_minutes" in error


def test_schedule_then_list_then_cancel(ctx):
    created = asyncio.run(schedule(ctx, what="drink water", in_minutes=30))
    assert "drink water" in created

    listed = asyncio.run(list_schedule(ctx))
    assert "drink water" in listed

    assert "Cancelled" in asyncio.run(cancel_schedule(ctx, reminder_id=1))
    assert "Nothing scheduled" in asyncio.run(list_schedule(ctx))


def test_bad_repeat_rejected(ctx):
    result = asyncio.run(schedule(ctx, what="x", in_minutes=5, repeat="fortnightly"))
    assert "Unsupported repeat" in result


def test_short_fragments_merge_for_tts():
    # Three-word sentences synthesised separately produce audible seams.
    assert split_sentences("Hi. Ok. Sure.") == ["Hi. Ok. Sure."]


def test_long_sentences_split_for_streaming():
    text = "The weather in Pune is clear and warm today. Tomorrow it will rain heavily."
    assert len(split_sentences(text)) == 2


def _tz(ctx):
    from zoneinfo import ZoneInfo

    return ZoneInfo(ctx.cfg["tools"]["weather"]["timezone"])
