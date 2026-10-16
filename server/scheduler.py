"""Reminder dispatcher.

A plain polling loop over the SQLite reminders table rather than an in-memory job
scheduler: reminders must survive a server restart, and the DB is already the
durable store. Polling every few seconds is accurate enough for spoken reminders.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable

log = logging.getLogger(__name__)

POLL_SECONDS = 10

PushFn = Callable[[str], Awaitable[None]]


class ReminderService:
    def __init__(self, memory, push: PushFn) -> None:
        self._memory = memory
        self._push = push
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="reminder-loop")
            log.info("reminder service started")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _run(self) -> None:
        while True:
            try:
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 — a bad reminder must not kill the loop
                log.exception("reminder tick failed")
            await asyncio.sleep(POLL_SECONDS)

    async def _tick(self) -> None:
        now = datetime.now(timezone.utc)
        for row in self._memory.pending_reminders(until=now):
            log.info("firing reminder %s: %s", row["id"], row["text"])
            await self._push(row["text"])
            self._memory.mark_fired(row["id"])

            repeat = (row["repeat"] or "").lower()
            delta = {"daily": timedelta(days=1), "weekly": timedelta(weeks=1),
                     "hourly": timedelta(hours=1)}.get(repeat)
            if delta:
                fire_at = datetime.fromisoformat(row["fire_at"]) + delta
                # If the server was off for a while, roll forward to the next
                # future occurrence instead of firing the whole backlog.
                while fire_at <= now:
                    fire_at += delta
                self._memory.add_reminder(fire_at, row["text"], repeat)
