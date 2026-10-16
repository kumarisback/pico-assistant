"""Alarms, timers and reminders — one tool for all three.

The model supplies either a relative offset or an absolute time, never a computed
timestamp: small models are unreliable at date arithmetic, so the arithmetic is
done here.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from dateutil import parser as date_parser

from . import Tool, ToolContext

REPEATS = {"none", "hourly", "daily", "weekly"}


def _local_tz(ctx: ToolContext) -> ZoneInfo:
    return ZoneInfo(ctx.cfg["tools"]["weather"]["timezone"])


def _resolve_time(
    ctx: ToolContext, in_minutes: float | None, at: str | None
) -> tuple[datetime | None, str]:
    tz = _local_tz(ctx)
    local_now = datetime.now(tz)

    if in_minutes is not None:
        if float(in_minutes) <= 0:
            return None, "The reminder time must be in the future."
        return local_now + timedelta(minutes=float(in_minutes)), ""

    if at:
        try:
            parsed = date_parser.parse(at, default=local_now, fuzzy=True)
        except (ValueError, OverflowError):
            return None, f"Could not understand the time '{at}'."
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=tz)
        # "8 am" said at 9 pm means tomorrow morning, not this morning.
        if parsed <= local_now:
            parsed += timedelta(days=1)
        return parsed, ""

    return None, "Provide either in_minutes or at."


async def schedule(
    ctx: ToolContext,
    what: str,
    in_minutes: float | None = None,
    at: str | None = None,
    repeat: str = "none",
) -> str:
    repeat = (repeat or "none").lower()
    if repeat not in REPEATS:
        return f"Unsupported repeat '{repeat}'. Use one of: {', '.join(sorted(REPEATS))}."

    fire_at, error = _resolve_time(ctx, in_minutes, at)
    if fire_at is None:
        return error

    reminder_id = ctx.memory.add_reminder(
        fire_at.astimezone(UTC), what, "" if repeat == "none" else repeat
    )
    when = fire_at.strftime("%I:%M %p on %A %d %B").lstrip("0")
    suffix = "" if repeat == "none" else f", repeating {repeat}"
    return f"Reminder {reminder_id} set for {when}{suffix}: {what}"


async def list_schedule(ctx: ToolContext, days: int = 7) -> str:
    tz = _local_tz(ctx)
    horizon = datetime.now(UTC) + timedelta(days=int(days))
    rows = ctx.memory.pending_reminders(until=horizon)
    if not rows:
        return f"Nothing scheduled in the next {days} days."

    lines = ["Upcoming reminders:"]
    for r in rows:
        local = datetime.fromisoformat(r["fire_at"]).astimezone(tz)
        when = local.strftime("%a %d %b at %I:%M %p").replace(" 0", " ")
        repeat = f" (repeats {r['repeat']})" if r["repeat"] else ""
        lines.append(f"- (id {r['id']}) {when}: {r['text']}{repeat}")
    return "\n".join(lines)


async def cancel_schedule(ctx: ToolContext, reminder_id: int) -> str:
    ok = ctx.memory.cancel_reminder(int(reminder_id))
    return f"Cancelled reminder {reminder_id}." if ok else f"No pending reminder {reminder_id}."


TOOLS = [
    Tool(
        name="schedule",
        description=(
            "Set a timer, alarm or reminder. Use for anything the user wants to be told "
            "about later. Give in_minutes for short offsets ('in 10 minutes'), or at for "
            "a clock time or date ('7:30 am', 'tomorrow 6 pm', '25 December 9 am'). "
            "Do not compute timestamps yourself."
        ),
        parameters={
            "type": "object",
            "properties": {
                "what": {
                    "type": "string",
                    "description": "What to say when it fires, phrased for speech.",
                },
                "in_minutes": {
                    "type": "number",
                    "description": "Minutes from now. Use for timers and short offsets.",
                },
                "at": {
                    "type": "string",
                    "description": "Clock time or date in plain words, e.g. '7:30 am'.",
                },
                "repeat": {
                    "type": "string",
                    "enum": ["none", "hourly", "daily", "weekly"],
                    "description": "Recurrence. Default none.",
                },
            },
            "required": ["what"],
        },
        handler=schedule,
    ),
    Tool(
        name="list_schedule",
        description="List upcoming reminders, timers and alarms.",
        parameters={
            "type": "object",
            "properties": {
                "days": {"type": "integer", "description": "How many days ahead. Default 7."}
            },
            "required": [],
        },
        handler=list_schedule,
    ),
    Tool(
        name="cancel_schedule",
        description="Cancel a pending reminder by its id, as shown by list_schedule.",
        parameters={
            "type": "object",
            "properties": {"reminder_id": {"type": "integer", "description": "Reminder id."}},
            "required": ["reminder_id"],
        },
        handler=cancel_schedule,
    ),
]
