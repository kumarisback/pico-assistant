"""Time and weather. Both keyless and both cheap."""

from __future__ import annotations

import logging
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx

from . import Tool, ToolContext

log = logging.getLogger(__name__)

WEATHER_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "freezing fog", 51: "light drizzle", 53: "drizzle",
    55: "heavy drizzle", 61: "light rain", 63: "rain", 65: "heavy rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 80: "rain showers",
    81: "heavy rain showers", 82: "violent rain showers",
    95: "thunderstorm", 96: "thunderstorm with hail", 99: "severe thunderstorm with hail",
}


async def now(ctx: ToolContext) -> str:
    tz = ZoneInfo(ctx.cfg["tools"]["weather"]["timezone"])
    t = datetime.now(tz)
    return t.strftime("%A, %d %B %Y, %I:%M %p %Z")


async def _geocode(place: str) -> tuple[float, float, str] | None:
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": place, "count": 1, "language": "en", "format": "json"},
        )
        resp.raise_for_status()
        results = resp.json().get("results") or []
    if not results:
        return None
    r = results[0]
    label = ", ".join(x for x in (r.get("name"), r.get("country")) if x)
    return float(r["latitude"]), float(r["longitude"]), label


async def weather(ctx: ToolContext, place: str | None = None) -> str:
    w = ctx.cfg["tools"]["weather"]

    if place:
        geo = await _geocode(place)
        if geo is None:
            return f"Could not find a place called {place}."
        lat, lon, label = geo
    else:
        lat, lon, label = float(w["latitude"]), float(w["longitude"]), w["default_place"]

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,apparent_temperature,relative_humidity_2m,weather_code",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
                "timezone": w["timezone"],
                "forecast_days": 1,
            },
        )
        resp.raise_for_status()
        data = resp.json()

    cur = data["current"]
    day = data["daily"]
    condition = WEATHER_CODES.get(cur["weather_code"], "unknown conditions")
    return (
        f"{label}: {round(cur['temperature_2m'])} degrees Celsius, {condition}, "
        f"feels like {round(cur['apparent_temperature'])}, "
        f"humidity {cur['relative_humidity_2m']} percent. "
        f"Today's range {round(day['temperature_2m_min'][0])} to "
        f"{round(day['temperature_2m_max'][0])}, "
        f"{day['precipitation_probability_max'][0]} percent chance of rain."
    )


TOOLS = [
    Tool(
        name="now",
        description=(
            "Get the current date, day of week and local time. Call this before any "
            "reasoning that depends on today's date, and whenever the user asks the time."
        ),
        parameters={"type": "object", "properties": {}, "required": []},
        handler=now,
    ),
    Tool(
        name="weather",
        description=(
            "Get current weather and today's forecast for a place. "
            "Omit 'place' to use the user's home location."
        ),
        parameters={
            "type": "object",
            "properties": {
                "place": {
                    "type": "string",
                    "description": "City name, e.g. 'Pune' or 'Tokyo'. Omit for home location.",
                }
            },
            "required": [],
        },
        handler=weather,
    ),
]
