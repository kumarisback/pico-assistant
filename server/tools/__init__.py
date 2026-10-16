"""Tool registry.

Deliberately few, broad tools. A 4B model picks correctly among eight general
tools far more reliably than among twenty narrow ones, so each tool here covers
a whole category rather than a single phrasing.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

Handler = Callable[..., Awaitable[str] | str]


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Handler

    def schema(self) -> dict[str, Any]:
        """OpenAI/llama.cpp function-calling schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolContext:
    """Everything a tool may need. Passed as the first argument to every handler."""

    cfg: Any
    memory: Any
    scheduler: Any = None


class ToolRegistry:
    def __init__(self, ctx: ToolContext) -> None:
        self._ctx = ctx
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools)

    async def call(self, name: str, args: dict[str, Any]) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return f"Error: no tool named {name}. Available tools: {', '.join(self._tools)}"

        try:
            result = tool.handler(self._ctx, **args)
            if inspect.isawaitable(result):
                result = await result
            return str(result)
        except TypeError as exc:
            # Wrong/missing arguments are a model mistake, not a crash — hand the
            # error back so it can retry with a corrected call.
            log.warning("tool %s bad arguments %s: %s", name, args, exc)
            return f"Error: invalid arguments for {name}: {exc}"
        except Exception as exc:  # noqa: BLE001 — a failing tool must not kill the turn
            log.exception("tool %s failed", name)
            return f"Error: {name} failed: {exc}"


def build_registry(ctx: ToolContext) -> ToolRegistry:
    from . import basics, notes, scheduling, web

    registry = ToolRegistry(ctx)
    for module in (basics, notes, scheduling, web):
        for tool in module.TOOLS:
            registry.register(tool)
    log.info("registered tools: %s", ", ".join(registry.names()))
    return registry
