"""Agent loop against llama-server's OpenAI-compatible endpoint.

llama-server must be started with --jinja, otherwise the `tools` field is ignored
and the model will hallucinate answers instead of calling tools.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from .config import Config
    from .memory import Memory
    from .tools import ToolRegistry

log = logging.getLogger(__name__)

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


class Agent:
    def __init__(self, cfg: Config, registry: ToolRegistry, memory: Memory) -> None:
        self._cfg = cfg
        self._registry = registry
        self._memory = memory

        llm = cfg["llm"]
        self._base_url = llm["base_url"].rstrip("/")
        self._model = llm["model"]
        self._temperature = float(llm["temperature"])
        self._max_tokens = int(llm["max_tokens"])
        self._max_rounds = int(llm["max_tool_rounds"])
        self._client = httpx.AsyncClient(timeout=float(llm["request_timeout_s"]))

    async def close(self) -> None:
        await self._client.aclose()

    async def health(self) -> bool:
        try:
            resp = await self._client.get(f"{self._base_url}/models", timeout=5)
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    def _system_prompt(self) -> str:
        parts = [self._cfg["persona"]["style"].strip()]

        profile = self._memory.get_profile()
        if profile:
            facts = "\n".join(f"- {k}: {v}" for k, v in profile.items())
            parts.append(f"What you know about the user:\n{facts}")

        parts.append(
            "You have tools. Call them instead of guessing. You do not know today's "
            "date, the weather, or any recent event unless a tool tells you."
        )
        return "\n\n".join(parts)

    async def _complete(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        payload = {
            "model": self._model,
            "messages": messages,
            "tools": self._registry.schemas(),
            "tool_choice": "auto",
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }
        resp = await self._client.post(f"{self._base_url}/chat/completions", json=payload)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]

    @staticmethod
    def _parse_args(raw: Any) -> dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            log.warning("unparseable tool arguments: %r", raw)
            return {}

    async def respond(self, user_text: str) -> str:
        """One user turn: reason, call tools as needed, return speakable text."""
        messages: list[dict[str, Any]] = [{"role": "system", "content": self._system_prompt()}]
        messages.extend(self._memory.history())
        messages.append({"role": "user", "content": user_text})

        self._memory.add_turn("user", user_text)

        reply = ""
        for round_no in range(self._max_rounds + 1):
            try:
                message = await self._complete(messages)
            except httpx.HTTPError as exc:
                log.error("llama-server request failed: %s", exc)
                return "I cannot reach my language model right now."

            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                reply = _THINK_BLOCK.sub("", message.get("content") or "").strip()
                break

            if round_no == self._max_rounds:
                # Out of budget: force a text answer from whatever has been gathered.
                messages.append(
                    {"role": "user", "content": "Answer now in one sentence using what you have."}
                )
                try:
                    final = await self._complete(messages)
                    reply = _THINK_BLOCK.sub("", final.get("content") or "").strip()
                except httpx.HTTPError:
                    reply = "I could not finish looking that up."
                break

            messages.append(message)
            for call in tool_calls:
                fn = call.get("function", {})
                name = fn.get("name", "")
                args = self._parse_args(fn.get("arguments"))
                log.info("tool call: %s(%s)", name, args)

                result = await self._registry.call(name, args)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id", name),
                        "name": name,
                        "content": result,
                    }
                )

        reply = reply or "Sorry, I did not catch that."
        self._memory.add_turn("assistant", reply)
        return reply
