"""Web search and page reading — the assistant's knowledge backstop.

A 4B model holds very little world knowledge. These two tools are what let it
answer factual questions at all, so their output format matters: results are
returned compactly, because every character becomes prompt tokens.
"""

from __future__ import annotations

import asyncio
import logging

import httpx

from . import Tool, ToolContext

log = logging.getLogger(__name__)


def _ddg_sync(query: str, max_results: int) -> list[dict]:
    from ddgs import DDGS

    with DDGS() as ddgs:
        return list(ddgs.text(query, max_results=max_results))


async def _searxng(url: str, query: str, max_results: int) -> list[dict]:
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(
            f"{url.rstrip('/')}/search",
            params={"q": query, "format": "json"},
            headers={"Accept": "application/json"},
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])[:max_results]
    return [
        {"title": r.get("title", ""), "href": r.get("url", ""), "body": r.get("content", "")}
        for r in results
    ]


async def web_search(ctx: ToolContext, query: str) -> str:
    s = ctx.cfg["tools"]["search"]
    max_results = int(s["max_results"])
    searxng_url = (s.get("searxng_url") or "").strip()

    try:
        if searxng_url:
            results = await _searxng(searxng_url, query, max_results)
        else:
            results = await asyncio.to_thread(_ddg_sync, query, max_results)
    except Exception as exc:  # noqa: BLE001
        log.warning("search failed: %s", exc)
        return f"Search unavailable: {exc}"

    if not results:
        return f"No results for '{query}'."

    lines = [f"Search results for '{query}':"]
    for i, r in enumerate(results, 1):
        snippet = (r.get("body") or "").strip().replace("\n", " ")[:300]
        lines.append(f"{i}. {r.get('title', 'Untitled')} — {snippet} [{r.get('href', '')}]")
    return "\n".join(lines)


async def fetch_page(ctx: ToolContext, url: str) -> str:
    import trafilatura

    f = ctx.cfg["tools"]["fetch"]
    try:
        async with httpx.AsyncClient(
            timeout=float(f["timeout_s"]),
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; PicoAssistant/1.0)"},
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            html = resp.text
    except Exception as exc:  # noqa: BLE001
        return f"Could not fetch {url}: {exc}"

    text = await asyncio.to_thread(
        trafilatura.extract, html, include_comments=False, include_tables=False
    )
    if not text:
        return f"No readable article text found at {url}."

    max_chars = int(f["max_chars"])
    if len(text) > max_chars:
        text = text[:max_chars] + "\n[truncated]"
    return f"Content of {url}:\n{text}"


TOOLS = [
    Tool(
        name="web_search",
        description=(
            "Search the web for current information: news, facts, prices, people, "
            "events, how-to answers. Use this whenever the answer is not something "
            "you already know for certain, or could have changed recently."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query, 2-8 keywords."}
            },
            "required": ["query"],
        },
        handler=web_search,
    ),
    Tool(
        name="fetch_page",
        description=(
            "Read the full text of a web page so you can summarise it. "
            "Use after web_search when a snippet is not enough detail."
        ),
        parameters={
            "type": "object",
            "properties": {"url": {"type": "string", "description": "Full URL including https://"}},
            "required": ["url"],
        },
        handler=fetch_page,
    ),
]
