"""Notes and personal facts — what makes this an assistant rather than a search box."""

from __future__ import annotations

from . import Tool, ToolContext


async def remember(ctx: ToolContext, text: str, tags: str = "") -> str:
    note_id = await ctx.memory.remember(text, tags)
    return f"Saved as note {note_id}."


async def recall(ctx: ToolContext, query: str) -> str:
    notes = await ctx.memory.recall(query)
    if not notes:
        return f"Nothing saved about '{query}'."
    lines = [f"Saved notes matching '{query}':"]
    lines += [f"- (note {n.id}, {n.created_at[:10]}) {n.text}" for n in notes]
    return "\n".join(lines)


async def forget(ctx: ToolContext, note_id: int) -> str:
    ok = await ctx.memory.forget(int(note_id))
    return f"Deleted note {note_id}." if ok else f"No note {note_id} found."


TOOLS = [
    Tool(
        name="remember",
        description=(
            "Save something for later: a note, an idea, a list item, a fact about the "
            "user or their preferences. Use whenever the user says to note, save, "
            "remember or write something down."
        ),
        parameters={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The note, written as a complete self-contained sentence.",
                },
                "tags": {
                    "type": "string",
                    "description": "Optional comma-separated tags, e.g. 'shopping,urgent'.",
                },
            },
            "required": ["text"],
        },
        handler=remember,
    ),
    Tool(
        name="recall",
        description=(
            "Search everything previously saved with 'remember'. Use when the user asks "
            "what they noted, or asks about their own preferences, plans or past notes."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look for, in plain words."}
            },
            "required": ["query"],
        },
        handler=recall,
    ),
    Tool(
        name="forget",
        description="Delete a saved note by its id. Confirm with the user before calling.",
        parameters={
            "type": "object",
            "properties": {"note_id": {"type": "integer", "description": "Id from recall output."}},
            "required": ["note_id"],
        },
        handler=forget,
    ),
]
