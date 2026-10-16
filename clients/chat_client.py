"""Text client — type instead of speaking.

The fastest way to test the agent and tools without a microphone or TTS in the
loop. Use this first when debugging tool selection.

    python -m clients.chat_client
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os

import websockets


async def run(url: str, token: str) -> None:
    async with websockets.connect(url, max_size=None) as ws:
        await ws.send(json.dumps({"type": "hello", "token": token, "device_id": "chat"}))

        done = asyncio.Event()

        async def recv() -> None:
            async for message in ws:
                if isinstance(message, bytes):
                    continue  # ignore TTS audio in text mode
                msg = json.loads(message)
                if msg.get("type") == "reply":
                    print(f"pico > {msg['text']}\n")
                    done.set()
                elif msg.get("type") == "push":
                    print(f"\npico * {msg['text']}\n")

        recv_task = asyncio.create_task(recv())
        loop = asyncio.get_running_loop()

        print("type a message, or 'quit'\n")
        while True:
            line = (await loop.run_in_executor(None, input, "you  > ")).strip()
            if line.lower() in {"quit", "exit"}:
                break
            if not line:
                continue
            done.clear()
            await ws.send(json.dumps({"type": "text", "text": line}))
            await done.wait()

        recv_task.cancel()


def main() -> None:
    parser = argparse.ArgumentParser(description="Text client for the Pico assistant")
    parser.add_argument("--url", default="ws://127.0.0.1:8080/ws")
    parser.add_argument("--token", default=os.environ.get("ASSISTANT_TOKEN", "change-me"))
    args = parser.parse_args()
    try:
        asyncio.run(run(args.url, args.token))
    except (KeyboardInterrupt, EOFError):
        print("\nbye")


if __name__ == "__main__":
    main()
