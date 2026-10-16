"""Laptop microphone client — stands in for the ESP32 until the hardware arrives.

Speaks the exact same wire protocol the firmware will, so anything that works here
works there.

    python -m clients.mic_client --url ws://127.0.0.1:8080/ws
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import queue
import sys

import sounddevice as sd
import websockets

SAMPLE_RATE = 16000
FRAME_SAMPLES = 320  # 20 ms
FRAME_BYTES = FRAME_SAMPLES * 2


async def run(url: str, token: str, device_id: str) -> None:
    mic_q: queue.Queue[bytes] = queue.Queue()

    def on_audio(indata, frames, time_info, status) -> None:  # noqa: ANN001
        if status:
            print(f"[mic] {status}", file=sys.stderr)
        mic_q.put(bytes(indata))

    print(f"connecting to {url}")
    async with websockets.connect(url, max_size=None, ping_interval=20) as ws:
        await ws.send(json.dumps({"type": "hello", "token": token, "device_id": device_id}))

        speaker = sd.RawOutputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=FRAME_SAMPLES
        )
        speaker.start()

        mic = sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=FRAME_SAMPLES,
            callback=on_audio,
        )
        mic.start()
        print("listening — just talk. Ctrl+C to quit.\n")

        async def send_mic() -> None:
            loop = asyncio.get_running_loop()
            while True:
                frame = await loop.run_in_executor(None, mic_q.get)
                await ws.send(frame)

        async def recv() -> None:
            async for message in ws:
                if isinstance(message, bytes):
                    speaker.write(message)
                    continue

                msg = json.loads(message)
                kind = msg.get("type")
                if kind == "transcript":
                    print(f"  you  > {msg['text']}")
                elif kind == "reply":
                    print(f"  pico > {msg['text']}\n")
                elif kind == "push":
                    print(f"  pico * {msg['text']}\n")
                elif kind == "state":
                    print(f"  [{msg['value']}]", end="\r")
                elif kind == "error":
                    print(f"  !! {msg.get('message')}")

        try:
            await asyncio.gather(send_mic(), recv())
        finally:
            mic.stop()
            mic.close()
            speaker.stop()
            speaker.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Microphone client for the Pico assistant")
    parser.add_argument("--url", default="ws://127.0.0.1:8080/ws")
    parser.add_argument("--token", default=os.environ.get("ASSISTANT_TOKEN", "change-me"))
    parser.add_argument("--device-id", default="laptop-mic")
    parser.add_argument("--list-devices", action="store_true", help="show audio devices and exit")
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    try:
        asyncio.run(run(args.url, args.token, args.device_id))
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
