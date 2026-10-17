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
import time

import numpy as np
import sounddevice as sd
import websockets

SAMPLE_RATE = 16000
FRAME_SAMPLES = 320  # 20 ms
FRAME_BYTES = FRAME_SAMPLES * 2


async def run(
    url: str,
    token: str,
    device_id: str,
    gain: float = 1.0,
    input_device: int | None = None,
    output_device: int | None = None,
    half_duplex: bool = True,
) -> None:
    mic_q: queue.Queue[bytes] = queue.Queue()
    last_meter_time = 0.0
    current_state = "listening"

    def on_audio(indata, frames, time_info, status) -> None:  # noqa: ANN001
        nonlocal last_meter_time
        if status:
            print(f"\n[mic status: {status}]", file=sys.stderr, flush=True)

        # Laptop mic + laptop speaker share a room: without this the assistant
        # hears its own voice and interrupts itself. Dropped at the source so
        # the queue does not build a backlog to replay afterwards.
        if half_duplex and current_state == "speaking":
            return

        if gain != 1.0:
            arr = np.frombuffer(indata, dtype=np.int16).astype(np.float32) * gain
            pcm = np.clip(arr, -32768, 32767).astype(np.int16).tobytes()
        else:
            pcm = bytes(indata)

        mic_q.put(pcm)

        now = time.time()
        if now - last_meter_time > 0.15:
            last_meter_time = now
            samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
            rms = int(np.sqrt(np.mean(samples**2)))
            bars = min(10, max(0, rms // 150))
            meter = "█" * bars + "░" * (10 - bars)
            sys.stdout.write(f"\r  [mic: {meter}] [{current_state}]    ")
            sys.stdout.flush()

    print(f"Connecting to {url} ...", flush=True)
    async with websockets.connect(url, max_size=None, ping_interval=20) as ws:
        await ws.send(json.dumps({"type": "hello", "token": token, "device_id": device_id}))

        speaker = sd.RawOutputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=FRAME_SAMPLES,
            device=output_device,
        )
        speaker.start()

        mic = sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=FRAME_SAMPLES,
            callback=on_audio,
            device=input_device,
        )
        mic.start()
        print(f"Connected (gain {gain}x). Talk into your microphone. Ctrl+C to quit.\n", flush=True)

        async def send_mic() -> None:
            loop = asyncio.get_running_loop()
            while True:
                frame = await loop.run_in_executor(None, mic_q.get)
                await ws.send(frame)

        async def recv() -> None:
            nonlocal current_state
            async for message in ws:
                if isinstance(message, bytes):
                    # RawOutputStream.write blocks until the ring buffer drains.
                    # Calling it inline stalls the event loop and backs the mic up.
                    await asyncio.to_thread(speaker.write, message)
                    continue

                msg = json.loads(message)
                kind = msg.get("type")
                if kind == "transcript":
                    print(f"\r\n  you  > {msg['text']}", flush=True)
                elif kind == "reply":
                    print(f"\r\n  pico > {msg['text']}\n", flush=True)
                elif kind == "push":
                    print(f"\r\n  pico * {msg['text']}\n", flush=True)
                elif kind == "state":
                    current_state = msg["value"]
                    sys.stdout.write(f"\r  [{current_state}]                 \r")
                    sys.stdout.flush()
                elif kind == "error":
                    print(f"\r\n  !! {msg.get('message')}", flush=True)

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
    parser.add_argument("--gain", type=float, default=1.0, help="mic digital gain multiplier")
    parser.add_argument("--input-device", type=int, default=None, help="index from --list-devices")
    parser.add_argument("--output-device", type=int, default=None, help="index from --list-devices")
    parser.add_argument("--list-devices", action="store_true", help="show audio devices and exit")
    parser.add_argument(
        "--barge-in",
        action="store_true",
        help="keep the mic open while the assistant speaks (needs headphones)",
    )
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    try:
        asyncio.run(
            run(
                args.url,
                args.token,
                args.device_id,
                gain=args.gain,
                input_device=args.input_device,
                output_device=args.output_device,
                half_duplex=not args.barge_in,
            )
        )
    except KeyboardInterrupt:
        print("\nbye", flush=True)


if __name__ == "__main__":
    main()
