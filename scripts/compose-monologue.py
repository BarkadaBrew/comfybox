#!/usr/bin/env python3
"""Compose a voice track whose silences land ON the chunk joins.

Todd, 2026-09-18: "use thoughtful pauses to span the joins."

`DirectorMath.spanningPauses` moves a join into whatever pause it can find.
That is a rescue, and it depends on the take happening to breathe near the
right place — in the first rendered monologue, two of five joins were
mid-speech because nothing had arranged otherwise.

This arranges otherwise. Give it one beat per chunk; it synthesises each beat
separately and lays them out so every join falls in the MIDDLE of a deliberate
silence. The pause is authored, not discovered.

    python3 scripts/compose-monologue.py out.wav \
        --beat "Morning. I'm early today." \
        --beat "I was thinking about what you said." \
        --beat "So I'm taking the long way round."

Prints where each join lands and how much silence surrounds it, and FAILS
rather than emit a track whose beats overrun their chunks — a beat that does
not fit is an authoring problem, and silently truncating or overlapping it
would put speech back across the join.
"""
import argparse
import json
import math
import struct
import sys
import urllib.request
import wave

LATENT_STRIDE = 8          # every chunk is 1 + 8k frames
MAX_CHUNK_STEPS = 36       # (289 - 1) / 8, the trained single-pass ceiling
MLX_SERVE = "http://127.0.0.1:11234/v1/audio/speech"


def snap_frames(frames):
    """1 + 8k, the only lengths the Director accepts."""
    return 1 + LATENT_STRIDE * max(1, math.ceil((frames - 1) / LATENT_STRIDE))


def chunk_layout(length_frames):
    """Mirrors DirectorMath.chunkLayout: fewest balanced chunks <= 289."""
    steps = (length_frames - 1) // LATENT_STRIDE
    count = max(1, math.ceil(steps / MAX_CHUNK_STEPS))
    base, extra = divmod(steps, count)
    spans, start = [], 0
    for i in range(count):
        frames = LATENT_STRIDE * (base + (1 if i < extra else 0)) + 1
        spans.append((start, frames))
        start += frames - 1
    return spans


def synthesize(text, voice, model):
    body = json.dumps({
        "model": model, "input": text, "voice": voice,
        "response_format": "wav", "speed": 1.0,
    }).encode()
    request = urllib.request.Request(
        MLX_SERVE, data=body, headers={"content-type": "application/json"})
    return urllib.request.urlopen(request, timeout=180).read()


def read_wav(data, path):
    with open(path, "wb") as handle:
        handle.write(data)
    with wave.open(path) as f:
        return (f.getframerate(), f.getnchannels(), f.getsampwidth(),
                f.readframes(f.getnframes()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output")
    ap.add_argument("--beat", action="append", required=True,
                    help="one beat per chunk, in order")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--silence", type=float, default=0.7,
                    help="seconds of deliberate pause at each join")
    ap.add_argument("--voice", default="af_nicole")
    ap.add_argument("--model", default="ddalcu/Kokoro-82M-MLX-Serve")
    args = ap.parse_args()

    beats = args.beat
    scratch = args.output + ".part"
    rendered = []
    rate = channels = width = None
    for index, text in enumerate(beats):
        r, c, w, frames = read_wav(synthesize(text, args.voice, args.model), scratch)
        if rate is None:
            rate, channels, width = r, c, w
        elif (r, c, w) != (rate, channels, width):
            sys.exit(f"beat {index} came back in a different format ({r},{c},{w})")
        rendered.append(frames)
        seconds = len(frames) / (rate * channels * width)
        print(f"  beat {index}: {seconds:5.2f}s  {text[:52]}")

    per_sample = channels * width
    silence_samples = int(args.silence * rate)
    silence = b"\x00" * (silence_samples * per_sample)

    # The Director splits a timeline into BALANCED chunks, so every chunk is
    # the same length and the binding constraint is the LONGEST beat, not the
    # sum. Size the timeline so the longest beat plus its pause fits in one
    # chunk; shorter beats simply get a longer pause, which is what a
    # thoughtful pause is anyway.
    speech = [len(f) // per_sample for f in rendered]
    total = len(beats) * (max(speech) + silence_samples)
    length_frames = snap_frames(math.ceil(total / rate * args.fps))
    spans = chunk_layout(length_frames)

    if len(spans) != len(beats):
        sys.exit(
            f"{len(beats)} beats but the timeline splits into {len(spans)} chunks "
            f"({length_frames} frames). Write one beat per chunk — merge or split "
            f"a beat and try again.")

    # Lay each beat inside its own chunk, centred, so the join between two
    # chunks is surrounded by the tail silence of one and the head silence of
    # the next.
    out = bytearray()
    joins, heads, tails = [], [], []
    for index, (start, frames) in enumerate(spans):
        chunk_samples = int(frames / args.fps * rate)
        pad = chunk_samples - speech[index]
        if pad < silence_samples:
            need = (speech[index] + silence_samples) / rate
            have = chunk_samples / rate
            sys.exit(
                f"beat {index} is {have:.2f}s of chunk but needs {need:.2f}s "
                f"(speech {speech[index]/rate:.2f}s + {args.silence:.2f}s pause).\n"
                f"Shorten it, or shorten another beat to free budget. "
                f"Refusing to emit a track that speaks across a join.")
        head = pad // 2
        tail = pad - head
        out += b"\x00" * (head * per_sample)
        out += rendered[index]
        out += b"\x00" * (tail * per_sample)
        heads.append(head / rate)
        tails.append(tail / rate)
        if index < len(spans) - 1:
            joins.append(start + frames - 1)

    with wave.open(args.output, "wb") as f:
        f.setnchannels(channels)
        f.setsampwidth(width)
        f.setframerate(rate)
        f.writeframes(bytes(out))

    print(f"\n{args.output}: {len(out)//per_sample/rate:.2f}s, "
          f"{length_frames} frames @ {args.fps} fps, {len(spans)} chunks")
    for index, frame in enumerate(joins):
        before, after = tails[index], heads[index + 1]
        print(f"  join at frame {frame}: {before:.2f}s silence before, "
              f"{after:.2f}s after — {before + after:.2f}s spanning it")
    print("\nEvery join sits inside an authored pause — nothing is spoken across one.")


if __name__ == "__main__":
    main()
