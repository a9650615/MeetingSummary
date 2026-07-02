"""Recorder: helper stdout -> crash-safe per-track PCM segments.

Frame protocol (helper -> recorder): little-endian header per chunk
    <B track><I length><payload length bytes>
track: 0 = system audio, 1 = mic. See docs spec component 2 + §5.
"""
import json
import struct
from pathlib import Path

TRACK_SYSTEM = 0
TRACK_MIC = 1

_HEADER = struct.Struct("<BI")
_TRACK_FILE = {TRACK_SYSTEM: "system.pcm", TRACK_MIC: "mic.pcm"}


def parse_frames(stream):
    """Yield (track, payload) from a binary stream. Stops on a truncated
    trailing frame (helper crashed mid-write) rather than raising."""
    while True:
        header = stream.read(_HEADER.size)
        if len(header) < _HEADER.size:
            return
        track, length = _HEADER.unpack(header)
        payload = stream.read(length)
        if len(payload) < length:
            return
        yield track, payload


async def aiter_frames(reader):
    """Async counterpart to parse_frames, for anything with an asyncio-style
    `await reader.readexactly(n)` (asyncio.StreamReader — e.g. a subprocess's
    stdout — or a fake in tests). Same truncated-tail handling: the helper
    dying mid-frame raises IncompleteReadError, which just ends the stream
    rather than propagating."""
    import asyncio

    while True:
        try:
            header = await reader.readexactly(_HEADER.size)
        except asyncio.IncompleteReadError:
            return
        track, length = _HEADER.unpack(header)
        try:
            payload = await reader.readexactly(length)
        except asyncio.IncompleteReadError:
            return
        yield track, payload


class SegmentWriter:
    """Append-only writer for one segment's two PCM tracks. Manifest is
    written on construction so a crash leaves recoverable metadata."""

    def __init__(self, seg_dir, sample_rate, channels, start_ts):
        seg_dir = Path(seg_dir)
        seg_dir.mkdir(parents=True, exist_ok=True)
        (seg_dir / "manifest.json").write_text(
            json.dumps(
                {"sample_rate": sample_rate, "channels": channels, "start_ts": start_ts}
            )
        )
        self._files = {
            track: open(seg_dir / name, "ab") for track, name in _TRACK_FILE.items()
        }

    def write(self, track, payload):
        self._files[track].write(payload)

    def flush(self):
        for f in self._files.values():
            f.flush()

    def close(self):
        for f in self._files.values():
            f.close()


def pcm_to_wav(pcm_bytes, sample_rate=16000, channels=1, bytes_per_sample=2):
    """Wrap raw PCM in a WAV container so a browser can play it back. Headers
    only — no resample/transcode."""
    import io
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(bytes_per_sample)
        w.setframerate(sample_rate)
        w.writeframes(pcm_bytes)
    return buf.getvalue()


def pcm_duration_s(path, sample_rate, channels=1, bytes_per_sample=2):
    """Duration recomputed from raw byte count — never trusts a written field
    (spec M2). 16-bit mono by default."""
    return Path(path).stat().st_size / (sample_rate * channels * bytes_per_sample)


def record_stream(stream, seg_dir, sample_rate, channels, start_ts):
    """Drain a helper stream into one segment. Flush is on close here;
    the live loop flushes every ~5 s.
    ponytail: flush-on-close, add periodic flush when wiring the live loop."""
    writer = SegmentWriter(seg_dir, sample_rate, channels, start_ts)
    try:
        for track, payload in parse_frames(stream):
            writer.write(track, payload)
    finally:
        writer.close()


def main():  # pragma: no cover - spawns the Swift helper, needs a real machine
    import argparse
    import subprocess
    import time

    p = argparse.ArgumentParser(description="Record one segment from the capture helper")
    p.add_argument("seg_dir")
    p.add_argument("--helper", default="capture/.build/release/capture")
    p.add_argument("--rate", type=int, default=16000)
    args = p.parse_args()

    proc = subprocess.Popen([args.helper], stdout=subprocess.PIPE)
    try:
        record_stream(proc.stdout, args.seg_dir, sample_rate=args.rate,
                      channels=1, start_ts=time.time())
    finally:
        proc.terminate()


if __name__ == "__main__":
    main()
