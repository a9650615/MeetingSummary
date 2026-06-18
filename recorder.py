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
