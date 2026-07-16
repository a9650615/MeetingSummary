import asyncio
import io
import json
import struct

import wave

import shutil

import pytest

from recorder import (
    aiter_frames,
    m4a_to_pcm,
    parse_frames,
    pcm_duration_s,
    pcm_to_m4a,
    pcm_to_wav,
    record_stream,
    SegmentWriter,
    TRACK_SYSTEM,
    TRACK_MIC,
)


def frame(track, payload):
    return struct.pack("<BI", track, len(payload)) + payload


def test_parse_frames_yields_tagged_payloads():
    stream = io.BytesIO(
        frame(TRACK_SYSTEM, b"\x01\x02") + frame(TRACK_MIC, b"\x03\x04\x05")
    )
    assert list(parse_frames(stream)) == [
        (TRACK_SYSTEM, b"\x01\x02"),
        (TRACK_MIC, b"\x03\x04\x05"),
    ]


def test_parse_frames_stops_on_truncated_tail():
    # Helper crashes mid-frame: header claims 10 bytes, only 2 present.
    good = frame(TRACK_SYSTEM, b"\xaa\xbb")
    truncated = struct.pack("<BI", TRACK_MIC, 10) + b"\x01\x02"
    assert list(parse_frames(io.BytesIO(good + truncated))) == [
        (TRACK_SYSTEM, b"\xaa\xbb"),
    ]


def test_segment_writer_appends_per_track(tmp_path):
    w = SegmentWriter(tmp_path, sample_rate=16000, channels=1, start_ts=123.0)
    w.write(TRACK_SYSTEM, b"\x01\x02")
    w.write(TRACK_MIC, b"\xaa")
    w.write(TRACK_SYSTEM, b"\x03")
    w.close()
    assert (tmp_path / "system.pcm").read_bytes() == b"\x01\x02\x03"
    assert (tmp_path / "mic.pcm").read_bytes() == b"\xaa"


def test_segment_writer_writes_manifest_at_start(tmp_path):
    # Manifest must hit disk on construction (segment start) — crash-safe.
    w = SegmentWriter(tmp_path, sample_rate=16000, channels=1, start_ts=123.5)
    w.close()
    assert json.loads((tmp_path / "manifest.json").read_text()) == {
        "sample_rate": 16000,
        "channels": 1,
        "start_ts": 123.5,
    }


def test_pcm_duration_from_byte_count(tmp_path):
    p = tmp_path / "system.pcm"
    p.write_bytes(b"\x00" * (16000 * 2))  # 1 s @ 16 kHz, 16-bit mono
    assert pcm_duration_s(p, sample_rate=16000, channels=1) == 1.0


def test_record_stream_writes_both_tracks(tmp_path):
    data = frame(TRACK_SYSTEM, b"\x01\x02") + frame(TRACK_MIC, b"\xaa\xbb")
    record_stream(
        io.BytesIO(data), tmp_path, sample_rate=16000, channels=1, start_ts=1.0
    )
    assert (tmp_path / "system.pcm").read_bytes() == b"\x01\x02"
    assert (tmp_path / "mic.pcm").read_bytes() == b"\xaa\xbb"


def test_aiter_frames_yields_tagged_payloads():
    # Async counterpart of parse_frames, fed from a real asyncio.StreamReader —
    # the same type /ws/native-capture's relay (a websocket-backed reader) is.
    async def run():
        reader = asyncio.StreamReader()
        reader.feed_data(frame(TRACK_SYSTEM, b"\x01\x02") + frame(TRACK_MIC, b"\x03\x04\x05"))
        reader.feed_eof()
        return [f async for f in aiter_frames(reader)]

    assert asyncio.run(run()) == [
        (TRACK_SYSTEM, b"\x01\x02"),
        (TRACK_MIC, b"\x03\x04\x05"),
    ]


def test_aiter_frames_stops_on_truncated_tail():
    async def run():
        reader = asyncio.StreamReader()
        good = frame(TRACK_SYSTEM, b"\xaa\xbb")
        truncated = struct.pack("<BI", TRACK_MIC, 10) + b"\x01\x02"
        reader.feed_data(good + truncated)
        reader.feed_eof()
        return [f async for f in aiter_frames(reader)]

    assert asyncio.run(run()) == [(TRACK_SYSTEM, b"\xaa\xbb")]


def test_pcm_to_wav_roundtrips(tmp_path):
    pcm = b"\x00\x01" * 16000  # 1 s @ 16 kHz 16-bit mono
    wav = pcm_to_wav(pcm, sample_rate=16000, channels=1)
    assert wav[:4] == b"RIFF" and wav[8:12] == b"WAVE"
    p = tmp_path / "out.wav"
    p.write_bytes(wav)
    with wave.open(str(p)) as w:
        assert w.getframerate() == 16000
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.readframes(w.getnframes()) == pcm


@pytest.mark.skipif(not shutil.which("afconvert"), reason="afconvert is macOS-only")
def test_pcm_m4a_roundtrip_compresses(tmp_path):
    import math

    sr = 16000
    pcm = b"".join(
        int(8000 * math.sin(2 * math.pi * 440 * i / sr)).to_bytes(2, "little", signed=True)
        for i in range(sr * 3)  # 3 s tone -> real audio afconvert can encode
    )
    src = tmp_path / "system.pcm"
    src.write_bytes(pcm)
    m4a = tmp_path / "system.m4a"
    pcm_to_m4a(str(src), str(m4a))
    assert m4a.stat().st_size < len(pcm) // 4  # AAC is >>4x smaller than raw PCM
    back = m4a_to_pcm(str(m4a))
    # AAC is lossy + adds encoder priming/padding: sample count is close, not exact
    assert abs(len(back) - len(pcm)) < sr * 2  # within ~0.5 s
