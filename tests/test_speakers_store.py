import struct

from store import Store


def _cen(*xs):
    return struct.pack(f"{len(xs)}f", *xs)


def test_global_speakers_roundtrip(tmp_path):
    s = Store(str(tmp_path / "m.db"))
    sid = s.add_speaker("說話者", _cen(1.0, 0.0))
    s.set_speaker_name(sid, "對方3")
    assert s.list_speakers()[0]["name"] == "對方3"
    s.update_speaker_centroid(sid, _cen(0.0, 1.0), 4)
    row = s.list_speakers()[0]
    assert row["count"] == 4 and struct.unpack("2f", row["centroid"]) == (0.0, 1.0)
    # rename propagates by name (unique placeholder) -> 1 row
    assert s.rename_global_speaker("對方3", "Scott") == 1
    assert s.list_speakers()[0]["name"] == "Scott"
