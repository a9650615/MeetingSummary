import zipfile
from store import Store
from plugins.remote_store import push


def test_build_and_push_posts_zip(tmp_path):
    store = Store(tmp_path / "s.db")
    mid = store.create_meeting("週會", 1721111111.0, "zh-TW")
    store.add_segment(mid, 0, "", 1721111111.0, 5.0, "live")
    store.add_transcript(mid, "accurate", "mixed", 0, 1000, "我", "hi")
    store.finalize_meeting(mid)

    def fake_assemble(_store, _mid, track):   # pretend one mixed track exists
        return b"\x00\x01" * 100 if track == "mixed" else None

    def fake_to_m4a(pcm_path, m4a_path):
        open(m4a_path, "wb").write(b"M4A:" + open(pcm_path, "rb").read()[:4])

    captured = {}

    class Resp:  # minimal requests.Response stand-in
        status_code = 200
        def json(self):
            return {"mid": 42}

    def fake_post(url, files=None, timeout=None):
        captured["url"] = url
        captured["zip"] = files["bundle"][1].read()
        return Resp()

    res = push.build_and_push(store, mid, "http://vm:5556",
                              assemble=fake_assemble, to_m4a=fake_to_m4a,
                              http_post=fake_post,
                              tracks=["mixed"])
    assert res["ok"] and res["status"] == 200 and res["mid"] == 42
    assert captured["url"] == "http://vm:5556/ingest-bundle"
    # the posted bytes are a valid zip carrying meeting.json + the track
    import io
    with zipfile.ZipFile(io.BytesIO(captured["zip"])) as z:
        names = z.namelist()
    assert "meeting.json" in names and "tracks/mixed.m4a" in names
