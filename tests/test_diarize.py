import numpy as np

import diarize
from diarize import SpeakerTracker, assign_speakers, match_speaker


def test_match_speaker_cosine():
    a = np.array([1.0, 0, 0], dtype=np.float32)
    b = np.array([0, 1.0, 0], dtype=np.float32)
    known = [(1, a), (2, b)]
    assert match_speaker(a, known, 0.55)[0] == 1            # same vector -> match
    assert match_speaker(b + np.array([0, 0, .1]), known, 0.55)[0] == 2
    assert match_speaker(np.array([0, 0, 1.0], np.float32), known, 0.55)[0] is None  # orthogonal -> new


def test_assign_speakers_uses_names_map():
    line = [{"start_ms": 0, "end_ms": 2000, "speaker": "x"}]
    segs = [{"start": 0.0, "end": 2.0, "speaker": 7}]   # cluster id 7
    out = assign_speakers(line, segs, names={7: "Alice"})
    assert out[0]["speaker"] == "Alice"


def test_resolve_models_prefers_existing_paths(tmp_path, monkeypatch):
    # explicit args that exist must win and never trigger a download
    seg = tmp_path / "seg.onnx"; seg.write_bytes(b"x")
    emb = tmp_path / "emb.onnx"; emb.write_bytes(b"x")
    monkeypatch.setattr(diarize, "_ensure_seg", lambda *a, **k: (_ for _ in ()).throw(AssertionError("downloaded")))
    monkeypatch.setattr(diarize, "_ensure_emb", lambda *a, **k: (_ for _ in ()).throw(AssertionError("downloaded")))
    assert diarize._resolve_models(str(seg), str(emb)) == (str(seg), str(emb))
    # missing explicit path -> falls back to the (mocked) provisioner
    monkeypatch.setattr(diarize, "_ensure_seg", lambda *a, **k: "SEG")
    monkeypatch.setattr(diarize, "_ensure_emb", lambda *a, **k: "EMB")
    assert diarize._resolve_models("/nope", "/nope") == ("SEG", "EMB")


def test_speaker_tracker_online_clustering():
    t = SpeakerTracker(threshold=0.5)
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    assert t.assign(a) == 0                       # first speaker
    assert t.assign(a + np.array([0.1, 0, 0])) == 0  # similar -> same
    assert t.assign(b) == 1                       # different -> new speaker
    assert t.assign(b + np.array([0, 0.1, 0])) == 1  # similar to 2nd -> same
    assert t.assign(a) == 0                       # back to speaker 0


def test_assign_speakers_by_time():
    transcripts = [
        {"start_ms": 500, "speaker": "對方", "text": "a"},
        {"start_ms": 3000, "speaker": "對方", "text": "b"},
        {"start_ms": 9000, "speaker": "對方", "text": "c"},
    ]
    segs = [{"start": 0.0, "end": 2.0, "speaker": 0},
            {"start": 2.0, "end": 5.0, "speaker": 1}]
    out = assign_speakers(transcripts, segs)
    assert out[0]["speaker"] == "說話者1"   # 0.5s -> seg 0
    assert out[1]["speaker"] == "說話者2"   # 3.0s -> seg 1
    assert out[2]["speaker"] == "對方"      # 9.0s -> no segment, unchanged
    assert out[0]["text"] == "a"           # other fields preserved


def test_assign_speakers_dominant_over_interjection():
    # a line spans [0,5]s; a 0.5s interjection by speaker B at the start, rest is
    # speaker A -> the line should be labeled A (max overlap), not B.
    line = [{"start_ms": 0, "end_ms": 5000, "speaker": "我", "text": "x"}]
    segs = [{"start": 0.0, "end": 0.5, "speaker": 1},   # B interjects briefly
            {"start": 0.5, "end": 5.0, "speaker": 0}]   # A dominates
    out = assign_speakers(line, segs)
    assert out[0]["speaker"] == "說話者1"  # speaker 0 -> id 1 (the dominant one)


def test_assign_speakers_remaps_ids_to_1_based():
    # sherpa cluster ids need not start at 0; remap distinct ids -> 1..N.
    out = assign_speakers([{"start_ms": 100, "speaker": "x"}],
                          [{"start": 0, "end": 1, "speaker": 2}], prefix="對方")
    assert out[0]["speaker"] == "對方1"  # the only cluster -> 1, not 3


def test_assign_speakers_empty_segments_noop():
    t = [{"start_ms": 0, "speaker": "我", "text": "hi"}]
    assert assign_speakers(t, []) == t


def test_assign_speakers_mark_overlap():
    # line [0,4]s split ~evenly between two speakers -> 2nd covers >=30% -> 🔀 tag
    line = [{"start_ms": 0, "end_ms": 4000, "speaker": "對方", "text": "x"}]
    segs = [{"start": 0.0, "end": 2.0, "speaker": 0},
            {"start": 2.0, "end": 4.0, "speaker": 1}]
    off = assign_speakers(line, segs)                       # default: dominant only
    assert off[0]["speaker"] == "說話者1" and "overlap" not in off[0]
    on = assign_speakers(line, segs, mark_overlap=True)
    assert on[0]["speaker"] == "🔀 說話者1+說話者2" and on[0]["overlap"] is True


def test_mark_overlap_ignores_brief_interjection():
    # 0.4s second speaker on a 5s line (<30%) -> still single dominant, no tag
    line = [{"start_ms": 0, "end_ms": 5000, "speaker": "我", "text": "x"}]
    segs = [{"start": 0.0, "end": 0.4, "speaker": 1},
            {"start": 0.4, "end": 5.0, "speaker": 0}]
    out = assign_speakers(line, segs, mark_overlap=True)
    assert out[0]["speaker"] == "說話者1" and "overlap" not in out[0]


def test_split_text_snaps_to_punctuation():
    from diarize import _split_text
    p = _split_text("愛實作啊，這個應該也得學實作。好，謝謝。", [0.7, 0.3])
    assert len(p) == 2 and p[0].endswith("。") and p[1] == "好，謝謝。"


def test_assign_speakers_splits_on_speaker_change():
    t = [{"id": 1, "track": "system", "profile": "accurate",
          "start_ms": 0, "end_ms": 10000, "text": "前半段講完了。後半段換人說。"}]
    segs = [{"start": 0, "end": 5, "speaker": "A"}, {"start": 5, "end": 10, "speaker": "B"}]
    out = assign_speakers(t, segs, prefix="對方", split=True)
    assert len(out) == 2                                  # one line -> two (speaker change)
    assert out[0]["speaker"] == "對方1" and out[1]["speaker"] == "對方2"
    assert out[0]["split"] and out[0]["src_id"] == 1
    assert out[0]["text"] and out[1]["text"]              # text distributed to both
    # a single-speaker line is NOT split
    one = assign_speakers(t, [{"start": 0, "end": 10, "speaker": "A"}],
                          prefix="對方", split=True)
    assert len(one) == 1 and not one[0].get("split")


def test_similar_pairs_skips_two_unnamed():
    import struct
    from diarize import similar_speaker_pairs
    Row = lambda i, n, x, y: {"id": i, "name": n,
                              "centroid": struct.pack("2f", x, y)}
    # 我3 ↔ 對方5 both auto-labels -> dropped; Alice ↔ 我3 kept (one is named)
    rows = [Row(1, "我3", 1.0, 0.0), Row(2, "對方5", 0.99, 0.14),
            Row(3, "Alice", 0.98, 0.2)]
    pairs = similar_speaker_pairs(rows, threshold=0.3)
    names = {frozenset((p["a"], p["b"])) for p in pairs}
    assert frozenset(("我3", "對方5")) not in names      # two un-named -> skipped
    assert any("Alice" in (p["a"], p["b"]) for p in pairs)  # named pair kept


def _spk_row(name, vec):
    v = np.asarray(vec, dtype=np.float32)
    v = v / np.linalg.norm(v)
    return {"id": 1, "name": name, "centroid": v.tobytes(), "count": 1}


def test_live_labeler_recognizes_named_voice_and_falls_back():
    alice = np.array([1.0, 0, 0, 0], dtype=np.float32)
    bob = np.array([0, 1.0, 0, 0], dtype=np.float32)
    rows = [_spk_row("Alice", alice)]
    cur = {"v": None}
    fn = diarize.live_speaker_labeler(lambda a: cur["v"], rows,
                                      match_threshold=0.62, min_secs=0)
    cur["v"] = alice
    assert fn(b"x" * 4000) == "Alice"          # matches the named global voiceprint
    cur["v"] = bob
    assert fn(b"x" * 4000) == "說話者1"          # unknown voice -> session-local label


def test_live_labeler_skips_placeholder_globals():
    alice = np.array([1.0, 0, 0, 0], dtype=np.float32)
    rows = [_spk_row("說話者5", alice)]               # auto placeholder, never human-named
    fn = diarize.live_speaker_labeler(lambda a: alice, rows,
                                      match_threshold=0.62, min_secs=0)
    assert fn(b"x" * 4000) == "說話者1"               # placeholder not used as a live name


def test_split_live_utterance_single_speaker_one_piece():
    audio = b"\x01\x00" * (16000 * 4)                  # 4s, one voice
    parts = diarize.split_live_utterance(audio, "完整的一句話。", lambda a: "Alice",
                                         window_s=1.5)
    assert len(parts) == 1
    assert parts[0][0] == "Alice"
    assert parts[0][1] == "完整的一句話。"


def test_split_live_utterance_splits_two_speakers():
    audio = b"\x01\x00" * (16000 * 6)                  # 6s -> 4 windows of 1.5s
    calls = {"n": 0}
    def lf(a):                                         # first half Alice, then Bob
        i = calls["n"]; calls["n"] += 1
        return "Alice" if i < 2 else "Bob"
    parts = diarize.split_live_utterance(audio, "你好嗎。我很好。", lf, window_s=1.5)
    assert [p[0] for p in parts] == ["Alice", "Bob"]   # one line per speaker, in order
    assert parts[0][2] == 0                            # first piece starts at 0
    assert parts[-1][3] <= 6000                         # last ends within the utterance
    assert "".join(p[1] for p in parts).replace(" ", "") == "你好嗎。我很好。"


def _labeler_with_peek(fn, peek):
    """A label_fn double exposing .peek, like live_speaker_labeler's real fn."""
    fn.peek = peek
    return fn


def test_split_live_utterance_uncertain_peek_never_spawns_mid_utterance():
    # Regression for the runaway-說話者N bug: split_live_utterance used to call
    # the full (match-or-spawn) label_fn independently on every ~1.5s window, so
    # a single real speaker's noisy windows could each spawn a fresh 說話者N.
    # peek() returning None ('no confident opinion') on every window must NOT
    # fragment the utterance or invoke the real decision more than once.
    audio = b"\x01\x00" * (16000 * 6)                  # 6s -> 4 windows
    calls = {"n": 0}
    def fn(a):
        calls["n"] += 1
        return "說話者1"
    lf = _labeler_with_peek(fn, lambda a: None)         # always uncertain
    parts = diarize.split_live_utterance(audio, "這應該只有一個人講話。", lf, window_s=1.5)
    assert len(parts) == 1                              # never fragmented
    assert calls["n"] == 1                               # decided ONCE, on the full utterance
    assert parts[0][0] == "說話者1"


def test_split_live_utterance_confident_peek_splits_and_commits_once_per_piece():
    # A window that CONFIDENTLY names a different already-established identity
    # still produces a real split — and the real label_fn is called once per
    # detected PIECE (full-quality audio), not once per raw window.
    audio = b"\x01\x00" * (16000 * 6)                  # 6s -> 4 windows: A,A,B,B
    commits = []
    def fn(a):
        commits.append(len(a))
        return "Alice" if len(commits) == 1 else "Bob"
    def peek(chunk):
        label = "Alice" if peek.n < 2 else "Bob"
        peek.n += 1
        return label
    peek.n = 0
    lf = _labeler_with_peek(fn, peek)
    parts = diarize.split_live_utterance(audio, "你好嗎。我很好。", lf, window_s=1.5)
    assert [p[0] for p in parts] == ["Alice", "Bob"]
    assert len(commits) == 2                            # one commit per piece, not per window


def test_live_speaker_labeler_promotes_session_cluster_to_named_voice():
    # A short, noisy single utterance from a NAMED speaker can land just under
    # match_threshold and spawn a session-local 說話者N. As that same cluster
    # accumulates more of the SAME speaker's utterances, its running-mean
    # centroid gets less noisy; once it crosses match_threshold, the cluster is
    # retroactively promoted to the real name for this and all future
    # occurrences (fixes named voices going permanently unrecognized).
    dim = 16
    rng = np.random.default_rng(0)
    alice_dir = rng.normal(size=dim)
    alice_dir /= np.linalg.norm(alice_dir)

    def extractor(_audio):
        v = alice_dir + 0.7 * rng.normal(size=dim)
        return v / np.linalg.norm(v)

    rows = [_spk_row("Alice", alice_dir)]
    fn = diarize.live_speaker_labeler(extractor, rows, session_threshold=0.4,
                                      match_threshold=0.62, min_secs=0)
    labels = [fn(b"x" * 4000) for _ in range(8)]
    assert labels[0] != "Alice"                         # first noisy sample: no individual match
    assert labels[-3:] == ["Alice", "Alice", "Alice"]    # promoted, and stays promoted


def test_live_speaker_labeler_on_promote_fires_with_old_cluster_label_and_name():
    # Live retroactive rename needs to know WHICH cluster label (說話者N) to
    # rewrite once it promotes — on_promote must fire exactly once, with the
    # cluster's old label and the newly recognized name, so the caller (
    # live_session.enable_diarization) can store.rename_speaker(mid, old, new).
    dim = 16
    rng = np.random.default_rng(0)
    alice_dir = rng.normal(size=dim)
    alice_dir /= np.linalg.norm(alice_dir)

    def extractor(_audio):
        v = alice_dir + 0.7 * rng.normal(size=dim)
        return v / np.linalg.norm(v)

    rows = [_spk_row("Alice", alice_dir)]
    promotions = []
    fn = diarize.live_speaker_labeler(extractor, rows, session_threshold=0.4,
                                      match_threshold=0.62, min_secs=0,
                                      on_promote=lambda old, new: promotions.append((old, new)))
    labels = [fn(b"x" * 4000) for _ in range(8)]
    assert labels[-1] == "Alice"                        # sanity: still promotes
    assert promotions == [("說話者1", "Alice")]           # fired once, old cluster -> new name


# --- interleaved-speaker robustness: change-point boundaries + piece-confirm ---
import array  # noqa: E402


def _pcm(value, secs, sr=16000):
    return array.array("h", [value] * int(secs * sr)).tobytes()


def _emb_labeler():
    """label_fn double like the real live_speaker_labeler: exposes embed +
    peek_from_emb (always None = unknown, so ONLY change-point can split) +
    a full-audio label_fn. Embedding is derived from the pcm sign: +value ->
    voice A vector, -value -> voice B vector (orthogonal -> cosine 0)."""
    def _vec(pcm):
        a = np.frombuffer(pcm, dtype=np.int16)
        return np.array([1.0, 0.0]) if a.mean() >= 0 else np.array([0.0, 1.0])
    def fn(pcm):
        return "A" if np.frombuffer(pcm, dtype=np.int16).mean() >= 0 else "B"
    fn.embed = _vec
    fn.peek_from_emb = lambda e: None          # never a confident named identity
    fn.peek = lambda pcm: None
    return fn


def test_group_by_peek_change_point_splits_unknown_speakers():
    # Two unknown speakers (peek None throughout) with orthogonal embeddings:
    # change-point must still cut, where the old confident-name-only logic merged.
    a, b = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    segs = [{"_peek": None, "_emb": a}, {"_peek": None, "_emb": a},
            {"_peek": None, "_emb": b}, {"_peek": None, "_emb": b}]
    diarize._group_by_peek(segs, change_sim=0.35)
    assert [s["speaker"] for s in segs] == [0, 0, 1, 1]
    # Without change_sim (embed absent) it stays merged — no false splits.
    segs2 = [{"_peek": None, "_emb": a}, {"_peek": None, "_emb": b}]
    diarize._group_by_peek(segs2, change_sim=None)
    assert [s["speaker"] for s in segs2] == [0, 0]


def test_confirm_pieces_merges_same_voice_keeps_distinct():
    # pieces alternating A/A/B on 1s each; A-A must merge, B stays.
    audio = _pcm(100, 1) + _pcm(100, 1) + _pcm(-100, 1)
    embed = lambda pcm: (np.array([1.0, 0.0]) if np.frombuffer(pcm, np.int16).mean() >= 0
                         else np.array([0.0, 1.0]))
    pieces = [(0, 0.0, 1.0), (1, 1.0, 2.0), (2, 2.0, 3.0)]
    out = diarize._confirm_pieces(audio, pieces, embed, merge_sim=0.62, sample_rate=16000)
    assert len(out) == 2                                  # A+A merged, B kept
    assert out[0][1] == 0.0 and out[0][2] == 2.0          # merged span covers both A pieces


def test_split_live_utterance_change_point_splits_two_unknown():
    # 3s voice A then 3s voice B, neither confidently named -> change-point splits.
    audio = _pcm(100, 3) + _pcm(-100, 3)
    parts = diarize.split_live_utterance(audio, "你好嗎。我很好。", _emb_labeler(), window_s=1.5)
    assert len(parts) == 2
    assert parts[0][0] == "A" and parts[1][0] == "B"


def test_split_live_utterance_embed_single_speaker_stays_one():
    # A single voice throughout must NOT be over-split by the change-point path.
    audio = _pcm(100, 5)
    parts = diarize.split_live_utterance(audio, "整段都是同一個人在講話。", _emb_labeler(), window_s=1.5)
    assert len(parts) == 1
    assert parts[0][0] == "A"


def test_consolidate_named_drops_outlier_and_merges():
    # One person named repeatedly: 3 coherent enrollments + 1 orthogonal garbage
    # row -> ONE consolidated centroid near the coherent mean, garbage dropped.
    # A placeholder (說話者N) is excluded; a lone name passes through.
    import numpy as np
    import diarize

    def row(name, vec, count):
        v = np.asarray(vec, dtype=np.float32)
        v = v / (np.linalg.norm(v) + 1e-9)
        return {"name": name, "centroid": v.tobytes(), "count": count}

    base = [1.0, 0.0, 0.0, 0.0]
    speakers = [
        row("Hank", base, 5), row("Hank", [0.95, 0.31, 0.0, 0.0], 3),
        row("Hank", [0.9, 0.2, 0.3, 0.0], 4), row("Hank", [0.0, 0.0, 0.0, 1.0], 1),  # orthogonal garbage
        row("Ann", [0.0, 1.0, 0.0, 0.0], 2),
        row("說話者3", base, 9),  # placeholder -> excluded
    ]
    out = diarize._consolidate_named(speakers)
    names = [n for n, _ in out]
    assert names.count("Hank") == 1              # 3 coherent merged, orphan dropped
    assert "Ann" in names and "說話者3" not in names
    hank = [v for n, v in out if n == "Hank"][0]
    base_u = np.asarray(base, dtype=np.float32)
    assert float(np.dot(hank, base_u)) > 0.9     # near coherent mean, not dragged to the orthogonal row


def test_live_speaker_labeler_continuity_keeps_last_speaker_on_noisy_utterance():
    # A recognized speaker's next short/noisy utterance can dip below match_threshold
    # (0.62) yet stay clearly the same voice (>= continuity 0.5). It must inherit the
    # last speaker, not flicker to 對方 / a fresh 說話者N.
    dim = 16
    rng = np.random.default_rng(1)
    alice = rng.normal(size=dim); alice /= np.linalg.norm(alice)
    orth = rng.normal(size=dim); orth -= (orth @ alice) * alice; orth /= np.linalg.norm(orth)
    noisy = 0.55 * alice + np.sqrt(1 - 0.55 ** 2) * orth    # cosine 0.55 to alice (unit)
    seq = [alice, noisy]
    calls = {"i": 0}

    def extractor(_audio):
        v = seq[min(calls["i"], len(seq) - 1)]; calls["i"] += 1
        return v

    rows = [_spk_row("Alice", alice)]
    fn = diarize.live_speaker_labeler(extractor, rows, session_threshold=0.4,
                                      match_threshold=0.62, min_secs=0,
                                      continuity_threshold=0.5)
    assert fn(b"x" * 4000) == "Alice"      # clean -> recognized
    assert fn(b"x" * 4000) == "Alice"      # noisy 0.55 -> continuity keeps Alice, no flicker
