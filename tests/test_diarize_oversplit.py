"""Post-meeting over-split: thin clusters must not become their own speakers.

Measured on a real 8-person meeting: the transcript came back with 24 distinct
speakers. Every extra one was a cluster of 1-4 short backchannels ("哦，" /
"好好好。") totalling 1.6-3.9s of audio, versus 3.6-8.7s per LINE for the clusters
that did get named. cluster_embeddings has no minimum, so a cluster with two
seconds of speech still produces an embedding, and that embedding is too noisy to
match anyone at the naming threshold — so each became its own 對方N.
"""
import numpy as np

import diarize as diar


def _u(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


A, B = _u([1.0, 0.0, 0.0]), _u([0.0, 1.0, 0.0])
A_ISH = _u([0.95, 0.31, 0.0])       # ~0.95 cosine to A — same person, noisy clip


def _segs(spans):                    # [(speaker, start, end), ...]
    return [{"speaker": s, "start": a, "end": b} for s, a, b in spans]


def test_thin_cluster_folds_into_the_voice_it_matches():
    segs = _segs([("s1", 0, 30), ("s2", 30, 60), ("s3", 60, 61.8)])  # s3: 1.8s
    embs = {"s1": A, "s2": B, "s3": A_ISH}
    segs2, embs2, info = diar.merge_tiny_clusters(segs, embs)
    assert info["merged"] == {"s3": "s1"}
    assert set(embs2) == {"s1", "s2"}                    # s3 is gone as an identity
    assert {s["speaker"] for s in segs2} == {"s1", "s2"}  # and its audio is s1's
    assert not info["weak"]


def test_substantial_clusters_are_never_merged():
    # Both have plenty of audio: even a close pair stays two speakers, because the
    # evidence for two people is real. Only thin clusters are folded.
    segs = _segs([("s1", 0, 30), ("s2", 30, 60)])
    embs = {"s1": A, "s2": A_ISH}
    _, embs2, info = diar.merge_tiny_clusters(segs, embs)
    assert info["merged"] == {} and set(embs2) == {"s1", "s2"}


def test_thin_cluster_that_matches_nobody_is_kept_but_flagged_weak():
    segs = _segs([("s1", 0, 30), ("s3", 30, 31.5)])
    embs = {"s1": A, "s3": B}                 # orthogonal -> below any sane bar
    _, embs2, info = diar.merge_tiny_clusters(segs, embs)
    assert info["merged"] == {}
    assert info["weak"] == {"s3"}             # caller must not give it a NUMBER
    assert set(embs2) == {"s1", "s3"}


def test_all_thin_is_left_alone():
    # Nothing substantial to merge into — a very short meeting must survive intact.
    segs = _segs([("s1", 0, 2), ("s2", 2, 3.5)])
    embs = {"s1": A, "s2": B}
    segs2, embs2, info = diar.merge_tiny_clusters(segs, embs)
    assert info["merged"] == {} and len(embs2) == 2 and len(segs2) == 2
    assert info["weak"] == {"s1", "s2"}


def test_duration_is_summed_across_a_cluster_s_segments():
    # 3 x 1.6s = 4.8s total, over the 4s floor -> substantial despite short turns.
    segs = _segs([("s1", 0, 30), ("s2", 30, 31.6), ("s2", 40, 41.6), ("s2", 50, 51.6)])
    embs = {"s1": A, "s2": A_ISH}
    _, _, info = diar.merge_tiny_clusters(segs, embs)
    assert info["merged"] == {} and not info["weak"]


def test_thresholds_are_tunable():
    segs = _segs([("s1", 0, 30), ("s3", 30, 31.8)])
    embs = {"s1": A, "s3": A_ISH}
    _, _, info = diar.merge_tiny_clusters(segs, embs, threshold=0.99)
    assert info["merged"] == {}          # bar too high -> no merge
    _, _, info = diar.merge_tiny_clusters(segs, embs, min_secs=0.5)
    assert info["merged"] == {}          # floor too low -> s3 counts as substantial
