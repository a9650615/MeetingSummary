"""Post-meeting multi-speaker diarization (聲紋分群).

Apple Silicon friendly: sherpa-onnx OfflineSpeakerDiarization (onnxruntime, no
CUDA/torch) — segmentation + speaker-embedding + clustering, all offline. Run as
a post-meeting pass on a saved track; relabel its transcripts by time overlap.

assign_speakers() is the pure relabeling step (tested); diarize_pcm() runs the
ONNX pipeline (lazy). Models auto-provision on first use: sherpa-onnx's own
pyannote-3-0 segmentation + 3dspeaker embedding, fetched from k2-fsa releases
into models/ (override via SHERPA_SEG_MODEL / SHERPA_EMB_MODEL). NOTE: the
community-1 onnx is NOT sherpa-compatible (its onnx lacks the 'sample_rate'
metadata sherpa requires) — don't wire it here."""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODELS = os.path.join(_HERE, "models")
# sherpa-onnx model zoo (public, no auth). The seg tar extracts to a dir with
# model.onnx; the emb is a single onnx. Sizes pinned in the comments match the
# files this app was validated against.
_SEG_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/"
            "speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2")
_EMB_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/"
            "speaker-recongition-models/"
            "3dspeaker_speech_eres2net_base_200k_sv_zh-cn_16k-common.onnx")
_SEG_PATH = os.path.join(_MODELS, "sherpa-onnx-pyannote-segmentation-3-0", "model.onnx")
_EMB_PATH = os.path.join(_MODELS, "emb_zh.onnx")


def _fetch(url, dst, progress=None):
    import urllib.request
    if progress:
        progress(f"下載 {os.path.basename(dst)}…")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, dst)


def _ensure_seg(progress=None):
    if os.path.exists(_SEG_PATH):
        return _SEG_PATH
    import tarfile
    tar = os.path.join(_MODELS, "seg.tar.bz2")
    if not os.path.exists(tar):
        _fetch(_SEG_URL, tar, progress)
    if progress:
        progress("解壓 segmentation…")
    with tarfile.open(tar) as tf:
        tf.extractall(_MODELS)  # -> models/sherpa-onnx-pyannote-segmentation-3-0/
    return _SEG_PATH


def _ensure_emb(progress=None):
    if not os.path.exists(_EMB_PATH):
        _fetch(_EMB_URL, _EMB_PATH, progress)
    return _EMB_PATH


def _resolve_models(seg_model=None, emb_model=None, progress=None):
    """(seg, emb) absolute paths — explicit arg > env > local models/ > download.
    Downloads the sherpa-validated models into models/ when nothing is present, so
    diarization works on a fresh machine with no manual model setup."""
    seg = seg_model or os.environ.get("SHERPA_SEG_MODEL")
    emb = emb_model or os.environ.get("SHERPA_EMB_MODEL")
    seg = seg if (seg and os.path.exists(seg)) else _ensure_seg(progress)
    emb = emb if (emb and os.path.exists(emb)) else _ensure_emb(progress)
    return seg, emb


def assign_speakers(transcripts, segments, *, prefix="說話者",
                    mark_overlap=False, overlap_ratio=0.3):
    """Relabel each transcript with the diarization speaker that has the MOST time
    overlap with the line's [start,end] window — so a short interjection at a line's
    start can't mislabel the whole line (dominant speaker wins). Falls back to the
    segment at the start point, then keeps the original. Cluster ids -> 1..N.

    mark_overlap (experimental): when a 2nd speaker also covers >= overlap_ratio of
    the line, label it "🔀 說話者A+B" and set overlap=True — flags turn-dense /
    likely-overlapping lines. sherpa collapses simultaneous speech to one speaker, so
    this is a heuristic from the clustered segments, not true overlap separation."""
    order = {raw: i + 1 for i, raw in
             enumerate(sorted({s["speaker"] for s in segments}))}
    out = []
    for t in transcripts:
        s0 = t.get("start_ms", 0) / 1000.0
        s1 = max(s0 + 0.01, t.get("end_ms", t.get("start_ms", 0)) / 1000.0)
        dur = s1 - s0
        ov = {}  # speaker -> total overlapping seconds with this line
        for seg in segments:
            o = min(s1, seg["end"]) - max(s0, seg["start"])
            if o > 0:
                ov[seg["speaker"]] = ov.get(seg["speaker"], 0.0) + o
        nt = dict(t)
        if ov:
            ranked = sorted(ov.items(), key=lambda kv: -kv[1])
            nt["speaker"] = f"{prefix}{order[ranked[0][0]]}"
            if (mark_overlap and len(ranked) >= 2 and dur > 0
                    and ranked[1][1] / dur >= overlap_ratio):
                ids = sorted(order[ranked[i][0]] for i in (0, 1))
                nt["speaker"] = "🔀 " + "+".join(f"{prefix}{n}" for n in ids)
                nt["overlap"] = True
        else:  # zero overlap (point line) -> segment covering the start, else keep
            best = next((seg["speaker"] for seg in segments
                         if seg["start"] <= s0 < seg["end"]), None)
            if best is not None:
                nt["speaker"] = f"{prefix}{order[best]}"
        out.append(nt)
    return out


class SpeakerTracker:
    """Online (live) speaker clustering. Feed an utterance embedding -> returns a
    0-based speaker id: nearest centroid by cosine if similar enough, else a new
    speaker. Less accurate than offline global clustering (no future context,
    threshold-sensitive) — the post-meeting /diarize pass can re-cluster to fix."""

    def __init__(self, threshold=0.4):
        # Lower threshold = merge more readily = fewer spurious speakers. 3D-Speaker
        # cosine on short utterances is noisy, so default conservatively to avoid
        # over-splitting (the common live failure). Tune via LIVE_DIAR_THRESHOLD.
        self.threshold = threshold
        self.centroids = []
        self.counts = []
        self.last_id = 0

    def assign(self, emb):
        import numpy as np  # noqa: PLC0415
        emb = np.asarray(emb, dtype=np.float32)
        emb = emb / (np.linalg.norm(emb) + 1e-9)
        if self.centroids:
            sims = [float(c @ emb) for c in self.centroids]
            best = max(range(len(sims)), key=sims.__getitem__)
            if sims[best] >= self.threshold:
                n = self.counts[best]
                c = (self.centroids[best] * n + emb) / (n + 1)
                self.centroids[best] = c / (np.linalg.norm(c) + 1e-9)
                self.counts[best] = n + 1
                self.last_id = best
                return best
        self.centroids.append(emb)
        self.counts.append(1)
        self.last_id = len(self.centroids) - 1
        return self.last_id


def embedding_extractor(model=None):
    """sherpa-onnx speaker embedding extractor: int16 PCM bytes -> vector. Lazy."""
    import numpy as np  # noqa: PLC0415
    import sherpa_onnx  # noqa: PLC0415

    _, emb = _resolve_models(emb_model=model)
    ext = sherpa_onnx.SpeakerEmbeddingExtractor(
        sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=emb))

    def _run(pcm_bytes, sample_rate=16000):
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        stream = ext.create_stream()
        stream.accept_waveform(sample_rate, samples)
        return np.asarray(ext.compute(stream), dtype=np.float32)

    return _run


def diarize_pcm(pcm_path, *, sample_rate=16000, num_speakers=-1,
                seg_model=None, emb_model=None, progress=None):
    """Cluster speakers in a 16-bit mono PCM file -> [{start,end,speaker}].
    num_speakers=-1 auto-detects the count. Models auto-provision into models/
    on first use (see _resolve_models); override via SHERPA_SEG/EMB_MODEL."""
    import numpy as np  # noqa: PLC0415
    import sherpa_onnx  # noqa: PLC0415

    seg, emb = _resolve_models(seg_model, emb_model, progress)

    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=seg)),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=emb),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=num_speakers),
    )
    sd = sherpa_onnx.OfflineSpeakerDiarization(config)
    audio = np.frombuffer(open(pcm_path, "rb").read(), dtype=np.int16)
    samples = audio.astype(np.float32) / 32768.0
    result = sd.process(samples).sort_by_start_time()
    return [{"start": s.start, "end": s.end, "speaker": s.speaker} for s in result]
