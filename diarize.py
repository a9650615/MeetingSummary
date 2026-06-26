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


def assign_speakers(transcripts, segments, *, prefix="說話者", names=None,
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
    # names (optional): cluster id -> persistent voiceprint label; else 說話者N.
    label = (lambda spk: names.get(spk, f"{prefix}{order[spk]}")) if names \
        else (lambda spk: f"{prefix}{order[spk]}")
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
            nt["speaker"] = label(ranked[0][0])
            if (mark_overlap and len(ranked) >= 2 and dur > 0
                    and ranked[1][1] / dur >= overlap_ratio):
                nt["speaker"] = "🔀 " + "+".join(label(ranked[i][0]) for i in (0, 1))
                nt["overlap"] = True
        else:  # zero overlap (point line) -> segment covering the start, else keep
            best = next((seg["speaker"] for seg in segments
                         if seg["start"] <= s0 < seg["end"]), None)
            if best is not None:
                nt["speaker"] = label(best)
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


def cluster_embeddings(pcm_path, segments, *, sample_rate=16000, max_secs=12,
                       model=None):
    """One L2-normalized speaker embedding per cluster id, from up to max_secs of
    that cluster's audio (concatenated). {} on no audio. Lazy (sherpa emb model)."""
    import numpy as np  # noqa: PLC0415
    audio = np.frombuffer(open(pcm_path, "rb").read(), dtype=np.int16)
    ext = embedding_extractor(model)
    spans = {}
    for seg in segments:
        spans.setdefault(seg["speaker"], []).append((seg["start"], seg["end"]))
    cap = int(max_secs * sample_rate)
    out = {}
    for spk, ss in spans.items():
        chunks, total = [], 0
        for s, e in ss:
            piece = audio[int(s * sample_rate):int(e * sample_rate)]
            if total + len(piece) > cap:
                piece = piece[:cap - total]
            if len(piece):
                chunks.append(piece)
                total += len(piece)
            if total >= cap:
                break
        if not chunks:
            continue
        emb = np.asarray(ext(np.concatenate(chunks).tobytes(), sample_rate),
                         dtype=np.float32)
        out[spk] = emb / (np.linalg.norm(emb) + 1e-9)
    return out


def _diar_worker(q, pcm_path, num_speakers, seg_model, emb_model, enroll):
    """Child-process entry: do ALL the GIL-holding sherpa work here, stream
    progress/result back over the queue."""
    try:
        segs = diarize_pcm(pcm_path, num_speakers=num_speakers, seg_model=seg_model,
                           emb_model=emb_model, on_progress=lambda d, t: q.put(("p", d, t)))
        embs = None
        if enroll:
            q.put(("phase", "enroll"))
            try:
                embs = cluster_embeddings(pcm_path, segs)
            except Exception as e:  # enroll is best-effort
                q.put(("warn", f"enroll skipped: {e}"))
        q.put(("done", segs, embs))
    except Exception as e:
        q.put(("err", repr(e)))


def diarize_with_progress(pcm_path, *, num_speakers=-1, seg_model=None, emb_model=None,
                          enroll=False, on_progress=None, on_phase=None):
    """Diarize in a SUBPROCESS. sherpa's OfflineSpeakerDiarization.process() does
    NOT release the GIL, so running it in a thread freezes the whole interpreter
    (incl. the asyncio event loop -> the server can't serve other pages while
    diarizing). A child process has its own GIL; the parent only blocks on
    queue.get() (which releases the GIL), so the event loop stays responsive.
    Returns (segments, embeddings|None). ponytail: spawn-per-call, trivial vs a
    minutes-long diarization."""
    import multiprocessing as mp  # noqa: PLC0415
    import sys  # noqa: PLC0415
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_diar_worker, daemon=True,
                    args=(q, pcm_path, num_speakers, seg_model, emb_model, enroll))
    p.start()
    segs, embs = None, None
    try:
        while True:
            msg = q.get()
            tag = msg[0]
            if tag == "p" and on_progress:
                on_progress(msg[1], msg[2])
            elif tag == "phase" and on_phase:
                on_phase(msg[1])
            elif tag == "warn":
                print(f"diarize: {msg[1]}", file=sys.stderr)
            elif tag == "done":
                segs, embs = msg[1], msg[2]
                break
            elif tag == "err":
                raise RuntimeError(msg[1])
    finally:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()
    return segs, embs


def match_speaker(emb, known, threshold=0.62):
    """known = [(id, centroid_np)]. Best cosine match >= threshold -> (id, sim),
    else (None, best_sim). Conservative threshold: prefer a new speaker over a
    wrong merge (3D-Speaker cosine: same voice ~0.7+, different <0.4; measured
    distinct speakers at 0.34-0.57, so 0.62 keeps them apart). Tune via the route."""
    import numpy as np  # noqa: PLC0415
    best_id, best = None, -1.0
    for sid, c in known:
        sim = float(np.dot(emb, c))
        if sim > best:
            best, best_id = sim, sid
    return (best_id if best >= threshold else None), best


def diarize_pcm(pcm_path, *, sample_rate=16000, num_speakers=-1,
                seg_model=None, emb_model=None, progress=None, on_progress=None):
    """Cluster speakers in a 16-bit mono PCM file -> [{start,end,speaker}].
    num_speakers=-1 auto-detects the count. Models auto-provision into models/
    on first use (see _resolve_models); override via SHERPA_SEG/EMB_MODEL.
    on_progress(done, total): optional, called as sherpa processes chunks."""
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
    cb = None
    if on_progress:
        def cb(done, total):  # sherpa: (processed_chunks, num_chunks)->int, !=0 aborts
            try:
                on_progress(int(done), int(total))
            except Exception:
                pass
            return 0
    raw = sd.process(samples, cb) if cb else sd.process(samples)
    result = raw.sort_by_start_time()
    return [{"start": s.start, "end": s.end, "speaker": s.speaker} for s in result]
