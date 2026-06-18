"""Post-meeting multi-speaker diarization (聲紋分群).

Apple Silicon friendly: sherpa-onnx OfflineSpeakerDiarization (onnxruntime, no
CUDA/torch) — segmentation + speaker-embedding + clustering, all offline. Run as
a post-meeting pass on a saved track; relabel its transcripts by time overlap.

assign_speakers() is the pure relabeling step (tested); diarize_pcm() runs the
ONNX pipeline (lazy, needs the model files — see SHERPA_* env / models/ dir)."""
import os


def assign_speakers(transcripts, segments, *, prefix="說話者"):
    """Relabel each transcript with the diarization speaker covering its time.
    Cluster ids are remapped to a contiguous 1..N (sherpa ids aren't 0-based), so
    labels are 說話者1/2/3. Transcripts with no overlapping segment keep theirs."""
    order = {raw: i + 1 for i, raw in
             enumerate(sorted({s["speaker"] for s in segments}))}
    out = []
    for t in transcripts:
        ts = t.get("start_ms", 0) / 1000.0
        spk = next((s["speaker"] for s in segments
                    if s["start"] <= ts < s["end"]), None)
        nt = dict(t)
        if spk is not None:
            nt["speaker"] = f"{prefix}{order[spk]}"
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

    emb = model or os.environ.get("SHERPA_EMB_MODEL") or "models/emb_zh.onnx"
    ext = sherpa_onnx.SpeakerEmbeddingExtractor(
        sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=emb))

    def _run(pcm_bytes, sample_rate=16000):
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        stream = ext.create_stream()
        stream.accept_waveform(sample_rate, samples)
        return np.asarray(ext.compute(stream), dtype=np.float32)

    return _run


def diarize_pcm(pcm_path, *, sample_rate=16000, num_speakers=-1,
                seg_model=None, emb_model=None):
    """Cluster speakers in a 16-bit mono PCM file -> [{start,end,speaker}].
    num_speakers=-1 auto-detects the count. Models default to env
    SHERPA_SEG_MODEL / SHERPA_EMB_MODEL (download once, see README)."""
    import numpy as np  # noqa: PLC0415
    import sherpa_onnx  # noqa: PLC0415

    seg = (seg_model or os.environ.get("SHERPA_SEG_MODEL")
           or "models/sherpa-onnx-pyannote-segmentation-3-0/model.onnx")
    emb = emb_model or os.environ.get("SHERPA_EMB_MODEL") or "models/emb_zh.onnx"
    if not (os.path.exists(seg) and os.path.exists(emb)):
        raise RuntimeError("diarization models missing — set SHERPA_SEG_MODEL "
                           "and SHERPA_EMB_MODEL (see README)")

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
