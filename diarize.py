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


_PUNCT = "。！？，、；：,!?;.…"


def _snap_cut(text, idx, window=10):
    """Nearest index to idx whose preceding char is sentence/clause punctuation
    (within ±window), else idx — so a split piece ends on a natural boundary."""
    n = len(text)
    idx = max(0, min(n, idx))
    best, bestd = None, 1e9
    for i in range(max(1, idx - window), min(n, idx + window) + 1):
        if text[i - 1] in _PUNCT and abs(i - idx) < bestd:
            best, bestd = i, abs(i - idx)
    return best if best is not None else idx


def _split_text(text, fracs):
    """Split text into len(fracs) pieces by cumulative time fraction, each cut
    snapped to the nearest punctuation so pieces read as whole clauses."""
    text = (text or "").strip()
    n = len(text)
    if n == 0 or len(fracs) <= 1:
        return [text]
    cuts, acc, prev = [], 0.0, 0
    for f in fracs[:-1]:
        acc += f
        cuts.append(_snap_cut(text, int(round(acc * n))))
    pieces = []
    for c in cuts + [n]:
        c = max(prev, min(n, c))
        pieces.append(text[prev:c].strip())
        prev = c
    return pieces


def _speaker_pieces(s0, s1, segments, min_piece=0.6):
    """The speaker timeline within [s0,s1]: consecutive same-speaker spans merged,
    tiny fragments (<min_piece s) absorbed into a neighbour. -> [(speaker, st, en)]."""
    spans = sorted(((max(s0, sg["start"]), min(s1, sg["end"]), sg["speaker"])
                    for sg in segments if min(s1, sg["end"]) > max(s0, sg["start"])),
                   key=lambda x: x[0])
    if not spans:
        return []
    merged = [list(spans[0])]
    for st, en, sp in spans[1:]:
        if sp == merged[-1][2]:
            merged[-1][1] = max(merged[-1][1], en)
        else:
            merged.append([st, en, sp])
    out = []
    for p in merged:
        if out and (p[1] - p[0]) < min_piece:
            out[-1][1] = p[1]            # absorb a tiny piece into the previous one
        else:
            out.append(p)
    if len(out) >= 2 and (out[0][1] - out[0][0]) < min_piece:
        out[1][0] = out[0][0]
        out.pop(0)
    return [(sp, st, en) for st, en, sp in out]


def assign_speakers(transcripts, segments, *, prefix="說話者", names=None,
                    mark_overlap=False, overlap_ratio=0.3, split=False):
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
        # split=True: if the line spans >=2 speaker TURNS, cut it at the boundaries
        # so each person's words land on their own line (text split by time fraction,
        # snapped to punctuation). mark_overlap takes precedence (can't do both).
        if split and not mark_overlap:
            pieces = _speaker_pieces(s0, s1, segments)
            if len(pieces) >= 2:
                texts = _split_text(t.get("text", ""), [(pe - ps) / dur for _, ps, pe in pieces])
                for (spk, ps, pe), txt in zip(pieces, texts):
                    if not txt:
                        continue
                    out.append({**t, "speaker": label(spk), "text": txt,
                                "start_ms": int(ps * 1000), "end_ms": int(pe * 1000),
                                "split": True, "src_id": t.get("id")})
                continue
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


def embedding_extractor(model=None, provider="cpu"):
    """sherpa-onnx speaker embedding extractor: int16 PCM bytes -> vector. Lazy.
    provider='coreml' offloads to the Neural Engine (power-efficient)."""
    import numpy as np  # noqa: PLC0415
    import sherpa_onnx  # noqa: PLC0415

    _, emb = _resolve_models(emb_model=model)
    ext = sherpa_onnx.SpeakerEmbeddingExtractor(
        sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=emb, provider=provider))

    def _run(pcm_bytes, sample_rate=16000):
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        stream = ext.create_stream()
        stream.accept_waveform(sample_rate, samples)
        return np.asarray(ext.compute(stream), dtype=np.float32)

    return _run


def cluster_embeddings(pcm_path, segments, *, sample_rate=16000, max_secs=12,
                       model=None, provider="cpu"):
    """One L2-normalized speaker embedding per cluster id, from up to max_secs of
    that cluster's audio (concatenated). {} on no audio. Lazy (sherpa emb model)."""
    import numpy as np  # noqa: PLC0415
    with open(pcm_path, "rb") as _f:
        audio = np.frombuffer(_f.read(), dtype=np.int16)
    ext = embedding_extractor(model, provider)
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


def _diar_worker(q, pcm_path, num_speakers, seg_model, emb_model, enroll, provider="cpu"):
    """Child-process entry: do ALL the GIL-holding sherpa work here, stream
    progress/result back over the queue."""
    try:
        segs = diarize_pcm(pcm_path, num_speakers=num_speakers, seg_model=seg_model,
                           emb_model=emb_model, provider=provider,
                           on_progress=lambda d, t: q.put(("p", d, t)))
        embs = None
        if enroll:
            q.put(("phase", "enroll"))
            try:
                embs = cluster_embeddings(pcm_path, segs, provider=provider)
            except Exception as e:  # enroll is best-effort
                q.put(("warn", f"enroll skipped: {e}"))
        q.put(("done", segs, embs))
    except Exception as e:
        q.put(("err", repr(e)))


def diarize_with_progress(pcm_path, *, num_speakers=-1, seg_model=None, emb_model=None,
                          enroll=False, on_progress=None, on_phase=None, provider="cpu"):
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
                    args=(q, pcm_path, num_speakers, seg_model, emb_model, enroll, provider))
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


def _is_placeholder(name):
    """Auto label not yet named by a human (我3 / 對方24 / 說話者5 / 混合8). 混合N is
    the 'both' track's auto label — mixed 2-speaker audio, so its centroid is
    garbage; leaving it out of _is_placeholder let 71/149 such rows pollute the
    live named-voiceprint match set (混合 vs 混合 ~0.8, 混合 vs a real name ~0.79)
    → 認錯人. Never match auto placeholders live."""
    import re  # noqa: PLC0415
    return bool(re.match(r"^(我|對方|說話者|混合)\d+$", (name or "").strip()))


def live_speaker_labeler(extractor, speakers, *, session_threshold=0.4,
                         match_threshold=0.62, min_secs=1.2, sample_rate=16000):
    """Factory for the LIVE per-utterance speaker label fn: fn(pcm_bytes) -> label.

    Recognizes voices the user already NAMED in past meetings (cosine-match the
    utterance embedding against the global speakers table) so live captions show
    real names; unknown voices get a session-local 說話者N. Read-only on the DB —
    live utterances are short + noisy, so enrollment/persistence is left to the
    post-meeting /diarize pass + 人工命名 (事後命名,自動記住). Only HUMAN-named
    voiceprints are matched (auto placeholders like 說話者5 are skipped), so live
    never shows a meaningless cross-meeting placeholder.

    speakers = store.list_speakers() rows (id, name, centroid bytes, count)."""
    import numpy as np  # noqa: PLC0415
    tr = SpeakerTracker(threshold=session_threshold)
    known = [(r["name"], np.frombuffer(r["centroid"], dtype=np.float32))
             for r in speakers
             if r["centroid"] and not _is_placeholder(r["name"])]
    min_bytes = int(min_secs * sample_rate) * 2
    promoted = {}  # session id -> name, once that cluster's running centroid matches

    def _label_for(sid):
        return promoted.get(sid, f"說話者{sid + 1}")

    def fn(audio):
        if len(audio) < min_bytes and tr.centroids:
            return _label_for(tr.last_id)            # too short -> reuse last, don't spawn
        emb = np.asarray(extractor(audio), dtype=np.float32)
        if known:
            e = emb / (np.linalg.norm(emb) + 1e-9)
            name, _sim = match_speaker(e, known, match_threshold)
            if name is not None:
                return name                          # recognized a named voice
        sid = tr.assign(emb)
        if sid not in promoted and known:
            # A short single utterance's own embedding can land just under
            # match_threshold by noise even for a truly named voice (measured:
            # separate short utterances from the SAME speaker don't always
            # individually clear 0.62). But tr's running-mean centroid for this
            # cluster gets less noisy as more of that speaker's utterances
            # accumulate into it — so retry the named match against THAT refined
            # centroid; once it crosses match_threshold, "promote" the cluster so
            # this and all its future occurrences resolve to the real name
            # (past occurrences already emitted stay as 說話者N — no retroactive
            # rewrite here, that's the post-meeting /diarize pass's job).
            name, _sim = match_speaker(tr.centroids[sid], known, match_threshold)
            if name is not None:
                promoted[sid] = name
                return name
        return _label_for(sid)                       # unknown -> session-local cluster

    def embed(audio):
        # STATE-FREE normalized embedding for a window (no spawn, no centroid
        # update). None when too short to embed reliably. split_live_utterance
        # uses these for change-point boundary detection between speakers that
        # aren't (yet) a confidently-named/established identity.
        if len(audio) < min_bytes:
            return None
        e = np.asarray(extractor(audio), dtype=np.float32)
        return e / (np.linalg.norm(e) + 1e-9)

    def peek_from_emb(e):
        # Confident identity for a precomputed embedding: a named voice, or an
        # ALREADY-ESTABLISHED session cluster. None = "no opinion" (treated as a
        # continuation, never a new speaker). NEVER spawns/updates centroids —
        # letting every noisy window run the full match-or-spawn fn() was the
        # runaway-說話者N bug. New identities are only ever created by fn()
        # committing on a full utterance/piece.
        if e is None:
            return None
        if known:
            name, _sim = match_speaker(e, known, match_threshold)
            if name is not None:
                return name
        if tr.centroids:
            sims = [float(c @ e) for c in tr.centroids]
            best = max(range(len(sims)), key=sims.__getitem__)
            if sims[best] >= tr.threshold:
                return _label_for(best)
        return None

    def peek(audio):
        return peek_from_emb(embed(audio))

    fn.peek = peek
    fn.embed = embed
    fn.peek_from_emb = peek_from_emb
    return fn


def _group_by_peek(segments, change_sim=None):
    """Assign local integer group ids from each window's '_peek' identity
    (consumed here, not persisted). Two boundary signals:

      1. CONFIDENT NAME CHANGE — a window confidently names a DIFFERENT identity
         than the last confident one seen (the original, safe signal). An
         unconfident window (peek None) extends the current group, never splits.
      2. CHANGE-POINT (only when change_sim is given + windows carry '_emb') —
         a LARGE embedding jump between adjacent windows (cosine < change_sim)
         marks a turn even between two speakers neither of which is a confident
         named/established identity. This alone over-splits a noisy single
         speaker (short-window cosine is noisy), so it's deliberately gated LOW
         and every resulting split is re-checked at piece quality by
         _confirm_pieces (which merges pieces that re-embed as the same voice).
    """
    gid, last_confident, prev_emb = 0, None, None
    for seg in segments:
        peeked = seg.pop("_peek", None)
        emb = seg.pop("_emb", None)
        cut = (peeked is not None and last_confident is not None and peeked != last_confident)
        if not cut and change_sim is not None and prev_emb is not None and emb is not None:
            cut = float(prev_emb @ emb) < change_sim
        if cut:
            gid += 1
        if peeked is not None:
            last_confident = peeked
        if emb is not None:
            prev_emb = emb
        seg["speaker"] = gid


def _confirm_pieces(audio, pieces, embed, merge_sim, sample_rate):
    """Guard against over-splitting: re-embed each candidate piece on its FULL
    (longer, cleaner) audio and merge adjacent pieces that come back as the SAME
    voice (cosine >= merge_sim). Piece-quality audio embeds far more reliably
    than the short windows that proposed the boundary, so this reliably undoes
    spurious change-point splits while keeping real turns. pieces = [(gid, s, e)]."""
    embs = []
    for _gid, ps, pe in pieces:
        pa = bytes(audio[int(ps * sample_rate) * 2:int(pe * sample_rate) * 2])
        embs.append(embed(pa) if pa else None)
    out = [list(pieces[0])]
    out_emb = [embs[0]]
    for i in range(1, len(pieces)):
        g, ps, pe = pieces[i]
        e, prev = embs[i], out_emb[-1]
        if e is not None and prev is not None and float(e @ prev) >= merge_sim:
            out[-1][2] = pe                      # same voice -> merge into previous piece
        else:
            out.append([g, ps, pe])
            out_emb.append(e)
    return [(g, ps, pe) for g, ps, pe in out]


def split_live_utterance(audio, text, label_fn, *, window_s=1.5,
                         sample_rate=16000, min_piece=0.6,
                         change_sim=0.35, merge_sim=0.62):
    """Split ONE finalized live utterance into per-speaker pieces when >1 speaker
    alternates within it (paragraph mode / no silence gap between turns).

    Segmentation and identity are deliberately two separate steps: this used to
    call the full label_fn (match-or-spawn) independently on every ~window_s
    slice, but a single 1.5s window's embedding is noisy enough that a real
    speaker's OWN windows regularly fell below both the named-match and
    session-cluster thresholds by chance — spawning a fresh 說話者N on nearly
    every window (see live_speaker_labeler). Instead: if label_fn exposes a
    state-free `.peek(chunk)` (live_speaker_labeler does), use it to find turn
    BOUNDARIES only where a window CONFIDENTLY names an already-established
    identity different from the current one (peek never spawns; an unconfident
    window just extends the current speaker rather than starting a new one).
    The real label_fn is then called ONCE per detected piece on that piece's
    full (longer, cleaner) audio — identity is decided at utterance-piece
    quality, not single-window quality. label_fn without `.peek` (e.g. test
    doubles) falls back to the original per-window-is-the-decision behaviour.

    Returns [(speaker, text, rel_start_ms, rel_end_ms)] — exactly one piece when a
    single speaker holds the utterance (so the caller can treat it as the normal
    one-line case)."""
    win_bytes = int(window_s * sample_rate) * 2
    total_s = len(audio) / (sample_rate * 2)
    total_ms = int(total_s * 1000)
    if win_bytes <= 0 or len(audio) < 2 * win_bytes:   # too short to hold a turn change
        return [(label_fn(audio), text, 0, total_ms)]
    peek = getattr(label_fn, "peek", None)
    embed = getattr(label_fn, "embed", None)
    peek_from_emb = getattr(label_fn, "peek_from_emb", None)
    segments, pos = [], 0
    while pos < len(audio):
        chunk = bytes(audio[pos:pos + win_bytes])
        if len(chunk) < win_bytes // 2 and segments:
            segments[-1]["end"] = total_s              # tail remnant -> extend last window
            break
        st = pos / (sample_rate * 2)
        seg = {"start": st, "end": min(total_s, st + window_s)}
        if embed is not None:
            # One extraction per window: reuse it for BOTH the confident-identity
            # peek and the change-point signal (no double embedding).
            e = embed(chunk)
            seg["_emb"] = e
            seg["_peek"] = peek_from_emb(e) if peek_from_emb is not None else (
                peek(chunk) if peek is not None else None)
        elif peek is not None:
            seg["_peek"] = peek(chunk)
        else:
            seg["speaker"] = label_fn(chunk)
        segments.append(seg)
        pos += win_bytes
    if peek is not None or embed is not None:
        _group_by_peek(segments, change_sim=change_sim if embed is not None else None)
    pieces = _speaker_pieces(0.0, total_s, segments, min_piece=min_piece)
    # Confirm change-point splits at piece quality: merge adjacent pieces that
    # re-embed as the same voice (undoes noisy over-splits; keeps real turns).
    if embed is not None and len(pieces) > 1:
        pieces = _confirm_pieces(audio, pieces, embed, merge_sim, sample_rate)
    if len(pieces) <= 1:
        if peek is not None or embed is not None:
            return [(label_fn(audio), text, 0, total_ms)]  # one turn -> decide on full audio
        spk = pieces[0][0] if pieces else (segments[0]["speaker"] if segments else None)
        return [(spk, text, 0, total_ms)]
    dur = total_s or 1.0
    texts = _split_text(text, [(pe - ps) / dur for _, ps, pe in pieces])
    if peek is None and embed is None:
        return [(spk, txt, int(ps * 1000), int(pe * 1000))
                for (spk, ps, pe), txt in zip(pieces, texts)]
    out = []
    for (_gid, ps, pe), txt in zip(pieces, texts):
        piece_audio = bytes(audio[int(ps * sample_rate) * 2:int(pe * sample_rate) * 2])
        spk = label_fn(piece_audio) if piece_audio else None
        out.append((spk, txt, int(ps * 1000), int(pe * 1000)))
    return out


def similar_speaker_pairs(rows, threshold=0.5, dismissed=()):
    """Global-speaker pairs whose voiceprints are close enough to maybe be the SAME
    person but didn't auto-merge — surfaced as 'might be a duplicate, merge?'
    suggestions. rows = store.list_speakers() (id, name, centroid bytes). `dismissed`
    = iterable of (name, name) pairs the user marked 'not the same person' — skipped."""
    import numpy as np  # noqa: PLC0415
    dismissed = {frozenset(p) for p in dismissed}
    vecs = []
    for r in rows:
        if not r["centroid"]:
            continue
        v = np.frombuffer(r["centroid"], dtype=np.float32)
        n = float(np.linalg.norm(v))
        if n > 0:
            vecs.append((r["id"], r["name"], v / n))
    out, seen = [], set()
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            a, b = vecs[i][1], vecs[j][1]
            if a == b or (a, b) in seen or (b, a) in seen:
                continue  # same name = same person; dedupe name pairs
            if _is_placeholder(a) and _is_placeholder(b):
                continue  # two un-named speakers -> can't judge, not a useful suggestion
            if frozenset((a, b)) in dismissed:
                continue  # user said 'not the same person'
            sim = float(np.dot(vecs[i][2], vecs[j][2]))
            if sim >= threshold:
                seen.add((a, b))
                out.append({"a": a, "b": b, "sim": round(sim, 3)})
    out.sort(key=lambda x: -x["sim"])
    return out


def diarize_pcm(pcm_path, *, sample_rate=16000, num_speakers=-1,
                seg_model=None, emb_model=None, progress=None, on_progress=None,
                provider="cpu"):
    """Cluster speakers in a 16-bit mono PCM file -> [{start,end,speaker}].
    num_speakers=-1 auto-detects the count. Models auto-provision into models/
    on first use (see _resolve_models); override via SHERPA_SEG/EMB_MODEL.
    on_progress(done, total): optional, called as sherpa processes chunks.
    provider='coreml' runs the seg + embedding onnx on the Neural Engine."""
    import numpy as np  # noqa: PLC0415
    import sherpa_onnx  # noqa: PLC0415

    seg, emb = _resolve_models(seg_model, emb_model, progress)

    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=seg),
            provider=provider),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=emb, provider=provider),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=num_speakers),
    )
    sd = sherpa_onnx.OfflineSpeakerDiarization(config)
    with open(pcm_path, "rb") as _f:
        audio = np.frombuffer(_f.read(), dtype=np.int16)
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
