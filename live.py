"""Live transcription: chunk streamed PCM, transcribe each chunk, emit segments
with absolute timestamps (spec Phase 2, profile=live).

Chunking is pluggable:
- FixedWindowChunker: cut every N seconds (deterministic, used in tests).
- VadChunker: cut at silence after speech (or a max-length ceiling) so windows
  hold whole phrases instead of slicing mid-word.

Each chunk is transcribed independently (no cross-chunk context) — live is a
preview; the accurate re-pass is the trusted transcript (spec M1)."""
import sys
import time

import numpy as np

import zhtw


def _rms(frame_bytes):
    if not frame_bytes:
        return 0.0
    a = np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(a * a))) if a.size else 0.0


def preprocess(audio):
    """Robustness for poor real-world recordings: strip DC offset and peak-
    normalize so quiet / off-level audio reaches whisper at a usable level.
    ponytail: no high-pass yet — add a one-pole HPF if low-freq rumble shows."""
    audio = audio - audio.mean()
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1e-4:
        audio = audio / peak * 0.95
    return audio


def _compress_ratio(stripped):
    import zlib  # noqa: PLC0415
    b = stripped.encode()
    return len(b) / max(1, len(zlib.compress(b, 9)))


def _is_repetition(text):
    """Drop degenerate ASR loops (whisper/qwen repeat a token on silence/noise:
    'segment segment…', '每日每日每日…'). The robust universal signal is COMPRESSION
    RATIO — a loop compresses far more than speech (measured: natural zh ~0.9, 每日×N
    ~4.2, 喬×N ~8.6; threshold 1.8 has a wide margin and catches loops behind a real
    prefix like 我們每日每日…). Cheap distinct-word/char checks handle the short cases
    where compression overhead is unreliable."""
    parts = text.split()
    if len(parts) >= 4 and len(set(parts)) <= 2:
        return True
    stripped = text.replace(" ", "")
    if len(stripped) >= 8 and len(set(stripped)) <= 3:
        return True
    return len(stripped) >= 12 and _compress_ratio(stripped) >= 1.8


# Whisper's stock hallucinations on breath/noise — drop when they ARE the whole
# (short) output. A real "yeah" inside a sentence won't match (text is longer).
_FILLERS = {"you", "yeah", "now", "ok", "okay", "so", "bye", "thank you",
            "thanks", "thank you.", "you.", "嗯", "啊", "呃", "謝謝", "謝謝觀看",
            "謝謝大家", "請不吝點贊訂閱", "字幕由", "下次再見"}

# Multi-word YouTube-outro hallucinations (Whisper trained on captioned video).
# Matched by substring with a coverage test so a real sentence merely *containing*
# "thank you" isn't dropped — only segments that ARE essentially the phrase.
_HALLUCINATION = [
    "thank you for watching", "thanks for watching", "thank you for your attention",
    "thank you for your watching", "please subscribe", "like and subscribe",
    "subscribe to my channel", "see you next time", "see you in the next video",
    "謝謝觀看", "謝謝大家的觀看", "請不吝點贊", "請訂閱", "點贊訂閱", "下次再見",
    "字幕由", "感謝您的觀看", "感謝觀看",
]

# Never-legitimate whisper-zh silence hallucinations (TV/stream intros learned from
# captioned video). Dropped on plain substring — NO coverage test — because they're
# garbage even when padded with extra hallucinated words (優悠獨播劇場——YoYo Television…).
_HALLUCINATION_STRONG = [
    "獨播劇場", "独播剧场", "yoyo television", "yoyo tv",
    "明鏡與點點", "明镜与点点", "中文字幕志愿者", "by 索绪尔", "索绪尔",
    "請不吝點贊 訂閱 轉發 打賞", "请不吝点赞",
]

# A finalized window that emits far more text than its measured speech duration
# supports is a silence/noise hallucination — whisper/qwen confabulate a fluent
# sentence over near-silence ("This is meeting tonight…" on the 對方 track). ~25
# chars/sec is already generous for fast zh OR en; above it the words could not
# have been spoken in the speech actually present. Universal — no per-model
# metadata, no language assumption, no phrase blocklist to keep chasing.
_MAX_CHARS_PER_SPEECH_S = 25


def _norm(text):
    return text.strip().lower().strip(" .,!?。，、！？、")


def _is_filler(text):
    return _norm(text) in _FILLERS


def _is_hallucination(text):
    t = _norm(text)
    if not t:
        return False
    if any(p in t for p in _HALLUCINATION_STRONG):  # never legit -> drop on substring
        return True
    for p in _HALLUCINATION:
        if p in t and len(p) >= 0.6 * len(t):  # phrase dominates the segment
            return True
    return False


_SILERO_SESS = {}  # path -> ort.InferenceSession, shared across tracks/sessions


def _silero_session(path):
    """Load the silero ONNX session ONCE per path and reuse it. The RNN state
    (h/c/buf) is per-SileroVad-instance (passed as run() inputs), so sharing the
    stateless session across tracks/live-sessions is safe and skips the per-
    session onnxruntime load that was blocking the /ws start handshake."""
    sess = _SILERO_SESS.get(path)
    if sess is None:
        import onnxruntime as ort  # noqa: PLC0415
        t = time.perf_counter()
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        _SILERO_SESS[path] = sess
        print(f"[live start] silero VAD load {(time.perf_counter()-t)*1000:.0f}ms (cold)",
              file=sys.stderr)
    return sess


class SileroVad:
    """Optional neural VAD (snakers4 silero v4, via onnxruntime — no torch). A
    callable frame_bytes->bool for TwoPassSession.speech_fn. Buffers to 512-sample
    chunks (v4 16k window), carries the h/c RNN state, returns the latest decision."""

    def __init__(self, path="models/silero_vad_v4.onnx", sample_rate=16000, threshold=0.5):
        self._s = _silero_session(path)
        self.sr = sample_rate
        self.th = threshold
        self._win = 512 if sample_rate == 16000 else 256
        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)
        self._buf = np.zeros(0, dtype=np.float32)
        self._last = 0.0

    def __call__(self, frame_bytes):
        x = np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        self._buf = np.concatenate([self._buf, x])
        while len(self._buf) >= self._win:
            chunk = self._buf[:self._win].astype(np.float32)
            self._buf = self._buf[self._win:]
            out, self._h, self._c = self._s.run(
                None, {"input": chunk[None, :], "sr": np.array(self.sr, dtype=np.int64),
                       "h": self._h, "c": self._c})
            self._last = float(out[0, 0])
        return self._last >= self.th


class FixedWindowChunker:
    def __init__(self, window_bytes):
        self.window_bytes = window_bytes
        self._buf = bytearray()

    def feed(self, pcm):
        self._buf.extend(pcm)
        out = []
        while len(self._buf) >= self.window_bytes:
            out.append(bytes(self._buf[:self.window_bytes]))
            del self._buf[:self.window_bytes]
        return out

    def flush(self):
        if not self._buf:
            return []
        w = bytes(self._buf)
        self._buf.clear()
        return [w]


class VadChunker:
    """Energy VAD: emit a window when a speech run is followed by >= silence_ms
    of quiet, or when the buffer hits max_window_s (latency ceiling).

    ponytail: RMS-threshold heuristic, no deps. Swap for silero-VAD if it
    mis-cuts on noisy input — the upgrade path is the same feed()/flush() API."""

    def __init__(self, sample_rate=16000, frame_ms=30, silence_ms=400,
                 max_window_s=8.0, rms_threshold=500):
        self.frame_bytes = int(sample_rate * frame_ms / 1000) * 2  # 16-bit mono
        self.silence_frames = max(1, silence_ms // frame_ms)
        self.max_bytes = int(max_window_s * sample_rate) * 2
        self.rms_threshold = rms_threshold
        self._buf = bytearray()
        self._scan = 0
        self._silence_run = 0
        self._has_speech = False

    def _reset(self):
        self._scan = 0
        self._silence_run = 0
        self._has_speech = False

    def _find_cut(self):
        n = len(self._buf)
        while self._scan + self.frame_bytes <= n:
            frame = self._buf[self._scan:self._scan + self.frame_bytes]
            self._scan += self.frame_bytes
            if _rms(frame) >= self.rms_threshold:
                self._has_speech = True
                self._silence_run = 0
            else:
                self._silence_run += 1
                if self._has_speech and self._silence_run >= self.silence_frames:
                    return self._scan  # cut after the silence that ended speech
        if n >= self.max_bytes:
            return self.max_bytes
        return None

    def feed(self, pcm):
        self._buf.extend(pcm)
        out = []
        while True:
            cut = self._find_cut()
            if cut is None:
                break
            had_speech = self._has_speech
            window = bytes(self._buf[:cut])
            del self._buf[:cut]
            self._reset()
            if had_speech:           # drop non-speech windows (force-cut noise)
                out.append(window)
        return out

    def flush(self):
        if not self._buf or not self._has_speech:
            self._buf.clear()
            self._reset()
            return []
        w = bytes(self._buf)
        self._buf.clear()
        self._reset()
        return [w]


class LiveSession:
    def __init__(self, *, backend, chunker=None, sample_rate=16000,
                 window_s=5.0, track="mic"):
        self.backend = backend
        self.sample_rate = sample_rate
        self.track = track
        self.chunker = chunker or FixedWindowChunker(int(window_s * sample_rate * 2))
        self._emitted_bytes = 0  # cumulative -> absolute timestamp offset

    def feed(self, pcm_bytes):
        out = []
        for window in self.chunker.feed(pcm_bytes):
            out.extend(self._transcribe(window))
        return out

    def flush(self):
        out = []
        for window in self.chunker.flush():
            out.extend(self._transcribe(window))
        return out

    def _transcribe(self, window):
        offset_ms = round(self._emitted_bytes / (self.sample_rate * 2) * 1000)
        self._emitted_bytes += len(window)
        out = []
        for s in self.backend(window):
            text = s["text"].strip()
            if not text or _is_repetition(text):
                continue
            out.append({
                "start_ms": round(s["start"] * 1000) + offset_ms,
                "end_ms": round(s["end"] * 1000) + offset_ms,
                "text": text,
                "track": self.track,
                "profile": "live",
            })
        return out


class TwoPassSession:
    """Teams-style live transcription. While you speak, a fast backend produces
    a tentative 'interim' line that updates in place. When the utterance ends
    (VAD silence) or hits a length ceiling, an accurate backend re-transcribes
    the whole utterance and emits a 'final' that replaces the interim.

    Events: {kind:'interim', text, track} (replaceable, not stored) and
    {kind:'final', text, track, start_ms, profile} (committed).

    ponytail: interim re-runs over the whole growing utterance each tick — cost
    grows with utterance length, bounded by max_utt_s. Go incremental only if
    long single utterances get sluggish."""

    def __init__(self, *, backend, interim_backend=None, sample_rate=16000,
                 frame_ms=30, silence_ms=400, max_utt_s=15.0, interim_s=0.6,
                 interim_tail_s=8.0, min_speech_ms=250, rms_threshold=80,
                 speech_factor=2.0, track="mic", speaker_fn=None, speech_fn=None,
                 splitter=None,
                 interim_duty=0.75, interim_min_s=0.4, interim_max_s=3.0, clock=None):
        self.final_backend = backend
        self.interim_backend = interim_backend
        self.speaker_fn = speaker_fn  # optional audio_bytes -> live speaker label
        # optional (audio_bytes, text) -> [(speaker, text, rel_start_ms, rel_end_ms)]:
        # when one finalized utterance holds >1 speaker (e.g. paragraph mode, no
        # silence gap), split it so each person's words land on their own line.
        self.splitter = splitter
        self.speech_fn = speech_fn    # optional frame_bytes -> bool (silero); else energy VAD
        self.sr = sample_rate
        self.track = track
        self.frame_bytes = int(sample_rate * frame_ms / 1000) * 2
        self.silence_frames = max(1, silence_ms // frame_ms)
        self.min_speech_frames = max(1, min_speech_ms // frame_ms)
        self.max_bytes = int(max_utt_s * sample_rate) * 2
        self.interim_bytes = int(interim_s * sample_rate) * 2
        # interim transcribes only the recent tail so its cost stays flat as the
        # utterance grows — keeps captions snappy ('live enough').
        self.interim_tail_bytes = int(interim_tail_s * sample_rate) * 2
        # Adaptive cadence: hold the interim ASR DUTY CYCLE (compute_time /
        # realtime) near interim_duty — there's no non-sudo GPU% on macOS, but
        # this fraction IS the GPU-bound MLX pipeline's effective load. After each
        # interim we set the next interval = last_compute / duty, so a fast model
        # transcribes more often (uses spare compute) and a slow one (1.7B) backs
        # off automatically instead of falling behind. Bounded both ways — the
        # ~1s floor (interim_min_s) deliberately caps how often the preview is
        # re-revised: faster than that just churns the displayed words (each
        # re-transcribe of the growing tail can flip them, "越修正越錯") and wastes
        # compute for no readability gain; the final pass is the accurate one.
        self.interim_duty = interim_duty
        self.interim_min_bytes = int(interim_min_s * sample_rate) * 2
        self.interim_max_bytes = int(interim_max_s * sample_rate) * 2
        self._interim_dyn = self.interim_bytes   # current threshold (adapts)
        self._interim_warm = True                # ignore 1st call (cold model load)
        self._clock = clock or time.monotonic
        self.min_floor = rms_threshold        # absolute noise-floor minimum
        self.speech_factor = speech_factor    # speech if rms >= noise * factor
        self._noise = float(rms_threshold)     # adaptive noise-floor estimate
        self._committed_bytes = 0
        self._reset_utt()

    def _is_speech(self, rms):
        # Adaptive VAD: fast attack down to a quieter floor, slow release up,
        # so a low/variable recording level still separates speech from noise.
        if rms < self._noise:
            self._noise = rms
        else:
            self._noise += (rms - self._noise) * 0.02
        self._noise = max(self.min_floor, self._noise)
        return rms >= self._noise * self.speech_factor

    def _reset_utt(self):
        self._utt = bytearray()
        self._scan = 0
        self._silence_run = 0
        self._speech_frames = 0
        self._has_speech = False
        self._last_interim_len = 0

    def _text(self, backend, audio):
        parts = [s["text"].strip() for s in backend(audio)]
        text = " ".join(p for p in parts if p).strip()
        if not text or _is_repetition(text) or _is_filler(text) or _is_hallucination(text):
            return ""
        return zhtw.to_tw(text)  # normalize 簡->繁(台灣)

    def _enough_speech(self):
        return self._speech_frames >= self.min_speech_frames

    def _warn_if_slow(self, audio, asr_s, diar_s):
        # Live stall diagnosis ("跑一跑卡住一陣子"): a finalize does final-ASR then
        # the per-utterance diarization embedding SERIALLY, both on the Neural
        # Engine — if they take longer than the utterance's own realtime the live
        # loop falls behind and backlog grows until the 45s trim. Log the split
        # (asr vs diar) ONLY when we're behind realtime, so a real session's log
        # says whether the stall is ASR, the embedding, or ANE contention between
        # them. Silent on the normal fast path (RTF<1) — no log spam.
        audio_s = len(audio) / (self.sr * 2)
        compute_s = asr_s + diar_s
        if audio_s > 0 and compute_s > audio_s:
            print(f"live SLOW [{self.track}] {audio_s:.1f}s audio -> "
                  f"asr {asr_s:.1f}s + diar {diar_s:.1f}s = {compute_s:.1f}s "
                  f"(RTF {compute_s / audio_s:.1f}x)", file=sys.stderr)

    def _finalize(self, want_diarize=True):
        # want_diarize=False (set by consume when the track is BEHIND realtime)
        # skips the per-utterance voiceprint embedding for this line to catch up
        # on ASR text — the embedding is the heavy per-line cost and contends with
        # ASR on the Neural Engine. The line just falls back to its side label
        # (對方/我); the post-meeting /diarize pass relabels accurately anyway.
        # Drop blips (cough/breath) BEFORE the ASR call — saves compute (perf).
        # But STILL advance the committed-byte clock by their length: the audio
        # file keeps those bytes, so skipping them would make every later
        # timestamp drift ahead of the audio (worse with paragraph-size windows).
        # Returns a LIST of events ([] / one / many) — the splitter can fan one
        # utterance into one line per speaker.
        if not self._enough_speech():
            self._committed_bytes += len(self._utt)
            self._reset_utt()
            return []
        audio = bytes(self._utt)
        _t_asr = self._clock()
        text = self._text(self.final_backend, audio)
        asr_s = self._clock() - _t_asr
        diar_s = 0.0
        # Silence-hallucination gate: drop text too long for the speech present.
        speech_s = self._speech_frames * self.frame_bytes / (self.sr * 2)
        if text and len(text) > 12 and len(text) > speech_s * _MAX_CHARS_PER_SPEECH_S:
            text = ""
        offset_ms = round(self._committed_bytes / (self.sr * 2) * 1000)
        end_ms = round((self._committed_bytes + len(audio)) / (self.sr * 2) * 1000)
        self._committed_bytes += len(audio)
        # Within-utterance split (multi-speaker in one VAD segment) takes precedence
        # over the single-label speaker_fn. The splitter labels every piece itself
        # (it owns the embedding), so use its pieces whenever it returns any — a
        # single piece is just the normal one-speaker case; >1 means a real split.
        if text and want_diarize and self.splitter:
            _t_diar = self._clock()
            try:
                parts = self.splitter(audio, text)
            except Exception:
                parts = None
            diar_s = self._clock() - _t_diar
            if parts:
                self._warn_if_slow(audio, asr_s, diar_s)
                self._reset_utt()
                multi = len(parts) > 1
                evs = []
                for pspk, ptext, rs, re in parts:
                    if not ptext:
                        continue
                    e = {"kind": "final", "text": ptext, "track": self.track,
                         "start_ms": offset_ms + rs, "end_ms": offset_ms + re,
                         "profile": "live"}
                    if pspk:
                        e["speaker"] = pspk
                    if multi:
                        e["split"] = True
                    evs.append(e)
                return evs
        spk = None
        if text and want_diarize and self.speaker_fn:
            _t_diar = self._clock()
            try:
                spk = self.speaker_fn(audio)  # online voiceprint speaker label
            except Exception:
                spk = None
            diar_s = self._clock() - _t_diar
        if text:
            self._warn_if_slow(audio, asr_s, diar_s)
        self._reset_utt()
        if not text:
            return []
        ev = {"kind": "final", "text": text, "track": self.track,
              "start_ms": offset_ms, "end_ms": end_ms, "profile": "live"}
        if spk:
            ev["speaker"] = spk
        return [ev]

    def feed(self, pcm, want_interim=True, want_diarize=True):
        self._utt.extend(pcm)
        events = []
        while self._scan + self.frame_bytes <= len(self._utt):
            frame = self._utt[self._scan:self._scan + self.frame_bytes]
            self._scan += self.frame_bytes
            speech = self.speech_fn(bytes(frame)) if self.speech_fn else self._is_speech(_rms(frame))
            if speech:
                self._has_speech = True
                self._speech_frames += 1
                self._silence_run = 0
            else:
                self._silence_run += 1
                if self._has_speech and self._silence_run >= self.silence_frames:
                    events.extend(self._finalize(want_diarize))
        if self._has_speech and len(self._utt) >= self.max_bytes:
            events.extend(self._finalize(want_diarize))
        # interim only once there's real sustained speech (no work on blips/silence)
        if (want_interim and self.interim_backend and self._enough_speech()
                and len(self._utt) - self._last_interim_len >= self._interim_dyn):
            self._last_interim_len = len(self._utt)
            tail = bytes(self._utt[-self.interim_tail_bytes:])
            t0 = self._clock()
            text = self._text(self.interim_backend, tail)
            self._adapt_interim(self._clock() - t0)
            if text:
                events.append({"kind": "interim", "text": text, "track": self.track})
        return [e for e in events if e]

    def _adapt_interim(self, compute_s):
        # Next interim fires after (compute / duty) seconds of new audio so the
        # fraction of realtime spent on interim ASR stays ~= interim_duty. Skip
        # the first measurement: the cold model-load call inflates it.
        if self._interim_warm:
            self._interim_warm = False
            return
        b = int(max(0.0, compute_s) / self.interim_duty * self.sr) * 2
        self._interim_dyn = max(self.interim_min_bytes,
                                min(self.interim_max_bytes, b))

    def flush(self):
        if self._has_speech:
            return self._finalize()
        return []


class AdaptiveBackend:
    """Wrap an ordered list of backends (slowest/best first). If the current one
    can't keep up — real-time factor (process_time / audio_seconds) over budget
    for `patience` consecutive calls — drop to the next faster tier. One-way:
    never flaps back up. Models load lazily on first use, so the slower tiers
    cost no RAM until reached."""

    def __init__(self, backends, models, *, sample_rate=16000, rtf_budget=0.8,
                 patience=2, clock=None, on_change=None):
        self.backends = backends
        self.models = models
        self.sr = sample_rate
        self.rtf_budget = rtf_budget
        self.patience = patience
        self.on_change = on_change  # called with the new model on downgrade
        self.idx = 0
        self._over = 0
        self._warmup = True  # first call per tier loads weights -> don't judge its RTF
        self._notice = None
        if clock is None:
            import time as _t
            clock = _t.monotonic
        self.clock = clock

    @property
    def current_model(self):
        return self.models[self.idx]

    def pop_notice(self):
        n, self._notice = self._notice, None
        return n

    def __call__(self, window_bytes):
        t0 = self.clock()
        out = self.backends[self.idx](window_bytes)
        dur = len(window_bytes) / (self.sr * 2)
        if dur > 0:
            rtf = (self.clock() - t0) / dur
            if self._warmup:
                self._warmup = False  # cold-load call — ignore its inflated RTF
            elif rtf > self.rtf_budget and self.idx < len(self.backends) - 1:
                self._over += 1
                if self._over >= self.patience:
                    self.idx += 1
                    self._over = 0
                    self._warmup = True  # new tier warms up before being judged
                    self._notice = f"模型跑不動,已切換較快模型:{self.models[self.idx]}"
                    if self.on_change:
                        self.on_change(self.models[self.idx])  # remember for next run
            else:
                self._over = 0
        return out


def mlx_whisper_live_backend(model="mlx-community/whisper-small-mlx", language=None):
    """Real live backend — Apple Silicon only, lazy import. Takes int16 PCM
    bytes for one window, returns whisper segments. language=None -> auto."""
    import mlx_whisper  # noqa: PLC0415
    lang = language or None

    def _run(window_bytes):
        if len(window_bytes) < 2:
            return []
        # Keep 16-bit alignment (a dropped/odd tail byte would crash frombuffer).
        if len(window_bytes) % 2:
            window_bytes = window_bytes[:-1]
        audio = np.frombuffer(window_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        audio = preprocess(audio)  # DC removal + normalize for poor recordings
        try:
            segs = mlx_whisper.transcribe(
                audio, path_or_hf_repo=model, language=lang,
                condition_on_previous_text=False,  # don't propagate a loop forward
            )["segments"]
        except Exception as e:  # one bad window must not kill the live session
            import sys
            print(f"live ASR error (skipped): {e}", file=sys.stderr)
            return []
        # Whisper's standard hallucination guards: drop non-speech / repetitive /
        # low-confidence segments before they reach the subtitle.
        return [s for s in segs
                if s.get("no_speech_prob", 0) < 0.6
                and s.get("compression_ratio", 0) < 2.4
                and s.get("avg_logprob", 0) > -1.0]

    return _run
