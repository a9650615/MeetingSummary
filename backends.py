"""Modular ASR backend registry + hot-reloadable live model.

route(model) decides which engine a model id belongs to. make_* build the actual
callables (lazy — weights load on first use, so hot-swapping is instant). The
LiveModelManager lets the running server switch the live model with no restart:
TwoPassSession calls `manager.backend` through a thin shim, so set_model() takes
effect on the next utterance — even mid-session."""
from live import AdaptiveBackend, mlx_whisper_live_backend


def route(model):
    """Engine for a model id. chatllm 1.7B GGUF -> chatllm (.cpp/Metal, persistent
    binding); q4-k-m -> femelo .cpp (Metal); other qwen3-asr -> transformers;
    whisper(-mlx) is the Apple-native live engine."""
    m = model.lower()
    if m.startswith("ane-"):  # ANE (Neural Engine) via the `speech` CLI
        return "ane"
    if model == "qwen3-asr-1.7b":
        return "chatllm"
    if "qwen3-asr" in m and "mlx-community" in m:  # MLX-native Qwen3-ASR (Metal, fast)
        return "qwen3mlx"
    if "q4-k-m" in m:
        return "qwen3cpp"
    if "qwen3-asr" in m:
        return "qwen3"
    if m == "firered" or "fire-red" in m:  # FireRedASR-AED via sherpa-onnx (CPU, batch)
        return "firered"
    return "whisper"


def qwen3_words_to_segments(words, text):
    """Map the sidecar's AlignedWord list -> [{start,end,text}] (seconds). Falls
    back to a single segment from the full text if alignment gave no words."""
    segs = [{"start": w["start"], "end": w["end"], "text": w["word"]}
            for w in words if w.get("word", "").strip()]
    if segs:
        return segs
    return [{"start": 0.0, "end": 0.0, "text": text}] if text.strip() else []


# femelo's Qwen3-ASR .cpp IGNORES the language arg — verified: ''/zh/ja/Chinese/zh-CN
# all auto-detect identically. So a forced language silently does nothing and short
# ambiguous clips misdetect (e.g. zh -> Japanese kana). whisper-MLX honors language
# strictly (zh->Chinese, ja->Japanese), so when the user FORCES a language we route the
# femelo path to whisper turbo-q4 (RTF~0.07, plenty fast). Auto mode keeps femelo.
_LANG_FALLBACK_MODEL = "mlx-community/whisper-large-v3-turbo-q4"


def _honor_language(model, language):
    if language and route(model) == "qwen3cpp":
        return _LANG_FALLBACK_MODEL
    return model


_QWEN3_MLX = {}  # repo -> loaded mlx-audio model (heavy; load once)
_QWEN3_MLX_EXEC = None  # single dedicated thread for ALL mlx-audio work


def _qwen3_mlx_exec():
    # MLX streams are thread-local; mlx-audio's Qwen3-ASR creates a non-default GPU
    # stream, so loading on one thread and running on another (live threadpool vs the
    # re-transcribe job thread) -> "no Stream(gpu, 1) in current thread" / crashes.
    # Pin load + every inference to ONE dedicated thread so the stream is consistent.
    global _QWEN3_MLX_EXEC
    if _QWEN3_MLX_EXEC is None:
        import concurrent.futures
        _QWEN3_MLX_EXEC = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="qwen3mlx")
    return _QWEN3_MLX_EXEC


def _stt_text(r):
    """Pull transcript text out of an mlx-audio STTOutput / dict / str.
    STTOutput.text is '' on silence — a VALID empty result. The old
    `getattr(r,'text',None) or str(r)` fell back to repr(STTOutput) on '' and
    stored the whole "STTOutput(text='', segments=[...])" string as transcript."""
    if hasattr(r, "text"):
        t = r.text
    elif isinstance(r, dict):
        t = r.get("text", "")
    else:
        t = str(r)
    return (t or "").strip()


def qwen3_mlx_backend(model="mlx-community/Qwen3-ASR-1.7B-8bit", language=None):
    """MLX-native Qwen3-ASR via mlx-audio (Metal, in-process). ~4x faster than the
    chatllm GGUF path (RTF ~0.13 vs 0.61 on M3). callable(audio) -> [{start,end,text}]
    (one segment; iter_transcribe windows for timestamps). All MLX work runs on a
    single dedicated thread (thread-local stream safety)."""

    _txt = _stt_text

    def _infer(audio):  # runs ON the dedicated thread
        import numpy as np  # noqa: PLC0415
        from mlx_audio.stt.generate import generate_transcription  # noqa: PLC0415
        from mlx_audio.stt.utils import load  # noqa: PLC0415
        m = _QWEN3_MLX.get(model)
        if m is None:
            m = load(model)
            _QWEN3_MLX[model] = m
        # live -> raw int16 PCM bytes; batch -> raw 16k mono .pcm path; upload ->
        # container file (m4a/mp3/wav). ALWAYS hand mlx-audio a 16k-mono f32
        # mx.array, never a path: its loader shells out to ffprobe, which is often
        # absent (ffmpeg-only installs) -> "ffprobe not found" crash on upload. We
        # decode containers ourselves with ffmpeg (no ffprobe needed).
        if isinstance(audio, (bytes, bytearray)):
            pcm = np.frombuffer(bytes(audio), dtype=np.int16)
        elif isinstance(audio, str) and audio.endswith(".pcm"):
            with open(audio, "rb") as _f:
                pcm = np.frombuffer(_f.read(), dtype=np.int16)
        elif isinstance(audio, str):
            import shutil  # noqa: PLC0415
            import subprocess  # noqa: PLC0415
            if not shutil.which("ffmpeg"):
                raise RuntimeError(f"ffmpeg needed to decode {audio}")
            raw = subprocess.run(
                ["ffmpeg", "-v", "error", "-i", audio, "-ar", "16000", "-ac", "1",
                 "-f", "s16le", "-"], stdout=subprocess.PIPE, check=True).stdout
            pcm = np.frombuffer(raw, dtype=np.int16)
        else:
            return [{"start": 0.0, "end": 0.0, "text": _txt(
                generate_transcription(model=m, audio=audio, verbose=False))}]
        import mlx.core as mx  # noqa: PLC0415

        def _gen(seg):  # one int16 window -> text; release MLX buffers after
            t = _txt(generate_transcription(
                model=m, audio=mx.array(seg.astype(np.float32) / 32768.0), verbose=False))
            mx.clear_cache()
            return t

        sr, win = 16000, 16000 * 30
        # Qwen3-ASR has NO internal long-audio chunking — past its encoder limit it
        # returns nothing (a 3-min upload came back empty). So window long inputs
        # (uploads) into ~30s with per-window timestamps. Short inputs (live windows,
        # re-transcribe's 30s .pcm) stay one untimed segment (caller already windows).
        if len(pcm) <= win:
            return [{"start": 0.0, "end": 0.0, "text": _gen(pcm)}]
        out = []
        for i in range(0, len(pcm), win):
            seg = pcm[i:i + win]
            if len(seg) < sr // 5:  # <0.2s tail
                continue
            t = _gen(seg)
            if t:
                out.append({"start": i / sr, "end": (i + len(seg)) / sr, "text": t})
        return out

    def _run(audio):
        return _qwen3_mlx_exec().submit(_infer, audio).result()

    return _run


def make_live_backend(model, language=None):
    """Live backend, callable(pcm_bytes) -> segments. language=None -> auto-detect;
    a code ("zh"/"en"/"ja"...) forces it. whisper-MLX (default), Qwen3-ASR .cpp via a
    persistent daemon (Metal), or transformers Qwen3-ASR."""
    if route(model) == "ane":  # persistent Neural-Engine helper (省電, off-GPU)
        return ane_live_backend()
    model = _honor_language(model, language)
    r = route(model)
    if r == "chatllm":
        return chatllm_live_backend(language)
    if r == "qwen3mlx":
        return qwen3_mlx_backend(model, language)
    if r == "qwen3cpp":
        return qwen3_cpp_live_backend(language)
    if r == "qwen3":
        return qwen3_live_backend(model, language)
    return mlx_whisper_live_backend(model, language)


class _ChatllmAsrDaemon:
    """Persistent chatllm.cpp Qwen3-ASR 1.7B (Metal) via the python ctypes binding
    — loads the GGUF once, in-process (no sidecar; binding is cp-version-agnostic).
    One request at a time (lock). PCM -> temp wav -> {{audio:path}} message ->
    transcription (text after <asr_text>). Lazy-loads on first call."""

    def __init__(self):
        import threading
        self._m = None
        self._acc = {"t": ""}
        self._lock = threading.Lock()
        self.lang = "auto"   # chatllm wants a NAME (Chinese/English/…) or auto

    def set_lang(self, lang):
        if lang != self.lang:
            self.lang = lang
            self._m = None   # respawn with the new --set language

    def _ensure(self):
        if self._m is not None:
            return
        import os
        import sys
        here = os.path.dirname(os.path.abspath(__file__))
        C = os.path.join(here, "chatllm.cpp")
        sys.argv[0] = os.path.join(C, "scripts", "_x.py")  # chatllm derives paths from argv[0]
        for p in (os.path.join(C, "bindings"), os.path.join(C, "scripts")):
            if p not in sys.path:
                sys.path.insert(0, p)
        from chatllm import ChatLLM, LibChatLLM  # noqa: PLC0415
        acc = self._acc

        class _Cap(ChatLLM):
            def callback_print(self, s):
                acc["t"] += s

            def callback_print_meta(self, s):
                pass

        lib = LibChatLLM(os.path.join(C, "bindings"))
        self._m = _Cap(lib, ["-m", os.path.join(C, "quantized", "qwen3-asr-1.7b.bin"),
                             "-mgl", "all", "999", "--multimedia_file_tags", "{{", "}}",
                             "--set", "language", self.lang])

    def transcribe(self, pcm_bytes):
        import os
        import sys
        import recorder  # noqa: PLC0415
        if len(pcm_bytes) < 3200:
            return []
        with self._lock:
            try:
                self._ensure()
                tmp = f"/tmp/_chatllm_{os.getpid()}.wav"
                with open(tmp, "wb") as f:
                    f.write(recorder.pcm_to_wav(pcm_bytes, sample_rate=16000, channels=1))
                self._acc["t"] = ""
                self._m.restart()
                self._m.chat("{{audio:%s}}" % tmp)
                out = self._acc["t"]
            except Exception as e:
                print(f"chatllm asr error: {e}", file=sys.stderr)
                self._m = None  # force reload next time
                return []
            finally:
                if 'tmp' in dir() and os.path.exists(tmp):
                    os.remove(tmp)
        if "<asr_text>" in out:
            out = out.split("<asr_text>", 1)[1]
        out = out.strip()
        return [{"start": 0.0, "end": 0.0, "text": out}] if out else []


_chatllm_daemon = None
# UI language code -> chatllm language NAME (chatllm wants names; "" -> auto).
_CHATLLM_LANG = {"": "auto", "zh": "Chinese", "en": "English", "ja": "Japanese",
                 "ko": "Korean", "yue": "Cantonese"}


def _chatllm_get(language):
    global _chatllm_daemon
    if _chatllm_daemon is None:
        _chatllm_daemon = _ChatllmAsrDaemon()
    _chatllm_daemon.set_lang(_CHATLLM_LANG.get(language or "", "auto"))
    return _chatllm_daemon


def chatllm_live_backend(language=None):
    """Live-final backend over the persistent chatllm 1.7B daemon (module singleton)."""
    return _chatllm_get(language).transcribe


def chatllm_batch_backend(model="qwen3-asr-1.7b", language=None):
    """Batch/accurate over the same daemon. callable(audio_path) -> segments
    (raw .pcm read to bytes; daemon wraps to wav)."""
    be = _chatllm_get(language).transcribe

    def _run(audio_path):
        with open(str(audio_path), "rb") as f:
            return be(f.read())

    return _run


def release_all():
    """Drop the heavy .cpp ASR runtimes (chatllm 1.7B in-process model, femelo
    sidecar subprocess) so idle RAM goes back. They lazy-reload on next use."""
    import gc
    freed = []
    if _chatllm_daemon is not None and _chatllm_daemon._m is not None:
        _chatllm_daemon._m = None          # drop ref -> GC frees the ggml model
        freed.append("chatllm-1.7b")
    if _qwen3_daemon is not None and getattr(_qwen3_daemon, "_proc", None) is not None:
        try:
            _qwen3_daemon._proc.terminate()
        except Exception:
            pass
        _qwen3_daemon._proc = None
        freed.append("qwen3cpp-0.6b")
    if _QWEN3_MLX:
        _QWEN3_MLX.clear()                     # drop MLX Qwen3-ASR weights
        freed.append("qwen3-asr-mlx")
    if _FIRERED:
        _FIRERED.clear()                       # drop FireRedASR onnx recognizer
        freed.append("firered")
    gc.collect()
    return freed


class _Qwen3CppDaemon:
    """Persistent subprocess (.venv-qwen314) holding the loaded GGUF model. One
    request at a time (lock); respawns if it dies. PCM bytes -> temp wav -> path
    line -> JSON line. Lets live use Qwen3-ASR .cpp without per-utterance reload."""

    def __init__(self):
        import threading
        self._proc = None
        self._lock = threading.Lock()

    def _ensure(self):
        import os
        import subprocess
        if self._proc is not None and self._proc.poll() is None:
            return
        here = os.path.dirname(os.path.abspath(__file__))
        py = os.path.join(here, ".venv-qwen314/bin/python")
        daemon = os.path.join(here, "qwen3_cpp_daemon.py")
        self._proc = subprocess.Popen(
            [py, daemon], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1)
        for line in self._proc.stdout:  # block until model loaded
            if line.strip() == "QWEN3READY":
                break

    def transcribe(self, pcm_bytes, language=""):
        import json
        import os
        import recorder  # noqa: PLC0415
        if len(pcm_bytes) < 3200:  # <0.1s -> nothing useful
            return []
        with self._lock:
            self._ensure()
            tmp = f"/tmp/_qwen3live_{os.getpid()}.wav"
            with open(tmp, "wb") as f:
                f.write(recorder.pcm_to_wav(pcm_bytes, sample_rate=16000, channels=1))
            try:
                self._proc.stdin.write(f"{language or ''}\t{tmp}\n")  # lang<TAB>path
                self._proc.stdin.flush()
                text = ""
                import select  # noqa: PLC0415
                while True:  # watchdog: never block the live thread forever on a hung daemon
                    rl, _, _ = select.select([self._proc.stdout], [], [], 30)
                    if not rl:                       # 30s with no output -> assume hung
                        self._proc.kill()
                        self._proc = None            # respawn on next call
                        return []
                    line = self._proc.stdout.readline()
                    if not line:                     # EOF (daemon exited)
                        break
                    if line.startswith("QWEN3JSON:"):
                        text = json.loads(line[len("QWEN3JSON:"):]).get("text", "")
                        break
            except Exception:
                self._proc = None  # force respawn next time
                return []
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)
        text = text.strip()
        return [{"start": 0.0, "end": 0.0, "text": text}] if text else []


_qwen3_daemon = None


def qwen3_cpp_live_backend(language=None):
    """Live-final backend over the persistent .cpp daemon (module singleton)."""
    global _qwen3_daemon
    if _qwen3_daemon is None:
        _qwen3_daemon = _Qwen3CppDaemon()
    return lambda pcm: _qwen3_daemon.transcribe(pcm, language or "")


def qwen3_live_backend(model="Qwen/Qwen3-ASR-0.6B", language=None):
    """Per-utterance Qwen3-ASR for live finals (experimental). Slower than
    whisper-MLX + ~63s cold load; best zh accuracy. Interim must stay whisper.
    Lazy-loads on first call so hot-swap (set_model) doesn't block the request."""
    import sys  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    state = {}
    lang = language or None

    def _run(window_bytes):
        if len(window_bytes) < 2:
            return []
        if len(window_bytes) % 2:
            window_bytes = window_bytes[:-1]
        audio = np.frombuffer(window_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        try:
            if "m" not in state:
                from qwen_asr import Qwen3ASRModel  # noqa: PLC0415
                state["m"] = Qwen3ASRModel.from_pretrained(model)
            out = state["m"].transcribe((audio, 16000), language=lang)
        except Exception as e:
            print(f"qwen3 live error (skipped): {e}", file=sys.stderr)
            return []
        text = " ".join(t.text for t in out).strip()
        return [{"start": 0.0, "end": 0.0, "text": text}] if text else []

    return _run


def denoise_file(src, *, raw_pcm=False):
    """Background-noise removal via `speech denoise` (DeepFilterNet3, CoreML/ANE).
    Returns a cleaned temp path, or `src` unchanged on any failure (graceful — a
    denoise hiccup must never block transcription). `raw_pcm=True`: `src` is 16k
    mono s16le PCM (wrapped to wav first), output is the same raw PCM. Else `src`
    is a normal audio file, output a 16k mono wav. DeepFilterNet3 only helps noisy
    input — on clean speech it adds errors, so this is an off-by-default toggle.
    ponytail: shells the CLI (reloads model per call); fine for a one-shot batch step."""
    import os  # noqa: PLC0415
    import shutil  # noqa: PLC0415
    import subprocess  # noqa: PLC0415
    import sys  # noqa: PLC0415
    import tempfile  # noqa: PLC0415
    speech = shutil.which("speech") or "/opt/homebrew/bin/speech"
    if not os.path.exists(speech):
        return src
    try:
        td = tempfile.mkdtemp(prefix="dn_")
        inp = src
        if raw_pcm:
            inp = os.path.join(td, "in.wav")
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "s16le",
                            "-ar", "16000", "-ac", "1", "-i", src, inp], check=True)
        clean = os.path.join(td, "clean.wav")
        env = {**os.environ, "SPEECH_COREML_COMPUTE_UNITS": "ane",
               "PATH": os.environ.get("PATH", "") + ":/opt/homebrew/bin"}
        subprocess.run([speech, "denoise", inp, "-o", clean],
                       env=env, check=True, capture_output=True)
        out = os.path.join(td, "out.pcm" if raw_pcm else "out.wav")
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", clean, "-ar", "16000", "-ac", "1"]
        cmd += (["-f", "s16le", out] if raw_pcm else [out])
        subprocess.run(cmd, check=True)
        return out
    except Exception as e:  # noqa: BLE001
        print(f"denoise failed, using original: {e}", file=sys.stderr)
        return src


def ane_speech_backend(engine="qwen3-coreml-full", model="0.6B", language=None):
    """ANE (Neural Engine) batch ASR via the `speech` CLI (homebrew, Qwen3-ASR
    CoreML). Power-efficient — runs off the GPU. The CoreML encoder is fixed at
    30s, and the CLI reloads the model per `transcribe` call, so we window into
    ~29s wavs in a temp dir and run `transcribe-batch` ONCE (model loaded once).
    callable(audio) -> [{start,end,text}]. Output simplified zh -> converted to TW."""
    import json  # noqa: PLC0415
    import os  # noqa: PLC0415
    import shutil  # noqa: PLC0415
    import subprocess  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415
    bin_ = shutil.which("speech") or "/opt/homebrew/bin/speech"

    def _decode(audio):
        if isinstance(audio, (bytes, bytearray)):
            return np.frombuffer(bytes(audio), dtype=np.int16)
        if isinstance(audio, str) and audio.endswith(".pcm"):
            with open(audio, "rb") as _f:
                return np.frombuffer(_f.read(), dtype=np.int16)
        raw = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(audio), "-ar", "16000", "-ac", "1",
             "-f", "s16le", "-"], stdout=subprocess.PIPE, check=True).stdout
        return np.frombuffer(raw, dtype=np.int16)

    def _run(audio):
        import recorder  # noqa: PLC0415
        import zhtw  # noqa: PLC0415
        pcm = _decode(audio)
        sr, win = 16000, 16000 * 29  # encoder caps at 30s -> 29s windows
        d = tempfile.mkdtemp(prefix="ane_asr_")
        idx = {}
        try:
            for i in range((len(pcm) + win - 1) // win or 1):
                seg = pcm[i * win:(i + 1) * win]
                if len(seg) < sr // 5:
                    continue
                name = f"{i:04d}"
                with open(os.path.join(d, name + ".wav"), "wb") as f:
                    f.write(recorder.pcm_to_wav(seg.tobytes(), sample_rate=sr, channels=1))
                idx[name] = i
            if not idx:
                return []
            cmd = [bin_, "transcribe-batch", d, "--engine", engine, "--model", model,
                   "--jsonl"]
            if language:
                cmd += ["--language", language]
            # Force CoreML onto the Neural Engine (encoder defaults to .all = GPU,
            # which is why 'ANE' still spun the GPU). ane = .cpuAndNeuralEngine.
            env = {**os.environ, "SPEECH_COREML_COMPUTE_UNITS": "ane"}
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=3600,
                                env=env).stdout
            segs = []
            for line in out.splitlines():
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    j = json.loads(line)
                except Exception:
                    continue
                i = idx.get(str(j.get("file")))
                t = (j.get("text") or "").strip()
                if i is None or not t:
                    continue
                segs.append((i, zhtw.to_tw(t)))
            segs.sort()
            return [{"start": i * win / sr, "end": min((i + 1) * win, len(pcm)) / sr,
                     "text": t} for i, t in segs]
        finally:
            shutil.rmtree(d, ignore_errors=True)
    return _run


# Synthetic model ids -> (engine, model) for the ANE `speech` CLI path.
_ANE_IDS = {
    "ane-qwen3-0.6b": ("qwen3-coreml-full", "0.6B"),     # pure ANE (coolest)
    "ane-qwen3-0.6b-hybrid": ("qwen3-coreml", "0.6B"),   # ANE encoder + GPU decoder (fastest)
}


_ANE_HELP = {"proc": None, "lock": None}


def ane_helper_bin():
    """Path to the prebuilt Swift ANE live helper, or None. Env QWEN3_ANE_BIN
    overrides (the .app bundles it elsewhere); else the repo build output."""
    import os  # noqa: PLC0415
    p = os.environ.get("QWEN3_ANE_BIN")
    if p and os.path.exists(p):
        return p
    here = os.path.dirname(os.path.abspath(__file__))
    cand = os.path.join(here, "swift", "qwen3-ane", ".build", "release", "qwen3-ane")
    return cand if os.path.exists(cand) else None


def floatpanel_bin():
    """Path to the prebuilt floating control-panel app, or None. Env FLOATPANEL_BIN
    overrides; else the repo build output."""
    import os  # noqa: PLC0415
    p = os.environ.get("FLOATPANEL_BIN")
    if p and os.path.exists(p):
        return p
    here = os.path.dirname(os.path.abspath(__file__))
    cand = os.path.join(here, "swift", "floatpanel", ".build", "release", "floatpanel")
    return cand if os.path.exists(cand) else None


def ane_live_backend():
    """Persistent ANE ASR as a live backend. Spawns the Swift helper ONCE (Qwen3-ASR
    CoreML on the Neural Engine — model loaded once, then warm/fast), and serves each
    live window over its stdin/stdout protocol: [4-byte BE len][int16-LE 16k PCM] ->
    JSON {"text":...}. One process, lock-serialized (live runs in a threadpool, dual
    tracks call concurrently). callable(pcm_bytes) -> [{start,end,text}]."""
    import json  # noqa: PLC0415
    import os  # noqa: PLC0415
    import struct  # noqa: PLC0415
    import subprocess  # noqa: PLC0415
    import threading  # noqa: PLC0415
    if _ANE_HELP["lock"] is None:
        _ANE_HELP["lock"] = threading.Lock()

    def _ensure():
        p = _ANE_HELP["proc"]
        if p is not None and p.poll() is None:
            return p
        b = ane_helper_bin()
        if b is None:
            raise RuntimeError("qwen3-ane helper not built (build_qwen3_ane.sh)")
        # Force CoreML onto the ANE (encoder defaults to .all = GPU). Read at load.
        env = {**os.environ, "SPEECH_COREML_COMPUTE_UNITS": "ane"}
        p = subprocess.Popen([b], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, bufsize=0, env=env)
        for line in p.stderr:  # block until model loaded (first time ~tens of seconds)
            if b"READY" in line:
                break
            if b"failed" in line.lower():
                p.kill()
                raise RuntimeError("qwen3-ane load failed: " + line.decode("utf8", "ignore"))
        threading.Thread(target=lambda: [None for _ in p.stderr], daemon=True).start()  # drain
        _ANE_HELP["proc"] = p
        return p

    import select  # noqa: PLC0415
    WIN = 29 * 16000 * 2  # 29s @ 16k mono int16 — CoreML encoder shape is fixed at 30s

    def _send(p, chunk):
        """Send one ≤29s window, watchdog the reply (30s), return text or None.
        None means the helper hung/errored — caller kills + respawns."""
        try:
            p.stdin.write(struct.pack(">I", len(chunk)) + chunk)
            p.stdin.flush()
            rl, _, _ = select.select([p.stdout], [], [], 30)
            if not rl:
                return None
            line = p.stdout.readline()
        except Exception:
            return None
        try:
            return (json.loads(line.decode("utf8", "ignore")).get("text") or "").strip()
        except Exception:
            return ""

    def _run(audio):
        if isinstance(audio, str):
            if audio.endswith(".pcm"):
                with open(audio, "rb") as _f:
                    audio = _f.read()
            else:
                audio = b""
        pcm = bytes(audio)
        if len(pcm) < 640:  # <0.02s -> skip
            return []
        # Slice long input (a long live utterance / the stop-flush of accumulated
        # system audio) into ≤29s windows so we never exceed the CoreML encoder's
        # 3000-mel-frame (30s) fixed shape; transcribe each + join. General + safe
        # for any length (mirrors the batch ANE path).
        texts = []
        with _ANE_HELP["lock"]:
            p = _ensure()
            for i in range(0, len(pcm), WIN):
                ch = pcm[i:i + WIN]
                if len(ch) < 640:
                    continue
                t = _send(p, ch)
                if t is None:  # hung/errored -> kill + respawn next call, stop here
                    try:
                        p.kill()
                    except Exception:
                        pass
                    _ANE_HELP["proc"] = None
                    break
                if t:
                    texts.append(t)
        text = " ".join(texts).strip()
        return [{"start": 0.0, "end": 0.0, "text": text}] if text else []
    return _run


_ANE_WARMED = {"v": False}


def ane_warm():
    """Pre-load the ANE helper in the background so the FIRST live utterance isn't
    blocked on the ~13s CoreML compile/load. Idempotent + best-effort — safe to call
    on live-page load, model selection, and at boot."""
    import threading  # noqa: PLC0415
    if _ANE_WARMED["v"] or ane_helper_bin() is None:
        return
    _ANE_WARMED["v"] = True

    def _w():
        try:
            ane_live_backend()(b"\x00" * 32000)  # 1s silence -> spawns helper + loads model once
        except Exception:  # noqa: BLE001
            _ANE_WARMED["v"] = False  # let a later trigger retry
    threading.Thread(target=_w, daemon=True).start()


def make_batch_backend(model, language=None):
    """Batch/accurate: route to Qwen3-ASR .cpp (fast, Metal, word-aligned) or
    transformers Qwen3-ASR or whisper-MLX. callable(audio_path) -> segments.
    language=None -> auto-detect; a code forces it."""
    if route(model) == "ane":  # ANE speech CLI (don't honor_language-reroute it)
        engine, m = _ANE_IDS.get(model, ("qwen3-coreml-full", "0.6B"))
        return ane_speech_backend(engine, m, language)
    model = _honor_language(model, language)  # femelo ignores language -> whisper
    r = route(model)
    if r == "chatllm":
        return chatllm_batch_backend(model, language)
    if r == "qwen3mlx":
        return qwen3_mlx_backend(model, language)
    if r == "qwen3cpp":
        return qwen3_cpp_batch_backend(model, language)
    if r == "qwen3":
        return qwen3_batch_backend(model, language)
    if r == "firered":
        return firered_batch_backend(model, language)
    import asr
    return asr.mlx_whisper_backend(model, language)


import os as _os

_MODELS = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "models")
# FireRedASR-AED-L (int8) from the sherpa-onnx model zoo (public, no auth). SOTA zh
# CER but CPU-only — re-recognition (offline batch) only, never live.
_FIRERED_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/"
                "asr-models/sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16.tar.bz2")
_FIRERED_DIR = _os.path.join(_MODELS, "sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16")
_FIRERED = {}  # lazy recognizer cache


def _ensure_firered(progress=None):
    """(encoder, decoder, tokens) paths — download+extract the ~1GB model on first
    use so a fresh machine needs no manual setup. Globs the extracted dir so a
    renamed onnx (int8 vs fp32) still resolves."""
    import glob
    import tarfile
    import urllib.request

    def _pick(paths):
        # prefer the int8 quant (smaller/faster) when both are present
        return next((p for p in paths if "int8" in p), paths[0]) if paths else None

    enc = _pick(glob.glob(_os.path.join(_FIRERED_DIR, "encoder*.onnx")))
    dec = _pick(glob.glob(_os.path.join(_FIRERED_DIR, "decoder*.onnx")))
    tok = _os.path.join(_FIRERED_DIR, "tokens.txt")
    if enc and dec and _os.path.exists(tok):
        return enc, dec, tok
    _os.makedirs(_MODELS, exist_ok=True)
    tar = _os.path.join(_MODELS, "firered.tar.bz2")
    if not _os.path.exists(tar):
        if progress:
            progress("下載 FireRedASR 模型（約 1GB，首次較久）…")
        tmp = tar + ".part"
        urllib.request.urlretrieve(_FIRERED_URL, tmp)
        _os.replace(tmp, tar)
    if progress:
        progress("解壓 FireRedASR…")
    with tarfile.open(tar) as tf:
        tf.extractall(_MODELS)
    enc = _pick(glob.glob(_os.path.join(_FIRERED_DIR, "encoder*.onnx")))
    dec = _pick(glob.glob(_os.path.join(_FIRERED_DIR, "decoder*.onnx")))
    if not (enc and dec and _os.path.exists(tok)):
        raise RuntimeError("FireRedASR model files missing after extract")
    return enc, dec, tok


def _firered_recognizer(progress=None):
    if "rec" not in _FIRERED:
        import sherpa_onnx  # noqa: PLC0415
        enc, dec, tok = _ensure_firered(progress)
        _FIRERED["rec"] = sherpa_onnx.OfflineRecognizer.from_fire_red_asr(
            encoder=enc, decoder=dec, tokens=tok,
            # never oversubscribe: 1-core boxes get 1 thread, not 2 fighting for it
            num_threads=max(1, (_os.cpu_count() or 4) - 2))
    return _FIRERED["rec"]


def firered_batch_backend(model="firered", language=None):
    """Batch FireRedASR-AED-L via sherpa-onnx (CPU). Highest zh accuracy of our
    options; no ANE/GPU so it's offline re-recognition only. Lazy: the ~1GB model
    downloads + loads on the first call (inside the transcribe job thread), so the
    endpoint stays fast. Windowed 30s chunks come from iter_transcribe; returns one
    untimed blob per window (asr.transcribe spreads it over clip_ms)."""
    def _run(audio_path):
        import numpy as np  # noqa: PLC0415
        rec = _firered_recognizer()
        with open(str(audio_path), "rb") as f:  # window temp is raw s16le pcm
            samples = np.frombuffer(f.read(), dtype=np.int16).astype(np.float32) / 32768.0
        if samples.size == 0:
            return []
        st = rec.create_stream()
        st.accept_waveform(16000, samples)
        rec.decode_stream(st)
        text = st.result.text.strip()
        return [{"start": 0.0, "end": 0.0, "text": text}] if text else []

    return _run


def qwen3_cpp_batch_backend(model="qwen3-asr-0.6b-q4-k-m", language=None):
    """Qwen3-ASR via the .cpp/GGUF sidecar (Metal, fast, word-level alignment).
    Runs in .venv-qwen314 as a subprocess (cp314 native module); the app is 3.10.
    Raw .pcm is wrapped to a temp wav (the .cpp loader needs a container)."""
    import json  # noqa: PLC0415
    import os  # noqa: PLC0415
    import subprocess  # noqa: PLC0415
    import sys  # noqa: PLC0415

    here = os.path.dirname(os.path.abspath(__file__))
    py = os.path.join(here, ".venv-qwen314/bin/python")
    cli = os.path.join(here, "qwen3_cpp_cli.py")

    def _run(audio_path):
        audio_path = str(audio_path)
        tmp = None
        if audio_path.endswith(".pcm"):
            import recorder  # noqa: PLC0415
            with open(audio_path, "rb") as f:
                wav = recorder.pcm_to_wav(f.read(), sample_rate=16000, channels=1)
            tmp = audio_path + ".qwav.wav"
            with open(tmp, "wb") as f:
                f.write(wav)
            audio_path = tmp
        try:
            p = subprocess.run([py, cli, audio_path, language or ""],
                               capture_output=True, text=True, timeout=1800)
            line = next((l for l in p.stdout.splitlines()
                         if l.startswith("QWEN3JSON:")), None)
            if line is None:
                print(f"qwen3cpp no output: {p.stderr[-300:]}", file=sys.stderr)
                return []
            d = json.loads(line[len("QWEN3JSON:"):])
            # .cpp zh word-alignment is coarse/unreliable (often one tiny-span
            # "word" for a whole sentence), so don't trust its times — return the
            # full text as one segment; iter_transcribe's per-window offset supplies
            # the timeline position. (qwen3_words_to_segments kept for future use.)
            text = d.get("text", "").strip()
            return [{"start": 0.0, "end": 0.0, "text": text}] if text else []
        except Exception as e:
            print(f"qwen3cpp error: {e}", file=sys.stderr)
            return []
        finally:
            if tmp and os.path.exists(tmp):
                os.remove(tmp)

    return _run


def qwen3_batch_backend(model="Qwen/Qwen3-ASR-0.6B", language=None):
    """Offline Qwen3-ASR via the qwen-asr transformers backend. Apple Silicon:
    torch on MPS/CPU (slow, not realtime) — for post-meeting accuracy only.
    Lazy import so the heavy torch/transformers stack isn't required otherwise."""
    state = {}
    lang = language or None

    def _run(audio_path):
        if "m" not in state:  # lazy: don't block startup / first request beyond load
            from qwen_asr import Qwen3ASRModel  # noqa: PLC0415
            state["m"] = Qwen3ASRModel.from_pretrained(model)
        out = state["m"].transcribe(str(audio_path), language=lang)
        text = " ".join(t.text for t in out).strip()
        return [{"start": 0.0, "end": 0.0, "text": text}] if text else []

    return _run


class LiveModelManager:
    """Holds the current live AdaptiveBackend and rebuilds it on set_model()."""

    def __init__(self, make, model, fallback=(), rtf_budget=0.8, on_change=None):
        self._make = make
        self._fallback = list(fallback)
        self._rtf_budget = rtf_budget
        self._on_change = on_change
        self.backend = None
        self.requested = None
        self.language = None    # None -> auto-detect; a code ("zh"/"en"/…) forces it
        self.set_model(model)

    def set_model(self, model):
        chain = [model] + [m for m in self._fallback if m != model]
        self.backend = AdaptiveBackend(
            [self._make(m, self.language) for m in chain], chain,
            rtf_budget=self._rtf_budget, on_change=self._on_change)
        self.requested = model
        return model

    def set_language(self, language):
        # Rebuild the chain so the new language is baked in (models stay cached).
        if (language or None) != self.language:
            self.language = language or None
            self.set_model(self.requested)
        return self.language

    @property
    def current(self):
        return self.backend.current_model

    def __call__(self, pcm_bytes):  # shim so a captured ref always hits the live one
        return self.backend(pcm_bytes)
