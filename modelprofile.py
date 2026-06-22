"""Background hardware/language profile -> auto-pick models. Zero config for the
user: detect RAM + chip + preferred language, recommend live/interim/accurate/
summary models that fit and stay realtime. The runtime AdaptiveBackend + adaptive
VAD handle the rest, so a wrong guess self-corrects rather than breaking.

Named modelprofile (not 'profile') to avoid shadowing the stdlib profiler."""
import json
import os
import platform
import subprocess

_TURBO = "mlx-community/whisper-large-v3-turbo"
_LARGE = "mlx-community/whisper-large-v3-mlx"
_SMALL = "mlx-community/whisper-small-mlx"
_BASE = "mlx-community/whisper-base-mlx"
_TINY = "mlx-community/whisper-tiny-mlx"
# 4-bit quantized = ~half the RAM, ~same model. Lighter live/interim/fallback.
_TURBO_Q4 = "mlx-community/whisper-large-v3-turbo-q4"
_SMALL_Q4 = "mlx-community/whisper-small-mlx-q4"
_BASE_Q4 = "mlx-community/whisper-base-mlx-q4"
_TINY_Q4 = "mlx-community/whisper-tiny-mlx-q4"
# BELLE Chinese-finetuned whisper: better Mandarin CER, BUT the mlx-community
# 8-bit ports are incompatible with mlx-whisper 0.4.3 (ModelDimensions rejects
# 'activation_dropout'). NOT auto-selected — opt in via LIVE_MODEL/ASR_MODEL once
# the runtime supports it. Startup probe skips it if still broken.
_BELLE_TURBO = "mlx-community/belle-whisper-large-v3-turbo-zh-8bit"
_BELLE_LARGE = "mlx-community/belle-whisper-large-v3-zh-8bit"


def _total_ram_bytes():
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return 16 * 10**9  # safe default


def _chip():
    try:
        return subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=1).stdout.strip() \
            or platform.processor()
    except Exception:
        return platform.processor() or platform.machine()


def detect_hardware():
    return {
        "arch": platform.machine(),
        "chip": _chip(),
        "ram_gb": round(_total_ram_bytes() / 1e9),
        "cores": os.cpu_count() or 1,
    }


def probe_models(candidates, *, audio_seconds, run, clock, target_rtf=0.5):
    """Empirical GPU-perf pick: time each model (best-first) on a probe clip and
    return the first whose real-time factor (process_time / audio_seconds) clears
    the budget. RTF is the truest measure of this machine's Metal throughput —
    better than guessing from chip name. Falls back to the smallest if all lag."""
    for model in candidates:
        t0 = clock()
        try:
            run(model)
        except Exception:
            continue  # broken/incompatible model (e.g. belle on this runtime)
        if (clock() - t0) / audio_seconds <= target_rtf:
            return model
    return candidates[-1]


def load_chosen(path):
    """Remembered best live model from a prior run (self-tuning across launches)."""
    try:
        with open(path) as f:
            return json.load(f).get("live")
    except (OSError, ValueError):
        return None


def save_chosen(path, model):
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump({"live": model}, f)
    except OSError:
        pass


def recommend(hw, lang="zh-TW"):
    """Pure: hardware + language -> model choices. zh-dominant prefers the BELLE
    Chinese-finetuned whisper (lower CER on Mandarin); the vanilla fallback chain
    covers English/code-switch if BELLE struggles."""
    ram = hw.get("ram_gb", 16)
    if ram >= 16:
        # q4 live/interim + 3B summary: several whisper tiers + the LLM can be
        # resident at once, so favor the lighter quantized variants to avoid OOM.
        return {
            "live": _TURBO_Q4,
            "interim": _SMALL_Q4,
            "accurate": _TURBO_Q4,
            "summary": "mlx-community/Qwen2.5-3B-Instruct-4bit",
            "fallback": [_SMALL_Q4, _BASE_Q4],
        }
    if ram >= 8:
        return {
            "live": _SMALL_Q4,
            "interim": _BASE_Q4,
            "accurate": _TURBO_Q4,
            "summary": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
            "fallback": [_BASE_Q4, _TINY_Q4],
        }
    return {
        "live": _BASE_Q4,
        "interim": _TINY_Q4,
        "accurate": _SMALL_Q4,
        "summary": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        "fallback": [_TINY_Q4],
    }
