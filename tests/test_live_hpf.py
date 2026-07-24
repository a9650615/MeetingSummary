"""HighPass (live ASR-input rumble removal): must kill sub-80Hz rumble while
leaving speech-band energy essentially intact, and filter continuously across
chunk boundaries (stateful zi) so live windows don't click at each seam."""
import numpy as np

from live import HighPass


def _tone(hz, secs=1.0, sr=16000, amp=10000):
    t = np.arange(int(secs * sr)) / sr
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.int16).tobytes()


def _rms(pcm):
    a = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)
    return float(np.sqrt(np.mean(a * a)))


def test_attenuates_low_freq_passes_speech():
    low_in = _tone(40)      # rumble (fan/AC)
    mid_in = _tone(300)     # speech band
    low_out = _rms(HighPass()(low_in))
    mid_out = _rms(HighPass()(mid_in))
    low_ratio = low_out / _rms(low_in)
    mid_ratio = mid_out / _rms(mid_in)
    assert low_ratio < 0.35, f"40Hz not attenuated enough: {low_ratio:.2f}"
    assert mid_ratio > 0.9, f"300Hz speech band damaged: {mid_ratio:.2f}"


def test_stateful_across_chunks():
    # Filtering [A|B] in two calls must equal filtering AB in one (continuous zi),
    # otherwise every live window seam gets a filter transient.
    sig = _tone(300, secs=0.5)
    half = len(sig) // 2
    half -= half % 2  # int16 sample boundary
    whole = HighPass()(sig)
    f = HighPass()
    piecewise = f(sig[:half]) + f(sig[half:])
    assert whole == piecewise


def test_empty_passthrough():
    assert HighPass()(b"") == b""
