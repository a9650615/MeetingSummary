"""Simplified -> Traditional (Taiwan) conversion for transcripts.

Whisper emits inconsistent 簡/繁 for Mandarin; normalize to zh-TW with OpenCC
's2twp' (also localizes vocab: 软件->軟體, 优化->最佳化). Module-level singleton,
lazy-loaded; no-op if opencc is missing so the app still runs.
ponytail: global converter, fine — it's stateless and read-only."""

_enabled = True
_cc = None  # None=unloaded, False=unavailable, else an OpenCC instance


def configure(enabled):
    global _enabled, _cc
    _enabled = enabled
    _cc = None  # force reload on next use


def to_tw(text):
    global _cc
    if not _enabled or not text:
        return text
    if _cc is None:
        try:
            from opencc import OpenCC
            _cc = OpenCC("s2twp")
        except Exception:
            _cc = False
    return _cc.convert(text) if _cc else text
