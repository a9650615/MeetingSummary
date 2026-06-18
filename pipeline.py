"""Batch pipeline: a saved audio file -> transcript -> summary (spec §7 proof).

Run on the M3 with real MLX backends:
    python -m pipeline meeting.m4a --title "季度會議" --kind minutes

mlx-whisper loads m4a/wav/mp3 directly, so this needs no Swift capture — it's
the end-to-end proof of transcribe + summary on a saved recording."""
import argparse
import time

import asr
from summarize import summarize


def run_pipeline(audio_path, *, store, title, lang, kind,
                 asr_backend, summary_backend, track="mic",
                 summary_model="mlx-lm"):
    mid = store.create_meeting(title, time.time(), lang)
    segs = asr.transcribe(audio_path, profile="accurate", track=track,
                          backend=asr_backend)
    for s in segs:
        store.add_transcript(mid, s["profile"], s["track"], s["start_ms"],
                             s["end_ms"], s["track"], s["text"])
    text = "\n".join(f"{s['track']}: {s['text']}" for s in segs)
    summary = summarize(text, kind=kind, lang=lang, backend=summary_backend)
    store.add_summary(mid, kind, lang, summary, summary_model, time.time())
    store.finalize_meeting(mid)
    return {"meeting_id": mid, "summary": summary, "transcripts": len(segs)}


def main():  # pragma: no cover - real-MLX entrypoint, run on the M3
    p = argparse.ArgumentParser()
    p.add_argument("audio")
    p.add_argument("--title", default="meeting")
    p.add_argument("--lang", default="zh-TW")
    p.add_argument("--kind", default="minutes", choices=["minutes", "bullets"])
    p.add_argument("--db", default="data/meetings.db")
    args = p.parse_args()

    from summarize import mlx_lm_backend
    from store import Store

    result = run_pipeline(
        args.audio, store=Store(args.db), title=args.title, lang=args.lang,
        kind=args.kind, asr_backend=asr.mlx_whisper_backend(),
        summary_backend=mlx_lm_backend(),
    )
    print(f"\n=== meeting {result['meeting_id']} "
          f"({result['transcripts']} segments) ===\n")
    print(result["summary"])


if __name__ == "__main__":
    main()
