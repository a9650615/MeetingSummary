"""Web app: FastAPI. Binds 127.0.0.1 only (privacy, spec G2) — no auth by
design, loopback-only. Backends injected so tests run without MLX.

Phase 1 = batch loop: create meeting -> transcribe saved PCM -> summarize -> view.
Live websocket + recording start/stop (Swift helper) land in Phase 2."""
import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import asr
from summarize import summarize

_INDEX = """<!doctype html><meta charset=utf-8><title>MeetingSummary</title>
<h1>MeetingSummary</h1>
<ul id=meetings></ul>
<script>
fetch('/meetings').then(r=>r.json()).then(ms=>{
  document.getElementById('meetings').innerHTML =
    ms.map(m=>`<li><a href="/meetings/${m.id}">${m.title}</a> (${m.status})</li>`).join('');
});
</script>"""


class MeetingIn(BaseModel):
    title: str
    lang: str = "zh-TW"


class SummaryIn(BaseModel):
    kind: str = "minutes"


def _rows(rows):
    return [dict(r) for r in rows]


def _transcript_text(rows):
    return "\n".join(f"{r['speaker']}: {r['text']}" for r in rows)


def create_app(store, *, summary_backend, asr_backend=None,
               summary_model="mlx-lm"):
    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    def index():
        return _INDEX

    @app.post("/meetings")
    def create_meeting(m: MeetingIn):
        return {"id": store.create_meeting(m.title, time.time(), m.lang)}

    @app.get("/meetings")
    def list_meetings():
        return _rows(store.list_meetings())

    @app.get("/meetings/{mid}")
    def get_meeting(mid: int):
        meeting = store.get_meeting(mid)
        if meeting is None:
            raise HTTPException(404, "meeting not found")
        return {
            "meeting": dict(meeting),
            "segments": _rows(store.list_segments(mid)),
            "transcripts": _rows(store.list_transcripts(mid)),
            "summaries": _rows(store.list_summaries(mid)),
        }

    @app.post("/meetings/{mid}/transcribe")
    def transcribe_meeting(mid: int):
        if store.get_meeting(mid) is None:
            raise HTTPException(404, "meeting not found")
        if asr_backend is None:
            raise HTTPException(503, "no ASR backend configured")
        n = 0
        for seg in store.list_segments(mid):
            for track_label, name in (("system", "system.pcm"), ("mic", "mic.pcm")):
                pcm = f"{seg['dir_path']}/{name}"
                for t in asr.transcribe(pcm, profile="accurate",
                                        track=track_label, backend=asr_backend):
                    store.add_transcript(mid, t["profile"], t["track"],
                                         t["start_ms"], t["end_ms"],
                                         t["track"], t["text"])
                    n += 1
        return {"transcripts": n}

    @app.post("/meetings/{mid}/summary")
    def summarize_meeting(mid: int, body: SummaryIn):
        meeting = store.get_meeting(mid)
        if meeting is None:
            raise HTTPException(404, "meeting not found")
        text = _transcript_text(store.list_transcripts(mid))
        out = summarize(text, kind=body.kind, lang=meeting["lang"],
                        backend=summary_backend)
        store.add_summary(mid, body.kind, meeting["lang"], out,
                          summary_model, time.time())
        return {"text": out, "kind": body.kind}

    return app


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    from store import Store
    app = create_app(
        Store("data/meetings.db"),
        summary_backend=__import__("summarize").mlx_lm_backend(),
        asr_backend=asr.mlx_whisper_backend(),
    )
    uvicorn.run(app, host="127.0.0.1", port=8000)  # loopback only (G2)
