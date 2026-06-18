"""Web app: FastAPI. Binds 127.0.0.1 only (privacy, spec G2) — no auth by
design, loopback-only. Backends injected so tests run without MLX.

Phase 1 = batch loop: create meeting -> transcribe saved PCM -> summarize -> view.
Live websocket + recording start/stop (Swift helper) land in Phase 2."""
import html
import os
import time

from fastapi import (FastAPI, File, Form, HTTPException, UploadFile, WebSocket,
                     WebSocketDisconnect)
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

import asr
from live import LiveSession, VadChunker
from summarize import summarize

_INDEX = """<!doctype html><meta charset=utf-8><title>MeetingSummary</title>
<h1>MeetingSummary</h1>
<form action="/ingest" method="post" enctype="multipart/form-data">
  <p>音檔 (wav/m4a/mp3): <input type="file" name="audio" required></p>
  <p>標題: <input name="title" value="實測"></p>
  <p>摘要型式:
    <select name="kind">
      <option value="minutes">會議記錄 minutes</option>
      <option value="bullets">條列 bullets</option>
    </select></p>
  <button type="submit">上傳並產生摘要</button>
</form>
<p>上傳後會跑 transcribe + summary，第一次會下載模型，請稍候。</p>
<p><a href="/live"><b>🔴 Live mode（即時逐字稿）</b></a></p>
<h2>會議</h2>
<ul id=meetings></ul>
<script>
fetch('/meetings').then(r=>r.json()).then(ms=>{
  document.getElementById('meetings').innerHTML =
    ms.map(m=>`<li><a href="/meetings/${m.id}">${m.title}</a> (${m.status})</li>`).join('');
});
</script>"""


def _result_page(title, summary, transcripts):
    lines = "".join(
        f"<tr><td>{r['track']}</td><td>{html.escape(r['text'])}</td></tr>"
        for r in transcripts
    )
    return f"""<!doctype html><meta charset=utf-8><title>{html.escape(title)}</title>
<p><a href="/">&larr; 回首頁</a></p>
<h1>{html.escape(title)}</h1>
<h2>摘要</h2><pre style="white-space:pre-wrap">{html.escape(summary)}</pre>
<h2>逐字稿</h2><table border=1 cellpadding=4>{lines}</table>"""


# Live page: browser mic -> 16 kHz Int16 PCM over websocket -> server ASR.
# ScriptProcessorNode is deprecated but needs no separate worklet file.
# ponytail: swap to AudioWorklet if latency/jank shows up.
_LIVE = """<!doctype html><meta charset=utf-8><title>Live</title>
<p><a href="/">&larr; 回首頁</a></p>
<h1>🔴 Live 逐字稿</h1>
<button id=start>開始</button> <button id=stop disabled>停止</button>
<span id=status></span>
<div id=caption style="margin:0.6em 0;padding:0.5em;min-height:1.6em;
  font-size:2em;font-weight:bold;background:#111;color:#fff;border-radius:6px"></div>
<div id=transcript style="margin-top:1em;font-size:1em;color:#444"></div>
<script>
let ws, ctx, node, gain, src, stream, mid;
const T=document.getElementById('transcript'), S=document.getElementById('status');
const C=document.getElementById('caption');
const startBtn=document.getElementById('start'), stopBtn=document.getElementById('stop');
startBtn.onclick = async () => {
  stream = await navigator.mediaDevices.getUserMedia({audio:true});
  ctx = new AudioContext();
  src = ctx.createMediaStreamSource(stream);
  node = ctx.createScriptProcessor(4096,1,1);
  gain = ctx.createGain(); gain.gain.value = 0;  // mute: no self-echo
  const ratio = ctx.sampleRate/16000;
  ws = new WebSocket(`ws://${location.host}/ws/live`);
  ws.binaryType='arraybuffer';
  ws.onmessage = e => {
    const m = JSON.parse(e.data);
    if(m.type==='meeting'){ mid=m.id; S.textContent=' 會議 #'+mid+' 錄製中…'; }
    else if(m.type==='segment'){
      C.textContent = m.text;  // big live caption
      T.innerHTML += `<p>[${(m.start_ms/1000).toFixed(1)}s] ${m.text}</p>`; }
    else if(m.type==='error'){ S.textContent=' 錯誤: '+m.msg; }
  };
  ws.onopen = () => {
    node.onaudioprocess = ev => {
      if(ws.readyState!==1) return;
      const input = ev.inputBuffer.getChannelData(0);
      const outLen = Math.floor(input.length/ratio);
      const pcm = new Int16Array(outLen);
      for(let i=0;i<outLen;i++){
        const s = Math.max(-1,Math.min(1,input[Math.floor(i*ratio)]));
        pcm[i] = s*32767;
      }
      ws.send(pcm.buffer);
    };
    src.connect(node); node.connect(gain); gain.connect(ctx.destination);
  };
  startBtn.disabled=true; stopBtn.disabled=false;
};
stopBtn.onclick = () => {
  if(node) node.disconnect();
  if(stream) stream.getTracks().forEach(t=>t.stop());
  if(ws) ws.close();
  if(ctx) ctx.close();
  S.textContent += mid ? ` 已停止。可到首頁對 #${mid} 產生摘要。` : ' 已停止。';
  startBtn.disabled=false; stopBtn.disabled=true;
};
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
               live_backend=None, summary_model="mlx-lm",
               live_silence_ms=400, live_max_window_s=8.0, live_rms_threshold=500):
    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    def index():
        return _INDEX

    @app.get("/live", response_class=HTMLResponse)
    def live_page():
        return _LIVE

    @app.websocket("/ws/live")
    async def ws_live(ws: WebSocket):
        await ws.accept()
        if live_backend is None:
            await ws.send_json({"type": "error", "msg": "no live backend"})
            await ws.close()
            return
        mid = store.create_meeting("Live", time.time(), "zh-TW")
        chunker = VadChunker(sample_rate=16000, silence_ms=live_silence_ms,
                             max_window_s=live_max_window_s,
                             rms_threshold=live_rms_threshold)
        sess = LiveSession(backend=live_backend, chunker=chunker, track="mic")
        await ws.send_json({"type": "meeting", "id": mid})

        def _store(seg):
            store.add_transcript(mid, "live", "mic", seg["start_ms"],
                                 seg["end_ms"], "我", seg["text"])
        try:
            while True:
                data = await ws.receive_bytes()
                for seg in await run_in_threadpool(sess.feed, data):
                    _store(seg)
                    await ws.send_json({"type": "segment", **seg})
        except WebSocketDisconnect:
            pass
        finally:
            for seg in await run_in_threadpool(sess.flush):
                _store(seg)
            store.finalize_meeting(mid)

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

    @app.post("/ingest", response_class=HTMLResponse)
    async def ingest(audio: UploadFile = File(...), title: str = Form("實測"),
                     kind: str = Form("minutes"), lang: str = Form("zh-TW")):
        if asr_backend is None:
            raise HTTPException(503, "no ASR backend configured")
        from pipeline import run_pipeline

        os.makedirs("data/uploads", exist_ok=True)
        path = os.path.join("data/uploads", audio.filename)
        with open(path, "wb") as f:
            f.write(await audio.read())
        result = run_pipeline(path, store=store, title=title, lang=lang,
                              kind=kind, asr_backend=asr_backend,
                              summary_backend=summary_backend,
                              summary_model=summary_model)
        mid = result["meeting_id"]
        return _result_page(title, result["summary"],
                            _rows(store.list_transcripts(mid)))

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
    from summarize import mlx_lm_backend

    asr_model = os.environ.get("ASR_MODEL", "mlx-community/whisper-large-v3-turbo")
    llm_model = os.environ.get("LLM_MODEL", "mlx-community/Qwen2.5-3B-Instruct-4bit")
    # Live subtitles: small model + VAD chunking (cut at pauses, not mid-word).
    # Tune: LIVE_MODEL (base/tiny for speed), LIVE_SILENCE_MS (lower = snappier,
    # more fragments), LIVE_MAX_WINDOW_S (latency ceiling), LIVE_RMS (mic level).
    live_model = os.environ.get("LIVE_MODEL", "mlx-community/whisper-small-mlx")
    live_silence = int(os.environ.get("LIVE_SILENCE_MS", "400"))
    live_max_window = float(os.environ.get("LIVE_MAX_WINDOW_S", "8.0"))
    live_rms = int(os.environ.get("LIVE_RMS", "500"))

    # Lazy-load the LLM on first request so the server starts instantly;
    # mlx-whisper already loads per-call. First /ingest downloads both models.
    _llm = {}

    def summary_backend(prompt):
        if "fn" not in _llm:
            _llm["fn"] = mlx_lm_backend(llm_model)
        return _llm["fn"](prompt)

    from live import mlx_whisper_live_backend

    app = create_app(
        Store("data/meetings.db"),
        summary_backend=summary_backend,
        asr_backend=asr.mlx_whisper_backend(asr_model),
        live_backend=mlx_whisper_live_backend(live_model),
        live_silence_ms=live_silence,
        live_max_window_s=live_max_window,
        live_rms_threshold=live_rms,
        summary_model=llm_model,
    )
    uvicorn.run(app, host="127.0.0.1", port=8000)  # loopback only (G2)
