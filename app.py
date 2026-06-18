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
from live import TwoPassSession
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
    ms.map(m=>`<li><a href="/m/${m.id}">${m.title}</a> (${m.status})</li>`).join('');
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
<label>來源:
  <select id=source>
    <option value="mic">麥克風(我)</option>
    <option value="system">系統音(對方)</option>
    <option value="both">兩者(混合)</option>
  </select>
</label>
<button id=start>開始</button> <button id=stop disabled>停止</button>
<span id=status></span>
<p style="color:#888">系統音/兩者:會跳出分享視窗,請選螢幕或分頁並<b>勾選「分享音訊」</b>。兩者建議戴耳機(否則麥克風會收到喇叭回音)。</p>
<div id=caption style="margin:0.6em 0;padding:0.5em;min-height:1.6em;
  font-size:2em;font-weight:bold;background:#111;color:#fff;border-radius:6px"></div>
<div id=transcript style="margin-top:1em;font-size:1em;color:#444"></div>
<script>
let ws, ctx, node, gain, streams=[], mid;
const T=document.getElementById('transcript'), S=document.getElementById('status');
const C=document.getElementById('caption');
const startBtn=document.getElementById('start'), stopBtn=document.getElementById('stop');

async function getStreams(source){
  if(source==='mic') return [await navigator.mediaDevices.getUserMedia({audio:true})];
  if(source==='system'){
    const s = await navigator.mediaDevices.getDisplayMedia({video:true,audio:true});
    s.getVideoTracks().forEach(t=>t.stop());
    if(!s.getAudioTracks().length) throw new Error('未取得系統音(分享時要勾選「分享音訊」)');
    return [s];
  }
  const mic = await navigator.mediaDevices.getUserMedia({audio:true});
  const sys = await navigator.mediaDevices.getDisplayMedia({video:true,audio:true});
  sys.getVideoTracks().forEach(t=>t.stop());
  return [mic, sys];
}

startBtn.onclick = async () => {
  const source = document.getElementById('source').value;
  try { streams = await getStreams(source); }
  catch(e){ S.textContent=' 取得音源失敗: '+e.message; return; }
  ctx = new AudioContext();
  node = ctx.createScriptProcessor(4096,1,1);
  gain = ctx.createGain(); gain.gain.value = 0;  // mute: no self-echo
  streams.forEach(st => ctx.createMediaStreamSource(st).connect(node));  // sum/mix
  const ratio = ctx.sampleRate/16000;
  ws = new WebSocket(`ws://${location.host}/ws/live?src=${source}`);
  ws.binaryType='arraybuffer';
  let tentative = null;  // current in-progress (interim) line, replaced on final
  ws.onmessage = e => {
    const m = JSON.parse(e.data);
    if(m.type==='meeting'){ mid=m.id; S.textContent=' 會議 #'+mid+' 錄製中…'; }
    else if(m.type==='interim'){
      C.textContent = m.text;
      if(!tentative){ tentative=document.createElement('p');
        tentative.style.color='#aaa'; T.insertAdjacentElement('afterbegin',tentative); }
      tentative.textContent = '… '+m.text;
    }
    else if(m.type==='final'){
      C.textContent = m.text;
      const line = `[${(m.start_ms/1000).toFixed(1)}s] ${m.text}`;
      if(tentative){ tentative.style.color=''; tentative.textContent=line; tentative=null; }
      else { T.insertAdjacentHTML('afterbegin', `<p>${line}</p>`); }  // 倒敘
    }
    else if(m.type==='notice'){ S.textContent=' ⚡ '+m.msg; }
    else if(m.type==='error'){ S.textContent=' 錯誤: '+m.msg; }
  };
  ws.onopen = () => {
    let hangover = 0;  // keep sending ~680ms after speech so server sees the pause
    node.onaudioprocess = ev => {
      if(ws.readyState!==1) return;
      const input = ev.inputBuffer.getChannelData(0);
      // client-side silence gate (perf): skip sending when idle
      let sum=0; for(let i=0;i<input.length;i++) sum+=input[i]*input[i];
      const rms = Math.sqrt(sum/input.length);
      if(rms > 0.01) hangover = 8; else if(hangover > 0) hangover--;
      if(rms <= 0.01 && hangover <= 0) return;  // silent -> don't transmit
      const outLen = Math.floor(input.length/ratio);
      const pcm = new Int16Array(outLen);
      for(let i=0;i<outLen;i++){
        const s = Math.max(-1,Math.min(1,input[Math.floor(i*ratio)]));
        pcm[i] = s*32767;
      }
      ws.send(pcm.buffer);
    };
    node.connect(gain); gain.connect(ctx.destination);
  };
  startBtn.disabled=true; stopBtn.disabled=false;
};
stopBtn.onclick = () => {
  if(node) node.disconnect();
  streams.forEach(st => st.getTracks().forEach(t=>t.stop()));
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


def _src_labels(src):
    """Map the live audio source to (track, speaker). mic = you, system = the
    other side, both = a client-mixed single stream."""
    return {
        "mic": ("mic", "我"),
        "system": ("system", "對方"),
        "both": ("mixed", "混合"),
    }.get(src, ("mic", "我"))


def _transcript_text(rows):
    return "\n".join(f"{r['speaker']}: {r['text']}" for r in rows)


def _detail_page(mid, meeting, transcripts, summaries):
    rows = "".join(
        f"<tr><td>{r['track']}</td><td>{html.escape(r['text'])}</td></tr>"
        for r in transcripts)
    sums = "".join(
        f"<h3>{html.escape(s['kind'])}</h3>"
        f"<pre style='white-space:pre-wrap'>{html.escape(s['text'])}</pre>"
        for s in summaries)
    return (
        "<!doctype html><meta charset=utf-8><title>"
        + html.escape(meeting["title"]) + "</title>"
        "<p><a href='/'>&larr; 回首頁</a></p>"
        "<h1>" + html.escape(meeting["title"])
        + " <small>(" + html.escape(meeting["status"]) + ")</small></h1>"
        "<h2>摘要</h2>"
        "<select id=kind><option value=minutes>會議記錄</option>"
        "<option value=bullets>條列</option></select> "
        "<button id=go>產生摘要</button> "
        "<button id=fin>完成會議</button> <span id=finmsg></span>"
        "<p style='color:#888'>※ 摘要只在你按下按鈕時才產生。停止錄音不會自動完成 —"
        " 校正/再跑都做完後再按「完成會議」。</p>"
        "<div id=out>" + sums + "</div>"
        "<h2>逐字稿</h2><table border=1 cellpadding=4>" + rows + "</table>"
        "<script>"
        "document.getElementById('go').onclick=async()=>{"
        "const k=document.getElementById('kind').value;"
        "const o=document.getElementById('out');o.textContent='產生中…';"
        "const r=await fetch('/meetings/" + str(mid) + "/summary',{method:'POST',"
        "headers:{'Content-Type':'application/json'},body:JSON.stringify({kind:k})});"
        "const j=await r.json();const p=document.createElement('pre');"
        "p.style.whiteSpace='pre-wrap';p.textContent=j.text;"
        "o.innerHTML='';o.appendChild(p);};"
        "document.getElementById('fin').onclick=async()=>{"
        "await fetch('/meetings/" + str(mid) + "/finalize',{method:'POST'});"
        "document.getElementById('finmsg').textContent=' 已完成。';};"
        "</script>"
    )


def create_app(store, *, summary_backend, asr_backend=None,
               live_backend=None, live_interim_backend=None,
               summary_model="mlx-lm", live_silence_ms=500, live_min_speech_ms=250,
               live_interim_s=1.2, live_max_utt_s=15.0, live_rms_threshold=500):
    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    def index():
        return _INDEX

    @app.get("/live", response_class=HTMLResponse)
    def live_page():
        return _LIVE

    @app.get("/m/{mid}", response_class=HTMLResponse)
    def meeting_page(mid: int):
        meeting = store.get_meeting(mid)
        if meeting is None:
            raise HTTPException(404, "meeting not found")
        return _detail_page(mid, dict(meeting), _rows(store.list_transcripts(mid)),
                            _rows(store.list_summaries(mid)))

    @app.websocket("/ws/live")
    async def ws_live(ws: WebSocket):
        await ws.accept()
        if live_backend is None:
            await ws.send_json({"type": "error", "msg": "no live backend"})
            await ws.close()
            return
        track, speaker = _src_labels(ws.query_params.get("src", "mic"))
        mid = store.create_meeting("Live", time.time(), "zh-TW")
        sess = TwoPassSession(
            backend=live_backend, interim_backend=live_interim_backend,
            sample_rate=16000, silence_ms=live_silence_ms,
            min_speech_ms=live_min_speech_ms, interim_s=live_interim_s,
            max_utt_s=live_max_utt_s, rms_threshold=live_rms_threshold, track=track)
        await ws.send_json({"type": "meeting", "id": mid})

        async def _emit(ev):
            if ev["kind"] == "final":
                store.add_transcript(mid, "live", track, ev["start_ms"],
                                     ev["start_ms"], speaker, ev["text"])
                await ws.send_json({"type": "final", **ev})
            else:
                await ws.send_json({"type": "interim", **ev})
        pop_notice = getattr(live_backend, "pop_notice", None)
        try:
            while True:
                data = await ws.receive_bytes()
                for ev in await run_in_threadpool(sess.feed, data):
                    await _emit(ev)
                if pop_notice and (msg := pop_notice()):  # model auto-downgraded
                    await ws.send_json({"type": "notice", "msg": msg})
        except WebSocketDisconnect:
            pass
        finally:
            for ev in await run_in_threadpool(sess.flush):
                if ev["kind"] == "final":
                    store.add_transcript(mid, "live", track, ev["start_ms"],
                                         ev["start_ms"], speaker, ev["text"])
            # Stop != finalize: leave the meeting unfinalized so it can be
            # resumed / re-run. Finalize only on explicit user action.

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

    @app.post("/meetings/{mid}/finalize")
    def finalize_meeting(mid: int):
        if store.get_meeting(mid) is None:
            raise HTTPException(404, "meeting not found")
        store.finalize_meeting(mid)  # explicit only — stopping live does not
        return {"status": "finalized"}

    return app


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    from store import Store
    from summarize import mlx_lm_backend

    asr_model = os.environ.get("ASR_MODEL", "mlx-community/whisper-large-v3-turbo")
    llm_model = os.environ.get("LLM_MODEL", "mlx-community/Qwen2.5-3B-Instruct-4bit")
    # Live two-pass: fast interim model while speaking, accurate final model per
    # finished utterance (Teams-style retro-correction). No ASR runs on silence.
    # Tune: LIVE_MODEL (final/accurate), LIVE_INTERIM_MODEL (fast; "" disables
    # interim), LIVE_SILENCE_MS (pause = sentence end), LIVE_MIN_SPEECH_MS (drop
    # blips), LIVE_INTERIM_S (interim cadence), LIVE_RMS (mic level).
    live_model = os.environ.get("LIVE_MODEL", "mlx-community/whisper-large-v3-turbo")
    live_interim_model = os.environ.get("LIVE_INTERIM_MODEL",
                                        "mlx-community/whisper-small-mlx")
    live_silence = int(os.environ.get("LIVE_SILENCE_MS", "500"))
    live_min_speech = int(os.environ.get("LIVE_MIN_SPEECH_MS", "250"))
    live_interim_s = float(os.environ.get("LIVE_INTERIM_S", "1.2"))
    live_rms = int(os.environ.get("LIVE_RMS", "500"))
    # Auto-fallback chain for the final model: if it can't keep up (RTF over
    # budget) it drops to the next faster model. LIVE_FALLBACK = comma list.
    live_fallback = os.environ.get(
        "LIVE_FALLBACK", "mlx-community/whisper-small-mlx,mlx-community/whisper-base-mlx")
    live_rtf_budget = float(os.environ.get("LIVE_RTF_BUDGET", "0.8"))

    # Lazy-load the LLM on first request so the server starts instantly;
    # mlx-whisper already loads per-call. First /ingest downloads both models.
    _llm = {}

    def summary_backend(prompt):
        if "fn" not in _llm:
            _llm["fn"] = mlx_lm_backend(llm_model)
        return _llm["fn"](prompt)

    from live import AdaptiveBackend, mlx_whisper_live_backend

    final_models = [live_model] + [m for m in live_fallback.split(",")
                                   if m and m != live_model]
    live_final = AdaptiveBackend(
        [mlx_whisper_live_backend(m) for m in final_models],
        final_models, rtf_budget=live_rtf_budget)

    app = create_app(
        Store("data/meetings.db"),
        summary_backend=summary_backend,
        asr_backend=asr.mlx_whisper_backend(asr_model),
        live_backend=live_final,
        live_interim_backend=(mlx_whisper_live_backend(live_interim_model)
                              if live_interim_model else None),
        live_silence_ms=live_silence,
        live_min_speech_ms=live_min_speech,
        live_interim_s=live_interim_s,
        live_rms_threshold=live_rms,
        summary_model=llm_model,
    )
    uvicorn.run(app, host="127.0.0.1", port=8000)  # loopback only (G2)
