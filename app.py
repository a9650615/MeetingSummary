"""Web app: FastAPI. Binds 127.0.0.1 only (privacy, spec G2) — no auth by
design, loopback-only. Backends injected so tests run without MLX.

Phase 1 = batch loop: create meeting -> transcribe saved PCM -> summarize -> view.
Live websocket + recording start/stop (Swift helper) land in Phase 2."""
import asyncio
import html
import os
import sys
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
    <option value="dual">分軌(我 + 對方,分開標示)</option>
  </select>
</label>
<button id=start>開始</button> <button id=stop disabled>停止</button>
<span id=status></span>
<p>
  即時模型:
  <select id=model>
    <option value="mlx-community/whisper-large-v3-turbo">turbo(較準)</option>
    <option value="mlx-community/whisper-small-mlx">small(快)</option>
    <option value="mlx-community/whisper-base-mlx">base(最快)</option>
    <option value="Qwen/Qwen3-ASR-0.6B">Qwen3-ASR(最準·較慢·首次載入久)</option>
  </select>
  <span id=curmodel style="color:#888"></span>
  <span style="color:#888"> · 精校: <b id=accmodel>-</b></span>
</p>
<p style="color:#888">系統音/兩者:會跳出分享視窗,請選螢幕或分頁並<b>勾選「分享音訊」</b>。兩者建議戴耳機(否則麥克風會收到喇叭回音)。模型可即時切換,免重啟。</p>
<div id=caption style="margin:0.6em 0;padding:0.5em;min-height:1.6em;
  font-size:2em;font-weight:bold;background:#111;color:#fff;border-radius:6px"></div>
<div id=transcript style="margin-top:1em;font-size:1em;color:#444"></div>
<script>
let ws, ctx, gain, streams=[], nodes=[], mid;
const T=document.getElementById('transcript'), S=document.getElementById('status');
const C=document.getElementById('caption');
const startBtn=document.getElementById('start'), stopBtn=document.getElementById('stop');
const modelSel=document.getElementById('model'), curModel=document.getElementById('curmodel');
const COLORS={'我':'#1565c0','對方':'#2e7d32'};  // speaker colors

function showModels(m){
  curModel.textContent = '(目前 '+(m.live||'-').split('/').pop()+')';
  if(m.live_requested) modelSel.value = m.live_requested;
  document.getElementById('accmodel').textContent = (m.accurate||'-').split('/').pop();
}
fetch('/models').then(r=>r.json()).then(showModels).catch(()=>{});
modelSel.onchange = () => {
  fetch('/models',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({live:modelSel.value})})
    .then(r=>r.json()).then(()=>fetch('/models').then(r=>r.json()).then(showModels));
};

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
  return [mic, sys];  // both / dual
}

// One ScriptProcessor for a stream. tag=null -> send raw PCM; tag=0/1 -> prepend
// a track byte (dual mode). Per-node silence gate + hangover so the server VAD
// still sees the pause.
function attach(stream, tag, ratio){
  const node = ctx.createScriptProcessor(4096,1,1);
  ctx.createMediaStreamSource(stream).connect(node);
  let hangover = 0;
  node.onaudioprocess = ev => {
    if(ws.readyState!==1) return;
    const input = ev.inputBuffer.getChannelData(0);
    let sum=0; for(let i=0;i<input.length;i++) sum+=input[i]*input[i];
    const rms = Math.sqrt(sum/input.length);
    if(rms > 0.01) hangover = 8; else if(hangover > 0) hangover--;
    if(rms <= 0.01 && hangover <= 0) return;
    const outLen = Math.floor(input.length/ratio);
    const pcm = new Int16Array(outLen);
    for(let i=0;i<outLen;i++){
      const s = Math.max(-1,Math.min(1,input[Math.floor(i*ratio)]));
      pcm[i] = s*32767;
    }
    if(tag===null){ ws.send(pcm.buffer); }
    else { const b=new Uint8Array(1+pcm.byteLength); b[0]=tag;
           b.set(new Uint8Array(pcm.buffer),1); ws.send(b.buffer); }
  };
  node.connect(gain);
  nodes.push(node);
}

startBtn.onclick = async () => {
  const source = document.getElementById('source').value;
  const dual = source==='dual';
  try { streams = await getStreams(source); }
  catch(e){ S.textContent=' 取得音源失敗: '+e.message; return; }
  ctx = new AudioContext();
  gain = ctx.createGain(); gain.gain.value = 0;  // mute: no self-echo
  gain.connect(ctx.destination);
  const ratio = ctx.sampleRate/16000;
  ws = new WebSocket(`ws://${location.host}/ws/live?src=${source}`);
  ws.binaryType='arraybuffer';
  const tentative = {};  // per-speaker in-progress line
  function colored(speaker){ return COLORS[speaker]||'#444'; }
  ws.onmessage = e => {
    const m = JSON.parse(e.data);
    if(m.type==='meeting'){ mid=m.id; S.textContent=' 會議 #'+mid+' 錄製中…'; }
    else if(m.type==='interim'){
      const sp=m.speaker||'';
      C.textContent = (sp?sp+': ':'')+m.text; C.style.color=colored(sp);
      if(!tentative[sp]){ const p=document.createElement('p');
        p.style.color='#aaa'; T.insertAdjacentElement('afterbegin',p); tentative[sp]=p; }
      tentative[sp].textContent = '… '+(sp?sp+': ':'')+m.text;
    }
    else if(m.type==='final'){
      const sp=m.speaker||'';
      C.textContent = (sp?sp+': ':'')+m.text; C.style.color=colored(sp);
      const line = `[${(m.start_ms/1000).toFixed(1)}s] ${sp?sp+': ':''}${m.text}`;
      const p = tentative[sp] || document.createElement('p');
      if(!tentative[sp]) T.insertAdjacentElement('afterbegin', p);
      p.style.color = colored(sp); p.textContent = line; tentative[sp]=null;
    }
    else if(m.type==='notice'){ S.textContent=' ⚡ '+m.msg;
      fetch('/models').then(r=>r.json()).then(showModels); }
    else if(m.type==='error'){ S.textContent=' 錯誤: '+m.msg; }
  };
  ws.onopen = () => {
    if(dual){ attach(streams[0],0,ratio); attach(streams[1],1,ratio); }  // 我 / 對方
    else { streams.forEach(st=>attach(st,null,ratio)); }  // mic/system/both(mixed)
  };
  startBtn.disabled=true; stopBtn.disabled=false;
};
stopBtn.onclick = () => {
  nodes.forEach(n=>n.disconnect()); nodes=[];
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


class ModelIn(BaseModel):
    live: str


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
               live_manager=None, live_interim_backend=None, model_names=None,
               on_model_change=None,
               summary_model="mlx-lm", live_silence_ms=400, live_min_speech_ms=250,
               live_interim_s=0.6, live_max_utt_s=15.0, live_rms_threshold=500,
               live_max_lag_s=4.0):
    app = FastAPI()

    @app.get("/health")
    def health():
        return {"status": "ok"}  # fast — supervisor probes this for liveness

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

    @app.get("/models")
    def get_models():
        info = dict(model_names or {})
        if live_manager is not None:
            info["live"] = live_manager.current        # may differ after auto-downgrade
            info["live_requested"] = live_manager.requested
        return info

    @app.post("/models")
    def set_models(body: ModelIn):
        if live_manager is None:
            raise HTTPException(503, "no live model manager")
        live_manager.set_model(body.live)  # hot reload — no restart
        if on_model_change:
            on_model_change(body.live)
        return {"live": live_manager.requested}

    @app.websocket("/ws/live")
    async def ws_live(ws: WebSocket):
        await ws.accept()
        if live_manager is None:
            await ws.send_json({"type": "error", "msg": "no live backend"})
            await ws.close()
            return
        src = ws.query_params.get("src", "mic")
        dual = src == "dual"
        t0 = time.time()
        mid = store.create_meeting("Live", t0, "zh-TW")
        await ws.send_json({"type": "meeting", "id": mid})

        # Per-track. Dual = separate tagged streams (0=mic/我, 1=system/對方);
        # otherwise one track. Frame in dual mode = [1 byte tag] + PCM.
        if dual:
            tracks = {0: ("mic", "我"), 1: ("system", "對方")}
        else:
            tracks = {0: _src_labels(src)}
        sessions = {tag: TwoPassSession(
            backend=live_manager, interim_backend=live_interim_backend,
            sample_rate=16000, silence_ms=live_silence_ms,
            min_speech_ms=live_min_speech_ms, interim_s=live_interim_s,
            max_utt_s=live_max_utt_s, rms_threshold=live_rms_threshold,
            track=lbl[0]) for tag, lbl in tracks.items()}
        buffers = {tag: bytearray() for tag in tracks}

        async def _emit(ev, label):
            track, speaker = label
            if ev["kind"] == "final":
                ev["start_ms"] = int((time.time() - t0) * 1000)  # wall-clock, monotonic
                store.add_transcript(mid, "live", track, ev["start_ms"],
                                     ev["start_ms"], speaker, ev["text"])
                await ws.send_json({"type": "final", "speaker": speaker, **ev})
            else:
                await ws.send_json({"type": "interim", "speaker": speaker, **ev})

        # Producer/consumer per track: receiver buffers (cheap, drops oldest beyond
        # the lag ceiling); consumer batches each track's queued audio into one ASR
        # call, skips interim when behind (final > interim).
        got = asyncio.Event()
        closed = False
        max_lag_bytes = int(live_max_lag_s * 16000) * 2
        interim_lag_bytes = int(2 * live_interim_s * 16000) * 2

        async def receiver():
            nonlocal closed
            try:
                while True:
                    data = await ws.receive_bytes()
                    tag, pcm = (data[0], data[1:]) if dual else (0, data)
                    buf = buffers.get(tag)
                    if buf is None:
                        continue
                    buf.extend(pcm)
                    if len(buf) > max_lag_bytes:
                        del buf[:len(buf) - max_lag_bytes]
                    got.set()
            except WebSocketDisconnect:
                closed = True
                got.set()

        rtask = asyncio.create_task(receiver())
        try:
            while True:
                await got.wait()
                got.clear()
                if closed:
                    break
                for tag, buf in buffers.items():
                    if not buf:
                        continue
                    chunk = bytes(buf)
                    buf.clear()
                    want_interim = len(chunk) <= interim_lag_bytes
                    try:
                        events = await run_in_threadpool(
                            sessions[tag].feed, chunk, want_interim)
                        for ev in events:
                            await _emit(ev, tracks[tag])
                    except WebSocketDisconnect:
                        raise
                    except Exception as e:  # transient ASR error -> keep going
                        print(f"live consumer error (continuing): {e}", file=sys.stderr)
                pop_notice = getattr(live_manager.backend, "pop_notice", None)
                if pop_notice and (msg := pop_notice()):
                    await ws.send_json({"type": "notice", "msg": msg})
        finally:
            rtask.cancel()
            for tag, s in sessions.items():
                for ev in await run_in_threadpool(s.flush):
                    if ev["kind"] == "final":
                        ts = int((time.time() - t0) * 1000)
                        store.add_transcript(mid, "live", tracks[tag][0], ts, ts,
                                             tracks[tag][1], ev["text"])
            # Stop != finalize — explicit only.

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
    import modelprofile as mp

    # Background profile: auto-pick models from hardware + language. Zero config.
    # The runtime AdaptiveBackend then measures real GPU throughput (RTF) and the
    # learned tier is remembered across runs (data/model_profile.json).
    hw = mp.detect_hardware()
    rec = mp.recommend(hw, lang=os.environ.get("LANG_PREF", "zh-TW"))
    profile_path = "data/model_profile.json"
    remembered = mp.load_chosen(profile_path)
    print(f"[profile] {hw} -> {rec} (remembered live={remembered})", flush=True)

    asr_model = os.environ.get("ASR_MODEL", rec["accurate"])
    llm_model = os.environ.get("LLM_MODEL", rec["summary"])
    live_model = os.environ.get("LIVE_MODEL", remembered or rec["live"])
    live_interim_model = os.environ.get("LIVE_INTERIM_MODEL", rec["interim"])
    live_silence = int(os.environ.get("LIVE_SILENCE_MS", "400"))
    live_min_speech = int(os.environ.get("LIVE_MIN_SPEECH_MS", "250"))
    live_interim_s = float(os.environ.get("LIVE_INTERIM_S", "0.6"))
    live_rms = int(os.environ.get("LIVE_RMS", "500"))
    live_fallback = os.environ.get("LIVE_FALLBACK", ",".join(rec["fallback"]))
    live_rtf_budget = float(os.environ.get("LIVE_RTF_BUDGET", "0.8"))
    live_max_lag = float(os.environ.get("LIVE_MAX_LAG_S", "4.0"))  # drop audio beyond this lag

    import zhtw
    zhtw.configure(os.environ.get("CONVERT_ZHTW", "1") != "0")  # 簡->繁(台灣)

    # Lazy-load the LLM on first request so the server starts instantly;
    # mlx-whisper already loads per-call. First /ingest downloads both models.
    _llm = {}

    def summary_backend(prompt):
        if "fn" not in _llm:
            _llm["fn"] = mlx_lm_backend(llm_model)
        return _llm["fn"](prompt)

    import backends
    from live import mlx_whisper_live_backend

    fallback = [m for m in live_fallback.split(",") if m and m != live_model]

    # Startup probe: VALIDATE each candidate loads+runs on a 1 s clip; pick the
    # first that WORKS (skips belle-type incompatibilities so a broken pick can't
    # silently kill finals). Validate-only — RTF here would include one-time model
    # load and wrongly favor the smallest; the runtime AdaptiveBackend measures
    # real steady-state throughput and downgrades if the chosen model can't keep up.
    import numpy as _np
    import time as _time
    _clip = _np.zeros(16000, dtype=_np.int16).tobytes()
    live_model = mp.probe_models(
        [live_model] + fallback, audio_seconds=1.0,
        run=lambda m: mlx_whisper_live_backend(m)(_clip),
        clock=_time.monotonic, target_rtf=float("inf"))
    print(f"[profile] probed live model -> {live_model}", flush=True)

    # Modular + hot-reloadable: manager rebuilds the live AdaptiveBackend on swap.
    live_manager = backends.LiveModelManager(
        make=backends.make_live_backend, model=live_model, fallback=fallback,
        rtf_budget=live_rtf_budget,
        on_change=lambda m: mp.save_chosen(profile_path, m))

    app = create_app(
        Store("data/meetings.db"),
        summary_backend=summary_backend,
        asr_backend=backends.make_batch_backend(asr_model),  # routes qwen3/whisper
        live_manager=live_manager,
        live_interim_backend=(mlx_whisper_live_backend(live_interim_model)
                              if live_interim_model else None),
        model_names={"interim": live_interim_model, "accurate": asr_model,
                     "summary": llm_model},
        on_model_change=lambda m: mp.save_chosen(profile_path, m),
        live_silence_ms=live_silence,
        live_min_speech_ms=live_min_speech,
        live_interim_s=live_interim_s,
        live_rms_threshold=live_rms,
        live_max_lag_s=live_max_lag,
        summary_model=llm_model,
    )
    uvicorn.run(app, host="127.0.0.1", port=8000)  # loopback only (G2)
