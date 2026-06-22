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
from fastapi.responses import FileResponse, HTMLResponse, Response
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

import asr
from live import TwoPassSession
from summarize import summarize

# Shared design system — calm zh-TW productivity aesthetic. Served pages (not a
# claude.ai Artifact), so a full self-styled doc is fine.
_STYLE = """
*{box-sizing:border-box}
:root{--bg:#f4f5f8;--surface:#fff;--ink:#16181d;--muted:#697086;--line:#e7e9ef;
 --accent:#4f46e5;--accent2:#6366f1;--me:#1565c0;--other:#2e7d32;--danger:#dc2626;
 --radius:12px;--shadow:0 1px 2px rgba(16,24,40,.04),0 6px 20px rgba(16,24,40,.06)}
body{margin:0;background:var(--bg);color:var(--ink);
 font:15px/1.6 -apple-system,BlinkMacSystemFont,"PingFang TC","Noto Sans TC","Segoe UI",Roboto,sans-serif}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
.wrap{max-width:880px;margin:0 auto;padding:22px 20px 72px}
header.top{display:flex;align-items:center;gap:12px;margin-bottom:20px}
.brand{font-weight:750;font-size:17px;letter-spacing:-.01em}
.brand .dot{color:var(--accent)}
.spacer{flex:1}
h1{font-size:24px;font-weight:750;letter-spacing:-.02em;margin:.1em 0 .7em}
h2{font-size:15px;font-weight:700;margin:26px 0 12px;color:var(--ink)}
h3{font-size:14px;font-weight:700;margin:14px 0 6px;color:var(--muted)}
.card{background:var(--surface);border:1px solid var(--line);border-radius:var(--radius);
 box-shadow:var(--shadow);padding:18px 20px;margin:14px 0}
.muted{color:var(--muted)}.small{font-size:13px}
.row{display:flex;flex-wrap:wrap;gap:12px;align-items:center}
label.fld{display:inline-flex;flex-direction:column;gap:5px;font-size:12px;color:var(--muted);font-weight:600}
label.chk{display:inline-flex;align-items:center;gap:6px;font-size:13px;color:var(--ink)}
input[type=text],input[type=file],input:not([type]),select{font:inherit;padding:.5em .65em;
 border:1px solid var(--line);border-radius:8px;background:var(--surface);color:var(--ink)}
select{cursor:pointer}
.btn{font:inherit;font-weight:600;padding:.55em 1em;border-radius:8px;border:1px solid var(--line);
 background:var(--surface);color:var(--ink);cursor:pointer;transition:.12s;display:inline-block}
.btn:hover{border-color:var(--accent);color:var(--accent)}
.btn:disabled{opacity:.45;cursor:not-allowed;border-color:var(--line);color:var(--ink)}
.btn.primary{background:var(--accent);border-color:var(--accent);color:#fff}
.btn.primary:hover{background:var(--accent2);color:#fff}
.btn.danger{color:var(--danger);border-color:#f0cccc}
.btn.danger:hover{background:var(--danger);color:#fff;border-color:var(--danger)}
.badge{display:inline-block;font-size:11px;font-weight:700;padding:.18em .6em;border-radius:999px;
 background:#eceef3;color:var(--muted)}
.badge.live{background:#fde7e7;color:#c0392b}.badge.done{background:#e6f5ec;color:#1e7d3a}
ul.meetings{list-style:none;margin:0;padding:0}
ul.meetings li{display:flex;align-items:center;gap:10px;padding:11px 2px;border-bottom:1px solid var(--line)}
ul.meetings li:last-child{border:0}ul.meetings a{font-weight:600;flex:1}
.caption{font-size:clamp(22px,4.2vw,34px);font-weight:750;line-height:1.32;background:#0f1115;
 color:#fff;border-radius:14px;padding:18px 22px;min-height:1.4em;margin:16px 0 8px}
.liveline{color:var(--muted);font-size:17px;min-height:1.5em;margin:6px 2px 14px}
.tline{padding:8px 2px;border-bottom:1px solid var(--line);display:flex;gap:12px}
.tline:last-child{border:0}
.tline .ts{color:var(--muted);font-size:12px;font-variant-numeric:tabular-nums;white-space:nowrap;padding-top:3px}
.tline .who{font-weight:700}
table.tx{width:100%;border-collapse:collapse}
table.tx th{text-align:left;font-size:12px;color:var(--muted);font-weight:600;padding:8px 10px;border-bottom:1px solid var(--line)}
table.tx td{padding:9px 10px;border-bottom:1px solid var(--line);vertical-align:top}
table.tx td.who{white-space:nowrap;font-weight:600;width:96px}
table.tx td.ts{white-space:nowrap;color:var(--muted);font-size:12px;font-variant-numeric:tabular-nums;width:52px}
table.tx tr[data-ts]:hover{background:#f7f8fb}
table.tx tr.active{background:#eef0ff!important;box-shadow:inset 3px 0 0 var(--accent)}
table.tx tr.active td{color:var(--ink)}
pre.sum{white-space:pre-wrap;font:inherit;background:#fafbfc;border:1px solid var(--line);
 border-radius:10px;padding:14px;margin:8px 0;overflow-x:auto}
.hint{color:var(--muted);font-size:13px;line-height:1.55}
.card.sticky{position:sticky;top:8px;z-index:20;backdrop-filter:blur(6px);
 background:rgba(255,255,255,.96)}
"""


def _shell(title, body, script="", back=False):
    nav = '<a href="/">&larr; 回首頁</a>' if back else ''
    return (
        "<!doctype html><html lang=zh-Hant><head><meta charset=utf-8>"
        "<meta name=viewport content='width=device-width,initial-scale=1'>"
        f"<title>{title}</title><style>{_STYLE}</style></head><body><div class=wrap>"
        "<header class=top><span class=brand>📝 Meeting<span class=dot>·</span>Summary</span>"
        f"<span class=spacer></span>{nav}</header>{body}</div>"
        + (f"<script>{script}</script>" if script else "")
        + "</body></html>"
    )


_INDEX = _shell("MeetingSummary", """
<h1>本地會議轉錄 · 摘要</h1>
<a class="btn primary" href="/live" style="font-size:16px;padding:.7em 1.2em">🔴 開始 Live 即時逐字稿</a>
<h2>上傳音檔</h2>
<div class=card>
  <form action="/ingest" method="post" enctype="multipart/form-data" class=row>
    <label class=fld>音檔 (wav/m4a/mp3)<input type=file name=audio required></label>
    <label class=fld>標題<input name=title value="實測"></label>
    <label class=fld>摘要型式<select name=kind>
      <option value=minutes>會議記錄 minutes</option>
      <option value=bullets>條列 bullets</option></select></label>
    <button class="btn primary" type=submit>上傳並產生摘要</button>
  </form>
  <p class=hint style="margin:.8em 0 0">上傳後會跑 transcribe + summary，第一次會下載模型，請稍候。</p>
</div>
<h2>會議紀錄</h2>
<div class=card>
  <div class=row style="margin-bottom:6px">
    <button class=btn id=mergebtn>整合相近的 live 會議</button>
    <span class="muted small" id=mergemsg></span>
  </div>
  <ul class=meetings id=meetings></ul>
</div>
""", script="""
fetch('/meetings').then(r=>r.json()).then(ms=>{
  document.getElementById('meetings').innerHTML = ms.length
    ? ms.map(m=>`<li><span title="${m.has_audio?'有音檔':'無音檔'}">${m.has_audio?'🔊':'🔇'}</span>
        <a href="/m/${m.id}">${m.title}</a>
        <span class="badge ${m.status==='finalized'?'done':'live'}">${m.status}</span></li>`).join('')
    : '<li class="muted small">尚無會議</li>';
});
document.getElementById('mergebtn').onclick = async () => {
  const mm=document.getElementById('mergemsg'); mm.textContent='整合中…';
  const r=await fetch('/meetings/merge-nearby?gap_min=10',{method:'POST'});
  const j=await r.json();
  mm.textContent=' 已整合 '+j.merged_groups+' 組';
  setTimeout(()=>location.reload(),600);
};
""")


def _result_page(title, summary, transcripts):
    lines = "".join(
        f"<tr><td class=who>{html.escape(str(r['track']))}</td>"
        f"<td>{html.escape(r['text'])}</td></tr>"
        for r in transcripts)
    body = (
        f"<h1>{html.escape(title)}</h1>"
        f"<div class=card><h2 style='margin-top:0'>摘要</h2>"
        f"<pre class=sum>{html.escape(summary)}</pre></div>"
        f"<div class=card><h2 style='margin-top:0'>逐字稿</h2>"
        f"<table class=tx><tr><th>軌</th><th>內容</th></tr>{lines}</table></div>")
    return _shell(html.escape(title), body, back=True)


# Live page: browser mic -> 16 kHz Int16 PCM over websocket -> server ASR.
# ScriptProcessorNode is deprecated but needs no separate worklet file.
# ponytail: swap to AudioWorklet if latency/jank shows up.
_LIVE_BODY = """
<h1>🔴 Live 即時逐字稿</h1>
<div class=card>
  <div class=row>
    <label class=fld>來源
      <select id=source>
        <option value="mic">麥克風(我)</option>
        <option value="system">系統音(對方)</option>
        <option value="both">兩者(混合)</option>
        <option value="dual">分軌(我 + 對方,分開標示)</option>
      </select></label>
    <label class=fld>辨識單元
      <select id=unit>
        <option value="sentence">句子(快)</option>
        <option value="paragraph">段落(較準,等較久)</option>
      </select></label>
    <label class=fld>即時模型
      <select id=model>
        <option value="mlx-community/whisper-large-v3-turbo-q4">turbo-q4(準·省一半)</option>
        <option value="mlx-community/whisper-large-v3-turbo">turbo(最準·較吃)</option>
        <option value="mlx-community/whisper-small-mlx-q4">small-q4(快·省)</option>
        <option value="mlx-community/whisper-base-mlx-q4">base-q4(更快)</option>
        <option value="mlx-community/whisper-tiny-mlx-q4">tiny-q4(最省·最快)</option>
        <option value="Qwen/Qwen3-ASR-0.6B">Qwen3-ASR(最準·較慢)</option>
      </select></label>
    <label class=chk style="align-self:end"><input type=checkbox id=diarize> 對方即時多人分群(實驗)</label>
  </div>
  <div class=row style="margin-top:14px">
    <button class="btn primary" id=start>開始</button>
    <button class=btn id=stop disabled>停止</button>
    <button class=btn id=newsess>新 session</button>
    <span class="muted small" id=status></span>
  </div>
  <p class=hint style="margin:.7em 0 0">
    <span id=curmodel></span> · 精校:<b id=accmodel>-</b><br>
    系統音/兩者會跳出分享視窗,請選螢幕或分頁並<b>勾選「分享音訊」</b>;兩者建議戴耳機。模型可即時切換,免重啟。</p>
</div>
<div class=caption id=caption></div>
<div class=liveline id=live></div>
<div class=tlist id=transcript></div>
"""

# Live page: browser mic -> 16 kHz Int16 PCM over websocket -> server ASR.
# ScriptProcessorNode is deprecated but needs no separate worklet file.
_LIVE_JS = """
let ws, ctx, gain, streams=[], nodes=[], mid, session=null;
const T=document.getElementById('transcript'), S=document.getElementById('status');
const C=document.getElementById('caption'), L=document.getElementById('live');
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
    // keep sending ~1.5s after speech so the server sees the full pause even in
    // paragraph mode (silence_ms up to 1000ms) — else finals never fire.
    if(rms > 0.01) hangover = 18; else if(hangover > 0) hangover--;
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
  const diar = document.getElementById('diarize').checked ? '&diarize=1' : '';
  // bigger unit = longer pause + bigger ceiling -> more context per accurate pass
  const unit = document.getElementById('unit').value==='paragraph'
    ? '&silence_ms=1000&max_utt_s=30' : '';
  const sess = session ? '&session='+session : '';
  ws = new WebSocket(`ws://${location.host}/ws/live?src=${source}${diar}${unit}${sess}`);
  ws.binaryType='arraybuffer';
  function colored(speaker){ return COLORS[speaker]||'#444'; }
  ws.onmessage = e => {
    const m = JSON.parse(e.data);
    if(m.type==='meeting'){ mid=m.id; session=m.id;  // bind session to this meeting
      S.textContent=' session #'+mid+' 錄製中…'; }
    else if(m.type==='interim'){
      const sp=m.speaker||'';
      C.textContent = m.text;                       // caption: words only
      L.textContent = '… '+(sp?sp+': ':'')+m.text;  // grey in-progress, above history
    }
    else if(m.type==='final'){
      const sp=m.speaker||'';
      C.textContent = m.text;
      const tstr = m.ts ? new Date(m.ts*1000).toLocaleTimeString() : (m.start_ms/1000).toFixed(1)+'s';
      const line=document.createElement('div'); line.className='tline';
      const ts=document.createElement('span'); ts.className='ts'; ts.textContent=tstr;
      const bd=document.createElement('span');
      if(sp){const w=document.createElement('b'); w.className='who';
        w.style.color=colored(sp); w.textContent=sp+'：'; bd.appendChild(w);}
      bd.appendChild(document.createTextNode(m.text));
      line.appendChild(ts); line.appendChild(bd);
      T.insertAdjacentElement('afterbegin', line);  // newest on top, below #live
      L.textContent='';                             // utterance committed
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
document.getElementById('newsess').onclick = () => {
  session=null; mid=null;       // next 開始 starts a fresh session
  T.innerHTML=''; L.textContent=''; C.textContent='';
  S.textContent=' 已開新 session';
};
stopBtn.onclick = () => {
  nodes.forEach(n=>n.disconnect()); nodes=[];
  streams.forEach(st => st.getTracks().forEach(t=>t.stop()));
  if(ws) ws.close();
  if(ctx) ctx.close();
  S.textContent += mid ? ` 已停止。可到首頁對 #${mid} 產生摘要。` : ' 已停止。';
  startBtn.disabled=false; stopBtn.disabled=true;
};
"""

_LIVE = _shell("Live · MeetingSummary", _LIVE_BODY, script=_LIVE_JS, back=True)


class MeetingIn(BaseModel):
    title: str
    lang: str = "zh-TW"


class SummaryIn(BaseModel):
    kind: str = "minutes"


class ModelIn(BaseModel):
    live: str


class MergeIn(BaseModel):
    ids: list[int]


class TranscribeIn(BaseModel):
    model: str | None = None   # None -> default accurate backend


class DiarizeIn(BaseModel):
    track: str = "system"      # diarize the 對方/system track by default
    num_speakers: int = -1     # -1 = auto-detect
    prefix: str = "說話者"


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


_TRACKS = ("system", "mic", "mixed")


def _assemble_track(store, mid, track, sample_rate=16000):
    """Build one continuous PCM for a track across ALL of a meeting's segments,
    each placed at its time offset (started_at - meeting.created_at) so audio
    lines up with the (merge-rebased) transcript timestamps. Segments sharing a
    dir (session resume) are placed once. None if the track has no audio."""
    meeting = store.get_meeting(mid)
    if meeting is None:
        return None
    base = meeting["created_at"]
    buf = bytearray()
    seen = set()
    found = False
    for seg in store.list_segments(mid):  # idx order -> first dir occurrence = earliest
        d = seg["dir_path"]
        if d in seen:
            continue
        pcm_path = os.path.join(d, f"{track}.pcm")
        if not os.path.exists(pcm_path) or os.path.getsize(pcm_path) == 0:
            continue
        seen.add(d)
        found = True
        off_bytes = max(0, int(round((seg["started_at"] - base) * sample_rate))) * 2
        with open(pcm_path, "rb") as f:
            data = f.read()
        end = off_bytes + len(data)
        if end > len(buf):
            buf.extend(b"\x00" * (end - len(buf)))
        buf[off_bytes:end] = data
    return bytes(buf) if found else None


def _track_wav_file(store, mid, track):
    """Assemble the track and write a cached .wav file, returning its path (or
    None if no audio). Served via FileResponse so the browser gets HTTP Range
    support and can SEEK — a plain in-memory Response can't be sought. Rebuilds
    only when a source pcm is newer than the cache."""
    import recorder
    cache = f"data/{mid}/_play_{track}.wav"
    srcs = [os.path.join(seg["dir_path"], f"{track}.pcm")
            for seg in store.list_segments(mid)]
    srcs = [p for p in srcs if os.path.exists(p) and os.path.getsize(p) > 0]
    if not srcs:
        return None
    newest = max(os.path.getmtime(p) for p in srcs)
    if not os.path.exists(cache) or os.path.getmtime(cache) < newest:
        pcm = _assemble_track(store, mid, track)
        if pcm is None:
            return None
        os.makedirs(f"data/{mid}", exist_ok=True)
        with open(cache, "wb") as f:
            f.write(recorder.pcm_to_wav(pcm, sample_rate=16000, channels=1))
    return cache


def _meeting_tracks(store, mid):
    """Tracks that have audio anywhere in the meeting's segments."""
    out = []
    for t in _TRACKS:
        for seg in store.list_segments(mid):
            p = os.path.join(seg["dir_path"], f"{t}.pcm")
            if os.path.exists(p) and os.path.getsize(p) > 0:
                out.append(t)
                break
    return out


def iter_transcribe(store, mid, backend, window_s=30, sample_rate=16000):
    """Generator: re-transcribe a meeting window-by-window, yielding progress
    events ({type:start,total} / {type:progress,done,total,text} / {type:done,n})
    and storing each result as it lands. Windowing gives granular progress even
    for a single long file; each window's text streams to the client live."""
    base = store.get_meeting(mid)["created_at"]
    win_bytes = int(window_s * sample_rate) * 2
    units = []  # (track, pcm_path, base_off_ms)
    for seg in store.list_segments(mid):
        seg_off = max(0, int((seg["started_at"] - base) * 1000))
        for track in ("system", "mic", "mixed"):
            p = os.path.join(seg["dir_path"], f"{track}.pcm")
            if not os.path.exists(p) or os.path.getsize(p) == 0:
                continue
            size = os.path.getsize(p)
            for bs in range(0, size, win_bytes):
                units.append((track, p, seg_off, bs, min(win_bytes, size - bs)))
    yield {"type": "start", "total": len(units)}
    n = 0
    for i, (track, p, seg_off, bs, bl) in enumerate(units):
        win_off_ms = seg_off + int(bs / 2 / sample_rate * 1000)
        tmp = f"{os.path.dirname(p)}/_win.pcm"
        with open(p, "rb") as f:
            f.seek(bs)
            with open(tmp, "wb") as w:
                w.write(f.read(bl))
        texts = []
        for t in asr.transcribe(tmp, profile="accurate", track=track, backend=backend):
            store.add_transcript(mid, "accurate", track,
                                 t["start_ms"] + win_off_ms,
                                 t["end_ms"] + win_off_ms, track, t["text"])
            texts.append(t["text"])
            n += 1
        try:
            os.remove(tmp)
        except OSError:
            pass
        yield {"type": "progress", "done": i + 1, "total": len(units),
               "text": " ".join(texts)[:60]}
    yield {"type": "done", "transcripts": n}


def _run_transcribe_job(store, mid, backend, jobs):
    """Run iter_transcribe to completion, recording progress in jobs[mid] so a
    page can poll it (survives client refresh). Transcripts are stored as they
    land, so even a server restart keeps partial work."""
    jobs[mid] = {"state": "running", "done": 0, "total": 0, "text": ""}
    try:
        for ev in iter_transcribe(store, mid, backend):
            if ev["type"] == "start":
                jobs[mid]["total"] = ev["total"]
            elif ev["type"] == "progress":
                jobs[mid].update(done=ev["done"], total=ev["total"], text=ev["text"])
            elif ev["type"] == "done":
                jobs[mid] = {"state": "done", "done": ev["transcripts"],
                             "total": jobs[mid].get("total", 0)}
    except Exception as e:
        jobs[mid] = {"state": "error", "msg": str(e)}


def _save_upload_pcm(src_path, mid, store):
    """Decode an uploaded file to data/<mid>/mic.pcm (16 kHz mono s16le) via ffmpeg
    so the meeting plays back like a live one. Best-effort: skip if ffmpeg missing
    or decode fails (playback simply won't appear)."""
    import shutil
    import subprocess
    if not shutil.which("ffmpeg"):
        return
    out_dir = f"data/{mid}-{int(time.time())}"  # unique: reused ids can't collide
    os.makedirs(out_dir, exist_ok=True)
    pcm = f"{out_dir}/mic.pcm"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", src_path,
             "-ar", "16000", "-ac", "1", "-f", "s16le", pcm],
            check=True, timeout=300)
    except Exception as e:
        print(f"upload pcm decode failed: {e}", file=sys.stderr)
        return
    store.add_segment(mid, idx=len(store.list_segments(mid)), dir_path=out_dir,
                      started_at=time.time(), duration_s=0, origin="recorded")


_TRACK_LABEL = {"system": "對方", "mic": "我", "mixed": "混合"}


def _detail_page(mid, meeting, transcripts, summaries, audio_tracks=()):
    def ts_str(ms):
        s = (ms or 0) // 1000
        return f"{s // 60:d}:{s % 60:02d}"
    rows = "".join(
        f"<tr data-track='{html.escape(str(r['track']))}' data-ts='{(r['start_ms'] or 0)/1000:.2f}'>"
        f"<td class=ts>{ts_str(r['start_ms'])}</td>"
        f"<td class=who>{html.escape(str(r['speaker']))}</td>"
        f"<td>{html.escape(r['text'])}</td></tr>"
        for r in transcripts) or "<tr><td colspan=3 class=muted>尚無逐字稿</td></tr>"
    sums = "".join(
        f"<h3>{html.escape(s['kind'])}</h3>"
        f"<pre class=sum>{html.escape(s['text'])}</pre>"
        for s in summaries)
    players = "".join(
        f"<div style='margin:6px 0'><span class='badge'>{_TRACK_LABEL.get(t, t)}</span> "
        f"<audio id='aud-{t}' controls preload=none style='vertical-align:middle;height:34px'"
        f" src='/meetings/{mid}/audio/{t}.wav'></audio></div>"
        for t in audio_tracks)
    audio_card = (f"<div class='card sticky'><h2 style='margin-top:0'>回放</h2>{players}"
                  "<p class=hint style='margin:.5em 0 0'>點逐字稿任一行可跳到該段落播放。"
                  "捲動時播放器固定在頂部。</p></div>") if audio_tracks else ""
    badge = "done" if meeting["status"] == "finalized" else "live"
    body = (
        f"<h1>{html.escape(meeting['title'])} "
        f"<span class='badge {badge}'>{html.escape(meeting['status'])}</span></h1>"
        + audio_card +
        "<div class=card><h2 style='margin-top:0'>摘要</h2>"
        "<div class=row>"
        "<select id=kind><option value=minutes>會議記錄</option>"
        "<option value=bullets>條列</option></select>"
        "<button class='btn primary' id=go>產生摘要</button>"
        "<button class=btn id=dia>多人分群(對方)</button>"
        "<button class=btn id=fin>完成會議</button>"
        "<button class='btn danger' id=del>刪除會議</button>"
        "<span class='muted small' id=finmsg></span></div>"
        "<p class=hint style='margin:.6em 0 0'>※ 摘要只在按下按鈕時才產生。停止錄音不會自動完成。"
        "「多人分群」用聲紋把對方那軌拆成說話者1/2/3(會後處理,需先有錄音)。</p>"
        f"<div id=out>{sums}</div></div>"
        "<div class=card><h2 style='margin-top:0'>逐字稿</h2>"
        "<div class=row style='margin-bottom:10px'>"
        "<select id=remodel>"
        "<option value='mlx-community/whisper-large-v3-turbo-q4'>turbo-q4(準·省)</option>"
        "<option value='mlx-community/whisper-large-v3-mlx'>large-v3(最準·吃)</option>"
        "<option value='mlx-community/whisper-small-mlx-q4'>small-q4(快)</option>"
        "<option value='Qwen/Qwen3-ASR-0.6B'>Qwen3-ASR(最準中文·慢)</option>"
        "</select>"
        "<button class=btn id=retr>重新語音辨識</button>"
        "<span class='muted small' id=remsg></span></div>"
        "<table class=tx><tr><th>時間</th><th>說話者</th><th>內容</th></tr>"
        f"{rows}</table></div>")
    script = (
        "document.getElementById('go').onclick=async()=>{"
        "const k=document.getElementById('kind').value;"
        "const o=document.getElementById('out');o.textContent='產生中…';"
        f"const r=await fetch('/meetings/{mid}/summary',{{method:'POST',"
        "headers:{'Content-Type':'application/json'},body:JSON.stringify({kind:k})});"
        "const j=await r.json();const p=document.createElement('pre');"
        "p.className='sum';p.textContent=j.text;o.innerHTML='';o.appendChild(p);};"
        "document.getElementById('fin').onclick=async()=>{"
        f"await fetch('/meetings/{mid}/finalize',{{method:'POST'}});"
        "document.getElementById('finmsg').textContent=' 已完成。';};"
        "document.getElementById('del').onclick=async()=>{"
        "if(!confirm('確定刪除這場會議?逐字稿與音檔都會移除,無法復原。'))return;"
        f"const r=await fetch('/meetings/{mid}',{{method:'DELETE'}});"
        "if(r.ok)location.href='/';"
        "else document.getElementById('finmsg').textContent=' 刪除失敗';};"
        # Background job + polling: progress is server-side, so a page refresh
        # reconnects to the running job instead of losing it. poll() self-manages
        # the interval — only ticks while a job is active.
        "const rm=document.getElementById('remsg'),retr=document.getElementById('retr');"
        "let poller=null,sawRunning=false;"  # only reload after we watched it finish
        "function poll(){"
        f"fetch('/meetings/{mid}/transcribe/progress').then(r=>r.json()).then(p=>{{"
        "if(p.state==='running'){sawRunning=true;retr.disabled=true;"
        "rm.textContent=` 處理中 ${p.done||0}/${p.total||'?'} — ${p.text||''}`;"
        "if(!poller)poller=setInterval(poll,1000);return;}"
        "if(poller){clearInterval(poller);poller=null;}"
        "if(p.state==='done'){rm.textContent=' 完成 '+p.done+' 段';"
        "if(sawRunning){sawRunning=false;setTimeout(()=>location.reload(),500);}}"  # reload once
        "else if(p.state==='error'){retr.disabled=false;rm.textContent=' 失敗: '+p.msg;}"
        "else retr.disabled=false;});}"
        "retr.onclick=()=>{const mdl=document.getElementById('remodel').value;"
        "rm.textContent=' 啟動中…';retr.disabled=true;sawRunning=true;"
        f"fetch('/meetings/{mid}/transcribe/start',{{method:'POST',"
        "headers:{'Content-Type':'application/json'},body:JSON.stringify({model:mdl})})"
        ".then(()=>poll());};"
        "poll();"  # resume on load if a job is already running
        "document.getElementById('dia').onclick=async()=>{"
        "const fm=document.getElementById('finmsg');fm.textContent=' 分群中…(會後聲紋,需稍候)';"
        f"const r=await fetch('/meetings/{mid}/diarize',{{method:'POST',"
        "headers:{'Content-Type':'application/json'},body:JSON.stringify({track:'system'})});"
        "if(r.ok){const j=await r.json();fm.textContent=' 分出 '+j.speakers+' 位說話者';"
        "location.reload();}else{fm.textContent=' 分群失敗: '+(await r.text());}};"
        "document.querySelectorAll('tr[data-ts]').forEach(tr=>{tr.style.cursor='pointer';"
        "tr.onclick=()=>{const a=document.getElementById('aud-'+tr.dataset.track);"
        "if(a){a.currentTime=parseFloat(tr.dataset.ts);a.play();}};});"
        # Follow-along: as a track's audio plays, highlight + scroll to the line
        # whose start time is the latest <= currentTime (end_ms==start_ms for live,
        # so use the next line's start as the implicit boundary).
        "document.querySelectorAll('audio[id^=aud-]').forEach(a=>{"
        "const trk=a.id.slice(4);"
        "const rows=[...document.querySelectorAll(`tr[data-track=\"${trk}\"]`)]"
        ".map(tr=>({tr,ts:parseFloat(tr.dataset.ts)})).sort((x,y)=>x.ts-y.ts);"
        "let last=null;"
        "a.ontimeupdate=()=>{const t=a.currentTime;let cur=null;"
        "for(const r of rows){if(r.ts<=t+0.05)cur=r;else break;}"
        "if(cur===last)return;"
        "rows.forEach(r=>r.tr.classList.toggle('active',r===cur));"
        "if(cur)cur.tr.scrollIntoView({block:'nearest',behavior:'smooth'});last=cur;};"
        "});")
    return _shell(html.escape(meeting["title"]), body, script=script, back=True)


def create_app(store, *, summary_backend, asr_backend=None,
               live_manager=None, live_interim_backend=None, model_names=None,
               on_model_change=None,
               summary_model="mlx-lm", live_silence_ms=400, live_min_speech_ms=250,
               live_interim_s=0.6, live_max_utt_s=15.0, live_rms_threshold=500,
               live_max_lag_s=4.0):
    app = FastAPI()
    transcribe_jobs = {}  # mid -> progress dict; survives page refresh (in-memory)

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
        # Tracks with retained audio (across all segments — handles merged meetings).
        audio_tracks = _meeting_tracks(store, mid)
        return _detail_page(mid, dict(meeting), _rows(store.list_transcripts(mid)),
                            _rows(store.list_summaries(mid)), audio_tracks)

    @app.get("/meetings/{mid}/audio/{track}.wav")
    def meeting_audio(mid: int, track: str):
        if track not in _TRACKS:
            raise HTTPException(404, "no audio")
        wav = _track_wav_file(store, mid, track)  # cached file -> Range/seek support
        if wav is None:
            raise HTTPException(404, "no audio")
        return FileResponse(wav, media_type="audio/wav")

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
        # Session: a recording binds to an existing meeting (token = meeting id) so
        # stop/resume stays in one session; a missing/unknown token starts a new one.
        session = ws.query_params.get("session")
        if session and session.isdigit() and store.get_meeting(int(session)) is not None:
            mid = int(session)
        else:
            mid = store.create_meeting("Live", t0, "zh-TW")
        await ws.send_json({"type": "meeting", "id": mid})

        # Per-track. Dual = separate tagged streams (0=mic/我, 1=system/對方);
        # otherwise one track. Frame in dual mode = [1 byte tag] + PCM.
        if dual:
            tracks = {0: ("mic", "我"), 1: ("system", "對方")}
        else:
            tracks = {0: _src_labels(src)}
        # Per-connection unit size: a longer finalize pause + bigger max means the
        # accurate pass transcribes a whole sentence/paragraph in one call (more
        # context -> better wording/punctuation, not cut mid-thought).
        q = ws.query_params
        sil = max(200, min(3000, int(q.get("silence_ms") or live_silence_ms)))
        maxu = max(5.0, min(40.0, float(q.get("max_utt_s") or live_max_utt_s)))
        sessions = {tag: TwoPassSession(
            backend=live_manager, interim_backend=live_interim_backend,
            sample_rate=16000, silence_ms=sil,
            min_speech_ms=live_min_speech_ms, interim_s=live_interim_s,
            max_utt_s=maxu, rms_threshold=live_rms_threshold,
            track=lbl[0]) for tag, lbl in tracks.items()}
        buffers = {tag: bytearray() for tag in tracks}

        # Save raw audio per track for the post-meeting diarization/playback pass.
        # Unique dir per connection (id + start ts): SQLite recycles deleted ids
        # after merges, so a plain data/<mid> could collide with a leftover dir and
        # append onto stale audio. _assemble_track stitches segments by time offset,
        # so a resumed session's new dir is just another ordered segment.
        audio_dir = f"data/{mid}-{int(t0)}"
        os.makedirs(audio_dir, exist_ok=True)
        store.add_segment(mid, idx=len(store.list_segments(mid)), dir_path=audio_dir,
                          started_at=t0, duration_s=0, origin="recorded")
        audio_files = {tag: open(f"{audio_dir}/{lbl[0]}.pcm", "wb")
                       for tag, lbl in tracks.items()}

        # Live multi-speaker (?diarize=1): per-track online voiceprint clustering
        # on each finalized utterance. Skip the 我/mic track (single person).
        if ws.query_params.get("diarize") == "1":
            try:
                import diarize as diar
                extractor = diar.embedding_extractor()
                thr = float(os.environ.get("LIVE_DIAR_THRESHOLD", "0.4"))
                min_bytes = int(1.2 * 16000) * 2  # <1.2s -> too short, reuse last

                def make_fn(tr):
                    def fn(audio):
                        if len(audio) < min_bytes and tr.centroids:
                            return f"說話者{tr.last_id + 1}"  # don't spawn on a blip
                        return f"說話者{tr.assign(extractor(audio)) + 1}"
                    return fn
                for tag, (trk, spk) in tracks.items():
                    if spk != "我":
                        sessions[tag].speaker_fn = make_fn(diar.SpeakerTracker(threshold=thr))
            except Exception as e:
                print(f"live diarize unavailable: {e}", file=sys.stderr)

        async def _emit(ev, label):
            track, speaker = label
            if ev["kind"] == "final":
                # start_ms = audio position (session's committed-bytes offset),
                # which matches the saved (silence-gated) pcm exactly -> seek +
                # follow-highlight line up. ts (wall-clock) is display only.
                ev["ts"] = time.time()
                spk = ev.get("speaker") or speaker        # online diarize overrides
                store.add_transcript(mid, "live", track, ev["start_ms"],
                                     ev.get("end_ms", ev["start_ms"]), spk, ev["text"])
                await ws.send_json({"type": "final", **ev, "speaker": spk})
            else:
                await ws.send_json({"type": "interim", **ev, "speaker": speaker})

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
                    audio_files[tag].write(pcm)  # full audio for diarization
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
                    if ev["kind"] == "final":  # audio-position offset, not wall-clock
                        spk = ev.get("speaker") or tracks[tag][1]
                        store.add_transcript(mid, "live", tracks[tag][0],
                                             ev["start_ms"],
                                             ev.get("end_ms", ev["start_ms"]), spk,
                                             ev["text"])
            for f in audio_files.values():
                f.close()
            # Stop != finalize — explicit only.

    @app.post("/meetings")
    def create_meeting(m: MeetingIn):
        return {"id": store.create_meeting(m.title, time.time(), m.lang)}

    @app.get("/meetings")
    def list_meetings():
        out = []
        for m in store.list_meetings():
            d = dict(m)
            d["has_audio"] = bool(_meeting_tracks(store, m["id"]))
            out.append(d)
        return out

    @app.delete("/meetings/{mid}")
    def delete_meeting(mid: int):
        if store.get_meeting(mid) is None:
            raise HTTPException(404, "meeting not found")
        import shutil
        orphans = store.delete_meeting(mid)  # only dirs no meeting references anymore
        for d in orphans:  # store-reported segment dirs (trusted, app-created)
            if d and d not in ("/", ".") and os.path.isdir(d):
                shutil.rmtree(d, ignore_errors=True)
        # Remove only this meeting's cache wav FILES (never a shared segment dir).
        for t in _TRACKS:
            cache = f"data/{mid}/_play_{t}.wav"
            if os.path.exists(cache):
                os.remove(cache)
        return {"deleted": mid}

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
    def transcribe_meeting(mid: int, body: TranscribeIn = TranscribeIn()):
        # Re-run accurate ASR over the saved audio, optionally with a chosen model.
        # Sync route -> FastAPI runs it in a threadpool, so /health stays responsive.
        if store.get_meeting(mid) is None:
            raise HTTPException(404, "meeting not found")
        if body.model:
            import backends
            backend = backends.make_batch_backend(body.model)
        elif asr_backend is not None:
            backend = asr_backend
        else:
            raise HTTPException(503, "no ASR backend configured")
        store.clear_transcripts(mid, profile="accurate")  # replace, don't duplicate
        base = store.get_meeting(mid)["created_at"]
        n = 0
        for seg in store.list_segments(mid):
            off_ms = max(0, int((seg["started_at"] - base) * 1000))  # align to timeline
            for track_label, name in (("system", "system.pcm"), ("mic", "mic.pcm")):
                pcm = f"{seg['dir_path']}/{name}"
                if not os.path.exists(pcm) or os.path.getsize(pcm) == 0:
                    continue  # track not present for this segment
                for t in asr.transcribe(pcm, profile="accurate",
                                        track=track_label, backend=backend):
                    store.add_transcript(mid, t["profile"], t["track"],
                                         t["start_ms"] + off_ms, t["end_ms"] + off_ms,
                                         t["track"], t["text"])
                    n += 1
        return {"transcripts": n}

    @app.post("/meetings/{mid}/transcribe/start")
    def transcribe_start(mid: int, body: TranscribeIn = TranscribeIn()):
        # Background job + server-side progress -> survives a page refresh.
        if store.get_meeting(mid) is None:
            raise HTTPException(404, "meeting not found")
        if transcribe_jobs.get(mid, {}).get("state") == "running":
            return {"state": "running"}  # already in progress
        if body.model:
            import backends
            backend = backends.make_batch_backend(body.model)
        elif asr_backend is not None:
            backend = asr_backend
        else:
            raise HTTPException(503, "no ASR backend configured")
        store.clear_transcripts(mid, profile="accurate")
        import threading
        threading.Thread(target=_run_transcribe_job,
                         args=(store, mid, backend, transcribe_jobs),
                         daemon=True).start()
        return {"state": "started"}

    @app.get("/meetings/{mid}/transcribe/progress")
    def transcribe_progress(mid: int):
        return transcribe_jobs.get(mid, {"state": "idle"})

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
        # Heavy (minutes of transcribe+summary). Run off the event loop so /health
        # stays responsive — otherwise the supervisor sees a hang and kills it.
        result = await run_in_threadpool(
            run_pipeline, path, store=store, title=title, lang=lang,
            kind=kind, asr_backend=asr_backend,
            summary_backend=summary_backend, summary_model=summary_model)
        mid = result["meeting_id"]
        # Decode the upload to data/<mid>/mic.pcm (16k mono) so playback uses the
        # same path as live recordings (player + click-to-seek work). Best-effort.
        await run_in_threadpool(_save_upload_pcm, path, mid, store)
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

    @app.post("/meetings/merge")
    def merge_meetings(body: MergeIn):
        if len(body.ids) < 2:
            raise HTTPException(400, "need >=2 meetings to merge")
        return {"target": store.merge_into_earliest(body.ids)}

    @app.post("/meetings/merge-nearby")
    def merge_nearby(gap_min: float = 10.0):
        from store import group_by_proximity
        meetings = [dict(m) for m in store.list_meetings()]
        groups = group_by_proximity(meetings, gap_s=gap_min * 60)
        for g in groups:
            store.merge_into_earliest(g)
        return {"merged_groups": len(groups),
                "merged_meetings": sum(len(g) for g in groups)}

    @app.post("/meetings/{mid}/diarize")
    def diarize_meeting(mid: int, body: DiarizeIn):
        if store.get_meeting(mid) is None:
            raise HTTPException(404, "meeting not found")
        import diarize as diar

        segs = store.list_segments(mid)
        if not segs:
            raise HTTPException(404, "no saved audio for this meeting")
        pcm = os.path.join(segs[0]["dir_path"], f"{body.track}.pcm")
        if not os.path.exists(pcm):
            raise HTTPException(404, f"no audio for track {body.track}")
        try:
            segments = diar.diarize_pcm(pcm, num_speakers=body.num_speakers)
        except Exception as e:
            raise HTTPException(503, f"diarization unavailable: {e}")
        rows = [dict(r) for r in store.list_transcripts(mid)
                if r["track"] == body.track]
        for r in diar.assign_speakers(rows, segments, prefix=body.prefix):
            store.update_speaker(r["id"], r["speaker"])
        speakers = sorted({s["speaker"] for s in segments})
        return {"track": body.track, "speakers": len(speakers),
                "segments": len(segments)}

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

    # No eager startup probe: loading/validating models here can download GBs and
    # block boot past the supervisor's health-poll patience -> kill/restart loop.
    # The AdaptiveBackend lazy-loads on first use (in a threadpool, /health stays
    # responsive) and returns [] on a bad model, so a broken pick can't crash boot.
    print(f"[profile] live={live_model} fallback={fallback}", flush=True)

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
