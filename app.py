"""Web app: FastAPI. Binds 127.0.0.1 only (privacy, spec G2) — no auth by
design, loopback-only. Backends injected so tests run without MLX.

Phase 1 = batch loop: create meeting -> transcribe saved PCM -> summarize -> view.
Live websocket + recording start/stop (Swift helper) land in Phase 2."""
import asyncio
import html
import json
import os
import shutil
import subprocess
import sys
import time
from typing import Optional  # not `X | None` — pydantic evals it, fails on 3.9 venvs
from urllib.parse import quote

from fastapi import (FastAPI, File, Form, HTTPException, UploadFile, WebSocket,
                     WebSocketDisconnect)
from fastapi.responses import (FileResponse, HTMLResponse, RedirectResponse,
                               Response)
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

import asr
import backends  # module-level: every `backends.x` call resolves (avoids per-fn import + NameError)
from live import TwoPassSession
from summarize import summarize

# Shared design system — calm zh-TW productivity aesthetic. Served pages (not a
# claude.ai Artifact), so a full self-styled doc is fine.
_STYLE = """
*{box-sizing:border-box}
:root{--bg:#f5f5f7;--surface:#fff;--surface2:#f0f0f3;--ink:#1d1d1f;--muted:#86868b;
 --line:#e3e3e8;--accent:#5b54e6;--accent2:#7c75ff;--accentsoft:#ecebfd;
 --me:#1565c0;--other:#2e7d32;--danger:#e5484d;--ok:#1e9e57;
 --radius:20px;--radius-sm:12px;--radius-lg:28px;
 --ease:cubic-bezier(.32,.72,0,1);
 --shadow:0 .5px 1.5px rgba(16,24,40,.04),0 6px 22px -12px rgba(16,24,40,.16);
 --shadow-lg:0 1px 2px rgba(16,24,40,.05),0 28px 64px -22px rgba(16,24,40,.26);
 --capbg:#1d1d1f;--capink:#f5f5f7}
:root[data-theme=dark]{
 --bg:#000;--surface:#1c1c1e;--surface2:#2c2c2e;--ink:#f5f5f7;--muted:#98989d;
 --line:#38383a;--accent:#9b96ff;--accent2:#b3aeff;--accentsoft:#27263d;
 --me:#6db3ff;--other:#74d99a;--danger:#ff6b6f;
 --shadow:0 .5px 1.5px rgba(0,0,0,.6),0 10px 30px -16px rgba(0,0,0,.7);
 --shadow-lg:0 1px 2px rgba(0,0,0,.6),0 28px 66px -24px rgba(0,0,0,.85);
 --capbg:#1c1c1e;--capink:#f5f5f7}
body{margin:0;min-height:100vh;background:var(--bg);color:var(--ink);-webkit-font-smoothing:antialiased;
 text-rendering:optimizeLegibility;
 font:16px/1.6 -apple-system,BlinkMacSystemFont,"SF Pro Text","PingFang TC","Noto Sans TC","Segoe UI",Roboto,sans-serif}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
.wrap{max-width:920px;margin:0 auto;padding:18px 20px 96px}
header.top{display:flex;align-items:center;gap:12px;margin:2px 0 26px;position:sticky;top:0;z-index:40;
 padding:12px 6px;backdrop-filter:saturate(1.8) blur(20px);-webkit-backdrop-filter:saturate(1.8) blur(20px);
 background:color-mix(in srgb,var(--bg) 72%,transparent);border-bottom:1px solid color-mix(in srgb,var(--line) 60%,transparent)}
.brand{font-weight:800;font-size:18px;letter-spacing:-.02em}
.brand .dot{color:var(--accent)}
.spacer{flex:1}
h1{font-size:34px;font-weight:800;letter-spacing:-.035em;margin:.1em 0 .7em;line-height:1.1}
h2{font-size:17px;font-weight:760;margin:28px 0 13px;letter-spacing:-.02em}
h3{font-size:12px;font-weight:760;margin:14px 0 6px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}
.card{background:var(--surface);border:1px solid var(--line);border-radius:var(--radius);
 box-shadow:var(--shadow);padding:22px 24px;margin:16px 0}
.muted{color:var(--muted)}.small{font-size:13px}
.row{display:flex;flex-wrap:wrap;gap:12px;align-items:end}
.row label.chk{align-self:center}
label.fld{display:inline-flex;flex-direction:column;gap:5px;font-size:11px;color:var(--muted);
 font-weight:700;text-transform:uppercase;letter-spacing:.04em}
label.chk{display:inline-flex;align-items:center;gap:6px;font-size:13px;color:var(--ink);text-transform:none}
input[type=text],input[type=search],input[type=file],input:not([type]),select,textarea{font:inherit;
 padding:.58em .7em;border:1px solid var(--line);border-radius:var(--radius-sm);
 background:var(--surface2);color:var(--ink);transition:.12s}
textarea{resize:vertical}
input:hover,select:hover,textarea:hover{border-color:color-mix(in srgb,var(--accent) 40%,var(--line))}
input:focus,select:focus,textarea:focus{outline:0;border-color:var(--accent);box-shadow:0 0 0 3px var(--accentsoft)}
input[type=search]{-webkit-appearance:none;appearance:none}
/* custom chevron so the native select arrow doesn't look unstyled */
select{cursor:pointer;-webkit-appearance:none;appearance:none;padding-right:2em;
 background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='%238b93a4' stroke-width='3' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M6 9l6 6 6-6'/%3E%3C/svg%3E");
 background-repeat:no-repeat;background-position:right .7em center}
/* the native file input "Choose File" button is the ugliest default — make it a chip */
input[type=file]{cursor:pointer;padding:.32em .5em}
input[type=file]::file-selector-button{font:inherit;font-weight:650;margin-right:10px;
 padding:.42em .9em;border:1px solid var(--line);border-radius:8px;background:var(--surface);
 color:var(--ink);cursor:pointer;transition:.13s}
input[type=file]::file-selector-button:hover{border-color:var(--accent);color:var(--accent)}
.btn{font:inherit;font-weight:600;padding:.55em 1.05em;border-radius:var(--radius-sm);
 border:1px solid var(--line);background:var(--surface);color:var(--ink);cursor:pointer;
 transition:transform .2s var(--ease),border-color .15s,color .15s,filter .15s,box-shadow .15s,background .15s;
 display:inline-block;line-height:1.25}
.btn:hover{border-color:var(--accent);color:var(--accent);transform:translateY(-1px)}
.btn:active{transform:scale(.96)}
.btn:disabled{opacity:.45;cursor:not-allowed;transform:none;border-color:var(--line);color:var(--muted)}
.btn.primary{background:var(--accent);border-color:transparent;border-radius:980px;
 color:#fff;padding:.6em 1.3em;box-shadow:0 8px 20px -8px var(--accent)}
.btn.primary:hover{background:var(--accent2);color:#fff;transform:translateY(-1px)}
.btn.danger{color:var(--danger);border-color:color-mix(in srgb,var(--danger) 35%,var(--line))}
.btn.danger:hover{background:var(--danger);color:#fff;border-color:var(--danger)}
.badge{display:inline-block;font-size:11px;font-weight:750;padding:.2em .65em;border-radius:999px;
 background:var(--surface2);color:var(--muted);border:1px solid var(--line)}
.badge.live{background:color-mix(in srgb,var(--danger) 15%,transparent);color:var(--danger);border-color:transparent}
.badge.done{background:color-mix(in srgb,var(--ok) 15%,transparent);color:var(--ok);border-color:transparent}
.tg{display:inline-block;font-size:11px;color:var(--muted);border:1px solid var(--line);
 border-radius:999px;padding:1px .55em;margin-left:5px;white-space:nowrap}
.tgbtn{cursor:pointer;margin:0}
.tgbtn:hover{border-color:var(--accent);color:var(--accent)}
.tgbtn.on{background:var(--accent);color:#fff;border-color:transparent}
.tg .x{cursor:pointer;margin-left:5px;opacity:.6}
.tg .x:hover{opacity:1;color:var(--danger)}
details.menu{position:relative;display:inline-block}
details.menu>summary{list-style:none}
details.menu>summary::-webkit-details-marker{display:none}
details.menu>summary::marker{content:""}
.menupop{position:absolute;z-index:30;top:calc(100% + 4px);right:0;left:auto;display:flex;flex-direction:column;
 gap:6px;padding:8px;min-width:170px;background:var(--surface);border:1px solid var(--line);
 border-radius:var(--radius-sm);box-shadow:0 10px 28px -10px rgba(0,0,0,.45)}
.menupop .btn{width:100%;text-align:left}
ul.meetings{list-style:none;margin:0;padding:0}
ul.meetings li{display:flex;align-items:center;gap:11px;padding:13px 6px;border-radius:var(--radius-sm);
 border-bottom:1px solid var(--line)}
ul.meetings li:last-child{border:0}ul.meetings li:hover{background:var(--surface2)}
ul.meetings a{font-weight:650;flex:1}
/* richer history rows */
.mdate{font-size:11px;font-weight:800;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;
 margin:16px 4px 4px;padding-top:8px;border-top:1px solid var(--line)}
.mdate:first-child{border-top:0;margin-top:4px}
li.mrow{padding:11px 8px}
.mico{font-size:15px;flex:none;width:1.4em;text-align:center}
.mmain{flex:1;min-width:0;display:flex;flex-direction:column;gap:2px}
.mmain .mtitle{font-weight:650;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.mmeta{font-size:12px;color:var(--muted);display:flex;align-items:center;gap:7px;flex-wrap:wrap}
.mmeta .dotsep{opacity:.5}
.mact{flex:none;display:flex;gap:4px;opacity:0;transition:.12s}
li.mrow:hover .mact,li.mrow:focus-within .mact{opacity:1}
.mact .btn{padding:.28em .5em;font-size:13px}
.msel{flex:none;width:16px;height:16px;accent-color:var(--accent);display:none}
ul.meetings.picking .msel{display:inline-block}
ul.meetings.picking .mact{display:none}
.selbar{display:none;align-items:center;gap:10px;margin:0 0 8px;padding:8px 10px;
 background:var(--accentsoft);border-radius:var(--radius-sm)}
.selbar.on{display:flex}
.caption{font-size:clamp(22px,4.2vw,34px);font-weight:800;line-height:1.34;background:var(--capbg);
 color:var(--capink);border-radius:var(--radius);padding:20px 24px;min-height:1.4em;margin:16px 0 8px;
 box-shadow:inset 0 0 0 1px rgba(255,255,255,.05)}
.caption:empty{display:none}  /* no ugly black bar before the first caption */
.liveline{color:var(--muted);font-size:17px;min-height:1.5em;margin:6px 2px 14px}
.liveline:empty{display:none}
.tline{padding:9px 2px;border-bottom:1px solid var(--line);display:flex;gap:12px}
.tline:last-child{border:0}
.tline .ts{color:var(--muted);font-size:12px;font-variant-numeric:tabular-nums;white-space:nowrap;padding-top:3px}
.tline .who{font-weight:750}
table.tx{width:100%;border-collapse:collapse}
table.tx th{text-align:left;font-size:11px;color:var(--muted);font-weight:700;text-transform:uppercase;
 letter-spacing:.04em;padding:8px 10px;border-bottom:1px solid var(--line)}
table.tx td{padding:10px;border-bottom:1px solid var(--line);vertical-align:top}
table.tx tr:last-child td{border-bottom:0}
table.tx td.who{white-space:nowrap;font-weight:650;width:96px}
table.tx td.ts{white-space:nowrap;color:var(--muted);font-size:12px;font-variant-numeric:tabular-nums;width:52px}
table.tx tr[data-ts]{cursor:pointer}
table.tx tr[data-ts]:hover{background:var(--surface2)}
table.tx tr.active{background:var(--accentsoft)!important;box-shadow:inset 3px 0 0 var(--accent)}
pre.sum{white-space:pre-wrap;font:inherit;background:var(--surface2);border:1px solid var(--line);
 border-radius:var(--radius-sm);padding:16px;margin:8px 0;overflow-x:auto}
.md{background:var(--surface2);border:1px solid var(--line);border-radius:var(--radius-sm);
 padding:4px 18px;margin:8px 0;overflow-x:auto}
.md h1{font-size:20px;margin:.7em 0 .4em}.md h2{font-size:16px;margin:.9em 0 .4em}
.md h3{font-size:13px;color:var(--muted);text-transform:none;margin:.8em 0 .3em}
.md h4,.md h5,.md h6{font-size:13px;margin:.7em 0 .3em}
.md p{margin:.4em 0}.md ul,.md ol{margin:.4em 0;padding-left:1.5em}.md li{margin:.2em 0}
.md code{background:var(--surface);border:1px solid var(--line);border-radius:5px;padding:.05em .4em;font-size:.92em}
.md strong{font-weight:750}
.hint{color:var(--muted);font-size:13px;line-height:1.55}
audio{filter:saturate(.9)}:root[data-theme=dark] audio{filter:invert(.92) hue-rotate(180deg)}
.card.sticky{position:sticky;top:60px;z-index:20;backdrop-filter:blur(8px);
 background:color-mix(in srgb,var(--surface) 92%,transparent)}
.progpop{position:fixed;right:18px;bottom:18px;width:330px;max-width:calc(100vw - 36px);
 display:flex;flex-direction:column;gap:8px;z-index:200}
.progpop[hidden]{display:none}
.pghead{font-size:12px;font-weight:750;color:var(--muted)}
.pgcard{background:var(--surface);border:1px solid var(--line);border-radius:var(--radius-sm);
 box-shadow:0 12px 32px -8px rgba(0,0,0,.3);padding:11px 13px;animation:pgin .25s ease}
@keyframes pgin{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
.pgline{display:flex;align-items:center;gap:8px;margin-bottom:5px}
.pgkind{font-size:11px;font-weight:750;color:#fff;background:var(--accent);
 border-radius:999px;padding:.12em .62em;flex:none}
.pgkind.done{background:#3a9d6e}.pgkind.err{background:#d9534f}
.pgtitle{font-size:13px;font-weight:650;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.pgmeta{font-size:11px;color:var(--muted);margin-bottom:7px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.pgrow{display:flex;align-items:center;gap:9px}
.pgbar{flex:1;height:9px;background:var(--surface2);border-radius:999px;overflow:hidden}
.pgbar i{display:block;height:100%;width:0;border-radius:999px;transition:width .4s ease;
 background:linear-gradient(90deg,var(--accent2),var(--accent))}
.pgbar.indet i{width:42%;animation:prind 1.15s infinite ease-in-out}
.pgbar.done i{background:#3a9d6e}
.pgpct{font-size:13px;font-weight:800;font-variant-numeric:tabular-nums;min-width:38px;text-align:right}
@keyframes prind{0%{margin-left:-42%}100%{margin-left:100%}}
"""


_THEME_JS = (
    "const _r=document.documentElement;"
    "_r.dataset.theme=localStorage.theme||(matchMedia('(prefers-color-scheme:dark)').matches?'dark':'light');"
    "function _toggleTheme(){const t=_r.dataset.theme==='dark'?'light':'dark';"
    "_r.dataset.theme=t;localStorage.theme=t;"
    "document.getElementById('themebtn').textContent=t==='dark'?'🌙':'☀️';}"
    "addEventListener('DOMContentLoaded',()=>{const b=document.getElementById('themebtn');"
    "if(b)b.textContent=_r.dataset.theme==='dark'?'🌙':'☀️';});"
    "async function _quitApp(){if(!confirm('結束服務並關閉 app？(錄製中的會議會停止)'))return;"
    "try{await fetch('/shutdown',{method:'POST'});}catch(e){}"
    "document.body.innerHTML='<div style=\"font:18px sans-serif;padding:60px;text-align:center\">"
    "✅ 已關閉服務，可關閉此視窗。</div>';}"
)


def _png_solid(size, rgb):
    """Minimal solid-color PNG (pure stdlib) — PWA icon, no image deps."""
    import struct
    import zlib

    def chunk(typ, data):
        c = typ + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xffffffff)
    ihdr = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)  # 8-bit RGB
    row = b"\x00" + bytes(rgb) * size
    idat = zlib.compress(row * size, 9)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", idat) + chunk(b"IEND", b""))


_MANIFEST = (
    '{"name":"Meeting·Summary","short_name":"Meeting","start_url":"/",'
    '"display":"standalone","background_color":"#0c0e13","theme_color":"#5b54e6",'
    '"icons":[{"src":"/icon-192.png","sizes":"192x192","type":"image/png"},'
    '{"src":"/icon-512.png","sizes":"512x512","type":"image/png","purpose":"any maskable"}]}'
)
# Minimal SW: a fetch handler (required for installability); network passthrough —
# the app needs the live server, so we don't cache dynamic responses.
_SW_JS = ("self.addEventListener('install',e=>self.skipWaiting());"
          "self.addEventListener('activate',e=>self.clients.claim());"
          "self.addEventListener('fetch',e=>{});")


# Meeting detection, surfaced IN the web UI (reliable) instead of via osascript
# (which is silently suppressed by macOS notification permission/Focus). Every page
# polls /detect; on a meeting (mic in use / meeting app) while not recording, shows a
# top banner + a Web Notification (PWA, permission-based — actually visible).
_DETECT_JS = (
    # Web Notifications need permission, and requestPermission() is IGNORED by
    # Safari (and unreliable in Chrome) unless it runs from a USER GESTURE — a
    # page-load call silently no-ops, so permission stays 'default', the granted
    # branch never fires, and no notification ever shows (works in some browsers,
    # not others). Ask on the first click/keydown instead. Per-origin: a grant on
    # one port (the .app's 8765) does NOT carry to another (a dev port).
    "function _detAsk(){if(window.Notification&&Notification.permission==='default')"
    "{try{Notification.requestPermission();}catch(e){}}}"
    "if(window.Notification&&Notification.permission==='default'){"
    "document.addEventListener('click',_detAsk,{once:true});"
    "document.addEventListener('keydown',_detAsk,{once:true});}"
    "let _detSeen=false,_detRec=false;"
    "async function _detTick(){try{const d=await(await fetch('/detect')).json();"
    "_detRec=!!d.recording;"
    "const _ri=document.getElementById('_recind');"
    "if(_ri)_ri.style.display=d.recording?'inline-flex':'none';"
    "const bar=document.getElementById('_detbar');if(bar){"
    "const show=d.meeting&&!d.recording&&sessionStorage.getItem('_detX')!=='1';"
    "bar.style.display=show?'flex':'none';"
    "if(show)document.getElementById('_detapp').textContent=d.app||'麥克風使用中';"
    "if(!d.meeting){sessionStorage.removeItem('_detX');_detSeen=false;}"
    "if(show&&!_detSeen){_detSeen=true;"
    "if(window.Notification&&Notification.permission==='granted')"
    "{try{new Notification('📝 偵測到會議',{body:(d.app||'麥克風使用中')+' — 點開始錄音'});}catch(e){}}}"
    "}}catch(e){}"
    "setTimeout(_detTick,_detRec?30000:10000);}"
    "function _detDismiss(){sessionStorage.setItem('_detX','1');"
    "document.getElementById('_detbar').style.display='none';}"
    "_detTick();")


# Global progress popout: on EVERY page, polls /jobs (all in-flight transcribe/
# diarize/summary across meetings) and renders a stack of cards with a prominent
# progress bar + %. Dispatches a 'meetingjobs' window event each tick so pages can
# react (home paints row chips; the meeting page reloads / renders on completion).
# window._jobsTick() forces an immediate poll (buttons call it right after start).
_PROG_JS = (
    "(function(){let prev=new Set(),timer=null,lastSig='';"
    "function esc(s){return (s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}"
    "function card(j){const pct=j.total?Math.round(j.done/j.total*100):null;"
    "const bar=pct!=null?`<div class=pgbar><i style=\"width:${pct}%\"></i></div>"
    "<span class=pgpct>${pct}%</span>`:`<div class=\"pgbar indet\"><i></i></div>`;"
    "return `<div class=pgcard><div class=pgline><span class=pgkind>${esc(j.kind)}</span>"
    "<span class=pgtitle>${esc(j.title||('#'+j.mid))}</span></div>"
    "<div class=pgmeta>${esc(j.text||'處理中…')}${j.total?(' · '+j.done+'/'+j.total):''}</div>"
    "<div class=pgrow>${bar}</div></div>`;}"
    "function key(j){return j.mid+'/'+j.kind;}"
    # Rebuild the DOM only when the SET of jobs changes (so the slide-in animation
    # fires once, not every poll = the flicker). Otherwise update bar/%/text IN
    # PLACE — width transitions smoothly, the indeterminate animation isn't restarted.
    "function render(jobs){const el=document.getElementById('progpop');if(!el)return;"
    "if(!jobs.length){el.hidden=true;el.innerHTML='';lastSig='';return;}"
    # sig includes D/I (has-total) so a job that flips indeterminate->determinate
    # (diarize: total unknown for the first tick, then 27/1361) rebuilds into a real
    # bar instead of staying a looping animation. NOTE: the event-diff key (mid/kind)
    # stays identity-only, so this never reads as a false 'finished'.
    "el.hidden=false;const sig=jobs.map(j=>key(j)+(j.total?'D':'I')).join('|');"
    "if(sig!==lastSig){el.innerHTML=`<div class=pghead>⏳ ${jobs.length} 項處理中</div>`"
    "+jobs.map(card).join('');lastSig=sig;return;}"
    "const cards=el.querySelectorAll('.pgcard');"
    "jobs.forEach((j,i)=>{const c=cards[i];if(!c)return;"
    "const pct=j.total?Math.round(j.done/j.total*100):null;"
    "const meta=c.querySelector('.pgmeta');"
    "if(meta)meta.textContent=(j.text||'處理中…')+(j.total?(' · '+j.done+'/'+j.total):'');"
    "const f=c.querySelector('.pgbar i'),p=c.querySelector('.pgpct');"
    "if(pct!=null){if(f)f.style.width=pct+'%';if(p)p.textContent=pct+'%';}});}"
    "function tick(){fetch('/jobs').then(r=>r.json()).then(d=>{const jobs=d.jobs||[];render(jobs);"
    "const cur=new Set(jobs.map(key));"
    "const finished=[...prev].filter(k=>!cur.has(k)),started=[...cur].filter(k=>!prev.has(k));"
    "prev=cur;"
    "window.dispatchEvent(new CustomEvent('meetingjobs',{detail:{jobs,finished,started}}));"
    "schedule(jobs.length?1500:4000);}).catch(()=>schedule(4000));}"
    "function schedule(ms){clearTimeout(timer);timer=setTimeout(tick,ms);}"
    "window._jobsTick=tick;tick();})();")


# Global quick-record: a floating button on every page (except /live). Click opens
# a small options panel (source / live model / record-only); then it captures in
# place (mic, system, or dual — same 16k box-averaged PCM + tag-byte protocol as
# /live) over its own ws. Self-contained. No bare // comments (ships in the shell,
# scanned by the single-line-script guard).
_REC_JS = (
    "(function(){if(location.pathname==='/live')return;"
    "let gws=null,gctx=null,nodes=[],streams=[],t0=0,timer=null,open=false,_busy=false;"
    "const fab=document.createElement('div');fab.id='_recfab';"
    "fab.style.cssText='position:fixed;right:18px;bottom:18px;z-index:300;display:flex;"
    "flex-direction:column;align-items:flex-end;gap:8px';document.body.appendChild(fab);"
    "function fmt(s){s=Math.floor(s);return (s/60|0)+':'+('0'+s%60).slice(-2);}"
    "const RB='border-radius:24px;padding:.6em 1.1em;box-shadow:0 10px 26px -10px rgba(0,0,0,.55)';"
    "const RED='background:#c0392b;border-color:#c0392b;color:#fff;font-weight:700;';"
    "function panel(){return `<div style=\"background:var(--surface);border:1px solid var(--line);"
    "border-radius:18px;box-shadow:var(--shadow-lg);padding:14px;width:236px;display:flex;"
    "flex-direction:column;gap:11px\"><div style=\"font-weight:760;font-size:14px\">快速錄音</div>"
    "<label class=fld>來源<select id=_rsrc><option value=mic>麥克風(我)</option>"
    "<option value=system>系統音(對方)</option><option value=dual>兩者</option></select></label>"
    "<label class=fld>即時模型<select id=_rmodel>"
    "<option value=\"qwen3-asr-0.6b-q4-k-m\">Qwen3-ASR 0.6B(快)</option>"
    "<option value=\"mlx-community/whisper-small-mlx-q4\">whisper small(省)</option>"
    "<option value=\"mlx-community/whisper-large-v3-turbo-q4\">whisper turbo(準)</option></select></label>"
    "<label class=chk><input type=checkbox id=_rro> 🪫 純錄音(不即時辨識)</label>"
    "<button class=\"btn primary\" id=_rgo style=\"width:100%\">● 開始錄音</button></div>`;}"
    "function renderIdle(elsewhere){fab.innerHTML=(open?panel():'')+"
    "(elsewhere?`<a class=btn href=/live style=\"${RED}${RB}\">● 錄音中（前往）</a>`"
    ":`<button class=btn id=_rtoggle style=\"${RB}\">● 快速錄音</button>`);"
    "const tg=document.getElementById('_rtoggle');if(tg)tg.onclick=()=>{open=!open;renderIdle(false);};"
    "const go=document.getElementById('_rgo');if(go)go.onclick=begin;}"
    "function renderRec(){open=false;fab.innerHTML=`<button class=btn id=_rstop style=\"${RED}${RB}\">"
    "■ 停止 <span id=_rt>0:00</span></button>`;document.getElementById('_rstop').onclick=stop;}"
    "async function begin(){if(_busy||gws)return;_busy=true;"  # guard double-start race
    "const src=document.getElementById('_rsrc').value;"
    "const model=document.getElementById('_rmodel').value,ro=document.getElementById('_rro').checked;"
    "try{if(src==='mic')streams=[await navigator.mediaDevices.getUserMedia({audio:true})];"
    "else if(src==='system'){const s=await navigator.mediaDevices.getDisplayMedia({video:true,audio:true});"
    "s.getVideoTracks().forEach(t=>t.stop());"
    "if(!s.getAudioTracks().length){alert('未取得系統音(分享時要勾「分享音訊」)');_busy=false;return;}streams=[s];}"
    "else{const mic=await navigator.mediaDevices.getUserMedia({audio:true});"
    "const sys=await navigator.mediaDevices.getDisplayMedia({video:true,audio:true});"
    "sys.getVideoTracks().forEach(t=>t.stop());streams=[mic,sys];}}"
    "catch(e){alert('無法取得音訊：'+e.message);_busy=false;return;}"
    "try{await fetch('/models',{method:'POST',headers:{'Content-Type':'application/json'},"
    "body:JSON.stringify({live:model})});}catch(e){}"
    "try{gctx=new AudioContext({sampleRate:16000});}catch(e){gctx=new AudioContext();}"
    "const ratio=gctx.sampleRate/16000,dual=(src==='dual');"
    "gws=new WebSocket(`ws://${location.host}/ws/live?src=${src}${ro?'&record_only=1':''}`);"
    "gws.binaryType='arraybuffer';"
    "gws.onopen=()=>{const g=gctx.createGain();g.gain.value=0;g.connect(gctx.destination);"
    "streams.forEach((st,i)=>{const node=gctx.createScriptProcessor(4096,1,1);"
    "gctx.createMediaStreamSource(st).connect(node);node.connect(g);const tag=dual?i:null;"
    "node.onaudioprocess=ev=>{if(!gws||gws.readyState!==1)return;"
    "const inp=ev.inputBuffer.getChannelData(0),outLen=Math.floor(inp.length/ratio),pcm=new Int16Array(outLen);"
    "for(let k=0;k<outLen;k++){const a=Math.floor(k*ratio),b=Math.floor((k+1)*ratio);let sm=0,n=0;"
    "for(let j=a;j<b&&j<inp.length;j++){sm+=inp[j];n++;}"
    "const v=Math.max(-1,Math.min(1,n?sm/n:0));pcm[k]=v*32767;}"
    "if(tag===null)gws.send(pcm.buffer);"
    "else{const bb=new Uint8Array(1+pcm.byteLength);bb[0]=tag;bb.set(new Uint8Array(pcm.buffer),1);gws.send(bb.buffer);}};"
    "nodes.push(node);});t0=Date.now();renderRec();clearInterval(timer);"
    "clearInterval(timer);timer=setInterval(()=>{const el=document.getElementById('_rt');if(el)el.textContent=fmt((Date.now()-t0)/1000);},1000);};"
    "gws.onclose=()=>cleanup();gws.onerror=()=>cleanup();}"
    "function stop(){if(gws){try{gws.close();}catch(e){}}cleanup();}"
    "function cleanup(){_busy=false;clearInterval(timer);timer=null;"
    "nodes.forEach(n=>{n.onaudioprocess=null;try{n.disconnect();}catch(e){}});nodes=[];"
    "streams.forEach(s=>s.getTracks().forEach(t=>t.stop()));streams=[];"
    "if(gctx){try{gctx.close();}catch(e){}gctx=null;}gws=null;refresh();}"
    "async function refresh(){if(gws){renderRec();return;}"
    "try{const d=await(await fetch('/detect')).json();renderIdle(!!d.recording);}catch(e){renderIdle(false);}}"
    "refresh();setInterval(()=>{if(!gws&&!open)refresh();},10000);})();")


def _md_html(text):
    """Minimal, safe Markdown -> HTML (offline, no dep) for changelog + summaries.
    HTML-escapes first, then headings / **bold** / `code` / - and 1. lists / paras."""
    import re  # noqa: PLC0415

    def inline(s):
        s = html.escape(s)
        s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
        return re.sub(r"`([^`]+?)`", r"<code>\1</code>", s)

    out, list_tag = [], None

    def close():
        nonlocal list_tag
        if list_tag:
            out.append(f"</{list_tag}>")
            list_tag = None

    for ln in (text or "").split("\n"):
        h = re.match(r"^(#{1,6})\s+(.*)$", ln)
        ul = re.match(r"^\s*[-*]\s+(.*)$", ln)
        ol = re.match(r"^\s*\d+[.)]\s+(.*)$", ln)
        if h:
            close()
            lvl = len(h.group(1))
            out.append(f"<h{lvl}>{inline(h.group(2))}</h{lvl}>")
        elif ul or ol:
            tag = "ul" if ul else "ol"
            if list_tag != tag:
                close()
                out.append(f"<{tag}>")
                list_tag = tag
            out.append(f"<li>{inline((ul or ol).group(1))}</li>")
        elif not ln.strip():
            close()
        else:
            close()
            out.append(f"<p>{inline(ln)}</p>")
    close()
    return "\n".join(out)


def _shell(title, body, script="", back=False):
    nav = '<a href="/">&larr; 回首頁</a>' if back else ''
    return (
        "<!doctype html><html lang=zh-Hant><head><meta charset=utf-8>"
        "<meta name=viewport content='width=device-width,initial-scale=1'>"
        "<link rel=manifest href='/manifest.webmanifest'>"
        "<meta name=theme-color content='#5b54e6'>"
        "<meta name=apple-mobile-web-app-capable content=yes>"
        "<link rel=apple-touch-icon href='/icon-192.png'>"
        f"<title>{title}</title><style>{_STYLE}</style>"
        f"<script>{_THEME_JS}</script>"
        "<script>if('serviceWorker' in navigator)"
        "addEventListener('load',()=>navigator.serviceWorker.register('/sw.js').catch(()=>{}));</script>"
        "</head><body>"
        "<div id=_detbar style=\"display:none;position:fixed;inset:0 0 auto 0;z-index:200;"
        "align-items:center;gap:12px;justify-content:center;padding:10px 16px;"
        "background:linear-gradient(180deg,var(--accent2),var(--accent));color:#fff;"
        "font-weight:650;box-shadow:0 6px 20px -8px rgba(0,0,0,.5)\">"
        "<span>📝 偵測到會議（<span id=_detapp></span>）</span>"
        "<a href='/live' style='color:#fff;text-decoration:underline'>開始錄音 →</a>"
        "<button onclick=_detDismiss() style='background:rgba(255,255,255,.2);border:0;"
        "color:#fff;border-radius:8px;padding:.2em .6em;cursor:pointer'>✕</button></div>"
        "<div class=wrap>"
        "<header class=top><span class=brand>📝 Meeting<span class=dot>·</span>Summary</span>"
        "<a id=_recind href=/live class=btn title='錄音進行中 — 點此回到錄音頁' "
        "style='display:none;background:#c0392b;border-color:#c0392b;color:#fff;font-weight:700;"
        "padding:.4em .7em'>● 錄音中</a>"
        "<span class=spacer></span>"
        "<button class=btn id=themebtn onclick=_toggleTheme() title='切換深淺色' "
        "style='padding:.4em .6em'>🌓</button>"
        "<button class='btn danger' onclick=_quitApp() title='結束服務' "
        "style='padding:.4em .6em'>⏻</button>"
        f"{nav}</header>{body}</div>"
        "<div id=progpop class=progpop hidden></div>"
        + (f"<script>{script}</script>" if script else "")
        + f"<script>{_DETECT_JS}</script>"
        + f"<script>{_PROG_JS}</script>"
        + f"<script>{_REC_JS}</script>"
        + "</body></html>"
    )


_INDEX = _shell("MeetingSummary", """
<h1>本地會議轉錄 · 摘要</h1>
<div class=row style="margin-bottom:4px">
<a class="btn primary" href="/live">🔴 開始 Live 即時逐字稿</a>
<a class="btn" href="/speakers/manage">👥 語者</a>
<a class="btn" href="/models/manage">⚙️ 設定</a>
<button class="btn" onclick="fetch('/floatpanel/open',{method:'POST'}).then(r=>r.json()).then(j=>{if(j&&j.detail)alert(j.detail)}).catch(()=>{})" title="置頂浮動控制面板（需先在設定安裝）">🪟 浮動面板</button>
</div>
<h2>上傳音檔</h2>
<div class=card>
  <form action="/ingest" method="post" enctype="multipart/form-data" class=row>
    <label class=fld>音檔 (wav/m4a/mp3)<input type=file name=audio required></label>
    <label class=fld>標題<input name=title value="實測"></label>
    <label class=fld>摘要型式<select name=kind>
      <option value=minutes>會議記錄 minutes</option>
      <option value=bullets>條列 bullets</option>
      <option value=actions>行動項目 actions</option></select></label>
    <button class="btn primary" type=submit>上傳並產生摘要</button>
  </form>
  <p class=hint style="margin:.8em 0 0">上傳後直接跳到該會議頁，辨識 + 摘要在背景進行（頁面會即時顯示進度、完成自動刷新）。第一次會下載模型。</p>
</div>
<h2>會議紀錄</h2>
<div class=card>
  <div class=row style="margin-bottom:8px">
    <input id=q type=search placeholder="搜尋標題 / 逐字稿 / 摘要…" style="flex:1;min-width:200px">
    <button class=btn id=pickbtn>選取</button>
    <button class=btn id=mergebtn>整合相近的 live 會議</button>
    <span class="muted small" id=mergemsg></span>
    <span class=spacer></span>
    <span class="badge live" id=jobhdr style="display:none"></span>
  </div>
  <div id=selbar class=selbar>
    <span id=selcount class="small">已選 0</span>
    <span class=spacer></span>
    <button class=btn id=selmerge>合併</button>
    <button class="btn danger" id=seldel>刪除</button>
    <button class=btn id=selcancel>取消</button>
  </div>
  <div id=tagbar class=row style="gap:6px;margin-bottom:8px"></div>
  <ul class=meetings id=meetings></ul>
</div>
""", script="""
let ALL=[], TAGFILTER=null, PICK=false; const SEL=new Set();
function esc(s){return (s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function chips(tags){return (tags||[]).map(t=>`<span class=tg>#${esc(t)}</span>`).join('');}
function fmtDur(s){s=Math.round(s||0);if(s<60)return s+'秒';const m=Math.round(s/60);return m<60?m+'分':(m/60).toFixed(1)+'時';}
function dayKey(ts){const d=new Date(ts*1000),n=new Date();
  const a=new Date(d.getFullYear(),d.getMonth(),d.getDate()),b=new Date(n.getFullYear(),n.getMonth(),n.getDate());
  const diff=Math.round((b-a)/86400000);
  if(diff<=0)return '今天';if(diff===1)return '昨天';if(diff<7)return diff+' 天前';
  return d.getFullYear()+'-'+String(d.getMonth()+1).padStart(2,'0')+'-'+String(d.getDate()).padStart(2,'0');}
function hhmm(ts){const d=new Date(ts*1000);return String(d.getHours()).padStart(2,'0')+':'+String(d.getMinutes()).padStart(2,'0');}
function rowHtml(m){
  const meta=[hhmm(m.created_at)];
  if(m.duration_s>0)meta.push(fmtDur(m.duration_s));
  meta.push(`<span class="badge ${m.status==='finalized'?'done':'live'}">${m.status}</span>`);
  return `<li class=mrow data-id="${m.id}">
    <input type=checkbox class=msel data-id="${m.id}"${SEL.has(m.id)?' checked':''}>
    <span class=mico title="${m.has_audio?'有音檔':'無音檔'}">${m.has_audio?'🔊':'🔇'}</span>
    <div class=mmain><a class=mtitle href="/m/${m.id}">${esc(m.title)}</a>
      <div class=mmeta>${meta.join('<span class=dotsep>·</span>')}${chips(m.tags)}<span class=mjob data-id="${m.id}"></span></div></div>
    <div class=mact><button class="btn mdel" data-id="${m.id}" title=刪除>🗑</button></div></li>`;}
function render(list){const el=document.getElementById('meetings');el.classList.toggle('picking',PICK);
  if(!list.length){el.innerHTML='<li class="muted small">尚無會議</li>';return;}
  let html='',last=null;
  for(const m of list){const k=dayKey(m.created_at);if(k!==last){html+=`<div class=mdate>${k}</div>`;last=k;}html+=rowHtml(m);}
  el.innerHTML=html;
  el.querySelectorAll('.mdel').forEach(b=>b.onclick=async(e)=>{e.preventDefault();e.stopPropagation();
    const id=+b.dataset.id;if(!confirm('刪除這場會議?逐字稿與音檔都會移除,無法復原。'))return;
    await fetch('/meetings/'+id,{method:'DELETE'});ALL=ALL.filter(x=>x.id!==id);SEL.delete(id);applyView();renderTagbar();});
  el.querySelectorAll('.msel').forEach(c=>{c.onclick=e=>{e.stopPropagation();
    c.checked?SEL.add(+c.dataset.id):SEL.delete(+c.dataset.id);updateSel();};});}
function renderHits(hits){const el=document.getElementById('meetings');el.classList.remove('picking');
  el.innerHTML = hits.length ? hits.map(h=>`<li class=mrow style="display:block"><a class=mtitle href="/m/${h.id}">${esc(h.title)}</a>
    <div class="mmeta" style="margin-top:2px">${esc(h.snippet)}</div></li>`).join('')
    : '<li class="muted small">查無結果</li>';}
function applyView(){render(TAGFILTER?ALL.filter(m=>(m.tags||[]).includes(TAGFILTER)):ALL);}
function updateSel(){document.getElementById('selcount').textContent='已選 '+SEL.size;}
function renderTagbar(){fetch('/tags').then(r=>r.json()).then(j=>{
  const bar=document.getElementById('tagbar');
  bar.innerHTML=j.tags.map(t=>`<span class="tg tgbtn${t.name===TAGFILTER?' on':''}" data-t="${esc(t.name)}">#${esc(t.name)} ${t.count}</span>`).join('');
  bar.querySelectorAll('.tgbtn').forEach(s=>s.onclick=()=>{
    TAGFILTER = TAGFILTER===s.dataset.t ? null : s.dataset.t; renderTagbar(); applyView();});});}
fetch('/meetings').then(r=>r.json()).then(ms=>{ALL=ms;applyView();renderTagbar();paintJobs();});
let JOBS={};
function jobChip(j){const pct=j.total?' '+Math.round(j.done/j.total*100)+'%':'';
  return `<span class="badge live" title="${esc(j.text||'')}" style="margin-left:6px">⏳ ${esc(j.kind)}${pct}</span>`;}
function paintJobs(){document.querySelectorAll('.mjob').forEach(s=>{
  const j=JOBS[+s.dataset.id];s.innerHTML=j?jobChip(j):'';});
  const n=Object.keys(JOBS).length,h=document.getElementById('jobhdr');
  if(h){h.style.display=n?'inline-block':'none';h.textContent='⏳ '+n+' 場處理中';}}
window.addEventListener('meetingjobs',e=>{
  const cur={};(e.detail.jobs||[]).forEach(j=>{cur[j.mid]=j;});JOBS=cur;
  if(e.detail.finished.length||e.detail.started.length)
    fetch('/meetings').then(r=>r.json()).then(ms=>{ALL=ms;applyView();renderTagbar();paintJobs();});
  else paintJobs();
});
let tmr; const qel=document.getElementById('q');
qel.oninput=()=>{clearTimeout(tmr);const q=qel.value.trim();
  if(!q){applyView();return;}
  tmr=setTimeout(async()=>{const j=await(await fetch('/search?q='+encodeURIComponent(q))).json();
    renderHits(j.results);},250);};
function setPick(on){PICK=on;SEL.clear();updateSel();
  document.getElementById('selbar').classList.toggle('on',on);
  document.getElementById('pickbtn').textContent=on?'結束選取':'選取';applyView();}
document.getElementById('pickbtn').onclick=()=>setPick(!PICK);
document.getElementById('selcancel').onclick=()=>setPick(false);
document.getElementById('seldel').onclick=async()=>{if(!SEL.size||!confirm('刪除選取的 '+SEL.size+' 場?無法復原。'))return;
  for(const id of [...SEL])await fetch('/meetings/'+id,{method:'DELETE'});
  ALL=ALL.filter(x=>!SEL.has(x.id));setPick(false);renderTagbar();};
document.getElementById('selmerge').onclick=async()=>{if(SEL.size<2){alert('合併需選 2 場以上');return;}
  await fetch('/meetings/merge',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({ids:[...SEL]})});
  setTimeout(()=>location.reload(),400);};
document.getElementById('mergebtn').onclick = async () => {
  const mm=document.getElementById('mergemsg'); mm.textContent='整合中…';
  const r=await fetch('/meetings/merge-nearby?gap_min=10',{method:'POST'});
  const j=await r.json();
  mm.textContent=' 已整合 '+j.merged_groups+' 組';
  setTimeout(()=>location.reload(),600);
};
""")


def _speakers_page():
    body = ("<h1>👥 語者</h1>"
            "<div class=card>"
            "<label class=chk><input type=checkbox id=persist> "
            "跨會議記住說話者(聲紋)：分群時把每個人的聲紋存進語者庫，下次會議自動認出同一人並沿用名字。</label>"
            "<p class=hint style='margin:.5em 0 0'>關閉後分群只在單場內進行，不建立／比對全域聲紋。</p>"
            "<div class=setrow style='margin-top:10px'><label class=hint>辨識門檻(cosine，越高越保守、寧可分開) "
            "<input id=thr type=number min=0.3 max=0.9 step=0.01 style='width:80px'></label></div></div>"
            "<div class=card id=sugcard style='display:none'>"
            "<h2 style='margin-top:0;font-size:15px'>可能是同一人</h2>"
            "<div id=sugs></div></div>"
            "<div class=card><table class=tx id=sp>"
            "<tr><th>名稱</th><th>會議</th><th>發言</th><th></th></tr></table>"
            "<p class=hint id=empty style='display:none'>還沒有任何已記住的語者。分群一場多人會議(會後處理)即可建立。</p></div>"
            "<div class=card id=uttcard style='display:none'>"
            "<h2 style='margin-top:0' id=utthead>發言</h2>"
            "<div id=utts></div></div>")
    script = """
let ALL=[],_au=null;
function esc(s){return (s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function play(name){try{if(_au)_au.pause();}catch(e){}
  _au=new Audio('/speakers/'+encodeURIComponent(name)+'/sample.wav');
  _au.play().catch(()=>alert('這位語者沒有可試聽的音訊片段'));}
function ts(ms){const s=Math.round((ms||0)/1000);return (s/60|0)+':'+String(s%60).padStart(2,'0');}
function fmtDay(t){const d=new Date(t*1000);return d.getFullYear()+'-'+String(d.getMonth()+1).padStart(2,'0')+'-'+String(d.getDate()).padStart(2,'0');}
async function load(){
  const j=await(await fetch('/speakers')).json();
  document.getElementById('persist').checked=j.persist;
  ALL=j.speakers||[];
  const t=document.getElementById('sp');
  t.querySelectorAll('tr:not(:first-child)').forEach(r=>r.remove());
  document.getElementById('empty').style.display=ALL.length?'none':'block';
  for(const s of ALL){
    const others=ALL.filter(o=>o.name!==s.name).map(o=>`<option value="${esc(o.name)}">${esc(o.name)}</option>`).join('');
    const tr=document.createElement('tr');
    tr.innerHTML=`<td><b>${esc(s.name)}</b></td><td>${s.meetings}</td><td>${s.utterances}</td>
      <td style="white-space:nowrap">
      ${s.has_sample?`<button class=btn data-act=play data-name="${esc(s.name)}" title=試聽>🔊</button>`:''}
      <button class=btn data-act=view data-name="${esc(s.name)}">發言</button>
      <button class=btn data-act=rename data-name="${esc(s.name)}">改名</button>
      ${others?`<select data-act=merge data-keep="${esc(s.name)}"><option value="">合併另一位到此…</option>${others}</select>`:''}
      <button class="btn danger" data-act=del data-name="${esc(s.name)}">刪除</button></td>`;
    t.appendChild(tr);
  }
  t.querySelectorAll('[data-act=rename]').forEach(b=>b.onclick=async()=>{
    const nw=prompt('語者改名(套用到所有會議):',b.dataset.name);if(nw==null||!nw.trim())return;
    await fetch('/speakers/rename',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({old:b.dataset.name,new:nw.trim()})});load();});
  t.querySelectorAll('[data-act=del]').forEach(b=>b.onclick=async()=>{
    if(!confirm('忘記這位語者的聲紋?(逐字稿名稱保留,只是之後不再自動認出)'))return;
    await fetch('/speakers/delete',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:b.dataset.name})});load();});
  t.querySelectorAll('[data-act=merge]').forEach(sel=>sel.onchange=async()=>{
    const drop=sel.value;if(!drop)return;
    if(!confirm(`把「${drop}」併入「${sel.dataset.keep}」?(同一個人)`)){sel.value='';return;}
    await fetch('/speakers/merge',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({keep:sel.dataset.keep,drop})});load();});
  t.querySelectorAll('[data-act=view]').forEach(b=>b.onclick=()=>view(b.dataset.name));
  t.querySelectorAll('[data-act=play]').forEach(b=>b.onclick=()=>play(b.dataset.name));
  loadSugs();
}
async function loadSugs(){
  const j=await(await fetch('/speakers/suggestions')).json();
  const pairs=j.pairs||[];
  document.getElementById('sugcard').style.display=pairs.length?'block':'none';
  document.getElementById('sugs').innerHTML=pairs.map(p=>
    `<div class=setrow style="gap:6px;margin:4px 0"><span>
     <button class=btn data-play="${esc(p.a)}" title=試聽>🔊</button> <b>${esc(p.a)}</b> ↔
     <button class=btn data-play="${esc(p.b)}" title=試聽>🔊</button> <b>${esc(p.b)}</b>
     <span class="muted small">相似度 ${p.sim}</span></span>
     <span style="display:flex;gap:6px">
     <button class=btn data-keep="${esc(p.a)}" data-drop="${esc(p.b)}">是同一人，合併</button>
     <button class=btn data-no-a="${esc(p.a)}" data-no-b="${esc(p.b)}">不是同一人</button></span></div>`).join('');
  document.querySelectorAll('#sugs [data-play]').forEach(b=>b.onclick=()=>play(b.dataset.play));
  document.querySelectorAll('#sugs [data-keep]').forEach(b=>b.onclick=async()=>{
    await fetch('/speakers/merge',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({keep:b.dataset.keep,drop:b.dataset.drop})});load();});
  document.querySelectorAll('#sugs [data-no-a]').forEach(b=>b.onclick=async()=>{
    await fetch('/speakers/nonmatch',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({keep:b.dataset.noA,drop:b.dataset.noB})});loadSugs();});
}
async function view(name){
  const j=await(await fetch(`/speakers/${encodeURIComponent(name)}/utterances`)).json();
  document.getElementById('utthead').textContent=`${name} 的發言（${j.utterances.length}）`;
  const by={};for(const u of j.utterances){(by[u.meeting_id]=by[u.meeting_id]||{title:u.title,at:u.created_at,items:[]}).items.push(u);}
  document.getElementById('utts').innerHTML=Object.entries(by).map(([mid,g])=>
    `<div style="margin:0 0 12px"><a href="/m/${mid}"><b>${esc(g.title)}</b></a> <span class="muted small">${fmtDay(g.at)}</span>
     <table class=tx>${g.items.map(u=>`<tr><td class=ts style="width:64px">${ts(u.start_ms)}</td><td>${esc(u.text)}</td></tr>`).join('')}</table></div>`).join('')
     || '<p class=muted>無發言</p>';
  document.getElementById('uttcard').style.display='block';
  document.getElementById('uttcard').scrollIntoView({behavior:'smooth',block:'nearest'});
}
document.getElementById('persist').onchange=async(e)=>{
  await fetch('/settings/persist_speakers',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({value:e.target.checked?'1':'0'})});};
async function loadThr(){const t=document.getElementById('thr');
  t.value=(await(await fetch('/settings/speaker_threshold')).json()).value;
  t.onchange=async()=>{await fetch('/settings/speaker_threshold',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({value:t.value})});loadSugs();};}
loadThr();load();
"""
    return _shell("語者 · MeetingSummary", body, script=script, back=True)


def _models_page():
    body = (
        "<style>"
        ".subnav{display:flex;gap:8px;flex-wrap:wrap;margin:0 0 18px}"
        ".subnav a{font-size:13px;font-weight:650;padding:.4em .9em;border-radius:999px;"
        "border:1px solid var(--line);color:var(--muted);text-decoration:none;transition:.12s}"
        ".subnav a:hover{border-color:var(--accent);color:var(--accent)}"
        "section{margin:0 0 26px}section>h2{margin:0 0 10px;font-size:15px}"
        ".setrow{display:flex;align-items:center;gap:12px;flex-wrap:wrap}"
        ".verpill{font-weight:750;font-size:13px;padding:.25em .7em;border-radius:999px;"
        "background:var(--surface2);border:1px solid var(--line)}"
        "</style>"
        "<h1>⚙️ 設定</h1>"
        "<nav class=subnav>"
        "<a href='#sec-sys'>系統 · 更新</a>"
        "<a href='#sec-models'>模型</a>"
        "<a href='#sec-data'>資料管理</a>"
        "<a href='#sec-rt'>加速 runtime</a>"
        "<a href='#sec-exp'>實驗性功能</a></nav>"

        "<section id=sec-sys><h2>系統 · 更新</h2>"
        "<div class=card>"
        "<div class=setrow style='margin-bottom:14px'>"
        "<span class=verpill>版本 <span id=ver>…</span></span>"
        "<span class=muted id=total></span></div>"
        "<div class=setrow>"
        "<button class='btn primary' id=upd>檢查更新</button>"
        "<a class=btn href='/changelog'>更新紀錄</a>"
        "<button class=btn id=free>釋放閒置記憶體</button></div>"
        "<p class=hint id=updmsg style='margin:.7em 0 0'>更新比對 GitHub Releases，由你決定是否更新並重啟；"
        "釋放會清掉閒置模型權重(下次用會重載)。</p></div></section>"

        "<section id=sec-models><h2>模型</h2>"
        "<div class=card><h3 style='margin:0 0 8px;font-size:13px;color:var(--muted)'>支援的模型</h3>"
        "<table class=tx id=sup><tr><th>模型</th><th>大小</th><th></th></tr></table></div>"
        "<div class=card><h3 style='margin:0 0 8px;font-size:13px;color:var(--muted)'>其他快取</h3>"
        "<table class=tx id=oth><tr><th>名稱</th><th>大小</th><th></th></tr></table></div></section>"

        "<section id=sec-data><h2>資料管理</h2>"
        "<div class=card>"
        "<div class=setrow style='margin-bottom:10px'><b>總用量</b>"
        "<span id=stotal class=muted>計算中…</span></div>"
        "<table class=tx id=stcat><tr><th>類別</th><th>大小</th></tr></table>"
        "<h3 style='margin:14px 0 6px;font-size:13px;color:var(--muted)'>最大的會議（音檔）</h3>"
        "<table class=tx id=stmtg><tr><th>會議</th><th>大小</th></tr></table>"
        "<p class=hint style='margin:.6em 0 0'>音檔（16k PCM）是主要佔空間來源；"
        "刪除會議會一併清掉其音檔。模型權重請到上方「模型」管理。</p>"
        "</div></section>"

        "<section id=sec-exp><h2>🧪 實驗性功能</h2>"
        "<div class=card>"
        "<label class=chk><input type=checkbox id=exp_overlap> "
        "分群時標記重疊/多人語音（同一句多位說話者會標「🔀」，啟發式，非真分離）</label>"
        "<p class=hint style='margin:.6em 0 0'>實驗性功能可能不穩定或不準確；於各會議的「多人分群」時生效。</p>"
        "</div>"
        + (("<div class=card style='margin-top:12px'>"
            "<label class=chk><input type=checkbox id=ane_opt> "
            "🧪 ANE 省電辨識（Apple Neural Engine·M 系列）：辨識（自動＋重新辨識）跑在 Neural Engine — "
            "省電、不發熱、不占 GPU（INT8，準度略降一點）。</label>"
            "<p class=hint style='margin:.6em 0 0'>開啟後上傳的自動辨識與「重新語音辨識」都改走 ANE。</p>"
            "<label class=chk style='margin-top:.6em'><input type=checkbox id=denoise_opt> "
            "🧪 嘈雜環境降噪（DeepFilterNet3·ANE）：辨識前先去背景噪音。</label>"
            "<p class=hint style='margin:.4em 0 0'>僅嘈雜錄音有用；乾淨音訊反而可能變差，預設關閉。"
            "於上傳辨識與「重新語音辨識」生效，會增加辨識耗時。</p></div>"
            if _ane_available() else
            "<div class=card style='margin-top:12px'><p class=hint>🧪 <b>ANE 省電辨識（M 系列）</b>："
            "先到下方「加速 runtime」一鍵安裝 <code>speech</code>，裝好後這裡會出現開關。</p></div>")
           if _apple_silicon() else "")
        + ("<div class=card style='margin-top:12px'>"
           "<label class=chk><input type=checkbox id=floatpanel_opt> "
           "🪟 錄音時自動開啟原生懸浮控制面板</label>"
           "<p class=hint style='margin:.6em 0 0'>開始錄音(含手動／快速錄音)時，自動開啟可置頂於其他 App 之上的"
           "原生小窗(狀態＋計時＋停止)。需先到下方「加速 runtime」安裝 <code>floatpanel</code>。</p></div>"
           if _apple_silicon() else "")
        + "</section>"

        "<section id=sec-rt><h2>加速 runtime（.cpp · Metal）</h2>"
        "<div class=card>"
        "<table class=tx id=rt><tr><th>Runtime</th><th>狀態</th><th></th></tr></table>"
        "<p class=hint style='margin:.5em 0 0'>一鍵安裝。chatllm 優先下載預編譯(免 cmake)，"
        "下載不到才原始碼編譯；femelo 需 python≥3.11；speech = ANE 省電辨識(Qwen3-ASR CoreML，brew)；"
        "audiocap = 原生系統音擷取、floatpanel = 浮動控制面板(皆為預編下載，不在本機編譯；首次用需授權「螢幕錄製」)。"
        "缺的環境會自動 brew 安裝。清除只移除編譯產物，模型權重保留。</p>"
        "</div></section>")
    script = r"""
    function human(mb){return mb>=1000?(mb/1000).toFixed(1)+' GB':mb+' MB';}
    function esc(s){return s.replace(/"/g,'&quot;');}
    function tmsg(t){if(!t)return '';
      const c=t.state==='error'?'#e66':t.state==='done'?'#3a3':'var(--muted)';
      const ic=t.state==='error'?'⚠ ':t.state==='running'?'⏳ ':t.state==='done'?'✓ ':'';
      return ` <span style="color:${c};font-size:11px">${ic}${esc(t.msg||t.state)}</span>`;}
    async function del(path,name){if(!confirm('刪除 '+name+' ？'))return false;
      const r=await fetch('/models/cache/delete',{method:'POST',
        headers:{'Content-Type':'application/json'},body:JSON.stringify({path})});return r.ok;}
    function load(){fetch('/models/status').then(r=>r.json()).then(d=>{
      document.getElementById('ver').textContent=d.version||'?';
      document.getElementById('total').textContent='模型快取共 '+human(d.total_mb);
      // runtimes
      document.getElementById('rt').innerHTML='<tr><th>Runtime</th><th>狀態</th><th></th></tr>'
        +Object.entries(d.runtimes).map(([k,ok])=>`<tr><td>${k}</td>`
          +`<td>${ok?'✅ 已安裝':'— 未安裝'}${tmsg(d.tasks[k])}</td>`
          +`<td><button class='btn' data-rt="${k}">${ok?'重新安裝':'安裝'}</button>`
          +`${ok?` <button class='btn danger' data-rtdel="${k}">清除</button>`:''}</td></tr>`).join('');
      document.querySelectorAll('#rt button[data-rtdel]').forEach(b=>b.onclick=async()=>{
        if(!confirm('清除 '+b.dataset.rtdel+' runtime？(模型權重保留,重新編譯可復原)'))return;
        b.disabled=true;b.textContent='清除中…';
        await fetch('/models/runtime/delete',{method:'POST',headers:{'Content-Type':'application/json'},
          body:JSON.stringify({runtime:b.dataset.rtdel})});
        load();});
      document.querySelectorAll('#rt button[data-rt]').forEach(b=>b.onclick=async()=>{
        b.disabled=true;b.textContent='處理中…';
        await fetch('/models/setup',{method:'POST',headers:{'Content-Type':'application/json'},
          body:JSON.stringify({runtime:b.dataset.rt})});
        setTimeout(load,800);});
      // supported — grouped by backend (MLX / .cpp / transformers) for clarity
      let lastG=null;
      document.getElementById('sup').innerHTML='<tr><th>模型</th><th>大小</th><th></th></tr>'
        +d.supported.map(m=>{
          const hdr=m.group!==lastG?(lastG=m.group,`<tr><td colspan=3 class=muted style="padding-top:12px;font-weight:700">${esc(m.group||'')}</td></tr>`):'';
          const act=m.cached
          ?`<button class='btn danger' data-p="${esc(m.path)}" data-n="${esc(m.label)}">刪除</button>`
          :`<button class='btn primary' data-id="${esc(m.id)}">下載</button>`;
          return hdr+`<tr><td>${m.label}<div class=muted style='font-size:11px'>${m.id} · ${m.kind}</div></td>`
            +`<td>${m.cached?human(m.size_mb):'—'}${tmsg(d.tasks[m.id])}</td><td>${act}</td></tr>`;}).join('');
      document.querySelectorAll('#sup button[data-id]').forEach(b=>b.onclick=async()=>{
        b.disabled=true;b.textContent='下載中…';
        const r=await fetch('/models/download',{method:'POST',headers:{'Content-Type':'application/json'},
          body:JSON.stringify({id:b.dataset.id})});
        if(!r.ok)b.textContent='失敗:'+(await r.text());else setTimeout(load,800);});
      document.querySelectorAll('#sup button[data-p]').forEach(b=>b.onclick=async()=>{
        if(await del(b.dataset.p,b.dataset.n))load();});
      // other cache
      document.getElementById('oth').innerHTML='<tr><th>名稱</th><th>大小</th><th></th></tr>'
        +(d.other.map(m=>`<tr><td title="${m.root}">${m.name}</td><td>${human(m.size_mb)}</td>`
          +`<td><button class='btn danger' data-p="${esc(m.path)}" data-n="${esc(m.name)}">刪除</button></td></tr>`).join('')
          ||'<tr><td colspan=3 class=muted>無</td></tr>');
      document.querySelectorAll('#oth button[data-p]').forEach(b=>b.onclick=async()=>{
        if(await del(b.dataset.p,b.dataset.n))load();});
      if(Object.values(d.tasks||{}).some(t=>t.state==='running'))setTimeout(load,3000);
    }).catch(()=>{const v=document.getElementById('ver');if(v)v.textContent='（無法連線伺服器）';});}
    document.getElementById('free').onclick=async()=>{
      const m=document.getElementById('updmsg');m.textContent=' 釋放中…';
      const r=await fetch('/models/free',{method:'POST'});const j=await r.json();
      m.textContent=' 已釋放: '+(j.freed.join(', ')||'無');setTimeout(load,400);};
    document.getElementById('upd').onclick=async()=>{
      const m=document.getElementById('updmsg');m.textContent=' 檢查中…';
      const j=await (await fetch('/update/check')).json();
      if(j.error){m.textContent=' 無法檢查: '+j.error;return;}
      if(!j.has_update){m.textContent=' 已是最新 ('+j.current+')';return;}
      m.innerHTML=' 有新版 '+j.latest+'（目前 '+j.current+'）';
      const b=document.createElement('button');b.className='btn primary';b.textContent='更新並重啟';
      b.style.marginLeft='8px';m.appendChild(b);
      b.onclick=async()=>{b.disabled=true;m.append(' 下載中…');
        await fetch('/update/apply',{method:'POST'});
        m.append(' 重啟中,稍候…');
        const wait=setInterval(async()=>{try{const h=await(await fetch('/health')).json();
          if(h){clearInterval(wait);location.reload();}}catch(e){}},1500);};};
    (function(){const xo=document.getElementById('exp_overlap');
      xo.checked=localStorage.getItem('exp_overlap')==='1';
      xo.onchange=()=>localStorage.setItem('exp_overlap',xo.checked?'1':'0');})();
    (function(){const a=document.getElementById('ane_opt');if(!a)return;
      fetch('/settings/ane').then(r=>r.json()).then(j=>a.checked=j.value==='1');
      a.onchange=()=>fetch('/settings/ane',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({value:a.checked?'1':'0'})});})();
    (function(){const d=document.getElementById('denoise_opt');if(!d)return;
      fetch('/settings/denoise').then(r=>r.json()).then(j=>d.checked=j.value==='1');
      d.onchange=()=>fetch('/settings/denoise',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({value:d.checked?'1':'0'})});})();
    (function(){const f=document.getElementById('floatpanel_opt');if(!f)return;
      fetch('/settings/float_panel').then(r=>r.json()).then(j=>f.checked=j.value==='1');
      f.onchange=()=>fetch('/settings/float_panel',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({value:f.checked?'1':'0'})});})();
    function fmtB(b){if(b>=1e9)return (b/1e9).toFixed(2)+' GB';
      if(b>=1e6)return (b/1e6).toFixed(1)+' MB';
      if(b>=1e3)return (b/1e3).toFixed(0)+' KB';return b+' B';}
    async function loadStorage(){
      const j=await(await fetch('/storage')).json();
      document.getElementById('stotal').textContent=fmtB(j.total);
      const ct=document.getElementById('stcat');
      ct.querySelectorAll('tr:not(:first-child)').forEach(r=>r.remove());
      (j.categories||[]).forEach(c=>{const tr=document.createElement('tr');
        tr.innerHTML='<td>'+esc(c.label)+'</td><td>'+fmtB(c.bytes)+'</td>';ct.appendChild(tr);});
      const mt=document.getElementById('stmtg');
      mt.querySelectorAll('tr:not(:first-child)').forEach(r=>r.remove());
      const ms=j.meetings||[];
      if(!ms.length){const tr=document.createElement('tr');
        tr.innerHTML='<td colspan=2 class=muted>沒有錄音資料</td>';mt.appendChild(tr);}
      ms.forEach(m=>{const a=document.createElement('a');a.href='/m/'+m.id;
        a.textContent=m.title||('#'+m.id);
        const t1=document.createElement('td');t1.appendChild(a);
        const t2=document.createElement('td');t2.textContent=fmtB(m.bytes);
        const tr=document.createElement('tr');tr.appendChild(t1);tr.appendChild(t2);mt.appendChild(tr);});
    }
    loadStorage();
    load();
    """
    return _shell("設定", body, script=script, back=True)


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
    <label class=fld>斷句 VAD
      <select id=vad>
        <option value="silero">silero(神經·較準·預設)</option>
        <option value="energy">能量(快·省)</option>
      </select></label>
    <label class=fld>語言
      <select id=lang>
        <option value="">自動偵測</option>
        <option value="zh">中文</option>
        <option value="en">English</option>
        <option value="ja">日本語</option>
        <option value="ko">한국어</option>
        <option value="yue">粵語</option>
      </select></label>
    <label class=fld>即時模型
      <select id=model>
        <optgroup label="🔧 .cpp · Metal">
        <option value="qwen3-asr-0.6b-q4-k-m">Qwen3-ASR 0.6B(預設·快)</option>
        <option value="qwen3-asr-1.7b">Qwen3-ASR 1.7B(chatllm·慢·備用)</option>
        </optgroup>
        <optgroup label="⚡ MLX · Metal/GPU">
        <option value="mlx-community/Qwen3-ASR-1.7B-8bit">Qwen3-ASR 1.7B(準·快)</option>
        <option value="mlx-community/whisper-small-mlx-q4">whisper small-q4(快·省)</option>
        <option value="mlx-community/whisper-large-v3-turbo-q4">whisper turbo-q4(較準)</option>
        <option value="mlx-community/whisper-large-v3-turbo">whisper turbo(最準·較吃)</option>
        <option value="mlx-community/whisper-base-mlx-q4">whisper base-q4(更快)</option>
        <option value="mlx-community/whisper-tiny-mlx-q4">whisper tiny-q4(最省)</option>
        </optgroup>
        <optgroup label="🐢 transformers · 慢">
        <option value="Qwen/Qwen3-ASR-0.6B">Qwen3-ASR 0.6B</option>
        </optgroup>
      </select></label>
    <label class=chk style="align-self:end"><input type=checkbox id=diarize> 對方即時多人分群(實驗)</label>
    <label class=chk style="align-self:end" title="只錄音不辨識，零推論、不發熱；事後再「重新語音辨識」"><input type=checkbox id=reconly> 🪫 純錄音(省電·不即時辨識)</label>
    <label class=chk style="align-self:end" title="嘈雜環境降噪，於事後重新辨識時生效"><input type=checkbox id=denoise_live> 🧪 降噪(重新辨識時)</label>
    <label class=chk style="align-self:end" title="用原生 ScreenCaptureKit 擷取系統音(對方)，免每次跳瀏覽器分享框；首次需授權螢幕錄製"><input type=checkbox id=nativesys> 🖥️ 系統音原生擷取(免分享框)</label>
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
<div class=panel style="margin-top:10px">
  <label style="font-weight:600">📝 現場筆記
    <span class="muted small" style="font-weight:400">— 邊開會邊記，會成為摘要的可信參考（人名／日期／決議）</span></label>
  <textarea id=notes rows=3 placeholder="例：負責人 Amy、下次 7/3 前交初稿…（自動儲存）"
    style="width:100%;box-sizing:border-box;margin-top:6px;font:inherit;padding:8px;border-radius:8px"></textarea>
  <span class="muted small" id=notestatus></span>
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
const notesEl=document.getElementById('notes'), noteStat=document.getElementById('notestatus');
let noteTimer;
function saveNotes(){
  if(!mid||!notesEl) return;  // no meeting yet -> buffered locally, flushed on 'meeting'
  fetch('/meetings/'+mid+'/notes',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({value:notesEl.value})})
    .then(()=>{noteStat.textContent='已儲存';setTimeout(()=>noteStat.textContent='',1500);})
    .catch(()=>{noteStat.textContent='儲存失敗';});
}
if(notesEl) notesEl.addEventListener('input',()=>{clearTimeout(noteTimer);noteTimer=setTimeout(saveNotes,800);});

function showModels(m){
  curModel.textContent = '(目前 '+(m.live||'-').split('/').pop()+')';
  if(m.live_requested) modelSel.value = m.live_requested;
  document.getElementById('accmodel').textContent = (m.accurate||'-').split('/').pop();
}
fetch('/models').then(r=>r.json()).then(showModels).catch(()=>{});
fetch('/live/prewarm',{method:'POST'}).catch(()=>{});  // warm ANE helper before first record
modelSel.onchange = () => {
  fetch('/models',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({live:modelSel.value})})
    .then(r=>r.json()).then(()=>fetch('/models').then(r=>r.json()).then(showModels));
};
// 純錄音 hot-toggle: flip mid-recording (e.g. when the machine overheats) without
// stop/restart — sends a control message to the live socket.
const recOnly=document.getElementById('reconly');
recOnly.onchange=()=>{
  if(ws&&ws.readyState===1) ws.send(JSON.stringify({type:'mode',record_only:recOnly.checked}));
  L.textContent=recOnly.checked?'🪫 純錄音中（不即時辨識，事後可重新辨識）':'';
};
// 降噪 toggle mirrors the global setting (applies when you re-transcribe).
const dnLive=document.getElementById('denoise_live');
fetch('/settings/denoise').then(r=>r.json()).then(j=>dnLive.checked=j.value==='1').catch(()=>{});
dnLive.onchange=()=>fetch('/settings/denoise',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({value:dnLive.checked?'1':'0'})});

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
const GATE=0.006;       // lower gate so quiet speech still transmits
// gate=true: silence-gate + pre-roll (perf, single-track). gate=false: send
// continuous so both dual tracks share one wall-clock timeline (else each track's
// gated/compressed clock drifts vs the other -> 我/對方 timestamps don't line up).
function attach(stream, tag, ratio, gate){
  const node = ctx.createScriptProcessor(4096,1,1);
  ctx.createMediaStreamSource(stream).connect(node);
  let hangover = 0, pre = [];
  function sendFloat(input){
    const outLen = Math.floor(input.length/ratio);
    const pcm = new Int16Array(outLen);
    for(let i=0;i<outLen;i++){
      // box-average the source block (cheap anti-alias) instead of nearest-sample
      // decimation — picking every Nth sample aliases HF -> garbled consonants/英文.
      const a = Math.floor(i*ratio), b = Math.floor((i+1)*ratio);
      let sum=0, n=0;
      for(let j=a;j<b && j<input.length;j++){ sum+=input[j]; n++; }
      const s = Math.max(-1,Math.min(1, n ? sum/n : 0));
      pcm[i] = s*32767;
    }
    if(tag===null){ ws.send(pcm.buffer); }
    else { const b=new Uint8Array(1+pcm.byteLength); b[0]=tag;
           b.set(new Uint8Array(pcm.buffer),1); ws.send(b.buffer); }
  }
  node.onaudioprocess = ev => {
    if(ws.readyState!==1) return;
    const input = ev.inputBuffer.getChannelData(0);
    if(!gate){ sendFloat(input); return; }  // continuous (dual): one shared clock
    let sum=0; for(let i=0;i<input.length;i++) sum+=input[i]*input[i];
    const rms = Math.sqrt(sum/input.length);
    if(rms > GATE){
      if(hangover===0){ pre.forEach(sendFloat); pre=[]; }  // flush pre-roll: save the onset
      hangover = 18;                                       // ~1.5s tail so finals fire
    } else if(hangover > 0){ hangover--; }
    if(rms <= GATE && hangover <= 0){                      // idle: keep a short pre-roll ring
      pre.push(new Float32Array(input)); if(pre.length>3) pre.shift();
      return;
    }
    sendFloat(input);
  };
  node.connect(gain);
  nodes.push(node);
}

startBtn.onclick = async () => {
  if(ws||startBtn.disabled) return;       // guard double-start during the async permission window
  startBtn.disabled=true;
  const source = document.getElementById('source').value;
  const nativeSys = document.getElementById('nativesys').checked;
  const dual = source==='dual';
  // native_sys: the SYSTEM track comes from ScreenCaptureKit server-side, so the
  // browser must NOT also capture it. dual -> browser does mic only; system -> none.
  let browserSrc = source;
  if(nativeSys) browserSrc = (source==='dual') ? 'mic' : (source==='system' ? 'none' : source);
  try { streams = browserSrc==='none' ? [] : await getStreams(browserSrc); }
  catch(e){ S.textContent=' 取得音源失敗: '+e.message; startBtn.disabled=false; return; }
  // Ask for a 16 kHz context so the browser's quality resampler does the work
  // (ratio≈1, no crude decimation). Falls back to the device rate if unsupported.
  try { ctx = new AudioContext({sampleRate:16000}); }
  catch(e){ ctx = new AudioContext(); }
  gain = ctx.createGain(); gain.gain.value = 0;  // mute: no self-echo
  gain.connect(ctx.destination);
  const ratio = ctx.sampleRate/16000;
  const diar = document.getElementById('diarize').checked ? '&diarize=1' : '';
  // bigger unit = longer pause + bigger ceiling -> more context per accurate pass
  const unit = document.getElementById('unit').value==='paragraph'
    ? '&silence_ms=1000&max_utt_s=30' : '';
  const sess = session ? '&session='+session : '';
  const vad = '&vad='+document.getElementById('vad').value;
  const lv = document.getElementById('lang').value;
  const lang = lv ? '&lang='+lv : '';
  const ro = document.getElementById('reconly').checked ? '&record_only=1' : '';
  const ns = document.getElementById('nativesys').checked ? '&native_sys=1' : '';
  ws = new WebSocket(`ws://${location.host}/ws/live?src=${source}${diar}${unit}${sess}${vad}${lang}${ro}${ns}`);
  ws.binaryType='arraybuffer';
  function colored(speaker){ return COLORS[speaker]||'#444'; }
  ws.onmessage = e => {
    const m = JSON.parse(e.data);
    if(m.type==='meeting'){ mid=m.id; session=m.id;  // bind session to this meeting
      S.textContent=' session #'+mid+' 錄製中…';
      if(notesEl&&notesEl.value) saveNotes(); }  // flush notes typed before record
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
    // guard each stream: with native_sys a track may be server-sourced (no browser stream)
    if(dual){ if(streams[0])attach(streams[0],0,ratio,false); if(streams[1])attach(streams[1],1,ratio,false); }
    else { streams.forEach(st=>attach(st,null,ratio,true)); }  // mic/system/both: gated
  };
  startBtn.disabled=true; stopBtn.disabled=false;
};
document.getElementById('newsess').onclick = () => {
  session=null; mid=null;       // next 開始 starts a fresh session
  T.innerHTML=''; L.textContent=''; C.textContent='';
  if(notesEl){notesEl.value=''; noteStat.textContent='';}
  S.textContent=' 已開新 session';
};
stopBtn.onclick = () => {
  clearTimeout(noteTimer); saveNotes();  // flush any pending notes edit
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


class PathIn(BaseModel):
    path: str


class DownloadIn(BaseModel):
    id: str   # must be one of the supported model ids


class SetupIn(BaseModel):
    runtime: str   # "femelo" | "chatllm"


class TitleIn(BaseModel):
    title: str


class SpeakerRenameIn(BaseModel):
    old: str
    new: str


class MergeIn(BaseModel):
    ids: list[int]


class SpeakerMergeIn(BaseModel):
    keep: str   # the name to keep
    drop: str   # the duplicate voiceprint/name to fold into keep


class NameIn(BaseModel):
    name: str


class SettingIn(BaseModel):
    value: str


class TranscribeIn(BaseModel):
    model: Optional[str] = None      # None -> default accurate backend
    language: Optional[str] = None   # None -> auto-detect; "zh"/"en"/"ja"… forces it


class DiarizeIn(BaseModel):
    track: str = "all"         # "all" -> every track with audio; or mic/system/mixed
    num_speakers: int = -1     # -1 = auto-detect
    seg_model: Optional[str] = None  # override segmentation onnx (e.g. community-1)
    emb_model: Optional[str] = None  # override speaker-embedding onnx
    mark_overlap: bool = False       # experimental: tag turn-dense/overlap lines
    enroll: bool = True              # persistent voiceprints: recognize voices across meetings


class TagIn(BaseModel):
    name: str
    remove: bool = False


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
    # Skip whisper silence-hallucination lines so the summary can't echo them
    # (e.g. 優優獨播劇場). Existing meetings keep the lines in the transcript view;
    # only the text FED TO the summarizer is cleaned.
    import live  # noqa: PLC0415
    return "\n".join(f"{r['speaker']}: {r['text']}" for r in rows
                     if not live._is_hallucination(r["text"]))


def _snippet(text, q, span=44):
    """A ~span-char window of text centered on the first case-insensitive hit of q."""
    text = (text or "").replace("\n", " ").strip()
    i = text.lower().find((q or "").lower())
    if i < 0:
        return text[:span * 2]
    a = max(0, i - span // 2)
    b = min(len(text), i + len(q) + span)
    return ("…" if a else "") + text[a:b] + ("…" if b < len(text) else "")


def _ts(ms):
    s = int(ms) // 1000
    return f"{s // 60:02d}:{s % 60:02d}"


def _export_text(meeting, transcripts, summaries):
    """Plain Markdown of a meeting: title + summaries + timestamped transcript."""
    import datetime
    when = datetime.datetime.fromtimestamp(meeting["created_at"]).strftime("%Y-%m-%d %H:%M")
    lines = [f"# {meeting['title']}", f"> {when}", ""]
    for s in summaries:
        lines += [f"## 摘要（{s['kind']}）", s["text"], ""]
    lines.append("## 逐字稿")
    for r in transcripts:
        lines.append(f"- `{_ts(r['start_ms'])}` **{r['speaker']}**：{r['text']}")
    return "\n".join(lines) + "\n"


_TRACKS = ("system", "mic", "mixed")


def _track_source_pcms(store, mid, track):
    """The pcm files that feed a track. For 'mixed' with no real mixed.pcm, that's
    the mic+system files (dual is mixed on demand for one unified player)."""
    segs = store.list_segments(mid)
    real = [os.path.join(s["dir_path"], f"{track}.pcm") for s in segs]
    real = [p for p in real if os.path.exists(p) and os.path.getsize(p) > 0]
    if track == "mixed" and not real:
        out = []
        for s in segs:
            for t in ("mic", "system"):
                p = os.path.join(s["dir_path"], f"{t}.pcm")
                if os.path.exists(p) and os.path.getsize(p) > 0:
                    out.append(p)
        return out
    return real


def _mix_pcm(parts):
    """Sum int16 PCM byte-strings (pad short to long), clip. None-safe."""
    import numpy as np
    arrs = [np.frombuffer(p, dtype=np.int16).astype(np.int32) for p in parts if p]
    if not arrs:
        return None
    n = max(len(a) for a in arrs)
    out = np.zeros(n, dtype=np.int32)
    for a in arrs:
        out[:len(a)] += a
    return np.clip(out, -32768, 32767).astype(np.int16).tobytes()


def _assemble_track(store, mid, track, sample_rate=16000):
    """Build one continuous PCM for a track across ALL of a meeting's segments,
    each placed at its time offset (started_at - meeting.created_at) so audio
    lines up with the (merge-rebased) transcript timestamps. Segments sharing a
    dir (session resume) are placed once. None if the track has no audio.
    'mixed' with no real mixed.pcm = mic+system summed (unified dual playback)."""
    meeting = store.get_meeting(mid)
    if meeting is None:
        return None
    if track == "mixed":  # derive from mic+system when no real mixed.pcm exists
        has_real = any(
            os.path.exists(p := os.path.join(s["dir_path"], "mixed.pcm"))
            and os.path.getsize(p) > 0 for s in store.list_segments(mid))
        if not has_real:
            mic = _assemble_track(store, mid, "mic", sample_rate)
            sysd = _assemble_track(store, mid, "system", sample_rate)
            return _mix_pcm([mic, sysd])
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
    srcs = _track_source_pcms(store, mid, track)  # mixed -> mic+system if derived
    if not srcs:
        return None
    # Fingerprint the source set (paths + sizes + mtimes). Rebuild whenever it
    # changes — covers merge (segment set changes but pcm mtimes are old), delete,
    # and live append. Without this, merged meetings played stale/partial audio.
    key = repr(sorted((p, os.path.getsize(p), int(os.path.getmtime(p))) for p in srcs))
    keyfile = cache + ".key"
    fresh = os.path.exists(cache) and os.path.exists(keyfile)
    if fresh:
        with open(keyfile) as _kf:
            fresh = _kf.read() == key
    if not fresh:
        pcm = _assemble_track(store, mid, track)
        if pcm is None:
            return None
        os.makedirs(f"data/{mid}", exist_ok=True)
        with open(cache, "wb") as f:
            f.write(recorder.pcm_to_wav(pcm, sample_rate=16000, channels=1))
        with open(keyfile, "w") as f:
            f.write(key)
    return cache


def _meeting_tracks(store, mid):
    """Playable tracks. Dual (mic+system, no real mixed) collapses to a single
    'mixed' track so playback is ONE unified player (the two one-sided players
    were the 'tracks don't line up' problem); transcripts still carry 我/對方."""
    present = []
    for t in _TRACKS:
        for seg in store.list_segments(mid):
            p = os.path.join(seg["dir_path"], f"{t}.pcm")
            if os.path.exists(p) and os.path.getsize(p) > 0:
                present.append(t)
                break
    if "mic" in present and "system" in present and "mixed" not in present:
        return ["mixed"]
    return present


# Model cache roots across the runtimes we use (HF for mlx/transformers, femelo
# .cpp gguf, chatllm.cpp gguf). The manage tab lists + deletes from these.
_MODEL_ROOTS = [
    os.path.expanduser("~/.cache/huggingface/hub"),
    os.path.expanduser("~/Library/Application Support/py_qwen3_asr_cpp/models"),
    os.path.abspath("chatllm.cpp/quantized"),
]


def _dir_size(path):
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            if os.path.islink(fp):
                continue  # HF snapshots/ symlinks -> blobs/; count the blob once
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total


def _scan_model_cache(roots=None):
    """List cached models (top-level entry per root) with size. blobs/refs etc.
    inside HF model dirs are summed, not listed."""
    out = []
    for root in (roots or _MODEL_ROOTS):
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            if name.startswith("."):
                continue
            p = os.path.join(root, name)
            out.append({"name": name, "path": p, "root": root,
                        "size_mb": round(_dir_size(p) / 1e6, 1)})
    return out


def _safe_model_path(path, roots=None):
    """True only if path is a real child of a known root (not the root, no
    traversal). Guards the delete route from rm-ing arbitrary paths."""
    rp = os.path.realpath(path)
    for root in (roots or _MODEL_ROOTS):
        rr = os.path.realpath(root)
        if rp != rr and os.path.commonpath([rp, rr]) == rr:
            return True
    return False


# Curated: the models the app actually uses. Download is constrained to these
# (kind decides the runtime). kind: hf (mlx/transformers), femelo (.cpp 0.6b),
# chatllm (.cpp 1.7b).
_SUPPORTED = [
    {"id": "mlx-community/whisper-small-mlx-q4", "label": "whisper small-q4（live 預設）", "kind": "hf", "group": "⚡ MLX · Metal/GPU"},
    {"id": "mlx-community/whisper-large-v3-turbo-q4", "label": "whisper turbo-q4（精校）", "kind": "hf", "group": "⚡ MLX · Metal/GPU"},
    {"id": "mlx-community/whisper-large-v3-turbo", "label": "whisper turbo", "kind": "hf", "group": "⚡ MLX · Metal/GPU"},
    {"id": "mlx-community/whisper-base-mlx-q4", "label": "whisper base-q4", "kind": "hf", "group": "⚡ MLX · Metal/GPU"},
    {"id": "mlx-community/whisper-tiny-mlx-q4", "label": "whisper tiny-q4", "kind": "hf", "group": "⚡ MLX · Metal/GPU"},
    {"id": "mlx-community/whisper-large-v3-mlx", "label": "whisper large-v3", "kind": "hf", "group": "⚡ MLX · Metal/GPU"},
    {"id": "mlx-community/Qwen2.5-3B-Instruct-4bit", "label": "Qwen2.5-3B（摘要）", "kind": "hf", "group": "⚡ MLX · Metal/GPU"},
    {"id": "mlx-community/Qwen3-ASR-1.7B-8bit", "label": "Qwen3-ASR 1.7B（準·快）", "kind": "hf", "group": "⚡ MLX · Metal/GPU"},
    {"id": "qwen3-asr-0.6b-q4-k-m", "label": "Qwen3-ASR 0.6B（femelo·快）", "kind": "femelo", "group": "🔧 .cpp · Metal"},
    {"id": "qwen3-asr-1.7b", "label": "Qwen3-ASR 1.7B（chatllm·慢·備用）", "kind": "chatllm", "group": "🔧 .cpp · Metal"},
    {"id": "Qwen/Qwen3-ASR-0.6B", "label": "Qwen3-ASR 0.6B", "kind": "hf", "group": "🐢 transformers · 慢"},
    {"id": "Qwen/Qwen3-ASR-1.7B", "label": "Qwen3-ASR 1.7B（最準·慢）", "kind": "hf", "group": "🐢 transformers · 慢"},
    # 說話者分群模型(sherpa pyannote-3-0 seg + 3dspeaker emb)由 diarize.py 首次分群時
    # 自動下載到 models/，不在此清單(community-1 onnx 與 sherpa 不相容,缺 sample_rate metadata)。
]
_FEMELO_DIR = os.path.expanduser("~/Library/Application Support/py_qwen3_asr_cpp/models")
_CHATLLM_DIR = os.path.abspath("chatllm.cpp/quantized")

# Background download/compile state, keyed by model id or runtime name, so the
# manage UI can show "進行中 / 完成 / 失敗:<reason>" instead of a silent thread
# whose error only ever hit the server log. {key: {state, msg}}.
_MODEL_TASKS = {}


def _hf_dir(model_id):
    return os.path.join(os.path.expanduser("~/.cache/huggingface/hub"),
                        "models--" + model_id.replace("/", "--"))


def _flatkey(s):
    return s.lower().replace("-", "").replace("_", "").replace(".", "")


def _model_cached(m):
    """(cached, size_mb, path) for a supported model, per its runtime's cache."""
    kind = m["kind"]
    if kind == "hf":
        p = _hf_dir(m["id"])
        return (os.path.isdir(p), round(_dir_size(p) / 1e6, 1) if os.path.isdir(p) else 0, p)
    if kind == "femelo":
        if os.path.isdir(_FEMELO_DIR):
            for f in os.listdir(_FEMELO_DIR):
                if _flatkey(m["id"]) in _flatkey(f):  # q4-k-m id vs q4_k_m.gguf file
                    p = os.path.join(_FEMELO_DIR, f)
                    return (True, round(os.path.getsize(p) / 1e6, 1), p)
        return (False, 0, _FEMELO_DIR)
    p = os.path.join(_CHATLLM_DIR, m["id"] + ".bin")  # chatllm
    return (os.path.isfile(p), round(os.path.getsize(p) / 1e6, 1) if os.path.isfile(p) else 0, p)


def _runtime_status():
    import platform  # noqa: PLC0415
    import shutil  # noqa: PLC0415
    st = {
        "femelo": os.path.isfile(os.path.abspath(".venv-qwen314/bin/python")),
        "chatllm": os.path.isfile(os.path.abspath("chatllm.cpp/bindings/libchatllm.dylib")),
    }
    if sys.platform == "darwin" and platform.machine() == "arm64":
        st["speech"] = shutil.which("speech") is not None  # ANE batch ASR (省電)
        import backends  # noqa: PLC0415
        st["qwen3-ane"] = backends.ane_helper_bin() is not None  # ANE live helper
        st["audiocap"] = backends.audiocap_bin() is not None     # native system audio
        st["floatpanel"] = backends.floatpanel_bin() is not None  # floating control panel
    return st


def _free_models():
    """Drop cached model weights + the Metal memory pool to reclaim RAM when idle.
    Best-effort. An in-flight transcribe just reloads on its next call (a latency
    blip, not a crash). ponytail: manual release beats a guessed idle-timer."""
    import gc
    freed = []
    try:
        import mlx_whisper.load_models as _lm  # whisper weights are lru_cached by repo
        if hasattr(_lm.load_model, "cache_clear"):
            _lm.load_model.cache_clear()
            freed.append("whisper")
    except Exception:
        pass
    try:
        import backends
        freed += backends.release_all()  # chatllm 1.7b + femelo sidecar
    except Exception:
        pass
    gc.collect()
    try:
        import mlx.core as mx
        (getattr(mx, "clear_cache", None) or getattr(mx.metal, "clear_cache"))()
        freed.append("mlx-pool")
    except Exception:
        pass
    return freed


def _apple_silicon():
    import platform  # noqa: PLC0415
    return sys.platform == "darwin" and platform.machine() == "arm64"


def _ane_available():
    """Apple Neural Engine ASR path is usable: Apple Silicon + the `speech` CLI
    installed. (On M-series but speech missing -> the 設定 install button shows.)"""
    import shutil  # noqa: PLC0415
    return _apple_silicon() and shutil.which("speech") is not None


def _persistent_names(store, embs, prefix, threshold=0.62):
    """cluster id -> persistent voiceprint label. Match each cluster embedding to the
    global speakers table (cosine); reuse the matched speaker's name + nudge its
    centroid, else enroll a new globally-unique speaker. So a voice you named "Alice"
    in one meeting auto-labels "Alice" in later ones; renaming propagates (see
    /speaker route -> rename_global_speaker)."""
    import numpy as np  # noqa: PLC0415
    import diarize as diar  # noqa: PLC0415
    try:
        threshold = float(store.get_setting("speaker_threshold", str(threshold)))
    except (ValueError, TypeError):
        pass
    known = [(r["id"], np.frombuffer(r["centroid"], dtype=np.float32))
             for r in store.list_speakers() if r["centroid"]]
    rows = {r["id"]: r for r in store.list_speakers()}
    names = {}
    for spk, emb in embs.items():
        mid, _sim = diar.match_speaker(emb, known, threshold)
        if mid is not None:
            r = rows[mid]
            n = r["count"] or 1
            cen = np.frombuffer(r["centroid"], dtype=np.float32) * n + emb
            cen = (cen / (np.linalg.norm(cen) + 1e-9)).astype(np.float32)
            store.update_speaker_centroid(mid, cen.tobytes(), n + 1)
            names[spk] = r["name"]
        else:
            sid = store.add_speaker(prefix, emb.astype(np.float32).tobytes())
            nm = f"{prefix}{sid}"  # globally-unique placeholder until the user renames
            store.set_speaker_name(sid, nm)
            names[spk] = nm
            known.append((sid, emb))
            rows[sid] = {"id": sid, "name": nm, "centroid": emb.tobytes(), "count": 1}
    return names


def _is_default_title(t):
    """A title the user hasn't meaningfully set -> safe to auto-replace on summary."""
    t = (t or "").strip()
    return (not t) or t in ("未命名", "實測") or t.startswith("錄音 ")


def _auto_title(summary_text, backend):
    """One short zh title derived from the summary via the LLM. Best-effort -> None."""
    if not (summary_text or "").strip():
        return None
    try:
        prompt = ("根據以下會議摘要，產生一個能代表主題的標題，12 字以內，"
                  "只輸出標題本身(不要引號、不要句號、不要說明):\n\n" + summary_text[:2000])
        line = backend(prompt).strip().splitlines()[0]
        return line.strip().strip('「」『』"\'。. ')[:40] or None
    except Exception:
        return None


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
    # Optional ANE denoise: clean each source PCM ONCE (cached), then window the
    # cleaned copy. Off by default — only helps genuinely noisy recordings.
    denoise_on = store.get_setting("denoise", "0") == "1" and _ane_available()
    _clean = {}

    def _src(path):
        if not denoise_on:
            return path
        if path not in _clean:
            _clean[path] = backends.denoise_file(path, raw_pcm=True)
        return _clean[path]

    for i, (track, p, seg_off, bs, bl) in enumerate(units):
        win_off_ms = seg_off + int(bs / 2 / sample_rate * 1000)
        tmp = f"{os.path.dirname(p)}/_win.pcm"
        with open(_src(p), "rb") as f:
            f.seek(bs)
            with open(tmp, "wb") as w:
                w.write(f.read(bl))
        texts = []
        speaker = _TRACK_LABEL.get(track, track)  # mic->我, system->對方
        try:
            for t in asr.transcribe(tmp, profile="accurate", track=track, backend=backend):
                store.add_transcript(mid, "accurate", track,
                                     t["start_ms"] + win_off_ms,
                                     t["end_ms"] + win_off_ms, speaker, t["text"])
                texts.append(t["text"])
                n += 1
        finally:
            try:
                os.remove(tmp)  # always clean the window temp, even if transcribe raises
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


def _run_diarize_job(store, mid, body, jobs):
    """Background speaker diarization with progress (jobs[mid], polled by the
    meeting page). sherpa's process() reports chunk progress via a callback;
    surface it as done/total. Multi-track meetings run one track at a time."""
    import diarize as diar  # noqa: PLC0415
    jobs[mid] = {"state": "running", "done": 0, "total": 0, "text": "準備中…"}
    try:
        playable = _meeting_tracks(store, mid)
        tracks = playable if body.track == "all" else [body.track]
        tracks = [t for t in tracks if t in playable]
        if not tracks:
            jobs[mid] = {"state": "error", "msg": "no saved audio for this meeting"}
            return
        total_spk = 0
        for ti, track in enumerate(tracks):
            pcm_bytes = _assemble_track(store, mid, track)
            if pcm_bytes is None:
                continue
            os.makedirs(f"data/{mid}", exist_ok=True)
            tmp = f"data/{mid}/_diar_{track}.pcm"
            with open(tmp, "wb") as f:
                f.write(pcm_bytes)
            label = _TRACK_LABEL.get(track, track)
            pre = f"[{ti + 1}/{len(tracks)}] " if len(tracks) > 1 else ""

            def on_prog(done, total, _p=pre, _l=label):
                jobs[mid].update(done=done, total=total, text=f"{_p}{_l} 分群中…")

            names = None
            # Persistent cross-meeting voiceprints: on unless the global toggle is off.
            enroll = body.enroll and store.get_setting("persist_speakers", "1") == "1"
            # Diarize stays on CPU: measured CoreML/ANE here is SLOWER (sherpa's
            # pyannote/3d-speaker onnx ops fall back), and diarize isn't a GPU-heat
            # source anyway. The ANE win is ASR-only (see the speech CLI backend).
            try:
                # Runs in a subprocess — sherpa's process() holds the GIL and would
                # otherwise freeze the event loop (no page switches while diarizing).
                segments, embs = diar.diarize_with_progress(
                    tmp, num_speakers=body.num_speakers, seg_model=body.seg_model,
                    emb_model=body.emb_model, enroll=enroll,
                    on_progress=on_prog,
                    on_phase=lambda ph: jobs[mid].update(text=f"{pre}{label} 建立聲紋…"))
                if embs:
                    names = _persistent_names(store, embs, label)
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)
            rows = [dict(r) for r in store.list_transcripts(mid) if r["track"] == track]
            # split=True: a line spanning >1 speaker turn is cut at the boundary so
            # each person gets their own line (force a break on speaker change).
            assigned = diar.assign_speakers(rows, segments, prefix=label, names=names,
                                            mark_overlap=body.mark_overlap, split=True)
            splits = {}
            for r in assigned:
                if r.get("split"):
                    splits.setdefault(r["src_id"], []).append(r)
                else:
                    store.update_speaker(r["id"], r["speaker"])
            for src_id, pieces in splits.items():  # replace 1 multi-speaker line with N
                store.delete_transcript(src_id)
                for p in pieces:
                    store.add_transcript(mid, p["profile"], p["track"], p["start_ms"],
                                         p["end_ms"], p["speaker"], p["text"])
            total_spk += len({s["speaker"] for s in segments})
        jobs[mid] = {"state": "done", "speakers": total_spk, "tracks": tracks}
    except Exception as e:
        jobs[mid] = {"state": "error", "msg": str(e)}


def _run_summary_job(store, mid, kind, summary_backend, summary_model, jobs):
    """Background summary generation (the LLM call is seconds-to-minutes). No
    token-stream progress, so it's indeterminate — jobs[mid] stays running with
    no total until done, then carries the text/title so the page renders inline
    without a reload."""
    jobs[mid] = {"state": "running", "done": 0, "total": 0, "text": "產生摘要中…"}
    try:
        meeting = store.get_meeting(mid)
        if meeting is None:
            jobs[mid] = {"state": "error", "msg": "meeting not found"}
            return
        text = _transcript_text(store.list_transcripts(mid))
        out = summarize(text, kind=kind, lang=meeting["lang"], backend=summary_backend,
                        notes=(meeting["notes"] or ""))
        store.add_summary(mid, kind, meeting["lang"], out, summary_model, time.time())
        title = None
        if _is_default_title(meeting["title"]):
            nt = _auto_title(out, summary_backend)
            if nt:
                store.update_title(mid, nt)
                title = nt
        jobs[mid] = {"state": "done", "text": out, "kind": kind, "title": title}
    except Exception as e:
        jobs[mid] = {"state": "error", "msg": str(e)}


def _run_upload_job(store, mid, audio_path, asr_backend, summary_backend,
                    summary_model, kind, title, jobs):
    """Background upload pipeline (mirrors run_pipeline, off the request thread):
    transcribe the uploaded file -> summarize -> auto-title -> finalize, writing
    coarse progress into jobs[mid] (same shape the detail page already polls, so
    refresh resumes). Transcribes the original file directly (mlx-whisper loads
    m4a/wav/mp3) so it does NOT depend on the best-effort ffmpeg PCM decode."""
    jobs[mid] = {"state": "running", "done": 0, "total": 0, "text": "辨識中…"}
    try:
        tx_path = audio_path  # denoise the ASR input only; playback uses the original
        if store.get_setting("denoise", "0") == "1" and _ane_available():
            jobs[mid]["text"] = "降噪中…"
            tx_path = backends.denoise_file(audio_path)
        segs = asr.transcribe(tx_path, profile="accurate", track="mic",
                              backend=asr_backend)
        for s in segs:
            store.add_transcript(mid, s["profile"], s["track"], s["start_ms"],
                                 s["end_ms"], s["track"], s["text"])
        jobs[mid].update(done=len(segs), total=len(segs), text="產生摘要中…")
        meeting = store.get_meeting(mid)
        if meeting is None:  # deleted mid-job (race with DELETE /meetings/{mid})
            jobs[mid] = {"state": "error", "msg": "meeting was deleted"}
            return
        text = _transcript_text(store.list_transcripts(mid))
        out = summarize(text, kind=kind, lang=meeting["lang"], backend=summary_backend,
                        notes=(meeting["notes"] or ""))
        store.add_summary(mid, kind, meeting["lang"], out, summary_model, time.time())
        if _is_default_title(title):
            nt = _auto_title(out, summary_backend)
            if nt:
                store.update_title(mid, nt)
        _save_upload_pcm(audio_path, mid, store)  # best-effort, for playback only
        store.finalize_meeting(mid)
        jobs[mid] = {"state": "done", "done": len(segs), "total": len(segs)}
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
    # Anchor at the meeting's created_at, NOT time.time(): an upload is a single
    # file that starts at offset 0. _assemble_track pads (started_at - created_at)
    # of silence, and decode runs minutes after create (whole transcribe first),
    # so time.time() would prepend minutes of silence and shove the audio out of
    # sync with the 0-based transcript timestamps (= "uploaded audio won't play").
    m = store.get_meeting(mid)
    base = m["created_at"] if m else time.time()
    store.add_segment(mid, idx=len(store.list_segments(mid)), dir_path=out_dir,
                      started_at=base, duration_s=0, origin="recorded")


_TRACK_LABEL = {"system": "對方", "mic": "我", "mixed": "混合"}


def _detail_page(mid, meeting, transcripts, summaries, audio_tracks=(), tags=(),
                 recognized=(), ane_on=False):
    try:
        m_notes = meeting["notes"] or ""
    except (KeyError, IndexError):  # older row / test stub without the column
        m_notes = ""

    def ts_str(ms):
        s = (ms or 0) // 1000
        return f"{s // 60:d}:{s % 60:02d}"
    rows = "".join(
        f"<tr data-track='{html.escape(str(r['track']))}' data-ts='{(r['start_ms'] or 0)/1000:.2f}'>"
        f"<td class=ts>{ts_str(r['start_ms'])}</td>"
        f"<td class=who data-spk='{html.escape(str(r['speaker']))}' "
        f"title='點擊改名'>{html.escape(str(r['speaker']))}</td>"
        f"<td>{html.escape(r['text'])}</td></tr>"
        for r in transcripts) or "<tr><td colspan=3 class=muted>尚無逐字稿</td></tr>"
    sums = "".join(
        f"<h3>{html.escape(s['kind'])}</h3>"
        f"<div class=md>{_md_html(s['text'])}</div>"
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
        f"<h1><span id=mtitle>{html.escape(meeting['title'])}</span> "
        f"<button class=btn id=edittitle title='改標題' style='padding:.25em .5em;font-size:13px'>✏️</button> "
        f"<span class='badge {badge}'>{html.escape(meeting['status'])}</span></h1>"
        "<div id=tags class=row style='gap:6px;margin:-4px 0 14px'></div>"
        + ((f"<div class=hint style='margin:-6px 0 12px'>🔁 本場認得："
            + "、".join(html.escape(n) for n in recognized)
            + " <a href='/speakers/manage'>管理語者</a></div>") if recognized else "")
        + audio_card +
        "<div class=card><h2 style='margin-top:0'>摘要</h2>"
        "<label class='muted small'>📝 筆記（摘要的可信參考：人名／日期／決議，自動儲存）</label>"
        "<textarea id=mnotes rows=2 placeholder='補充重點…' "
        "style='width:100%;box-sizing:border-box;font:inherit;padding:8px;"
        "border-radius:8px;margin:.2em 0 .6em'>"
        f"{html.escape(m_notes)}</textarea>"
        "<div class=row>"
        "<select id=kind><option value=minutes>會議記錄</option>"
        "<option value=bullets>條列</option>"
        "<option value=actions>行動項目</option></select>"
        "<button class='btn primary' id=go>產生摘要</button>"
        "<details class=menu><summary class=btn>⋯ 更多</summary><div class=menupop>"
        "<button class=btn id=dia>多人分群</button>"
        f"<a class=btn href='/meetings/{mid}/export'>匯出 Markdown</a>"
        "<button class=btn id=cpsum>複製摘要</button>"
        "<button class=btn id=cptx>複製逐字稿</button>"
        "<button class=btn id=fin>完成會議</button>"
        "<button class='btn danger' id=del>刪除會議</button>"
        "</div></details>"
        "<span class='muted small' id=finmsg></span></div>"
        "<p class=hint style='margin:.6em 0 0'>※ 摘要只在按下按鈕時才產生。停止錄音不會自動完成。"
        "「多人分群」用聲紋把每條音軌(我/對方)各自拆成多位說話者(會後處理,需先有錄音)。</p>"
        f"<div id=out>{sums}</div></div>"
        "<div class=card><div class=setrow style='margin-bottom:8px'>"
        "<h2 style='margin:0'>截圖</h2>"
        "<button class=btn id=shot>📸 截圖（擷取畫面）</button>"
        "<span class='muted small' id=shotmsg></span></div>"
        "<div id=shots class=shotgrid></div>"
        "<p class=hint style='margin:.6em 0 0'>由瀏覽器擷取畫面（如投影片）附加到此會議；會跳出分享畫面選擇（瀏覽器自己的權限，免額外授權）。</p></div>"
        "<div class=card><h2 style='margin-top:0'>逐字稿</h2>"
        "<div class=row style='margin-bottom:10px'>"
        "<select id=remodel>"
        + ("<optgroup label='🧠 NPU · ANE 省電'>"
           "<option value='ane-qwen3-0.6b'>Qwen3-ASR 0.6B(省電·M系列)</option>"
           "<option value='ane-qwen3-0.6b-hybrid'>Qwen3-ASR 0.6B(混合·快)</option>"
           "</optgroup>" if ane_on else "") +
        "<optgroup label='⚡ MLX · Metal/GPU'>"
        "<option value='mlx-community/whisper-large-v3-turbo-q4'>whisper turbo-q4(準·省)</option>"
        "<option value='mlx-community/whisper-large-v3-mlx'>whisper large-v3(最準·吃)</option>"
        "<option value='mlx-community/whisper-small-mlx-q4'>whisper small-q4(快)</option>"
        "<option value='mlx-community/Qwen3-ASR-1.7B-8bit'>Qwen3-ASR 1.7B(準·快)</option>"
        "</optgroup>"
        "<optgroup label='🔧 .cpp · Metal'>"
        "<option value='qwen3-asr-0.6b-q4-k-m'>Qwen3-ASR 0.6B(快)</option>"
        "<option value='qwen3-asr-1.7b'>Qwen3-ASR 1.7B(chatllm·慢·備用)</option>"
        "</optgroup>"
        "<optgroup label='🐢 transformers · 慢'>"
        "<option value='Qwen/Qwen3-ASR-0.6B'>Qwen3-ASR 0.6B</option>"
        "<option value='Qwen/Qwen3-ASR-1.7B'>Qwen3-ASR 1.7B(很慢)</option>"
        "</optgroup>"
        "</select>"
        "<select id=relang>"
        "<option value=''>語言:自動</option><option value='zh'>中文</option>"
        "<option value='en'>English</option><option value='ja'>日本語</option>"
        "<option value='ko'>한국어</option><option value='yue'>粵語</option>"
        "</select>"
        "<button class=btn id=retr>重新語音辨識</button>"
        "<span class='muted small' id=remsg></span></div>"
        "<table class=tx><tr><th>時間</th><th>說話者</th><th>內容</th></tr>"
        f"{rows}</table></div>")
    script = (
        "function _mdEsc(s){return s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}"
        "function _mdHtml(t){const L=(t||'').split('\\n');let o=[],lt=null;"
        "const cl=()=>{if(lt){o.push('</'+lt+'>');lt=null;}};"
        "const il=s=>_mdEsc(s).replace(/\\*\\*(.+?)\\*\\*/g,'<strong>$1</strong>').replace(/`([^`]+?)`/g,'<code>$1</code>');"
        "for(const ln of L){const h=ln.match(/^(#{1,6})\\s+(.*)$/),u=ln.match(/^\\s*[-*]\\s+(.*)$/),ol=ln.match(/^\\s*\\d+[.)]\\s+(.*)$/);"
        "if(h){cl();const n=h[1].length;o.push('<h'+n+'>'+il(h[2])+'</h'+n+'>');}"
        "else if(u||ol){const tg=u?'ul':'ol';if(lt!==tg){cl();o.push('<'+tg+'>');lt=tg;}o.push('<li>'+il((u||ol)[1])+'</li>');}"
        "else if(!ln.trim()){cl();}else{cl();o.push('<p>'+il(ln)+'</p>');}}cl();return o.join('');}"
        "document.getElementById('fin').onclick=async()=>{"
        f"await fetch('/meetings/{mid}/finalize',{{method:'POST'}});"
        "document.getElementById('finmsg').textContent=' 已完成。';};"
        "function _cp(txt,btn){navigator.clipboard.writeText(txt).then(()=>{"
        "const o=btn.textContent;btn.textContent='已複製';setTimeout(()=>btn.textContent=o,1200);});}"
        "document.getElementById('cpsum').onclick=(e)=>_cp(document.getElementById('out').innerText.trim(),e.target);"
        "document.getElementById('cptx').onclick=(e)=>_cp("
        "[...document.querySelectorAll('tr[data-ts]')].map(tr=>tr.innerText.replace(/\\t/g,' ')).join('\\n'),e.target);"
        "document.getElementById('del').onclick=async()=>{"
        "if(!confirm('確定刪除這場會議?逐字稿與音檔都會移除,無法復原。'))return;"
        f"const r=await fetch('/meetings/{mid}',{{method:'DELETE'}});"
        "if(r.ok)location.href='/';"
        "else document.getElementById('finmsg').textContent=' 刪除失敗';};"
        # Async jobs (辨識/分群/摘要) DISPLAY in the global progress popout (_shell).
        # Buttons just start the job + force an immediate global poll. The page only
        # handles side-effects on completion, via the 'meetingjobs' window event:
        # reload for transcribe/diarize (new rows/labels), render inline for summary.
        "const _kick=()=>window._jobsTick&&window._jobsTick();"
        "document.getElementById('retr').onclick=()=>{"
        "const mdl=document.getElementById('remodel').value,lng=document.getElementById('relang').value;"
        f"fetch('/meetings/{mid}/transcribe/start',{{method:'POST',"
        "headers:{'Content-Type':'application/json'},body:JSON.stringify({model:mdl,language:lng})}).then(_kick);};"
        "document.getElementById('dia').onclick=()=>{const ov=localStorage.getItem('exp_overlap')==='1';"
        f"fetch('/meetings/{mid}/diarize',{{method:'POST',"
        "headers:{'Content-Type':'application/json'},body:JSON.stringify({track:'all',mark_overlap:ov})}).then(_kick);};"
        "(function(){const n=document.getElementById('mnotes');if(!n)return;let t;"
        f"const save=()=>fetch('/meetings/{mid}/notes',{{method:'POST',"
        "headers:{'Content-Type':'application/json'},body:JSON.stringify({value:n.value})});"
        "n.addEventListener('input',()=>{clearTimeout(t);t=setTimeout(save,800);});"
        "window._saveNotes=save;})();"
        "document.getElementById('go').onclick=()=>{const k=document.getElementById('kind').value;"
        # flush notes FIRST so the summary uses them, then kick the summary job
        "Promise.resolve(window._saveNotes?window._saveNotes():0).then(()=>"
        f"fetch('/meetings/{mid}/summary',{{method:'POST',"
        "headers:{'Content-Type':'application/json'},body:JSON.stringify({kind:k})}).then(_kick));};"
        "window.addEventListener('meetingjobs',e=>{for(const k of e.detail.finished){"
        f"const pp=k.split('/');if(+pp[0]!=={mid})continue;"
        "if(pp[1]==='摘要'){"
        f"fetch('/meetings/{mid}/summary/progress').then(r=>r.json()).then(s=>{{"
        "if(s.state==='done'){document.getElementById('out').innerHTML='<div class=md>'+_mdHtml(s.text)+'</div>';"
        "if(s.title)document.getElementById('mtitle').textContent=s.title;}});}"
        "else{setTimeout(()=>location.reload(),700);}}});"
        # Player for a row: its own track, else the single (unified mixed) player.
        "const onlyAudio=document.querySelector('audio[id^=aud-]');"
        "function rowAudio(trk){return document.getElementById('aud-'+trk)||onlyAudio;}"
        "document.querySelectorAll('tr[data-ts]').forEach(tr=>{tr.style.cursor='pointer';"
        "tr.onclick=()=>{const a=rowAudio(tr.dataset.track);"
        "if(a){a.currentTime=parseFloat(tr.dataset.ts);a.play();}};});"
        # Follow-along: as a track's audio plays, highlight + scroll to the line
        # whose start time is the latest <= currentTime (end_ms==start_ms for live,
        # so use the next line's start as the implicit boundary).
        # Manual scroll pauses auto-follow for 8s (else it yanks you back). Highlight
        # still tracks; only the scroll is suppressed while you're browsing.
        "let pauseFollow=0;"
        "['wheel','touchmove','keydown'].forEach(ev=>window.addEventListener(ev,"
        "()=>{pauseFollow=Date.now()+8000;},{passive:true}));"
        "document.querySelectorAll('audio[id^=aud-]').forEach(a=>{"
        "const trk=a.id.slice(4);"
        # mixed/unified player follows ALL rows (both 我/對方); per-track only its own.
        "const sel=document.getElementById('aud-'+trk)&&document.querySelector(`tr[data-track=\"${trk}\"]`)"
        "?`tr[data-track=\"${trk}\"]`:'tr[data-ts]';"
        "const rows=[...document.querySelectorAll(sel)]"
        ".map(tr=>({tr,ts:parseFloat(tr.dataset.ts)})).sort((x,y)=>x.ts-y.ts);"
        "let last=null;"
        "a.ontimeupdate=()=>{const t=a.currentTime;let cur=null;"
        "for(const r of rows){if(r.ts<=t+0.05)cur=r;else break;}"
        "if(cur===last)return;"
        "rows.forEach(r=>r.tr.classList.toggle('active',r===cur));"
        "if(cur&&Date.now()>pauseFollow)cur.tr.scrollIntoView({block:'nearest',behavior:'smooth'});"
        "last=cur;};"
        "});"
        # Edit title (✏️) -> POST /title.
        "document.getElementById('edittitle').onclick=async()=>{"
        "const cur=document.getElementById('mtitle').textContent;"
        "const t=prompt('會議標題',cur);if(t==null||!t.trim())return;"
        f"const r=await fetch('/meetings/{mid}/title',{{method:'POST',"
        "headers:{'Content-Type':'application/json'},body:JSON.stringify({title:t.trim()})});"
        "if(r.ok)document.getElementById('mtitle').textContent=(await r.json()).title;};"
        # Click a speaker name -> rename that person across the whole meeting.
        # Speaker cell: click -> inline input backed by a datalist of known names
        # (pick a past speaker instead of retyping). Enter/blur commits, Esc cancels.
        "document.querySelectorAll('td.who[data-spk]').forEach(td=>{td.style.cursor='pointer';"
        "td.onclick=(e)=>{e.stopPropagation();if(td.querySelector('input'))return;"
        "const old=td.dataset.spk;let cancelled=false;"
        "const inp=document.createElement('input');inp.value=old;inp.setAttribute('list','spkdl');"
        "inp.autocomplete='off';inp.style.cssText='font:inherit;width:7em';"
        "td.textContent='';td.appendChild(inp);inp.focus();inp.select();"
        "const commit=async()=>{const nw=inp.value.trim();"
        "if(cancelled||!nw||nw===old){td.textContent=old;return;}"
        f"const r=await fetch('/meetings/{mid}/speaker',{{method:'POST',"
        "headers:{'Content-Type':'application/json'},body:JSON.stringify({old:old,new:nw})});"
        "if(r.ok)location.reload();else td.textContent=old;};"
        "inp.onkeydown=(ev)=>{if(ev.key==='Enter')inp.blur();"
        "else if(ev.key==='Escape'){cancelled=true;inp.blur();}};inp.onblur=commit;};});"
        "(function(){fetch('/speakers').then(r=>r.json()).then(j=>{"
        "let dl=document.getElementById('spkdl');if(!dl){dl=document.createElement('datalist');"
        "dl.id='spkdl';document.body.appendChild(dl);}"
        "dl.innerHTML=(j.speakers||[]).map(s=>`<option value=\"${_esc(s.name)}\">`).join('');});})();"
        # Tags: chips + remove (✕) + an add input (Enter to add).
        "function _esc(s){return (s||'').replace(/[&<>\"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;'}[c]));}"
        f"const MID={mid};"
        "async function _postTag(name,remove){return (await fetch(`/meetings/${MID}/tags`,{method:'POST',"
        "headers:{'Content-Type':'application/json'},body:JSON.stringify({name,remove})})).json();}"
        "function renderTags(list){const el=document.getElementById('tags');"
        "el.innerHTML=list.map(t=>`<span class=tg>#${_esc(t)}<span class=x data-rm=\"${_esc(t)}\">✕</span></span>`).join('')"
        "+`<input id=tagin placeholder='+ 標籤' list=tagdl autocomplete=off style='font-size:12px;padding:2px 8px;width:88px'>`;"
        "el.querySelectorAll('.x').forEach(s=>s.onclick=async()=>renderTags((await _postTag(s.dataset.rm,true)).tags));"
        "const ti=document.getElementById('tagin');ti.onkeydown=async(e)=>{"
        "if(e.key==='Enter'&&ti.value.trim()){const j=await _postTag(ti.value.trim(),false);renderTags(j.tags);}};}"
        f"renderTags({json.dumps(list(tags))});"
        # tag autocomplete: suggest previously-used tags (across all meetings)
        "fetch('/tags').then(r=>r.json()).then(j=>{"
        "let dl=document.getElementById('tagdl');if(!dl){dl=document.createElement('datalist');"
        "dl.id='tagdl';document.body.appendChild(dl);}"
        "dl.innerHTML=(j.tags||[]).map(t=>`<option value=\"${_esc(t.name||t)}\">`).join('');});"
        # screenshots: capture screen -> attach to meeting; grid with delete
        "function loadShots(){const el=document.getElementById('shots');if(!el)return;"
        "el.style.cssText='display:flex;flex-wrap:wrap;gap:10px';"
        "fetch(`/meetings/${MID}/shots`).then(r=>r.json()).then(j=>{"
        "el.innerHTML=(j.shots||[]).map(n=>`<div style=\"position:relative;width:180px\">"
        "<a href=\"/meetings/${MID}/shots/${n}\" target=_blank>"
        "<img src=\"/meetings/${MID}/shots/${n}\" loading=lazy style=\"max-width:100%;border-radius:8px;display:block\"></a>"
        "<button class=btn data-shot=\"${n}\" style=\"position:absolute;top:4px;right:4px;padding:.1em .4em\">✕</button></div>`).join('');"
        "el.querySelectorAll('[data-shot]').forEach(b=>b.onclick=async()=>{"
        "await fetch(`/meetings/${MID}/shots/delete`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:b.dataset.shot})});loadShots();});});}"
        "const _sb=document.getElementById('shot');if(_sb)_sb.onclick=async()=>{"
        "const mm=document.getElementById('shotmsg');"
        # capture in the browser (it owns the screen-share permission), then upload
        "let stream;try{stream=await navigator.mediaDevices.getDisplayMedia({video:true});}"
        "catch(e){mm.textContent='已取消';setTimeout(()=>{mm.textContent='';},2000);return;}"
        "mm.textContent='擷取中…';"
        "const v=document.createElement('video');v.srcObject=stream;await v.play();"
        "await new Promise(r=>setTimeout(r,250));"
        "const cv=document.createElement('canvas');cv.width=v.videoWidth;cv.height=v.videoHeight;"
        "cv.getContext('2d').drawImage(v,0,0);stream.getTracks().forEach(t=>t.stop());"
        "const blob=await new Promise(res=>cv.toBlob(res,'image/png'));"
        "const fd=new FormData();fd.append('img',blob,'shot.png');mm.textContent='上傳中…';"
        "try{const r=await fetch(`/meetings/${MID}/screenshot`,{method:'POST',body:fd});"
        "if(r.ok){mm.textContent='已擷取';loadShots();}else{mm.textContent='失敗：'+r.status;}}"
        "catch(err){mm.textContent='網路錯誤：'+err.message;}"
        "setTimeout(()=>{mm.textContent='';},2500);};"
        "loadShots();")
    return _shell(html.escape(meeting["title"]), body, script=script, back=True)


def create_app(store, *, summary_backend, asr_backend=None,
               live_manager=None, live_interim_backend=None, model_names=None,
               on_model_change=None,
               summary_model="mlx-lm", live_silence_ms=500, live_min_speech_ms=150,
               live_interim_s=1.2, live_max_utt_s=15.0, live_rms_threshold=500,
               live_max_lag_s=4.0, live_interim_duty=0.75):
    app = FastAPI()
    transcribe_jobs = {}  # mid -> progress dict; survives page refresh (in-memory)
    diarize_jobs = {}     # mid -> diarization progress dict (same shape)
    summary_jobs = {}     # mid -> summary-generation progress dict (same shape)

    # Idle auto-release: free loaded models after N seconds with no activity and no
    # live connection. Loaded weights lazy-reload on next use.
    idle = {"last": time.time(), "live": 0}
    live_active = {}  # mid -> open live connections; surfaced in /jobs (global popout)
    live_stop = set()  # mids the float control panel asked to stop (server-side)
    _panel = {"p": None}  # the floating control-panel subprocess (singleton)

    def _open_panel():
        """Launch the native float panel if installed; idempotent (reuse if alive)."""
        binp = backends.floatpanel_bin()
        if not binp:
            return False
        p = _panel["p"]
        if p and p.poll() is None:
            return True  # already running
        try:
            _panel["p"] = subprocess.Popen(
                [binp], start_new_session=True,
                env={**os.environ, "MEETING_PORT": os.environ.get("MEETING_PORT", "8765")})
            return True
        except Exception:  # noqa: BLE001
            return False
    idle_release_s = int(os.environ.get("LIVE_IDLE_RELEASE_S", "600"))

    def _touch():
        idle["last"] = time.time()

    def _idle_loop():
        import time as _t
        released = False
        while True:
            _t.sleep(60)
            if idle["live"] == 0 and not released \
                    and _t.time() - idle["last"] > idle_release_s:
                freed = _free_models()
                if freed:
                    print(f"[idle] released after {idle_release_s}s: {freed}", file=sys.stderr)
                released = True
            elif idle["live"] > 0 or _t.time() - idle["last"] <= idle_release_s:
                released = False  # re-arm once active again

    import threading as _threading
    _threading.Thread(target=_idle_loop, daemon=True).start()

    @app.get("/health")
    def health():
        # recording flag lets the meeting-watcher suppress notifications while live.
        return {"status": "ok", "recording": idle["live"] > 0}

    @app.get("/detect")
    def detect():
        # "In a meeting" = mic ACTUALLY in use. Teams/Slack/etc. run in the
        # background all day, so a process match alone false-fires (reported:
        # HUD said "MSTeams 進行中" with no mic). So when the micbusy helper is
        # available, mic is the real signal and the app name only LABELS it; the
        # process list is the trigger ONLY as a fallback when micbusy is absent.
        import meeting_watch as mw
        app_name = None
        try:
            app_name = mw.meeting_app_running()
        except Exception:
            pass
        mic = False
        try:
            mic = mw.mic_in_use()
        except Exception:
            pass
        have_mic = os.path.exists(mw._MICBUSY)
        meeting = mic if have_mic else bool(app_name)
        label = app_name or ("麥克風使用中" if mic else None)
        return {"meeting": bool(meeting), "app": label,
                "recording": idle["live"] > 0}

    @app.get("/live/state")
    def live_state():
        # For the floating control panel: is anything recording + which meeting.
        mids = list(live_active)
        return {"recording": bool(mids), "mid": mids[0] if mids else None,
                "count": len(mids)}

    @app.post("/live/stop")
    def live_stop_all():
        # Float panel "停止": ask every active live session to end (server-side);
        # each ws_live loop sees its mid in live_stop, breaks, flushes + finalizes.
        n = 0
        for m in list(live_active):
            live_stop.add(m)
            n += 1
        return {"stopping": n}

    @app.post("/floatpanel/open")
    def floatpanel_open():
        # Launch the native floating control panel (detached, idempotent).
        if not _open_panel():
            raise HTTPException(400, "floatpanel 未安裝（設定 → 加速 runtime → 安裝）")
        return {"opened": True}

    @app.post("/shutdown")
    def shutdown():
        # Clean in-app quit: kill the supervisor (so it won't restart us) + the
        # watcher, then exit. For the .app this ends its whole process -> app quits.
        def _kill():
            import subprocess
            import time as _t
            _t.sleep(0.3)  # let the HTTP response flush first
            # SIGKILL the supervisor FIRST so it can't restart the server mid-teardown.
            for pat in ("supervise.sh", "meeting_watch.py", "bootstrap.py"):
                subprocess.run(["pkill", "-9", "-f", pat], capture_output=True)
            port = os.environ.get("MEETING_PORT", "8765")
            subprocess.run(f"lsof -ti tcp:{port} | xargs kill -9", shell=True,
                           capture_output=True)  # frees port (kills any server incl self)
            os._exit(0)
        import threading as _th
        _th.Thread(target=_kill, daemon=True).start()
        return {"shutdown": True}

    @app.get("/manifest.webmanifest")
    def manifest():
        return Response(_MANIFEST, media_type="application/manifest+json")

    @app.get("/sw.js")
    def service_worker():
        return Response(_SW_JS, media_type="application/javascript")

    @app.get("/icon-{size}.png")
    def icon(size: int):
        return Response(_png_solid(size, (0x5b, 0x54, 0xe6)), media_type="image/png")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return _INDEX

    @app.get("/live", response_class=HTMLResponse)
    def live_page():
        # Live ANE (省電): show only on Apple Silicon with the prebuilt helper + the
        # toggle on. Selecting it -> /models -> make_live_backend('ane-live') -> the
        # persistent Neural-Engine helper (off the GPU).
        import backends as _b
        if (_apple_silicon() and _b.ane_helper_bin() is not None
                and store.get_setting("ane", "0") == "1"):
            body = _LIVE_BODY.replace(
                "<select id=model>",
                "<select id=model><optgroup label='🧠 NPU · ANE 省電'>"
                "<option value='ane-live'>Qwen3-ASR(ANE·省電·M系列)</option></optgroup>", 1)
            return _shell("Live · MeetingSummary", body, script=_LIVE_JS, back=True)
        return _LIVE

    @app.get("/m/{mid}", response_class=HTMLResponse)
    def meeting_page(mid: int):
        meeting = store.get_meeting(mid)
        if meeting is None:
            raise HTTPException(404, "meeting not found")
        # Tracks with retained audio (across all segments — handles merged meetings).
        audio_tracks = _meeting_tracks(store, mid)
        transcripts = _rows(store.list_transcripts(mid))
        # "本場認得" = speakers here that are global voiceprints also seen in OTHER
        # meetings (cross-meeting recognition), so the user sees persistence working.
        here = {r["speaker"] for r in transcripts}
        known = {s["name"]: s for s in store.speakers_with_stats()}
        recognized = sorted(n for n in here if known.get(n, {}).get("meetings", 0) > 1)
        ane_on = _ane_available() and store.get_setting("ane", "0") == "1"
        return _detail_page(mid, dict(meeting), transcripts,
                            _rows(store.list_summaries(mid)), audio_tracks,
                            tags=store.tags_for(mid), recognized=recognized, ane_on=ane_on)

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
        if backends.route(body.live) == "ane":  # pre-warm the helper (~13s load) off-thread
            backends.ane_warm()
        return {"live": live_manager.requested}

    @app.post("/live/prewarm")
    def live_prewarm():
        # Called on live-page load: warm the ANE helper ahead of the first record
        # so it isn't blocked on the ~13s CoreML load. No-op unless ANE is current.
        if live_manager is not None and backends.route(live_manager.current) == "ane":
            backends.ane_warm()
            return {"warming": True}
        return {"warming": False}

    @app.get("/models/manage", response_class=HTMLResponse)
    def models_manage():
        return _models_page()

    @app.get("/speakers/manage", response_class=HTMLResponse)
    def speakers_manage():
        return _speakers_page()

    @app.get("/changelog", response_class=HTMLResponse)
    def changelog_page():
        here = os.path.dirname(os.path.abspath(__file__))
        try:
            with open(os.path.join(here, "CHANGELOG.md")) as _f:
                md = _f.read()
        except Exception:
            md = "（找不到 CHANGELOG.md）"
        import updater
        ver = updater.current_version(here)
        body = (f"<h1>📝 變更紀錄 <span class=muted>v{html.escape(ver)}</span></h1>"
                f"<div class=card><div class=md>{_md_html(md)}</div></div>")
        return _shell("變更紀錄", body, back=True)

    @app.get("/models/status")
    def models_status():
        scan = _scan_model_cache()
        supported = []
        for m in _SUPPORTED:
            cached, size_mb, path = _model_cached(m)
            supported.append({**m, "cached": cached, "size_mb": size_mb, "path": path})
        sup_paths = {os.path.realpath(s["path"]) for s in supported if s["cached"]}
        other = [e for e in scan if os.path.realpath(e["path"]) not in sup_paths]
        import updater
        return {"runtimes": _runtime_status(), "supported": supported,
                "other": other, "tasks": _MODEL_TASKS,
                "version": updater.current_version(os.path.dirname(os.path.abspath(__file__))),
                "total_mb": round(sum(e["size_mb"] for e in scan), 1)}

    @app.post("/models/cache/delete")
    def models_cache_delete(body: PathIn):
        if not _safe_model_path(body.path):
            raise HTTPException(400, "path not under a known model cache root")
        import shutil
        if os.path.isdir(body.path):
            shutil.rmtree(body.path, ignore_errors=True)
        elif os.path.exists(body.path):
            os.remove(body.path)
        else:
            raise HTTPException(404, "not found")
        return {"deleted": body.path}

    @app.post("/models/download")
    def models_download(body: DownloadIn):
        m = next((x for x in _SUPPORTED if x["id"] == body.id), None)
        if m is None:
            raise HTTPException(400, "not a supported model")
        rt = _runtime_status()
        if m["kind"] in ("femelo", "chatllm") and not rt[m["kind"]]:
            raise HTTPException(409, f"{m['kind']} runtime not installed — 先按編譯安裝")

        def _dl():
            import subprocess
            _MODEL_TASKS[m["id"]] = {"state": "running", "msg": "下載中…"}
            try:
                if m["kind"] == "hf":
                    from huggingface_hub import snapshot_download
                    snapshot_download(m["id"])
                elif m["kind"] == "femelo":
                    subprocess.run([".venv-qwen314/bin/python", "-c",
                                    "from py_qwen3_asr_cpp.model import Qwen3ASRModel as M;"
                                    f"M(asr_model='{m['id']}')"],
                                   check=True, timeout=3600,
                                   capture_output=True, text=True)
                else:  # chatllm: model_downloader resolves :id and fetches
                    subprocess.run([".venv/bin/python",
                                    "chatllm.cpp/scripts/model_downloader.py",
                                    ":qwen3-asr:1.7b"],
                                   check=True, timeout=3600,
                                   capture_output=True, text=True)
                _MODEL_TASKS[m["id"]] = {"state": "done", "msg": "完成"}
            except Exception as e:
                detail = getattr(e, "stderr", "") or str(e)
                detail = detail.strip().splitlines()[-1] if detail.strip() else str(e)
                _MODEL_TASKS[m["id"]] = {"state": "error", "msg": detail[:300]}
                print(f"download failed {m['id']}: {e}", file=sys.stderr)
        import threading
        threading.Thread(target=_dl, daemon=True).start()
        return {"downloading": m["id"]}

    @app.post("/models/free")
    def models_free():
        return {"freed": _free_models()}

    @app.get("/update/check")
    def update_check():
        import updater
        return updater.check(os.environ.get("MEETING_REPO", "a9650615/MeetingSummary"),
                             os.path.dirname(os.path.abspath(__file__)))

    @app.post("/update/apply")
    def update_apply():
        import updater
        here = os.path.dirname(os.path.abspath(__file__))
        info = updater.apply(os.environ.get("MEETING_REPO", "a9650615/MeetingSummary"), here)
        if info.get("applied"):
            # restart so the new code loads (supervisor relaunches; bare run exits).
            def _restart():
                import time as _t
                _t.sleep(0.4)
                os._exit(3)   # supervisor relaunches with the new code
            import threading as _th
            _th.Thread(target=_restart, daemon=True).start()
        return info

    @app.post("/models/setup")
    def models_setup(body: SetupIn):
        if body.runtime not in ("femelo", "chatllm", "speech", "qwen3-ane",
                                "audiocap", "floatpanel"):
            raise HTTPException(400, "unknown runtime")
        # native helpers are prebuilt — downloaded, not compiled on the client
        _download_only = body.runtime in ("audiocap", "floatpanel")

        def _build():
            import subprocess
            _MODEL_TASKS[body.runtime] = {
                "state": "running",
                "msg": "下載中…" if _download_only else "安裝中…(可能數分鐘)"}
            try:
                r = subprocess.run(["bash", "setup_runtime.sh", body.runtime],
                                   timeout=3600, capture_output=True, text=True)
                if r.returncode != 0:
                    tail = (r.stderr or r.stdout).strip().splitlines()
                    raise RuntimeError(tail[-1] if tail else f"exit {r.returncode}")
                _MODEL_TASKS[body.runtime] = {"state": "done", "msg": "完成"}
            except Exception as e:
                _MODEL_TASKS[body.runtime] = {"state": "error", "msg": str(e)[:300]}
                print(f"setup {body.runtime} failed: {e}", file=sys.stderr)
        import threading
        threading.Thread(target=_build, daemon=True).start()
        return {"building": body.runtime}

    @app.post("/models/runtime/delete")
    def models_runtime_delete(body: SetupIn):
        # Uninstall a .cpp runtime = remove its build dir -> status flips to 未安裝.
        # Re-compile re-fetches/builds. Model weights live elsewhere, untouched.
        import shutil
        if body.runtime == "speech":  # brew formula, not a build dir
            import subprocess
            subprocess.run(["brew", "uninstall", "speech"], capture_output=True)
            return {"runtime": "speech", "removed": True}
        targets = {"femelo": ".venv-qwen314", "chatllm": "chatllm.cpp",
                   "qwen3-ane": "swift/qwen3-ane/.build"}
        d = targets.get(body.runtime)
        if not d:
            raise HTTPException(400, "unknown runtime")
        p = os.path.abspath(d)
        existed = os.path.isdir(p)
        shutil.rmtree(p, ignore_errors=True)
        _MODEL_TASKS.pop(body.runtime, None)
        return {"removed": existed, "runtime": body.runtime}

    @app.websocket("/ws/live")
    async def ws_live(ws: WebSocket):
        await ws.accept()
        if live_manager is None:
            await ws.send_json({"type": "error", "msg": "no live backend"})
            await ws.close()
            return
        idle["live"] += 1  # block idle-release while recording
        _touch()
        src = ws.query_params.get("src", "mic")
        dual = src == "dual"
        t0 = time.time()
        # Session: a recording binds to an existing meeting (token = meeting id) so
        # stop/resume stays in one session; a missing/unknown token starts a new one.
        session = ws.query_params.get("session")
        if session and session.isdigit() and store.get_meeting(int(session)) is not None:
            mid = int(session)
        else:  # default title = local date-time (distinct per session, sortable)
            title = time.strftime("錄音 %Y-%m-%d %H:%M", time.localtime(t0))
            mid = store.create_meeting(title, t0, "zh-TW")
        # On resume, this connection's audio is placed at (t0 - meeting.created_at)
        # in the assembled track; live start_ms is per-connection (0-based), so add
        # the same offset or resumed transcripts collide with the first session at 0.
        conn_offset_ms = max(0, int((t0 - store.get_meeting(mid)["created_at"]) * 1000))
        live_active[mid] = live_active.get(mid, 0) + 1  # show in the global popout
        await ws.send_json({"type": "meeting", "id": mid})
        if store.get_setting("float_panel", "0") == "1":  # auto-open native panel
            _open_panel()

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
        live_manager.set_language(q.get("lang") or None)  # ""/absent -> auto-detect

        def _mk_speech_fn():  # silero VAD per track (own RNN state); None -> energy
            # silero is the DEFAULT now — neural VAD endpoints far better than energy
            # RMS (energy reads soft trailing syllables as silence -> cuts words). Only
            # ?vad=energy opts out; missing/broken model falls back to energy.
            if q.get("vad") == "energy":
                return None
            try:
                from live import SileroVad
                return SileroVad("models/silero_vad_v4.onnx")
            except Exception as e:
                print(f"silero vad unavailable, using energy: {e}", file=sys.stderr)
                return None

        # ANE live (省電): skip the MLX whisper-small interim entirely — that's the
        # GPU draw during live. Final-only on the Neural Engine = minimal GPU.
        import backends as _bk  # noqa: PLC0415
        ane_live = _bk.route(live_manager.current) == "ane"
        interim_be = None if ane_live else live_interim_backend
        sessions = {tag: TwoPassSession(
            backend=live_manager, interim_backend=interim_be,
            sample_rate=16000, silence_ms=sil,
            min_speech_ms=live_min_speech_ms, interim_s=live_interim_s,
            max_utt_s=maxu, rms_threshold=live_rms_threshold,
            interim_duty=live_interim_duty,
            track=lbl[0], speech_fn=_mk_speech_fn()) for tag, lbl in tracks.items()}
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
                store.add_transcript(mid, "live", track,
                                     ev["start_ms"] + conn_offset_ms,
                                     ev.get("end_ms", ev["start_ms"]) + conn_offset_ms,
                                     spk, ev["text"])
                await ws.send_json({"type": "final", **ev, "speaker": spk})
            else:
                await ws.send_json({"type": "interim", **ev, "speaker": speaker})

        # Wall-clock continuous recording: on every received chunk, pad EVERY track
        # with silence up to now-t0, then append the chunk to its track. This makes
        # all tracks the same length (= session duration) and on one shared clock,
        # regardless of when each browser stream starts or how it's gated — so 我/對方
        # timestamps and lengths line up. File + session buffer get the same bytes, so
        # committed-bytes == file position == wall-clock (seek/highlight aligned).
        # ponytail: no lag-drop — it broke the invariant; small-q4 keeps up. Add a
        # smarter back-pressure cap only if a slow model makes the buffer balloon.
        got = asyncio.Event()
        closed = False
        # 純錄音 (record-only): capture + save PCM but run NO ASR — zero inference,
        # zero heat (for "電腦快炸" / quick capture). Re-transcribe later. Hot-
        # toggleable mid-recording via a {type:'mode'} control message.
        rec = {"on": ws.query_params.get("record_only") == "1"}
        interim_lag_bytes = int(2 * live_interim_s * 16000) * 2
        written = {tag: 0 for tag in tracks}  # bytes laid down per track (wall-clock)

        def _pad_to(now):
            target = int((now - t0) * 16000) * 2
            for tag in tracks:
                gap = target - written[tag]
                gap -= gap % 2
                if gap > 0:
                    sil = bytes(gap)
                    audio_files[tag].write(sil)
                    buffers[tag].extend(sil)
                    written[tag] += gap

        async def receiver():
            nonlocal closed
            try:
                while True:
                    msg = await ws.receive()
                    if msg["type"] == "websocket.disconnect":
                        break
                    txt = msg.get("text")
                    if txt is not None:                 # control message (mode toggle)
                        try:
                            o = json.loads(txt)
                            if o.get("type") == "mode":
                                rec["on"] = bool(o.get("record_only"))
                        except Exception:
                            pass
                        continue
                    data = msg.get("bytes")
                    if data is None:
                        continue
                    tag, pcm = (data[0], data[1:]) if dual else (0, data)
                    if tag not in buffers:
                        continue
                    _pad_to(time.time())          # keep all tracks at wall-clock
                    audio_files[tag].write(pcm)
                    buffers[tag].extend(pcm)
                    written[tag] += len(pcm)
                    got.set()
            except WebSocketDisconnect:
                pass
            closed = True
            got.set()

        rtask = asyncio.create_task(receiver())
        # Native system audio (ScreenCaptureKit): when ?native_sys=1, the SERVER
        # captures system audio via the audiocap helper and feeds it into the
        # system/對方 track — no per-session browser "share screen + audio" dialog.
        # Same event loop as receiver() (asyncio subprocess) -> no thread races.
        nrtask = None
        nstask = None
        audiocap_proc = None
        if ws.query_params.get("native_sys") == "1":
            sys_tag = next((t for t, (trk, _l) in tracks.items() if trk == "system"), None)
            binp = _bk.audiocap_bin()
            if sys_tag is not None and binp:
                audiocap_proc = await asyncio.create_subprocess_exec(
                    binp, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)

                async def native_reader(tag=sys_tag, proc=audiocap_proc):
                    try:
                        while True:
                            data = await proc.stdout.read(8192)
                            if not data:
                                break
                            _pad_to(time.time())
                            audio_files[tag].write(data)
                            buffers[tag].extend(data)
                            written[tag] += len(data)
                            got.set()
                    except Exception as e:  # noqa: BLE001
                        print(f"native_sys reader stopped: {e}", file=sys.stderr)

                async def native_stderr(proc=audiocap_proc):
                    # Drain helper stderr; surface permission/errors to the live page
                    # (denial -> no audio -> the consumer loop never wakes, so the
                    # notice MUST come from here, not the main loop).
                    try:
                        while True:
                            ln = await proc.stderr.readline()
                            if not ln:
                                break
                            s = ln.decode("utf8", "ignore").strip()
                            if not s or s == "READY":
                                continue
                            print(f"audiocap: {s}", file=sys.stderr)
                            if any(k in s for k in ("NOPERM", "ERR", "-3801", "TCC")):
                                try:
                                    await ws.send_json({"type": "notice",
                                                        "msg": "原生系統音擷取失敗：" + s[:180]})
                                except Exception:  # noqa: BLE001
                                    pass
                    except Exception:  # noqa: BLE001
                        pass
                nrtask = asyncio.create_task(native_reader())
                nstask = asyncio.create_task(native_stderr())
            else:
                await ws.send_json({"type": "notice",
                                    "msg": "原生系統音擷取不可用（缺 helper 或需螢幕錄製權限）"})
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
                    if rec["on"]:
                        continue  # 純錄音: PCM already saved; skip ASR (no inference)
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
                if mid in live_stop:  # remote stop — AFTER draining this round's buffers
                    live_stop.discard(mid)
                    break
        finally:
            rtask.cancel()
            if nrtask:
                nrtask.cancel()
            if nstask:
                nstask.cancel()
            if audiocap_proc and audiocap_proc.returncode is None:
                try:
                    audiocap_proc.terminate()
                except ProcessLookupError:
                    pass
            _pad_to(time.time())  # equalize track lengths to the final wall-clock
            for tag, s in sessions.items():
                for ev in await run_in_threadpool(s.flush):
                    if ev["kind"] == "final":  # audio-position offset, not wall-clock
                        spk = ev.get("speaker") or tracks[tag][1]
                        store.add_transcript(mid, "live", tracks[tag][0],
                                             ev["start_ms"] + conn_offset_ms,
                                             ev.get("end_ms", ev["start_ms"]) + conn_offset_ms,
                                             spk, ev["text"])
            for f in audio_files.values():
                f.close()
            idle["live"] = max(0, idle["live"] - 1)
            if live_active.get(mid, 0) <= 1:
                live_active.pop(mid, None)
            else:
                live_active[mid] -= 1
            _touch()  # idle countdown starts when recording stops
            # Stop != finalize — explicit only.

    @app.post("/meetings")
    def create_meeting(m: MeetingIn):
        return {"id": store.create_meeting(m.title, time.time(), m.lang)}

    @app.get("/meetings")
    def list_meetings():
        out = []
        for m in store.list_meetings():
            d = dict(m)
            segs = store.list_segments(m["id"])
            d["has_audio"] = bool(_meeting_tracks(store, m["id"]))
            d["tags"] = store.tags_for(m["id"])
            d["duration_s"] = round(sum((s["duration_s"] or 0) for s in segs))
            d["n_segments"] = len(segs)
            out.append(d)
        return out

    @app.get("/tags")
    def list_tags():
        return {"tags": store.all_tags()}

    @app.post("/meetings/{mid}/tags")
    def tag_meeting(mid: int, body: TagIn):
        if store.get_meeting(mid) is None:
            raise HTTPException(404, "meeting not found")
        if body.remove:
            store.remove_tag(mid, body.name)
        else:
            store.add_tag(mid, body.name)
        return {"tags": store.tags_for(mid)}

    @app.get("/search")
    def search(q: str = ""):
        results = []
        for r in store.search(q):
            snip = r["t_snip"] or r["s_snip"] or r["title"]
            results.append({"id": r["id"], "title": r["title"],
                            "created_at": r["created_at"],
                            "snippet": _snippet(snip, q)})
        return {"q": q, "results": results}

    @app.get("/meetings/{mid}/export")
    def export_meeting(mid: int):
        meeting = store.get_meeting(mid)
        if meeting is None:
            raise HTTPException(404, "meeting not found")
        md = _export_text(dict(meeting), store.list_transcripts(mid),
                          store.list_summaries(mid))
        safe = "".join(c for c in meeting["title"] if c.isalnum() or c in " -_")[:40].strip()
        fname = (safe or f"meeting-{mid}") + ".md"
        return Response(md, media_type="text/markdown; charset=utf-8",
                        headers={"Content-Disposition":
                                 f"attachment; filename*=UTF-8''{quote(fname)}"})

    @app.post("/meetings/{mid}/title")
    def rename_meeting(mid: int, body: TitleIn):
        if store.get_meeting(mid) is None:
            raise HTTPException(404, "meeting not found")
        store.update_title(mid, body.title.strip() or "未命名")
        return {"title": body.title.strip() or "未命名"}

    @app.post("/meetings/{mid}/speaker")
    def rename_speaker(mid: int, body: SpeakerRenameIn):
        if store.get_meeting(mid) is None:
            raise HTTPException(404, "meeting not found")
        new = body.new.strip()
        n = store.rename_speaker(mid, body.old, new)
        # propagate to the global voiceprint so future meetings auto-use the new name
        store.rename_global_speaker(body.old, new)
        return {"renamed": n}

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

    def _default_asr():
        # The auto/default ASR backend. When the ANE 省電 toggle is on (Apple
        # Silicon), the automatic pipeline (upload + default re-transcribe) also
        # runs on the Neural Engine — off the GPU — not just manual dropdown picks.
        if (asr_backend is not None and _ane_available()
                and store.get_setting("ane", "0") == "1"):
            import backends
            return backends.make_batch_backend("ane-qwen3-0.6b")
        return asr_backend

    @app.post("/meetings/{mid}/transcribe")
    def transcribe_meeting(mid: int, body: TranscribeIn = TranscribeIn()):
        # Re-run accurate ASR over the saved audio, optionally with a chosen model.
        # Sync route -> FastAPI runs it in a threadpool, so /health stays responsive.
        if store.get_meeting(mid) is None:
            raise HTTPException(404, "meeting not found")
        if body.model:
            import backends
            backend = backends.make_batch_backend(body.model, body.language)
        else:
            backend = _default_asr()
        if backend is None:
            raise HTTPException(503, "no ASR backend configured")
        store.clear_transcripts(mid)  # full replace (incl live) -> one coherent set
        base = store.get_meeting(mid)["created_at"]
        n = 0
        for seg in store.list_segments(mid):
            off_ms = max(0, int((seg["started_at"] - base) * 1000))  # align to timeline
            for track_label, name in (("system", "system.pcm"), ("mic", "mic.pcm")):
                pcm = f"{seg['dir_path']}/{name}"
                if not os.path.exists(pcm) or os.path.getsize(pcm) == 0:
                    continue  # track not present for this segment
                speaker = _TRACK_LABEL.get(track_label, track_label)
                for t in asr.transcribe(pcm, profile="accurate",
                                        track=track_label, backend=backend):
                    store.add_transcript(mid, t["profile"], t["track"],
                                         t["start_ms"] + off_ms, t["end_ms"] + off_ms,
                                         speaker, t["text"])
                    n += 1
        return {"transcripts": n}

    @app.post("/meetings/{mid}/transcribe/start")
    def transcribe_start(mid: int, body: TranscribeIn = TranscribeIn()):
        # Background job + server-side progress -> survives a page refresh.
        _touch()
        if store.get_meeting(mid) is None:
            raise HTTPException(404, "meeting not found")
        if transcribe_jobs.get(mid, {}).get("state") == "running":
            return {"state": "running"}  # already in progress
        if body.model:
            import backends
            backend = backends.make_batch_backend(body.model, body.language)
        else:
            backend = _default_asr()
        if backend is None:
            raise HTTPException(503, "no ASR backend configured")
        store.clear_transcripts(mid)  # full replace (incl live) -> one coherent set
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
        _touch()
        if asr_backend is None:
            raise HTTPException(503, "no ASR backend configured")
        os.makedirs("data/uploads", exist_ok=True)
        # strip any directory components from the client-supplied filename — Starlette
        # does NOT, so "../../x" / absolute paths would escape data/uploads.
        safe = os.path.basename(audio.filename or "") or "upload"
        path = os.path.join("data/uploads", safe)
        with open(path, "wb") as f:
            f.write(await audio.read())
        mid = store.create_meeting(title, time.time(), lang)
        # Async: transcribe+summary is minutes of work. Kick it to a background
        # thread and redirect to the meeting page, which polls the same progress
        # job re-transcribe uses (refresh-safe) and reloads when done. The POST
        # returns at once instead of holding the connection (and tripping the
        # supervisor's hang check) for the whole job.
        import threading
        transcribe_jobs[mid] = {"state": "running", "done": 0, "total": 0,
                                "text": "辨識中…"}
        threading.Thread(
            target=_run_upload_job,
            args=(store, mid, path, _default_asr(), summary_backend, summary_model,
                  kind, title, transcribe_jobs),
            daemon=True).start()
        return RedirectResponse(f"/m/{mid}", status_code=303)

    @app.post("/meetings/{mid}/summary")
    def summarize_meeting(mid: int, body: SummaryIn):
        # Async: the LLM call is seconds-to-minutes. Background thread (progress ->
        # summary_jobs, polled by the page) + return at once. The done payload
        # carries text/title so the page renders inline (no reload).
        _touch()
        if store.get_meeting(mid) is None:
            raise HTTPException(404, "meeting not found")
        if summary_jobs.get(mid, {}).get("state") == "running":
            return {"started": True}
        import threading
        summary_jobs[mid] = {"state": "running", "done": 0, "total": 0,
                             "text": "產生摘要中…"}
        threading.Thread(target=_run_summary_job,
                         args=(store, mid, body.kind, summary_backend,
                               summary_model, summary_jobs), daemon=True).start()
        return {"started": True}

    @app.get("/meetings/{mid}/summary/progress")
    def summary_progress(mid: int):
        return summary_jobs.get(mid, {"state": "idle"})

    @app.post("/meetings/{mid}/notes")
    def set_notes_route(mid: int, body: SettingIn):
        if store.get_meeting(mid) is None:
            raise HTTPException(404, "meeting not found")
        store.set_notes(mid, body.value)
        return {"ok": True}

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
        # Async: diarization is minutes of sherpa clustering. Kick it to a
        # background thread (progress -> diarize_jobs, polled by the page) and
        # return at once so the POST doesn't hold the connection / look hung.
        if store.get_meeting(mid) is None:
            raise HTTPException(404, "meeting not found")
        if diarize_jobs.get(mid, {}).get("state") == "running":
            return {"started": True}
        import threading
        diarize_jobs[mid] = {"state": "running", "done": 0, "total": 0,
                             "text": "準備中…"}
        threading.Thread(target=_run_diarize_job,
                         args=(store, mid, body, diarize_jobs), daemon=True).start()
        return {"started": True}

    @app.get("/meetings/{mid}/diarize/progress")
    def diarize_progress(mid: int):
        return diarize_jobs.get(mid, {"state": "idle"})

    @app.get("/jobs")
    def jobs_status():
        # Global view of in-flight background work, so the home list can flag which
        # recordings are processing. Upload transcribe+summary writes to
        # transcribe_jobs, so an uploading meeting shows under 辨識.
        out = []
        for mid, n in list(live_active.items()):  # live recording = ongoing 辨識
            if n > 0:
                m = store.get_meeting(mid)
                out.append({"mid": mid, "kind": "錄音", "done": 0, "total": 0,
                            "text": "🔴 即時辨識中", "title": m["title"] if m else None})
        for kind, jd in (("辨識", transcribe_jobs), ("分群", diarize_jobs),
                         ("摘要", summary_jobs)):
            for mid, j in list(jd.items()):
                if j.get("state") == "running":
                    m = store.get_meeting(mid)
                    out.append({"mid": mid, "kind": kind, "done": j.get("done", 0),
                                "total": j.get("total", 0), "text": j.get("text", ""),
                                "title": m["title"] if m else None})
        return {"jobs": out}

    # --- persistent global speakers (cross-meeting voiceprint library) ---
    @app.get("/speakers")
    def speakers_list():
        return {"speakers": _rows(store.speakers_with_stats()),
                "persist": store.get_setting("persist_speakers", "1") == "1"}

    @app.post("/speakers/rename")
    def speaker_rename(body: SpeakerRenameIn):
        # Name-centric: rename every voiceprint + transcript of `old` to `new`.
        n = store.rename_speaker_global(body.old, body.new.strip())
        return {"renamed": n, "name": body.new.strip()}

    @app.post("/speakers/merge")
    def speaker_merge(body: SpeakerMergeIn):
        return {"moved": store.merge_speakers(body.keep.strip(), body.drop.strip())}

    @app.post("/speakers/delete")
    def speaker_delete(body: NameIn):
        store.delete_speakers_by_name(body.name)
        return {"deleted": body.name}

    @app.get("/speakers/{name}/utterances")
    def speaker_utterances(name: str):
        return {"name": name, "utterances": _rows(store.speaker_utterances(name))}

    @app.get("/speakers/{name}/sample.wav")
    def speaker_sample(name: str):
        # A short voice clip so you can 試聽 who a speaker is before naming/merging.
        # Longest utterance for that name, clipped from the (created_at-aligned)
        # track it was diarized on — same timeline as the transcript timestamps.
        import recorder
        span = store.speaker_best_span(name)
        if span is None:
            raise HTTPException(404, "no audio sample for this speaker")
        pcm = _assemble_track(store, span["meeting_id"], span["track"])
        if pcm is None:
            raise HTTPException(404, "no audio")
        sr = 16000
        a = int(span["start_ms"] / 1000 * sr) * 2
        b = int(min(span["end_ms"], span["start_ms"] + 6000) / 1000 * sr) * 2
        clip = pcm[a:max(a + 2, b)]
        if not clip:
            raise HTTPException(404, "empty clip")
        return Response(recorder.pcm_to_wav(clip, sample_rate=sr, channels=1),
                        media_type="audio/wav")

    @app.get("/speakers/suggestions")
    def speaker_suggestions():
        import diarize as diar
        thr = float(store.get_setting("merge_suggest_threshold", "0.5"))
        # Only suggest merging speakers you can actually 試聽 — a voiceprint with no
        # playable sample (>800ms utterance) can't be verified by ear, so comparing
        # it is useless ("沒有語音就不應該拿來比較是不是同一人").
        sampled = {s["name"] for s in store.speakers_with_stats() if s["has_sample"]}
        rows = [r for r in store.list_speakers() if r["name"] in sampled]
        return {"pairs": diar.similar_speaker_pairs(
            rows, thr, dismissed=store.list_speaker_nonmatches())}

    @app.post("/speakers/nonmatch")
    def speaker_nonmatch(body: SpeakerMergeIn):
        # "不是同一人" — remember the pair so it stops being suggested.
        store.add_speaker_nonmatch(body.keep.strip(), body.drop.strip())
        return {"ok": True}

    @app.get("/storage")
    def storage():
        # Disk usage of app-generated data, so you can see what's eating space.
        def dsize(p):
            if not p or not os.path.exists(p):
                return 0
            if os.path.isfile(p):
                try:
                    return os.path.getsize(p)
                except OSError:
                    return 0
            tot = 0
            for root, _d, files in os.walk(p):
                for f in files:
                    try:
                        tot += os.path.getsize(os.path.join(root, f))
                    except OSError:
                        pass
            return tot
        db_b = dsize(store.db_path)
        data_b = dsize("data")
        # db often lives under data/ -> don't double-count it in the data category
        if os.path.abspath(store.db_path).startswith(os.path.abspath("data") + os.sep):
            data_b = max(0, data_b - db_b)
        cats = [
            {"label": "錄音／逐字稿音檔 (data/)", "bytes": data_b},
            {"label": "資料庫", "bytes": db_b},
            {"label": "語音模型 (models/)", "bytes": dsize("models")},
            {"label": "MLX 模型 (mlx_models/)", "bytes": dsize("mlx_models")},
        ]
        meetings = []
        for m in store.list_meetings():
            b = sum(dsize(s["dir_path"]) for s in store.list_segments(m["id"]))
            if b:
                meetings.append({"id": m["id"], "title": m["title"], "bytes": b})
        meetings.sort(key=lambda x: -x["bytes"])
        return {"categories": cats, "total": sum(c["bytes"] for c in cats),
                "meetings": meetings[:12]}

    @app.post("/meetings/{mid}/screenshot")
    async def screenshot(mid: int, img: UploadFile = File(...)):
        # The browser captures the frame (it owns the screen-share permission —
        # macOS Screen-Recording is per-app/TCC, so a headless server can't run
        # `screencapture`). We just store the uploaded PNG against the meeting.
        if store.get_meeting(mid) is None:
            raise HTTPException(404, "meeting not found")
        data = await img.read()
        if not data:
            raise HTTPException(400, "empty image")
        d = f"data/{mid}/shots"
        os.makedirs(d, exist_ok=True)
        name = f"{int(time.time() * 1000)}.png"
        with open(os.path.join(d, name), "wb") as f:
            f.write(data)
        return {"name": name}

    @app.get("/meetings/{mid}/shots")
    def list_shots(mid: int):
        d = f"data/{mid}/shots"
        shots = sorted(os.listdir(d)) if os.path.isdir(d) else []
        return {"shots": [s for s in shots if s.endswith(".png")]}

    @app.get("/meetings/{mid}/shots/{name}")
    def get_shot(mid: int, name: str):
        if "/" in name or ".." in name:  # path-traversal guard
            raise HTTPException(400, "bad name")
        path = f"data/{mid}/shots/{name}"
        if not os.path.exists(path):
            raise HTTPException(404, "not found")
        return FileResponse(path, media_type="image/png")

    @app.post("/meetings/{mid}/shots/delete")
    def delete_shot(mid: int, body: NameIn):
        name = body.name
        if "/" in name or ".." in name:
            raise HTTPException(400, "bad name")
        path = f"data/{mid}/shots/{name}"
        if os.path.exists(path):
            os.remove(path)
        return {"deleted": name}

    _SETTINGS = {"persist_speakers": "1", "speaker_threshold": "0.62", "ane": "0",
                 "denoise": "0", "float_panel": "0"}

    @app.get("/settings/{key}")
    def get_setting_route(key: str):
        if key not in _SETTINGS:
            raise HTTPException(404, "unknown setting")
        return {"value": store.get_setting(key, _SETTINGS[key])}

    @app.post("/settings/{key}")
    def set_setting_route(key: str, body: SettingIn):
        if key not in _SETTINGS:
            raise HTTPException(404, "unknown setting")
        v = body.value
        if key == "persist_speakers":
            v = "1" if v in ("1", "true", "on") else "0"
        store.set_setting(key, v)
        return {"value": store.get_setting(key, _SETTINGS[key])}

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
    # Fresh install (no .cpp sidecar built yet)? Don't silently default to a runtime
    # that isn't installed — fall back to whisper so live works out of the box.
    import backends as _b
    _rt = _runtime_status()
    if (_b.route(live_model) == "qwen3cpp" and not _rt["femelo"]) or \
       (_b.route(live_model) == "chatllm" and not _rt["chatllm"]):
        print(f"[profile] {live_model} runtime not installed -> whisper-small-mlx-q4",
              file=sys.stderr)
        live_model = "mlx-community/whisper-small-mlx-q4"
    live_interim_model = os.environ.get("LIVE_INTERIM_MODEL", rec["interim"])
    live_silence = int(os.environ.get("LIVE_SILENCE_MS", "500"))  # 500: less aggressive
    # split than 400 (measured: 350->4 frags, 450/500->1, 800->0 but +400ms lag)
    live_min_speech = int(os.environ.get("LIVE_MIN_SPEECH_MS", "150"))  # keep short words
    live_interim_s = float(os.environ.get("LIVE_INTERIM_S", "1.2"))  # ~1/s preview revise (not 2-3/s)
    # target interim ASR duty cycle (compute/realtime); cadence auto-adapts to hold it
    live_interim_duty = float(os.environ.get("LIVE_INTERIM_DUTY", "0.75"))
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
    if backends.route(live_model) == "ane":  # default is ANE -> warm at boot
        backends.ane_warm()

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
        live_interim_duty=live_interim_duty,
        live_rms_threshold=live_rms,
        live_max_lag_s=live_max_lag,
        summary_model=llm_model,
    )
    port = int(os.environ.get("MEETING_PORT", "8765"))  # 8000 left free for dev
    uvicorn.run(app, host="127.0.0.1", port=port)  # loopback only (G2)
