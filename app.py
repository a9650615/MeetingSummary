"""Web app: FastAPI. Binds 127.0.0.1 only (privacy, spec G2) — no auth by
design, loopback-only. Backends injected so tests run without MLX.

Phase 1 = batch loop: create meeting -> transcribe saved PCM -> summarize -> view.
Live websocket + recording start/stop (Swift helper) land in Phase 2."""
import asyncio
import html
import json
import os
import sys
import time
from typing import Optional  # not `X | None` — pydantic evals it, fails on 3.9 venvs
from urllib.parse import quote

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
:root{--bg:#eef1f6;--surface:#fff;--surface2:#f7f8fb;--ink:#11141a;--muted:#6b7384;
 --line:#e4e7ef;--accent:#5b54e6;--accent2:#7c75ff;--accentsoft:#ecebfd;
 --me:#1565c0;--other:#2e7d32;--danger:#e5484d;--ok:#1e9e57;
 --radius:16px;--radius-sm:10px;
 --shadow:0 1px 2px rgba(16,24,40,.05),0 10px 30px -12px rgba(16,24,40,.18);
 --capbg:#0d1017;--capink:#f3f5fa}
:root[data-theme=dark]{
 --bg:#0c0e13;--surface:#15181f;--surface2:#1b1f28;--ink:#e9edf4;--muted:#8b93a4;
 --line:#262b36;--accent:#8b85ff;--accent2:#a39dff;--accentsoft:#23223a;
 --me:#6db3ff;--other:#74d99a;--danger:#ff6b6f;
 --shadow:0 1px 2px rgba(0,0,0,.4),0 14px 34px -14px rgba(0,0,0,.6);
 --capbg:#05070b;--capink:#f3f5fa}
body{margin:0;min-height:100vh;background:var(--bg);color:var(--ink);-webkit-font-smoothing:antialiased;
 font:15px/1.62 -apple-system,BlinkMacSystemFont,"PingFang TC","Noto Sans TC","Segoe UI",Roboto,sans-serif}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
.wrap{max-width:900px;margin:0 auto;padding:20px 20px 80px}
header.top{display:flex;align-items:center;gap:12px;margin:6px 0 22px;position:sticky;top:0;z-index:40;
 padding:10px 4px;backdrop-filter:saturate(1.4) blur(10px)}
.brand{font-weight:800;font-size:17px;letter-spacing:-.01em}
.brand .dot{color:var(--accent)}
.spacer{flex:1}
h1{font-size:26px;font-weight:820;letter-spacing:-.025em;margin:.1em 0 .7em}
h2{font-size:15px;font-weight:750;margin:26px 0 12px;letter-spacing:-.01em}
h3{font-size:13px;font-weight:750;margin:14px 0 6px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.card{background:var(--surface);border:1px solid var(--line);border-radius:var(--radius);
 box-shadow:var(--shadow);padding:20px 22px;margin:14px 0}
.muted{color:var(--muted)}.small{font-size:13px}
.row{display:flex;flex-wrap:wrap;gap:12px;align-items:end}
.row label.chk{align-self:center}
label.fld{display:inline-flex;flex-direction:column;gap:5px;font-size:11px;color:var(--muted);
 font-weight:700;text-transform:uppercase;letter-spacing:.04em}
label.chk{display:inline-flex;align-items:center;gap:6px;font-size:13px;color:var(--ink);text-transform:none}
input[type=text],input[type=search],input[type=file],input:not([type]),select{font:inherit;
 padding:.58em .7em;border:1px solid var(--line);border-radius:var(--radius-sm);
 background:var(--surface2);color:var(--ink);transition:.12s}
input:hover,select:hover{border-color:color-mix(in srgb,var(--accent) 40%,var(--line))}
input:focus,select:focus{outline:0;border-color:var(--accent);box-shadow:0 0 0 3px var(--accentsoft)}
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
.btn{font:inherit;font-weight:650;padding:.58em 1.05em;border-radius:var(--radius-sm);
 border:1px solid var(--line);background:var(--surface);color:var(--ink);cursor:pointer;
 transition:.13s;display:inline-block;line-height:1.2}
.btn:hover{border-color:var(--accent);color:var(--accent);transform:translateY(-1px)}
.btn:active{transform:translateY(0)}
.btn:disabled{opacity:.45;cursor:not-allowed;transform:none;border-color:var(--line);color:var(--muted)}
.btn.primary{background:linear-gradient(180deg,var(--accent2),var(--accent));border-color:transparent;
 color:#fff;box-shadow:0 6px 16px -6px var(--accent)}
.btn.primary:hover{filter:brightness(1.06);color:#fff}
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
.liveline{color:var(--muted);font-size:17px;min-height:1.5em;margin:6px 2px 14px}
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
    "if(window.Notification&&Notification.permission==='default')"
    "{try{Notification.requestPermission();}catch(e){}}"
    "let _detSeen=false;"
    "async function _detTick(){try{const d=await(await fetch('/detect')).json();"
    "const bar=document.getElementById('_detbar');if(!bar)return;"
    "const show=d.meeting&&!d.recording&&sessionStorage.getItem('_detX')!=='1';"
    "bar.style.display=show?'flex':'none';"
    "if(show)document.getElementById('_detapp').textContent=d.app||'麥克風使用中';"
    "if(!d.meeting){sessionStorage.removeItem('_detX');_detSeen=false;}"
    "if(show&&!_detSeen){_detSeen=true;"
    "if(window.Notification&&Notification.permission==='granted')"
    "{try{new Notification('📝 偵測到會議',{body:(d.app||'麥克風使用中')+' — 點開始錄音'});}catch(e){}}}"
    "}catch(e){}}"
    "function _detDismiss(){sessionStorage.setItem('_detX','1');"
    "document.getElementById('_detbar').style.display='none';}"
    "setInterval(_detTick,10000);_detTick();")


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
        "<span class=spacer></span>"
        "<button class=btn id=themebtn onclick=_toggleTheme() title='切換深淺色' "
        "style='padding:.4em .6em'>🌓</button>"
        "<button class='btn danger' onclick=_quitApp() title='結束服務' "
        "style='padding:.4em .6em'>⏻</button>"
        f"{nav}</header>{body}</div>"
        + (f"<script>{script}</script>" if script else "")
        + f"<script>{_DETECT_JS}</script>"
        + "</body></html>"
    )


_INDEX = _shell("MeetingSummary", """
<h1>本地會議轉錄 · 摘要</h1>
<div class=row style="margin-bottom:4px">
<a class="btn primary" href="/live">🔴 開始 Live 即時逐字稿</a>
<a class="btn" href="/models/manage">⚙️ 設定</a>
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
  <p class=hint style="margin:.8em 0 0">上傳後會跑 transcribe + summary，第一次會下載模型，請稍候。</p>
</div>
<h2>會議紀錄</h2>
<div class=card>
  <div class=row style="margin-bottom:8px">
    <input id=q type=search placeholder="搜尋標題 / 逐字稿 / 摘要…" style="flex:1;min-width:200px">
    <button class=btn id=pickbtn>選取</button>
    <button class=btn id=mergebtn>整合相近的 live 會議</button>
    <span class="muted small" id=mergemsg></span>
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
      <div class=mmeta>${meta.join('<span class=dotsep>·</span>')}${chips(m.tags)}</div></div>
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
fetch('/meetings').then(r=>r.json()).then(ms=>{ALL=ms;applyView();renderTagbar();});
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

        "<section id=sec-exp><h2>🧪 實驗性功能</h2>"
        "<div class=card>"
        "<label class=chk><input type=checkbox id=exp_overlap> "
        "分群時標記重疊/多人語音（同一句多位說話者會標「🔀」，啟發式，非真分離）</label>"
        "<p class=hint style='margin:.6em 0 0'>實驗性功能可能不穩定或不準確；於各會議的「多人分群」時生效。</p>"
        "</div></section>"

        "<section id=sec-rt><h2>加速 runtime（.cpp · Metal）</h2>"
        "<div class=card>"
        "<table class=tx id=rt><tr><th>Runtime</th><th>狀態</th><th></th></tr></table>"
        "<p class=hint style='margin:.5em 0 0'>一鍵安裝。chatllm 優先下載預編譯(免 cmake)，"
        "下載不到才原始碼編譯；femelo 需 python≥3.11。缺的環境會自動 brew 安裝。"
        "清除只移除編譯產物，模型權重保留。</p>"
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
          +`<td><button class='btn' data-rt="${k}">${ok?'重新編譯':'編譯安裝'}</button>`
          +`${ok?` <button class='btn danger' data-rtdel="${k}">清除</button>`:''}</td></tr>`).join('');
      document.querySelectorAll('#rt button[data-rtdel]').forEach(b=>b.onclick=async()=>{
        if(!confirm('清除 '+b.dataset.rtdel+' runtime？(模型權重保留,重新編譯可復原)'))return;
        b.disabled=true;b.textContent='清除中…';
        await fetch('/models/runtime/delete',{method:'POST',headers:{'Content-Type':'application/json'},
          body:JSON.stringify({runtime:b.dataset.rtdel})});
        load();});
      document.querySelectorAll('#rt button[data-rt]').forEach(b=>b.onclick=async()=>{
        b.disabled=true;b.textContent='編譯中…';
        await fetch('/models/setup',{method:'POST',headers:{'Content-Type':'application/json'},
          body:JSON.stringify({runtime:b.dataset.rt})});
        setTimeout(load,800);});
      // supported
      document.getElementById('sup').innerHTML='<tr><th>模型</th><th>大小</th><th></th></tr>'
        +d.supported.map(m=>{const act=m.cached
          ?`<button class='btn danger' data-p="${esc(m.path)}" data-n="${esc(m.label)}">刪除</button>`
          :`<button class='btn primary' data-id="${esc(m.id)}">下載</button>`;
          return `<tr><td>${m.label}<div class=muted style='font-size:11px'>${m.id} · ${m.kind}</div></td>`
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
    });}
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
    load();
    """
    return _shell("設定", body, script=script, back=True)


def _result_page(title, summary, transcripts):
    lines = "".join(
        f"<tr><td class=who>{html.escape(str(r['track']))}</td>"
        f"<td>{html.escape(r['text'])}</td></tr>"
        for r in transcripts)
    body = (
        f"<h1>{html.escape(title)}</h1>"
        f"<div class=card><h2 style='margin-top:0'>摘要</h2>"
        f"<div class=md>{_md_html(summary)}</div></div>"
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
        <option value="qwen3-asr-0.6b-q4-k-m">Qwen3-ASR 0.6B(.cpp·Metal·預設)</option>
        <option value="mlx-community/Qwen3-ASR-1.7B-8bit">Qwen3-ASR 1.7B(mlx·Metal·準·快)</option>
        <option value="qwen3-asr-1.7b">Qwen3-ASR 1.7B(chatllm·Metal·慢·備用)</option>
        <option value="mlx-community/whisper-small-mlx-q4">whisper small-q4(快·省)</option>
        <option value="mlx-community/whisper-large-v3-turbo-q4">whisper turbo-q4(較準)</option>
        <option value="mlx-community/whisper-large-v3-turbo">whisper turbo(最準·較吃)</option>
        <option value="mlx-community/whisper-base-mlx-q4">whisper base-q4(更快)</option>
        <option value="mlx-community/whisper-tiny-mlx-q4">whisper tiny-q4(最省)</option>
        <option value="Qwen/Qwen3-ASR-0.6B">Qwen3-ASR 0.6B(transformers·慢)</option>
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
  const source = document.getElementById('source').value;
  const dual = source==='dual';
  try { streams = await getStreams(source); }
  catch(e){ S.textContent=' 取得音源失敗: '+e.message; return; }
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
  ws = new WebSocket(`ws://${location.host}/ws/live?src=${source}${diar}${unit}${sess}${vad}${lang}`);
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
    if(dual){ attach(streams[0],0,ratio,false); attach(streams[1],1,ratio,false); }  // 我/對方, continuous
    else { streams.forEach(st=>attach(st,null,ratio,true)); }  // mic/system/both: gated
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
    fresh = (os.path.exists(cache) and os.path.exists(keyfile)
             and open(keyfile).read() == key)
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
    {"id": "mlx-community/whisper-small-mlx-q4", "label": "whisper small-q4（live 預設）", "kind": "hf"},
    {"id": "mlx-community/whisper-large-v3-turbo-q4", "label": "whisper turbo-q4（精校）", "kind": "hf"},
    {"id": "mlx-community/whisper-large-v3-turbo", "label": "whisper turbo", "kind": "hf"},
    {"id": "mlx-community/whisper-base-mlx-q4", "label": "whisper base-q4", "kind": "hf"},
    {"id": "mlx-community/whisper-tiny-mlx-q4", "label": "whisper tiny-q4", "kind": "hf"},
    {"id": "mlx-community/whisper-large-v3-mlx", "label": "whisper large-v3", "kind": "hf"},
    {"id": "mlx-community/Qwen2.5-3B-Instruct-4bit", "label": "Qwen2.5-3B（摘要）", "kind": "hf"},
    {"id": "Qwen/Qwen3-ASR-0.6B", "label": "Qwen3-ASR 0.6B（transformers）", "kind": "hf"},
    {"id": "Qwen/Qwen3-ASR-1.7B", "label": "Qwen3-ASR 1.7B（transformers·最準·慢）", "kind": "hf"},
    {"id": "mlx-community/Qwen3-ASR-1.7B-8bit", "label": "Qwen3-ASR 1.7B（mlx·Metal·準·快）", "kind": "hf"},
    {"id": "qwen3-asr-0.6b-q4-k-m", "label": "Qwen3-ASR .cpp 0.6B（femelo·Metal·快）", "kind": "femelo"},
    {"id": "qwen3-asr-1.7b", "label": "Qwen3-ASR .cpp 1.7B（chatllm·Metal·慢·備用）", "kind": "chatllm"},
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
    return {
        "femelo": os.path.isfile(os.path.abspath(".venv-qwen314/bin/python")),
        "chatllm": os.path.isfile(os.path.abspath("chatllm.cpp/bindings/libchatllm.dylib")),
    }


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


def _persistent_names(store, embs, prefix, threshold=0.62):
    """cluster id -> persistent voiceprint label. Match each cluster embedding to the
    global speakers table (cosine); reuse the matched speaker's name + nudge its
    centroid, else enroll a new globally-unique speaker. So a voice you named "Alice"
    in one meeting auto-labels "Alice" in later ones; renaming propagates (see
    /speaker route -> rename_global_speaker)."""
    import numpy as np  # noqa: PLC0415
    import diarize as diar  # noqa: PLC0415
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
    for i, (track, p, seg_off, bs, bl) in enumerate(units):
        win_off_ms = seg_off + int(bs / 2 / sample_rate * 1000)
        tmp = f"{os.path.dirname(p)}/_win.pcm"
        with open(p, "rb") as f:
            f.seek(bs)
            with open(tmp, "wb") as w:
                w.write(f.read(bl))
        texts = []
        speaker = _TRACK_LABEL.get(track, track)  # mic->我, system->對方
        for t in asr.transcribe(tmp, profile="accurate", track=track, backend=backend):
            store.add_transcript(mid, "accurate", track,
                                 t["start_ms"] + win_off_ms,
                                 t["end_ms"] + win_off_ms, speaker, t["text"])
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


def _detail_page(mid, meeting, transcripts, summaries, audio_tracks=(), tags=()):
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
        + audio_card +
        "<div class=card><h2 style='margin-top:0'>摘要</h2>"
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
        "<div class=card><h2 style='margin-top:0'>逐字稿</h2>"
        "<div class=row style='margin-bottom:10px'>"
        "<select id=remodel>"
        "<option value='mlx-community/whisper-large-v3-turbo-q4'>turbo-q4(準·省)</option>"
        "<option value='mlx-community/whisper-large-v3-mlx'>large-v3(最準·吃)</option>"
        "<option value='mlx-community/whisper-small-mlx-q4'>small-q4(快)</option>"
        "<option value='mlx-community/Qwen3-ASR-1.7B-8bit'>Qwen3-ASR 1.7B(mlx·Metal·準·快)</option>"
        "<option value='qwen3-asr-0.6b-q4-k-m'>Qwen3-ASR 0.6B(.cpp·Metal·快)</option>"
        "<option value='qwen3-asr-1.7b'>Qwen3-ASR 1.7B(chatllm·Metal·慢·備用)</option>"
        "<option value='Qwen/Qwen3-ASR-0.6B'>Qwen3-ASR 0.6B(transformers·慢)</option>"
        "<option value='Qwen/Qwen3-ASR-1.7B'>Qwen3-ASR 1.7B(transformers·很慢)</option>"
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
        "document.getElementById('go').onclick=async()=>{"
        "const k=document.getElementById('kind').value;"
        "const o=document.getElementById('out');o.textContent='產生中…';"
        f"const r=await fetch('/meetings/{mid}/summary',{{method:'POST',"
        "headers:{'Content-Type':'application/json'},body:JSON.stringify({kind:k})});"
        "const j=await r.json();o.innerHTML='<div class=md>'+_mdHtml(j.text)+'</div>';"
        "if(j.title)document.getElementById('mtitle').textContent=j.title;};"
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
        "const lng=document.getElementById('relang').value;"
        "rm.textContent=' 啟動中…';retr.disabled=true;sawRunning=true;"
        f"fetch('/meetings/{mid}/transcribe/start',{{method:'POST',"
        "headers:{'Content-Type':'application/json'},body:JSON.stringify({model:mdl,language:lng})})"
        ".then(()=>poll());};"
        "poll();"  # resume on load if a job is already running
        "document.getElementById('dia').onclick=async()=>{"
        "const fm=document.getElementById('finmsg');fm.textContent=' 分群中…(會後聲紋,需稍候)';"
        "const ov=localStorage.getItem('exp_overlap')==='1';"
        f"const r=await fetch('/meetings/{mid}/diarize',{{method:'POST',"
        "headers:{'Content-Type':'application/json'},body:JSON.stringify({track:'all',mark_overlap:ov})});"
        "if(r.ok){const j=await r.json();fm.textContent=' 分出 '+j.speakers+' 位說話者';"
        "location.reload();}else{fm.textContent=' 分群失敗: '+(await r.text());}};"
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
        "document.querySelectorAll('td.who[data-spk]').forEach(td=>{td.style.cursor='pointer';"
        "td.onclick=async(e)=>{e.stopPropagation();const old=td.dataset.spk;"
        "const nw=prompt('把「'+old+'」改成(套用到整場該說話者):',old);"
        "if(nw==null||!nw.trim()||nw.trim()===old)return;"
        f"const r=await fetch('/meetings/{mid}/speaker',{{method:'POST',"
        "headers:{'Content-Type':'application/json'},body:JSON.stringify({old:old,new:nw.trim()})});"
        "if(r.ok)location.reload();};});"
        # Tags: chips + remove (✕) + an add input (Enter to add).
        "function _esc(s){return (s||'').replace(/[&<>\"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;'}[c]));}"
        f"const MID={mid};"
        "async function _postTag(name,remove){return (await fetch(`/meetings/${MID}/tags`,{method:'POST',"
        "headers:{'Content-Type':'application/json'},body:JSON.stringify({name,remove})})).json();}"
        "function renderTags(list){const el=document.getElementById('tags');"
        "el.innerHTML=list.map(t=>`<span class=tg>#${_esc(t)}<span class=x data-rm=\"${_esc(t)}\">✕</span></span>`).join('')"
        "+`<input id=tagin placeholder='+ 標籤' style='font-size:12px;padding:2px 8px;width:88px'>`;"
        "el.querySelectorAll('.x').forEach(s=>s.onclick=async()=>renderTags((await _postTag(s.dataset.rm,true)).tags));"
        "const ti=document.getElementById('tagin');ti.onkeydown=async(e)=>{"
        "if(e.key==='Enter'&&ti.value.trim()){const j=await _postTag(ti.value.trim(),false);renderTags(j.tags);}};}"
        f"renderTags({json.dumps(list(tags))});")
    return _shell(html.escape(meeting["title"]), body, script=script, back=True)


def create_app(store, *, summary_backend, asr_backend=None,
               live_manager=None, live_interim_backend=None, model_names=None,
               on_model_change=None,
               summary_model="mlx-lm", live_silence_ms=500, live_min_speech_ms=150,
               live_interim_s=0.6, live_max_utt_s=15.0, live_rms_threshold=500,
               live_max_lag_s=4.0):
    app = FastAPI()
    transcribe_jobs = {}  # mid -> progress dict; survives page refresh (in-memory)

    # Idle auto-release: free loaded models after N seconds with no activity and no
    # live connection. Loaded weights lazy-reload on next use.
    idle = {"last": time.time(), "live": 0}
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
        # Meeting = mic in use (CoreAudio via micbusy) or a known meeting app running.
        # Polled by every page (_DETECT_JS) -> banner + Web Notification.
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
        label = app_name or ("麥克風使用中" if mic else None)
        return {"meeting": bool(mic or app_name), "app": label,
                "recording": idle["live"] > 0}

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
        return _LIVE

    @app.get("/m/{mid}", response_class=HTMLResponse)
    def meeting_page(mid: int):
        meeting = store.get_meeting(mid)
        if meeting is None:
            raise HTTPException(404, "meeting not found")
        # Tracks with retained audio (across all segments — handles merged meetings).
        audio_tracks = _meeting_tracks(store, mid)
        return _detail_page(mid, dict(meeting), _rows(store.list_transcripts(mid)),
                            _rows(store.list_summaries(mid)), audio_tracks,
                            tags=store.tags_for(mid))

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

    @app.get("/models/manage", response_class=HTMLResponse)
    def models_manage():
        return _models_page()

    @app.get("/changelog", response_class=HTMLResponse)
    def changelog_page():
        here = os.path.dirname(os.path.abspath(__file__))
        try:
            md = open(os.path.join(here, "CHANGELOG.md")).read()
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
        if body.runtime not in ("femelo", "chatllm"):
            raise HTTPException(400, "unknown runtime")

        def _build():
            import subprocess
            _MODEL_TASKS[body.runtime] = {"state": "running", "msg": "編譯中…(數分鐘)"}
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
        targets = {"femelo": ".venv-qwen314", "chatllm": "chatllm.cpp"}
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

        sessions = {tag: TwoPassSession(
            backend=live_manager, interim_backend=live_interim_backend,
            sample_rate=16000, silence_ms=sil,
            min_speech_ms=live_min_speech_ms, interim_s=live_interim_s,
            max_utt_s=maxu, rms_threshold=live_rms_threshold,
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
                    data = await ws.receive_bytes()
                    tag, pcm = (data[0], data[1:]) if dual else (0, data)
                    if tag not in buffers:
                        continue
                    _pad_to(time.time())          # keep all tracks at wall-clock
                    audio_files[tag].write(pcm)
                    buffers[tag].extend(pcm)
                    written[tag] += len(pcm)
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

    @app.post("/meetings/{mid}/transcribe")
    def transcribe_meeting(mid: int, body: TranscribeIn = TranscribeIn()):
        # Re-run accurate ASR over the saved audio, optionally with a chosen model.
        # Sync route -> FastAPI runs it in a threadpool, so /health stays responsive.
        if store.get_meeting(mid) is None:
            raise HTTPException(404, "meeting not found")
        if body.model:
            import backends
            backend = backends.make_batch_backend(body.model, body.language)
        elif asr_backend is not None:
            backend = asr_backend
        else:
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
        elif asr_backend is not None:
            backend = asr_backend
        else:
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
        # Auto-title from the summary unless the user typed a non-default title.
        if _is_default_title(title):
            nt = await run_in_threadpool(_auto_title, result["summary"], summary_backend)
            if nt:
                store.update_title(mid, nt)
                title = nt
        # Decode the upload to data/<mid>/mic.pcm (16k mono) so playback uses the
        # same path as live recordings (player + click-to-seek work). Best-effort.
        await run_in_threadpool(_save_upload_pcm, path, mid, store)
        return _result_page(title, result["summary"],
                            _rows(store.list_transcripts(mid)))

    @app.post("/meetings/{mid}/summary")
    def summarize_meeting(mid: int, body: SummaryIn):
        _touch()
        meeting = store.get_meeting(mid)
        if meeting is None:
            raise HTTPException(404, "meeting not found")
        text = _transcript_text(store.list_transcripts(mid))
        out = summarize(text, kind=body.kind, lang=meeting["lang"],
                        backend=summary_backend)
        store.add_summary(mid, body.kind, meeting["lang"], out,
                          summary_model, time.time())
        # Summary done -> derive a real title (unless the user set one of their own).
        new_title = None
        if _is_default_title(meeting["title"]):
            nt = _auto_title(out, summary_backend)
            if nt:
                store.update_title(mid, nt)
                new_title = nt
        return {"text": out, "kind": body.kind, "title": new_title}

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

        tracks = _meeting_tracks(store, mid) if body.track == "all" else [body.track]
        tracks = [t for t in tracks if t in _meeting_tracks(store, mid)]
        if not tracks:
            raise HTTPException(404, "no saved audio for this meeting")
        total = 0
        for track in tracks:
            pcm_bytes = _assemble_track(store, mid, track)  # offset-aligned, all segments
            if pcm_bytes is None:
                continue
            tmp = f"data/{mid}/_diar_{track}.pcm"
            os.makedirs(f"data/{mid}", exist_ok=True)
            with open(tmp, "wb") as f:
                f.write(pcm_bytes)
            names = None
            try:
                segments = diar.diarize_pcm(tmp, num_speakers=body.num_speakers,
                                            seg_model=body.seg_model,
                                            emb_model=body.emb_model)
                if body.enroll:  # persistent cross-meeting voiceprints (best-effort)
                    try:
                        embs = diar.cluster_embeddings(tmp, segments)
                        names = _persistent_names(
                            store, embs, _TRACK_LABEL.get(track, track))
                    except Exception as e:
                        print(f"speaker enroll skipped: {e}", file=sys.stderr)
            except Exception as e:
                raise HTTPException(503, f"diarization unavailable: {e}")
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)
            rows = [dict(r) for r in store.list_transcripts(mid) if r["track"] == track]
            # prefix per track so 我-side and 對方-side speakers don't collide
            for r in diar.assign_speakers(rows, segments,
                                          prefix=_TRACK_LABEL.get(track, track),
                                          names=names, mark_overlap=body.mark_overlap):
                store.update_speaker(r["id"], r["speaker"])
            total += len({s["speaker"] for s in segments})
        return {"tracks": tracks, "speakers": total}

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
    port = int(os.environ.get("MEETING_PORT", "8765"))  # 8000 left free for dev
    uvicorn.run(app, host="127.0.0.1", port=port)  # loopback only (G2)
