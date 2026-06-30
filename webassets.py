"""Static web assets for MeetingSummary — CSS, theme/manifest/service-worker,
the shell JS blobs (meeting-detect HUD, progress popout, quick-record FAB), and
the PWA icon generator. Pure presentation, no app logic — keeps app.py focused
on routes + the live pipeline. None interpolate app state (all plain strings)."""

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

