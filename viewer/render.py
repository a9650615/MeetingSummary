# viewer/render.py
"""Pure render helpers for the remote viewer. No I/O, no Apple/ASR deps."""
import html
import time


def pick_transcripts(transcripts, prefer="local"):
    """Default to the LOCAL (Mac-pushed) transcript. FireRed is a side-by-side
    comparison, never a silent auto-upgrade — it isn't always more accurate than
    local, so firered is returned only when the viewer explicitly asks for it
    (prefer='firered'). firered_staging (in-progress) is never shown."""
    fr = [dict(t) for t in transcripts if dict(t).get("profile") == "firered"]
    local = [dict(t) for t in transcripts
             if dict(t).get("profile") not in ("firered", "firered_staging")]
    chosen = fr if (prefer == "firered" and fr) else local
    return sorted(chosen, key=lambda t: t.get("start_ms") or 0)


def has_firered(transcripts):
    return any(dict(t).get("profile") == "firered" for t in transcripts)


def group_lines(transcripts, prefer="local"):
    return [{"speaker": t.get("speaker") or "", "text": t.get("text") or "",
             "start_ms": t.get("start_ms") or 0, "track": t.get("track")}
            for t in pick_transcripts(transcripts, prefer)]


def export_md(meeting, transcripts, summaries):
    out = [f"# {dict(meeting).get('title','')}", ""]
    for s in summaries:
        s = dict(s)  # sqlite3.Row has no .get(); normalize
        out += [f"## 摘要（{s.get('kind','')}）", s.get("text") or "", ""]
    out += ["## 逐字稿", ""]
    for line in group_lines(transcripts):
        out.append(f"{line['speaker']}：{line['text']}")
    return "\n".join(out) + "\n"


def _e(s):
    return html.escape(str(s) if s is not None else "")


def _ts(ms):
    s = int((ms or 0) / 1000)
    return f"{s // 60}:{s % 60:02d}"


_CSS = """
:root{--bg:#fff;--fg:#1a1a1a;--muted:#6b7280;--card:#f6f7f9;--border:#e5e7eb;
--accent:#2563eb;--hi:#fff7d6;--hiedge:#f5c518;--chip:#eef2ff;--chipfg:#3730a3}
@media(prefers-color-scheme:dark){:root{--bg:#0f1115;--fg:#e7e9ee;--muted:#9aa3b2;
--card:#171a21;--border:#262b36;--accent:#6ea8fe;--hi:#2a2410;--hiedge:#8a6d1a;
--chip:#1e2436;--chipfg:#a9b7ff}}
*{box-sizing:border-box}
body{font-family:system-ui,'PingFang TC','Noto Sans TC',sans-serif;max-width:860px;
margin:0 auto;padding:1rem 1rem 6rem;line-height:1.65;color:var(--fg);background:var(--bg)}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
h1{font-size:1.5rem;margin:.6rem 0}h2{font-size:1.05rem;margin:1.4rem 0 .6rem}
.badge{font-size:.72rem;background:var(--card);border:1px solid var(--border);
border-radius:999px;padding:.15rem .55rem;margin-left:.4rem;color:var(--muted);white-space:nowrap}
.card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:.6rem .9rem;margin:.6rem 0}
ul.meetings{list-style:none;padding:0}ul.meetings li{padding:.6rem .2rem;border-bottom:1px solid var(--border)}
form.search{display:flex;gap:.5rem;margin:1rem 0}
form.search input{flex:1;padding:.55rem .7rem;border:1px solid var(--border);border-radius:10px;background:var(--bg);color:var(--fg)}
form.search button,.btn{padding:.55rem .9rem;border:1px solid var(--border);border-radius:10px;
background:var(--bg);color:var(--fg);cursor:pointer}
.player{position:sticky;top:0;z-index:5;background:var(--bg);padding:.5rem 0;border-bottom:1px solid var(--border);margin-bottom:.6rem}
.player .trk{font-size:.8rem;color:var(--muted);margin:.15rem 0}
audio{width:100%;height:34px}
.tx{margin-top:.3rem}
.line{display:flex;gap:.6rem;padding:.32rem .5rem;margin:.1rem -.5rem;border-radius:8px;cursor:pointer;
scroll-margin-top:120px;border-left:3px solid transparent}
.line:hover{background:var(--card)}
.line.active{background:var(--hi);border-left-color:var(--hiedge)}
.line .t{color:var(--muted);font-variant-numeric:tabular-nums;font-size:.8rem;min-width:3.2rem;padding-top:.1rem}
.line .spk{color:var(--chipfg);background:var(--chip);border-radius:6px;padding:0 .4rem;
font-size:.78rem;white-space:nowrap;align-self:flex-start;margin-top:.05rem}
.line .tt{flex:1}
"""


def _page(title, body):
    return (f"<!doctype html><html lang='zh-Hant'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{_e(title)}</title><style>{_CSS}</style></head><body>{body}</body></html>")


def render_index(meetings):
    items = []
    for m in meetings:
        d = dict(m)
        ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(d.get("created_at") or 0))
        items.append(f"<li><a href='/m/{d['id']}'>{_e(d.get('title') or '未命名')}</a>"
                     f" <span class='badge'>{ts}</span></li>")
    body = ("<h1>會議</h1>"
            "<form class='search' action='/search'><input name='q' placeholder='搜尋會議…'>"
            "<button>搜尋</button></form>"
            f"<ul class='meetings'>{''.join(items) or '<li>（無會議）</li>'}</ul>")
    return _page("會議", body)


def render_search(q, results):
    items = [f"<li><a href='/m/{r['id']}'>{_e(r['title'])}</a> "
             f"<small>{_e(r.get('snippet') or '')}</small></li>" for r in results]
    return _page(f"搜尋 {q}",
                 f"<h1>搜尋：{_e(q)}</h1><p><a href='/'>← 全部</a></p>"
                 f"<ul>{''.join(items) or '<li>無結果</li>'}</ul>")


def render_detail(meeting, transcripts, summaries, tracks, tags):
    m = dict(meeting)
    has_fr = has_firered(transcripts)
    progress = (f"<div id='fr-prog' class='badge' style='display:none'></div>"
                f"<script>(function(){{var el=document.getElementById('fr-prog');"
                f"function tick(){{fetch('/meetings/{m['id']}/firered/progress')"
                f".then(r=>r.json()).then(p=>{{"
                f"if(p.state==='running'||p.state==='paused'){{el.style.display='';"
                f"el.textContent='FireRed 校正 '+(p.done||0)+'/'+(p.total||0)+"
                f"(p.state==='paused'?'（已暫停）':'…');setTimeout(tick,3000);}}"
                f"else if(p.state==='done'){{el.style.display='';el.textContent='FireRed 校正完成';}}"
                f"}}).catch(()=>{{}});}}tick();}})();</script>")
    _TRK = {"mic": "🎙 我", "system": "🔊 對方", "mixed": "🎧 混合"}
    audio = "<div class='player'>" + "".join(
        f"<div class='trk'>{_e(_TRK.get(t, t))}</div>"
        f"<audio id='au-{_e(t)}' data-track='{_e(t)}' controls preload='none' "
        f"src='/meetings/{m['id']}/audio/{_e(t)}.m4a'></audio>" for t in tracks) + "</div>"
    sums = "".join(f"<h2>摘要（{_e(dict(s).get('kind',''))}）</h2>"
                   f"<div class='card'>{_e(dict(s).get('text'))}</div>" for s in summaries)

    def _lines_html(prefer):
        out = []
        for l in group_lines(transcripts, prefer):
            trk, st = _e(l["track"]), int(l["start_ms"] or 0)
            out.append(f"<div class='line' data-track='{trk}' data-start='{st}'>"
                       f"<span class='t'>{_ts(st)}</span>"
                       f"<span class='spk'>{_e(l['speaker'])}</span>"
                       f"<span class='tt'>{_e(l['text'])}</span></div>")
        return "".join(out)

    # Default view = local (Mac Qwen). FireRed is a comparison, not an override:
    # a toggle swaps between the two; local never gets silently replaced.
    if has_fr:
        toggle = (
            "<button id='tx-toggle' class='badge' style='cursor:pointer' onclick=\""
            "var l=document.getElementById('tx-local'),f=document.getElementById('tx-firered'),"
            "b=document.getElementById('tx-toggle');var showF=l.style.display!=='none';"
            "l.style.display=showF?'none':'';f.style.display=showF?'':'none';"
            "b.textContent=showF?'← 顯示本地版':'顯示 FireRed 校正版 →';\">顯示 FireRed 校正版 →</button>")
        tx = (f"<h2>逐字稿 <small>本地版（預設）</small> {toggle}</h2>"
              f"<div id='tx-local'>{_lines_html('local')}</div>"
              f"<div id='tx-firered' style='display:none'>{_lines_html('firered')}</div>")
    else:
        tx = f"<h2>逐字稿</h2><div id='tx-local'>{_lines_html('local')}</div>"

    body = (f"<p><a href='/'>← 全部</a></p><h1>{_e(m.get('title') or '未命名')}{progress}</h1>"
            f"<p><a href='/meetings/{m['id']}/export'>下載逐字稿 (.md)</a></p>"
            f"{audio}{sums}{tx}{_TIMELINE_JS}")
    return _page(m.get("title") or "會議", body)


# Timeline sync: highlight the line being spoken as audio plays; click a line to
# seek. Binds every track's <audio> to its own lines; falls back to all lines when a
# playing track has none. Vanilla JS, no deps (the viewer is a static-ish server).
_TIMELINE_JS = """
<script>(function(){
function visLines(track){
  var all=[].slice.call(document.querySelectorAll('.line'));
  var vis=all.filter(function(el){return el.offsetParent!==null;});
  var t=vis.filter(function(el){return el.dataset.track===track;});
  return t.length?t:vis;   // fall back to all visible lines if track has none
}
function sync(au){
  var lines=visLines(au.dataset.track);if(!lines.length)return;
  var ms=au.currentTime*1000,cur=null;
  for(var i=0;i<lines.length;i++){if(+lines[i].dataset.start<=ms)cur=lines[i];else break;}
  document.querySelectorAll('.line.active').forEach(function(el){if(el!==cur)el.classList.remove('active');});
  if(cur&&!cur.classList.contains('active')){cur.classList.add('active');
    cur.scrollIntoView({block:'nearest',behavior:'smooth'});}
}
document.querySelectorAll('audio').forEach(function(au){
  au.addEventListener('timeupdate',function(){sync(au);});
});
document.querySelectorAll('.line').forEach(function(el){
  el.addEventListener('click',function(){
    var au=document.getElementById('au-'+el.dataset.track)||document.querySelector('audio');
    if(!au)return;au.currentTime=(+el.dataset.start)/1000;au.play().catch(function(){});
  });
});
})();</script>"""
