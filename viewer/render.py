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


def _page(title, body):
    return (f"<!doctype html><html lang='zh-Hant'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{_e(title)}</title><style>body{{font-family:system-ui,"
            f"'PingFang TC',sans-serif;max-width:820px;margin:2rem auto;padding:0 1rem;"
            f"line-height:1.6}}a{{color:#06c;text-decoration:none}}.badge{{font-size:.75rem;"
            f"background:#eee;border-radius:6px;padding:.1rem .4rem;margin-left:.5rem}}"
            f".line{{margin:.3rem 0}}.spk{{color:#888;margin-right:.4rem}}"
            f"audio{{width:100%;margin:.5rem 0}}</style></head><body>{body}</body></html>")


def render_index(meetings):
    items = []
    for m in meetings:
        d = dict(m)
        ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(d.get("created_at") or 0))
        items.append(f"<li><a href='/m/{d['id']}'>{_e(d.get('title') or '未命名')}</a>"
                     f" <span class='badge'>{ts}</span></li>")
    body = ("<h1>會議</h1>"
            "<form action='/search'><input name='q' placeholder='搜尋…'>"
            "<button>搜尋</button></form>"
            f"<ul>{''.join(items) or '<li>（無會議）</li>'}</ul>")
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
    audio = "".join(
        f"<div><b>{_e(t)}</b><audio controls preload='none' "
        f"src='/meetings/{m['id']}/audio/{_e(t)}.m4a'></audio></div>" for t in tracks)
    sums = "".join(f"<h2>摘要（{_e(dict(s).get('kind',''))}）</h2>"
                   f"<p>{_e(dict(s).get('text'))}</p>" for s in summaries)

    def _lines_html(prefer):
        return "".join(f"<div class='line'><span class='spk'>{_e(l['speaker'])}</span>"
                       f"{_e(l['text'])}</div>" for l in group_lines(transcripts, prefer))

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
            f"{audio}{sums}{tx}")
    return _page(m.get("title") or "會議", body)
