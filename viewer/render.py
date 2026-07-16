# viewer/render.py
"""Pure render helpers for the remote viewer. No I/O, no Apple/ASR deps."""
import html
import time


def pick_transcripts(transcripts):
    rows = [dict(t) for t in transcripts]
    fr = [t for t in rows if t.get("profile") == "firered"]
    chosen = fr if fr else [t for t in rows if t.get("profile") != "firered"]
    return sorted(chosen, key=lambda t: t.get("start_ms") or 0)


def active_profile(transcripts):
    return "firered" if any(dict(t).get("profile") == "firered"
                            for t in transcripts) else "local"


def group_lines(transcripts):
    return [{"speaker": t.get("speaker") or "", "text": t.get("text") or "",
             "start_ms": t.get("start_ms") or 0, "track": t.get("track")}
            for t in pick_transcripts(transcripts)]


def export_md(meeting, transcripts, summaries):
    out = [f"# {meeting['title']}", ""]
    for s in summaries:
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
    prof = active_profile(transcripts)
    badge = ("<span class='badge'>FireRed 校正版</span>" if prof == "firered"
             else "<span class='badge'>本地版（校正中…）</span>")
    audio = "".join(
        f"<div><b>{_e(t)}</b><audio controls preload='none' "
        f"src='/meetings/{m['id']}/audio/{_e(t)}.m4a'></audio></div>" for t in tracks)
    sums = "".join(f"<h2>摘要（{_e(s.get('kind',''))}）</h2><p>{_e(s.get('text'))}</p>"
                   for s in summaries)
    lines = "".join(f"<div class='line'><span class='spk'>{_e(l['speaker'])}</span>"
                    f"{_e(l['text'])}</div>" for l in group_lines(transcripts))
    body = (f"<p><a href='/'>← 全部</a></p><h1>{_e(m.get('title') or '未命名')}{badge}</h1>"
            f"<p><a href='/meetings/{m['id']}/export'>下載逐字稿 (.md)</a></p>"
            f"{audio}{sums}<h2>逐字稿</h2>{lines}")
    return _page(m.get("title") or "會議", body)
