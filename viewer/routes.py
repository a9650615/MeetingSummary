"""Read-only viewer routes over a Store, serving M4A directly. Linux-safe."""
import glob
import os
from urllib.parse import quote

from fastapi import HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response

from viewer import render


def meeting_tracks(data_dir, mid):
    paths = sorted(glob.glob(os.path.join(data_dir, str(mid), "*.m4a")))
    order = {"mixed": 0, "system": 1, "mic": 2}
    names = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    return sorted(names, key=lambda n: order.get(n, 9))


def mount_viewer(app, store, data_dir):
    @app.get("/", response_class=HTMLResponse)
    def index():
        return render.render_index(store.list_meetings())

    @app.get("/search", response_class=HTMLResponse)
    def search(q: str = ""):
        results = [{"id": r["id"], "title": r["title"],
                    "snippet": (r["t_snip"] or r["s_snip"] or "")}
                   for r in store.search(q)]
        return render.render_search(q, results)

    @app.get("/meetings")
    def list_meetings():
        out = []
        for m in store.list_meetings():
            d = dict(m)
            segs = store.list_segments(m["id"])
            d["tracks"] = meeting_tracks(data_dir, m["id"])
            d["duration_s"] = round(sum((s["duration_s"] or 0) for s in segs))
            d["n_segments"] = len(segs)
            out.append(d)
        return out

    @app.get("/m/{mid}", response_class=HTMLResponse)
    def detail(mid: int):
        m = store.get_meeting(mid)
        if m is None:
            raise HTTPException(404, "meeting not found")
        return render.render_detail(dict(m), store.list_transcripts(mid),
                                    store.list_summaries(mid),
                                    meeting_tracks(data_dir, mid),
                                    store.tags_for(mid))

    @app.get("/meetings/{mid}/audio/{track}.m4a")
    def audio(mid: int, track: str):
        path = os.path.join(data_dir, str(mid), f"{os.path.basename(track)}.m4a")
        if not os.path.exists(path):
            raise HTTPException(404, "track not found")
        return FileResponse(path, media_type="audio/mp4")  # FileResponse => Range

    @app.get("/meetings/{mid}/export")
    def export(mid: int):
        m = store.get_meeting(mid)
        if m is None:
            raise HTTPException(404, "meeting not found")
        md = render.export_md(dict(m), store.list_transcripts(mid),
                              store.list_summaries(mid))
        safe = "".join(c for c in m["title"] if c.isalnum() or c in " -_")[:40].strip()
        fname = (safe or f"meeting-{mid}") + ".md"
        return Response(md, media_type="text/markdown; charset=utf-8",
                        headers={"Content-Disposition":
                                 f"attachment; filename*=UTF-8''{quote(fname)}"})
