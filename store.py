"""Store: SQLite metadata for meetings/segments/transcripts/summaries.
Schema per docs spec §4. Audio bytes live on disk under data/."""
import sqlite3
from pathlib import Path


def group_by_proximity(meetings, gap_s=600):
    """Group meetings whose consecutive start times are within gap_s. Each input
    is a dict/row with id + created_at. Returns groups of >=2 ids (sorted by
    time); lone meetings are dropped — only mergeable clusters come back."""
    ms = sorted(meetings, key=lambda m: m["created_at"])
    groups, cur = [], []
    for m in ms:
        if cur and m["created_at"] - cur[-1]["created_at"] > gap_s:
            if len(cur) >= 2:
                groups.append([x["id"] for x in cur])
            cur = []
        cur.append(m)
    if len(cur) >= 2:
        groups.append([x["id"] for x in cur])
    return groups

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meetings(
    id INTEGER PRIMARY KEY, title TEXT, created_at REAL, lang TEXT,
    status TEXT NOT NULL DEFAULT 'recording');
CREATE TABLE IF NOT EXISTS segments(
    id INTEGER PRIMARY KEY, meeting_id INTEGER, idx INTEGER, dir_path TEXT,
    started_at REAL, duration_s REAL, origin TEXT);
CREATE TABLE IF NOT EXISTS transcripts(
    id INTEGER PRIMARY KEY, meeting_id INTEGER, profile TEXT, track TEXT,
    start_ms INTEGER, end_ms INTEGER, speaker TEXT, text TEXT);
CREATE TABLE IF NOT EXISTS summaries(
    id INTEGER PRIMARY KEY, meeting_id INTEGER, kind TEXT, lang TEXT,
    text TEXT, model TEXT, created_at REAL);
"""


class Store:
    def __init__(self, db_path):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        # ponytail: single shared connection across FastAPI's threadpool;
        # sqlite serializes writes by default. Per-request connections only
        # if write contention ever shows up (single-user local app — unlikely).
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(_SCHEMA)

    def _insert(self, sql, params):
        cur = self.db.execute(sql, params)
        self.db.commit()
        return cur.lastrowid

    def create_meeting(self, title, created_at, lang):
        return self._insert(
            "INSERT INTO meetings(title, created_at, lang) VALUES(?,?,?)",
            (title, created_at, lang),
        )

    def get_meeting(self, meeting_id):
        return self.db.execute(
            "SELECT * FROM meetings WHERE id=?", (meeting_id,)
        ).fetchone()

    def list_meetings(self):
        return self.db.execute(
            "SELECT * FROM meetings ORDER BY created_at DESC"
        ).fetchall()

    def finalize_meeting(self, meeting_id):
        self.db.execute(
            "UPDATE meetings SET status='finalized' WHERE id=?", (meeting_id,)
        )
        self.db.commit()

    def list_unfinalized(self):
        return self.db.execute(
            "SELECT * FROM meetings WHERE status != 'finalized'"
        ).fetchall()

    def add_segment(self, meeting_id, idx, dir_path, started_at, duration_s, origin):
        return self._insert(
            "INSERT INTO segments(meeting_id, idx, dir_path, started_at, "
            "duration_s, origin) VALUES(?,?,?,?,?,?)",
            (meeting_id, idx, dir_path, started_at, duration_s, origin),
        )

    def list_segments(self, meeting_id):
        return self.db.execute(
            "SELECT * FROM segments WHERE meeting_id=? ORDER BY idx", (meeting_id,)
        ).fetchall()

    def add_transcript(self, meeting_id, profile, track, start_ms, end_ms, speaker, text):
        return self._insert(
            "INSERT INTO transcripts(meeting_id, profile, track, start_ms, "
            "end_ms, speaker, text) VALUES(?,?,?,?,?,?,?)",
            (meeting_id, profile, track, start_ms, end_ms, speaker, text),
        )

    def list_transcripts(self, meeting_id):
        return self.db.execute(
            "SELECT * FROM transcripts WHERE meeting_id=? ORDER BY start_ms",
            (meeting_id,),
        ).fetchall()

    def update_speaker(self, transcript_id, speaker):
        self.db.execute("UPDATE transcripts SET speaker=? WHERE id=?",
                        (speaker, transcript_id))
        self.db.commit()

    def update_title(self, meeting_id, title):
        self.db.execute("UPDATE meetings SET title=? WHERE id=?", (title, meeting_id))
        self.db.commit()

    def rename_speaker(self, meeting_id, old, new):
        """Rename a speaker across a meeting (人工命名, e.g. 說話者1 -> Scott).
        Returns rows changed."""
        cur = self.db.execute(
            "UPDATE transcripts SET speaker=? WHERE meeting_id=? AND speaker=?",
            (new, meeting_id, old))
        self.db.commit()
        return cur.rowcount

    def add_summary(self, meeting_id, kind, lang, text, model, created_at):
        return self._insert(
            "INSERT INTO summaries(meeting_id, kind, lang, text, model, "
            "created_at) VALUES(?,?,?,?,?,?)",
            (meeting_id, kind, lang, text, model, created_at),
        )

    def list_summaries(self, meeting_id):
        return self.db.execute(
            "SELECT * FROM summaries WHERE meeting_id=? ORDER BY created_at",
            (meeting_id,),
        ).fetchall()

    def merge_meetings(self, target_id, source_ids):
        """Fold sources into target: transcripts/segments reassigned, transcript
        start_ms rebased by the created_at gap so the timeline stays ordered.
        Sources (and their summaries) are deleted. Audio files stay in place —
        segment dir_path still points at them."""
        target = self.get_meeting(target_id)
        if target is None:
            raise ValueError("target meeting not found")
        base = target["created_at"]
        next_idx = len(self.list_segments(target_id))
        for sid in source_ids:
            if sid == target_id:
                continue
            src = self.get_meeting(sid)
            if src is None:
                continue
            offset_ms = int((src["created_at"] - base) * 1000)
            self.db.execute(
                "UPDATE transcripts SET meeting_id=?, start_ms=start_ms+?, "
                "end_ms=end_ms+? WHERE meeting_id=?",
                (target_id, offset_ms, offset_ms, sid))
            for seg in self.list_segments(sid):
                self.db.execute("UPDATE segments SET meeting_id=?, idx=? WHERE id=?",
                                (target_id, next_idx, seg["id"]))
                next_idx += 1
            self.db.execute("DELETE FROM summaries WHERE meeting_id=?", (sid,))
            self.db.execute("DELETE FROM meetings WHERE id=?", (sid,))
        self.db.commit()
        return target_id

    def delete_meeting(self, meeting_id):
        """Delete a meeting + its transcripts/summaries/segments. Returns the set
        of segment dir_paths that are no longer referenced by ANY meeting (caller
        removes the audio files — a dir shared via merge stays if still in use)."""
        dirs = {seg["dir_path"] for seg in self.list_segments(meeting_id)}
        self.db.execute("DELETE FROM transcripts WHERE meeting_id=?", (meeting_id,))
        self.db.execute("DELETE FROM summaries WHERE meeting_id=?", (meeting_id,))
        self.db.execute("DELETE FROM segments WHERE meeting_id=?", (meeting_id,))
        self.db.execute("DELETE FROM meetings WHERE id=?", (meeting_id,))
        self.db.commit()
        still = {r["dir_path"] for r in
                 self.db.execute("SELECT DISTINCT dir_path FROM segments").fetchall()}
        return dirs - still

    def clear_transcripts(self, meeting_id, profile=None):
        if profile is None:
            self.db.execute("DELETE FROM transcripts WHERE meeting_id=?", (meeting_id,))
        else:
            self.db.execute("DELETE FROM transcripts WHERE meeting_id=? AND profile=?",
                            (meeting_id, profile))
        self.db.commit()

    def merge_into_earliest(self, ids):
        """Merge a set of meetings into whichever started first."""
        rows = [self.get_meeting(i) for i in ids]
        rows = [r for r in rows if r is not None]
        target = min(rows, key=lambda r: r["created_at"])["id"]
        return self.merge_meetings(target, [r["id"] for r in rows if r["id"] != target])
