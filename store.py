"""Store: SQLite metadata for meetings/segments/transcripts/summaries.
Schema per docs spec §4. Audio bytes live on disk under data/."""
import sqlite3
from pathlib import Path

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
