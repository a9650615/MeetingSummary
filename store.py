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
CREATE TABLE IF NOT EXISTS tags(id INTEGER PRIMARY KEY, name TEXT UNIQUE);
CREATE TABLE IF NOT EXISTS meeting_tags(
    meeting_id INTEGER, tag_id INTEGER, PRIMARY KEY(meeting_id, tag_id));
CREATE TABLE IF NOT EXISTS speakers(
    id INTEGER PRIMARY KEY, name TEXT, centroid BLOB, count INTEGER DEFAULT 1);
CREATE TABLE IF NOT EXISTS settings(k TEXT PRIMARY KEY, v TEXT);
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
        # ponytail: idempotent column add for DBs created before notes existed.
        try:
            self.db.execute("ALTER TABLE meetings ADD COLUMN notes TEXT DEFAULT ''")
            self.db.commit()
        except sqlite3.OperationalError:
            pass  # column already present

    def set_notes(self, meeting_id, notes):
        self.db.execute("UPDATE meetings SET notes=? WHERE id=?", (notes, meeting_id))
        self.db.commit()

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

    def delete_transcript(self, transcript_id):
        self.db.execute("DELETE FROM transcripts WHERE id=?", (transcript_id,))
        self.db.commit()

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

    def search(self, q, limit=50):
        """Find meetings whose title, transcript, or summary contains q. LIKE is
        correct for CJK substring matching (no whitespace tokenization needed);
        FTS5-trigram only if this ever gets slow on a large archive. Returns rows
        with a best-effort matching snippet + where it hit."""
        q = (q or "").strip()
        if not q:
            return []
        like = f"%{q}%"
        return self.db.execute(
            """SELECT m.id, m.title, m.created_at,
                 (SELECT t.text FROM transcripts t WHERE t.meeting_id=m.id
                    AND t.text LIKE ? ORDER BY t.start_ms LIMIT 1) AS t_snip,
                 (SELECT s.text FROM summaries s WHERE s.meeting_id=m.id
                    AND s.text LIKE ? LIMIT 1) AS s_snip
               FROM meetings m
               WHERE m.title LIKE ?
                 OR EXISTS(SELECT 1 FROM transcripts t WHERE t.meeting_id=m.id AND t.text LIKE ?)
                 OR EXISTS(SELECT 1 FROM summaries s WHERE s.meeting_id=m.id AND s.text LIKE ?)
               ORDER BY m.created_at DESC LIMIT ?""",
            (like, like, like, like, like, limit)).fetchall()

    # --- global speaker voiceprints (recognize the same voice across meetings) ---
    def list_speakers(self):
        return self.db.execute(
            "SELECT id, name, centroid, count FROM speakers").fetchall()

    def add_speaker(self, name, centroid_bytes):
        return self._insert("INSERT INTO speakers(name, centroid, count) VALUES(?,?,1)",
                            (name, centroid_bytes))

    def set_speaker_name(self, speaker_id, name):
        self.db.execute("UPDATE speakers SET name=? WHERE id=?", (name, speaker_id))
        self.db.commit()

    def update_speaker_centroid(self, speaker_id, centroid_bytes, count):
        self.db.execute("UPDATE speakers SET centroid=?, count=? WHERE id=?",
                        (centroid_bytes, count, speaker_id))
        self.db.commit()

    def rename_global_speaker(self, old, new):
        """Rename every global voiceprint currently labeled `old` -> `new`, so a
        rename in one meeting propagates to future auto-labels. Returns rowcount."""
        cur = self.db.execute("UPDATE speakers SET name=? WHERE name=?", (new, old))
        self.db.commit()
        return cur.rowcount

    def rename_speaker_global(self, old, new):
        """Rename a speaker EVERYWHERE: all meetings' transcripts + the voiceprint.
        (rename_global_speaker only touches the voiceprint — that's for propagating
        a per-meeting rename forward; this is the management-page global rename.)"""
        new = (new or "").strip()
        if not old or not new or old == new:
            return 0
        cur = self.db.execute("UPDATE transcripts SET speaker=? WHERE speaker=?",
                              (new, old))
        self.db.execute("UPDATE speakers SET name=? WHERE name=?", (new, old))
        self.db.commit()
        return cur.rowcount

    def speakers_with_stats(self):
        """ONE row per name (a name = a person; a person may carry several
        voiceprints). Multiple voiceprints sharing a name would otherwise show as
        duplicate rows with identical per-name stats. + distinct meetings/utterances."""
        out = []
        for s in self.db.execute(
                "SELECT name, COUNT(*) vp FROM speakers GROUP BY name ORDER BY name"):
            r = self.db.execute(
                "SELECT COUNT(DISTINCT meeting_id) m, COUNT(*) u, "
                "MAX(end_ms - start_ms) span FROM transcripts WHERE speaker=?",
                (s["name"],)).fetchone()
            if not r["u"]:
                continue  # no transcript references this name -> orphan voiceprint, hide it
            out.append({"name": s["name"], "voiceprints": s["vp"],
                        "meetings": r["m"], "utterances": r["u"],
                        "has_sample": (r["span"] or 0) > 800})  # matches speaker_best_span
        return out

    def delete_speakers_by_name(self, name):
        """Forget every voiceprint for a name (the management page is per-name)."""
        self.db.execute("DELETE FROM speakers WHERE name=?", (name,))
        self.db.commit()

    def merge_speakers(self, keep_name, drop_name):
        """Two voiceprints are the same person: move all of drop's transcripts to
        keep (every meeting) and delete drop's voiceprint(s). Returns rows moved."""
        if not keep_name or not drop_name or keep_name == drop_name:
            return 0
        cur = self.db.execute("UPDATE transcripts SET speaker=? WHERE speaker=?",
                              (keep_name, drop_name))
        self.db.execute("DELETE FROM speakers WHERE name=?", (drop_name,))
        self.db.commit()
        return cur.rowcount

    def delete_speaker(self, speaker_id):
        """Forget a voiceprint (future meetings won't match it). Transcripts keep
        their existing text label."""
        self.db.execute("DELETE FROM speakers WHERE id=?", (speaker_id,))
        self.db.commit()

    def speaker_best_span(self, name, min_ms=800):
        """Longest single utterance for a speaker (audio preview / 試聽). Returns
        (meeting_id, track, start_ms, end_ms) or None if none has a real duration."""
        return self.db.execute(
            "SELECT meeting_id, track, start_ms, end_ms FROM transcripts "
            "WHERE speaker=? AND end_ms > start_ms + ? "
            "ORDER BY (end_ms - start_ms) DESC LIMIT 1", (name, min_ms)).fetchone()

    def speaker_utterances(self, name, limit=500):
        """Every utterance by a speaker ACROSS meetings (newest first)."""
        return self.db.execute(
            "SELECT t.meeting_id, m.title, m.created_at, t.start_ms, t.text "
            "FROM transcripts t JOIN meetings m ON m.id=t.meeting_id "
            "WHERE t.speaker=? ORDER BY m.created_at DESC, t.start_ms LIMIT ?",
            (name, limit)).fetchall()

    # --- key/value settings (e.g. the global persistent-speaker toggle) ---
    def get_setting(self, k, default=None):
        r = self.db.execute("SELECT v FROM settings WHERE k=?", (k,)).fetchone()
        return r["v"] if r else default

    def set_setting(self, k, v):
        self.db.execute(
            "INSERT INTO settings(k,v) VALUES(?,?) "
            "ON CONFLICT(k) DO UPDATE SET v=excluded.v", (k, str(v)))
        self.db.commit()

    # --- tags (many-to-many: a meeting can carry #客戶 #1on1 #產品) ---
    def add_tag(self, meeting_id, name):
        name = (name or "").strip().lstrip("#").strip()
        if not name:
            return False
        self.db.execute("INSERT OR IGNORE INTO tags(name) VALUES(?)", (name,))
        tid = self.db.execute("SELECT id FROM tags WHERE name=?", (name,)).fetchone()[0]
        self.db.execute("INSERT OR IGNORE INTO meeting_tags(meeting_id, tag_id) VALUES(?,?)",
                        (meeting_id, tid))
        self.db.commit()
        return True

    def remove_tag(self, meeting_id, name):
        self.db.execute(
            "DELETE FROM meeting_tags WHERE meeting_id=? AND tag_id="
            "(SELECT id FROM tags WHERE name=?)", (meeting_id, (name or "").strip().lstrip("#")))
        self.db.commit()

    def tags_for(self, meeting_id):
        return [r["name"] for r in self.db.execute(
            "SELECT t.name FROM tags t JOIN meeting_tags mt ON mt.tag_id=t.id "
            "WHERE mt.meeting_id=? ORDER BY t.name", (meeting_id,)).fetchall()]

    def all_tags(self):
        """[{name, count}] over meetings that still exist, most-used first."""
        return [dict(r) for r in self.db.execute(
            "SELECT t.name, COUNT(mt.meeting_id) AS count FROM tags t "
            "JOIN meeting_tags mt ON mt.tag_id=t.id "
            "JOIN meetings m ON m.id=mt.meeting_id "
            "GROUP BY t.id ORDER BY count DESC, t.name").fetchall()]

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
        self.db.execute("DELETE FROM meeting_tags WHERE meeting_id=?", (meeting_id,))
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
