# Caption-Only Ephemeral Live Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a floatpanel-only "字幕限定" toggle: when on, `/ws/native-capture`
runs a fully ephemeral session — no `meetings` row, no `transcripts` rows, no
audio on disk. Captions still stream live to the panel over the existing
websocket push channel. Text lost when the ASR engine can't keep up is simply
gone (no audio backup to recover it from) — that is the intended behavior,
not a bug to guard against.

**Architecture:** A new `mode="caption"` value recognized only by
`/ws/native-capture` (never a persisted `live_mode` setting — floatpanel sends
it explicitly via its own new toggle). The session gets a synthetic negative
"pseudo-mid" instead of a real `meetings.id`, so every existing bookkeeping
dict (`live_active`, `native_sessions`, `live_paused`, `live_stop`) keeps
working unchanged — they're plain dicts/sets keyed by any int, never actual
foreign keys. A new `live_session.make_caption_only_emit()` mirrors
`make_store_emit`'s same-speaker line-merging logic entirely in memory (no
`store` calls), pushing over the same websocket channel the floatpanel
already listens on for live captions.

**Tech Stack:** Python (FastAPI, existing `live_session.py`/`app.py`
pipeline), Swift/SwiftUI (`floatpanel`).

## Global Constraints

- Floatpanel-only. Web `/live` page, the shared `/settings/live_mode` picker,
  and its validation whitelist (`app.py`'s `_SETTINGS`/`set_setting_route`,
  currently `v if v in ("both", "record", "transcribe") else "both"`) are
  **not touched** — `"caption"` must never become a persistable value there.
  It only ever arrives as an explicit `?mode=caption` query param from
  floatpanel's own new toggle.
- No diarization in this mode. The server must force `diarize = False`
  whenever `mode == "caption"`, regardless of what the client sent — this is
  the single authoritative guard, not a client-side courtesy.
- No new drop/backpressure mechanism. Lost text under load is the existing,
  unmodified consequence of not keeping an audio backup — do not add queue
  caps, drop-oldest logic, or any new mechanism framed as "handling" this.
- No reconnect/resume support for this mode — every `/ws/native-capture`
  connection with `mode=caption` always mints a fresh pseudo-mid, even if a
  `session=` query param is present.
- **`app.py` git hygiene (recurring risk in this repo):** `app.py` routinely
  carries the user's own unrelated uncommitted work in its working tree.
  Before staging, run `git diff app.py` and confirm you can see BOTH your own
  edits AND a separate unrelated block — the unrelated block is not yours.
  Never run `git add app.py`, `git add -A`, or `git add .`. Stage only your
  hunks via `git add -p app.py` (answer `y` only to your hunks) or a
  hand-built patch applied with `git apply --cached`. After staging, run
  `git diff --cached app.py` and confirm it shows ONLY your intended change.
  Commit with a bare `git commit` (no pathspec) — `git commit -- app.py ...`
  re-stages the full working-tree version of that path and silently
  reintroduces whatever you just excluded.
- Swift side: this repo has no XCTest target or `swift test` infrastructure
  (`Package.swift` only declares an `executableTarget`). Do not add one for
  this feature — verify the Swift changes by running `swift build -c
  release` from `swift/floatpanel/` and manual testing, as the rest of this
  file's Swift work already does.

---

### Task 1: `live_session.make_caption_only_emit()` + `flush_sessions(persist=...)`

**Files:**
- Modify: `live_session.py` (insert new function after `make_store_emit`,
  currently ending at line 332; modify `flush_sessions`, currently
  `live_session.py:496-527`)
- Test: `tests/test_live_session.py`

**Interfaces:**
- Produces: `live_session.make_caption_only_emit(push=None) -> Callable[[dict,
  tuple[str, str]], Awaitable[None]]` — an `emit(ev, label)` coroutine with
  the exact same call signature `make_store_emit`'s returned `emit` has
  (`ev` is `{"kind": "interim"|"final", "text": str, "start_ms": int,
  "end_ms": int, "speaker": str|None, ...}`; `label` is `(track, side_label)`
  tuple), so `live_session.consume()` can be handed either interchangeably.
  Unlike `make_store_emit`, it takes no `mid`/`conn_offset_ms`/`store` —
  everything it needs is either in `ev`/`label` or its own closure state.
- Produces: `live_session.flush_sessions(sessions, tracks, mid,
  conn_offset_ms, store, skip=(), persist=True)` — new keyword-only
  `persist` (default `True` preserves current behavior for every existing
  caller). `persist=False` skips the `store.add_transcript` call for every
  flushed final; the rest of the function (per-tag `flush()` timeout
  handling, the peak-RSS log line) runs unchanged.
- Consumes: `live_session.store_speaker` (existing, `live_session.py:53`),
  `live_session._MERGE_GAP_MS` / `_MERGE_MAX_MS` / `_MERGE_MAX_CHARS`
  (existing module constants, `live_session.py:36-38`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_live_session.py` (after `test_make_store_emit_persists_only_final`,
currently ending at line 434):

```python
def _caption_finals(evs):
    pushed = []

    async def push(payload):
        pushed.append(payload)

    emit = live_session.make_caption_only_emit(push=push)

    async def run():
        for ev, label in evs:
            await emit({"kind": "final", **ev}, label)
    asyncio.run(run())
    return [p for p in pushed if p["type"] == "final"]


def test_caption_only_emit_never_touches_the_store(monkeypatch):
    # The whole point of this mode: zero DB calls. Fail loudly if anything
    # tries to reach a `store` object at all (there isn't one to pass in).
    emit = live_session.make_caption_only_emit()

    async def run():
        await emit({"kind": "interim", "text": "hi", "start_ms": 0, "end_ms": 0},
                   ("mic", "我"))
        await emit({"kind": "final", "text": "hi", "start_ms": 0, "end_ms": 500},
                   ("mic", "我"))
    asyncio.run(run())  # would raise if it touched anything DB-shaped


def test_caption_only_emit_streams_interim_and_final():
    pushed = []

    async def push(p):
        pushed.append(p)

    emit = live_session.make_caption_only_emit(push=push)

    async def run():
        await emit({"kind": "interim", "text": "暫定", "start_ms": 0, "end_ms": 0},
                   ("system", "對方"))
        await emit({"kind": "final", "text": "定稿", "start_ms": 0, "end_ms": 1000},
                   ("system", "對方"))
    asyncio.run(run())
    assert pushed[0]["type"] == "interim" and pushed[0]["text"] == "暫定"
    assert pushed[1] == {"type": "final", "track": "system", "speaker": "對方",
                         "text": "定稿"}


def test_caption_only_emit_merges_same_speaker_short_gap():
    rows = _caption_finals([
        ({"text": "你好", "start_ms": 0, "end_ms": 1000}, ("system", "說話者1")),
        ({"text": "今天", "start_ms": 1500, "end_ms": 2500}, ("system", "說話者1")),
    ])
    assert len(rows) == 2                 # one push per emit call...
    assert rows[-1]["text"] == "你好今天"  # ...but the second carries the MERGED text


def test_caption_only_emit_speaker_change_starts_new_line():
    rows = _caption_finals([
        ({"text": "你好", "start_ms": 0, "end_ms": 1000}, ("system", "說話者1")),
        ({"text": "我是", "start_ms": 1200, "end_ms": 2000}, ("system", "說話者2")),
    ])
    assert [r["text"] for r in rows] == ["你好", "我是"]  # not merged


def test_caption_only_emit_long_gap_starts_new_line():
    rows = _caption_finals([
        ({"text": "你好", "start_ms": 0, "end_ms": 1000}, ("system", "說話者1")),
        ({"text": "再來", "start_ms": 6000, "end_ms": 7000}, ("system", "說話者1")),
    ])
    assert [r["text"] for r in rows] == ["你好", "再來"]  # gap > 3s -> not merged


def test_flush_sessions_persist_false_skips_the_store(tmp_path):
    store = Store(tmp_path / "m.db")
    mid = store.create_meeting("t", 0.0, "zh-TW")
    tracks = {"t": ("mic", "我")}
    session = StubSession(flush_events=[{"kind": "final", "start_ms": 0, "end_ms": 500,
                                         "text": "hello"}])
    asyncio.run(live_session.flush_sessions({"t": session}, tracks, mid, conn_offset_ms=0,
                                            store=store, persist=False))
    assert store.latest_transcript(mid) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_live_session.py -k "caption_only or persist_false" -v`
Expected: FAIL — `AttributeError: module 'live_session' has no attribute
'make_caption_only_emit'` (and the `persist_false` test fails on the
unexpected `persist` keyword argument).

- [ ] **Step 3: Implement `make_caption_only_emit()`**

Insert this function into `live_session.py` immediately after
`make_store_emit` ends (after its closing `return emit` and the two blank
lines, currently right before `async def consume(...)` at line 335):

```python
def make_caption_only_emit(push=None):
    """Same same-speaker line-merging behavior as make_store_emit, but for
    the fully ephemeral caption-only mode (mode="caption"): there is no
    meeting row and no transcripts table to read/write, so the "previous
    line" lives in a plain dict scoped to this closure instead of a
    store.last_live_row() round-trip. push is the ONLY output — the
    floatpanel reads captions exclusively from these pushed events (see
    docs/superpowers/specs/2026-08-11-caption-only-mode-design.md), never
    from a DB-backed poll, so there is nothing else for this function to do."""
    last = {}  # track -> {"speaker", "text", "start_ms", "end_ms"}

    async def emit(ev, label):
        track, speaker = label
        if ev["kind"] != "final":
            if push:
                await push({"type": "interim", "track": track,
                            "speaker": speaker, "text": ev["text"]})
            return
        spk = store_speaker(ev.get("speaker"), speaker)
        start = ev["start_ms"]
        end = ev.get("end_ms", ev["start_ms"])
        prev = last.get(track)
        shown = ev["text"]
        if prev is not None and prev["speaker"] == spk:
            gap = start - prev["end_ms"]
            joined = (prev["text"] or "") + (ev["text"] or "")
            if (0 <= gap <= _MERGE_GAP_MS
                    and (end - prev["start_ms"]) <= _MERGE_MAX_MS
                    and len(joined) <= _MERGE_MAX_CHARS):
                last[track] = {"speaker": spk, "text": joined,
                               "start_ms": prev["start_ms"], "end_ms": end}
                shown = joined
            else:
                last[track] = {"speaker": spk, "text": ev["text"],
                               "start_ms": start, "end_ms": end}
        else:
            last[track] = {"speaker": spk, "text": ev["text"],
                           "start_ms": start, "end_ms": end}
        if push:
            await push({"type": "final", "track": track,
                        "speaker": spk, "text": shown})
    return emit
```

- [ ] **Step 4: Implement `flush_sessions(persist=True)`**

In `live_session.py`, change the `flush_sessions` signature and the
`store.add_transcript` call (currently lines 496 and 514-520):

```python
async def flush_sessions(sessions, tracks, mid, conn_offset_ms, store, skip=(),
                         persist=True):
    """Final flush on stop: drain each TwoPassSession's tail (a partial
    utterance) and persist any 'final' it yields. Timed out per-track so a
    wedged backend can't hang stop forever — the backend's own watchdog
    reaps the thread. `skip`: tags consume() couldn't safely hand off (its
    feed() call was still running) — touching their session here would race
    that leftover thread, so skip flush for just those tags. `persist=False`
    (caption-only mode): still runs each session's flush() to drain its
    internal buffer cleanly, but never writes the result anywhere — there is
    no meeting row for it to belong to."""
    for tag, s in sessions.items():
        if tag in skip:
            continue
        t = time.perf_counter()
        try:
            evs = await asyncio.wait_for(run_in_threadpool(s.flush), timeout=15)
        except Exception as e:  # noqa: BLE001  (TimeoutError or backend error)
            print(f"live flush skipped ({tag}): {e}", file=sys.stderr)
            evs = []
        print(f"[live stop] flush {tag} {(time.perf_counter()-t)*1000:.0f}ms",
              file=sys.stderr)
        if not persist:
            continue
        for ev in evs:
            if ev["kind"] == "final":  # audio-position offset, not wall-clock
                spk = store_speaker(ev.get("speaker"), tracks[tag][1])
                store.add_transcript(mid, "live", tracks[tag][0],
                                      ev["start_ms"] + conn_offset_ms,
                                      ev.get("end_ms", ev["start_ms"]) + conn_offset_ms,
                                      spk, ev["text"])
    try:  # diagnostic: peak process RSS, to catch a live memory blowup (macOS: bytes)
        import resource  # noqa: PLC0415
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
        print(f"[live stop] peak RSS {rss:.0f}MB", file=sys.stderr)
    except Exception:  # noqa: BLE001
        pass
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_live_session.py -v`
Expected: all PASS, including the new tests and every pre-existing test in
this file (no regression in `test_flush_sessions_persists_finals` etc. —
`persist` defaults to `True`).

- [ ] **Step 6: Commit**

`live_session.py` has no known WIP-contamination risk (only `app.py` does
per the Global Constraints) — stage normally.

```bash
git add live_session.py tests/test_live_session.py
git commit -m "feat(live_session): add caption-only emit + non-persisting flush"
```

---

### Task 2: Server wiring in `app.py`

**Files:**
- Modify: `app.py` — module-level session state block (currently
  `app.py:2952-2975`), `/live/state` (currently `app.py:3052-3085`),
  `/ws/native-capture` (currently `app.py:3113-3312`)
- Test: `tests/test_live_modes.py`

**Interfaces:**
- Consumes: `live_session.make_caption_only_emit` and
  `live_session.flush_sessions(..., persist=...)` from Task 1.
- Produces: no new public function — this task only changes route bodies. No
  other task depends on new names from this one.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_live_modes.py` (after `test_transcribe_mode_keeps_no_audio_and_no_segment`,
currently ending at line 37):

```python
def test_caption_mode_creates_no_meeting_and_no_transcripts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = Store(tmp_path / "m.db")
    app = create_app(store, summary_backend=lambda p: "x", asr_backend=None,
                     live_manager=_FakeLiveManager())
    before = len(store.list_meetings())
    with TestClient(app) as c:
        with c.websocket_connect(
                "/ws/native-capture?source=system&mode=caption&diarize=1") as ws:
            mid = ws.receive_json()["id"]
            assert mid < 0  # pseudo-mid, never a real autoincrement id
            ws.send_bytes(_frames())
            assert _wait_until(lambda: c.get("/live/state").json()["recording"] is True)
            state = c.get("/live/state").json()
            assert state["mid"] == mid
            assert state["title"] == "字幕限定"  # fallback, not a DB row
        assert _wait_until(lambda: c.get("/live/state").json()["recording"] is False)

    assert len(store.list_meetings()) == before  # NOT ONE new meeting row
    assert list((tmp_path / "data").glob("*/*.pcm")) == []  # and no audio
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_live_modes.py::test_caption_mode_creates_no_meeting_and_no_transcripts -v`
Expected: FAIL — currently `mode=caption` falls through the existing
`if mode == "transcribe": ... else: <records audio + creates a real
meeting>` branch, so a real (positive) `mid` is created and `store.list_meetings()`
grows by one.

- [ ] **Step 3: Add the pseudo-mid counter + fallback session dict**

In `app.py`, in the module-scope block inside `create_app` (currently lines
2952-2975), the block currently ends with:

```python
    _live_rev = {}
    _CAPTION_IDLE_S = 6  # clear the panel caption after this much silence
```

Change it to add two new pieces of state right after `_live_rev = {}`:

```python
    _live_rev = {}
    # Fully ephemeral live captions (mode="caption", floatpanel only — see
    # docs/superpowers/specs/2026-08-11-caption-only-mode-design.md): no
    # meeting row exists, so /live/state needs somewhere else to read
    # title/started_at from. Keyed by the same pseudo-mid used everywhere
    # else (live_active, native_sessions, ...). Popped when the session ends.
    _caption_sessions = {}
    _caption_mid_ctr = [0]  # next pseudo-mid = decrement then use; negative,
    # so it can never collide with a real (positive, autoincrement) meetings.id

    def _next_caption_mid():
        _caption_mid_ctr[0] -= 1
        return _caption_mid_ctr[0]

    _CAPTION_IDLE_S = 6  # clear the panel caption after this much silence
```

- [ ] **Step 4: Add the `/live/state` fallback**

In `app.py`'s `live_state()` (currently lines 3052-3085), change:

```python
        if mid is not None:
            m = store.get_meeting(mid)
            title = m["title"] if m else None
            started_at = m["created_at"] if m else None  # session start epoch, for /live attach-on-load's timer
```

to:

```python
        if mid is not None:
            # A caption-only session (mode="caption") has no meetings row —
            # fall back to the in-memory session dict app._caption_sessions
            # keeps for exactly this (see its definition for why).
            m = store.get_meeting(mid) or _caption_sessions.get(mid)
            title = m["title"] if m else None
            started_at = m["created_at"] if m else None  # session start epoch, for /live attach-on-load's timer
```

(The rest of `live_state()` — `caption`/`captions`/`last_caption_end_ms`
queries — needs no change: they're plain `WHERE meeting_id=?` lookups against
a pseudo-mid that matches no row, so they naturally return `None`/`[]`
without error. The floatpanel's caption display doesn't read these fields
anyway — see the design doc.)

- [ ] **Step 5: Wire `mode == "caption"` into `/ws/native-capture`**

In `app.py`'s `ws_native_capture` (currently lines 3113-3201), the current
code reads:

```python
        qp = ws.query_params
        source = qp.get("source", "mic")
        if source not in ("mic", "system", "both"):
            source = "mic"
        diarize = qp.get("diarize") == "1"

        import recorder  # noqa: PLC0415
        t0 = time.time()
        # Reconnect/resume: floatpanel remembers the meeting id from the
        # {"type":"meeting"} message below and passes it back as ?session=
        # after a dropped relay socket, so the resumed audio lands in the SAME
        # meeting instead of fragmenting into a new one each reconnect. Same
        # pattern as ws_live (see there for the offset rationale).
        session = qp.get("session")
        if session and session.isdigit() and store.get_meeting(int(session)) is not None:
            mid = int(session)
            conn_offset_ms = max(0, int((t0 - store.get_meeting(mid)["created_at"]) * 1000))
        else:
            title = time.strftime("錄音 %Y-%m-%d %H:%M", time.localtime(t0))
            mid = store.create_meeting(title, t0, "zh-TW")
            conn_offset_ms = 0
        await ws.send_json({"type": "meeting", "id": mid})
        # Session mode (see _SETTINGS['live_mode']). The floatpanel sends no mode
        # of its own, so it inherits the saved setting — same trick live_language
        # uses to reach the panel without a Swift change.
        mode = ws.query_params.get("mode") or store.get_setting("live_mode", "both")
        if source == "mic":
            tracks = {recorder.TRACK_MIC: ("mic", "我")}
        elif source == "system":
            tracks = {recorder.TRACK_SYSTEM: ("system", "對方")}
        else:
            tracks = {recorder.TRACK_MIC: ("mic", "我"),
                      recorder.TRACK_SYSTEM: ("system", "對方")}
        if mode == "transcribe":
            # 純字幕: no recording kept. Skip the segment row too — an empty audio
            # dir is otherwise indistinguishable from a recording that failed.
            audio_files = {tag: live_session.NullSink() for tag in tracks}
        else:
            audio_dir = f"data/{mid}-{int(t0)}"
            os.makedirs(audio_dir, exist_ok=True)
            store.add_segment(mid, idx=len(store.list_segments(mid)), dir_path=audio_dir,
                              started_at=t0, duration_s=0, origin="recorded")
            audio_files = {tag: open(f"{audio_dir}/{lbl[0]}.pcm", "wb")
                           for tag, lbl in tracks.items()}
```

Change it to:

```python
        qp = ws.query_params
        source = qp.get("source", "mic")
        if source not in ("mic", "system", "both"):
            source = "mic"
        # Session mode (see _SETTINGS['live_mode']). The floatpanel sends no mode
        # of its own for the normal cases (it inherits the saved setting — same
        # trick live_language uses to reach the panel without a Swift change) —
        # EXCEPT "caption", which floatpanel's own toggle always sends explicitly
        # and which is deliberately never a persistable _SETTINGS value (it must
        # never leak into a browser /ws/live session via the shared setting).
        mode = qp.get("mode") or store.get_setting("live_mode", "both")
        # No diarization in caption-only mode (see the design doc) — enforced
        # here regardless of what the client sent, since this is the one place
        # that decides whether any `store` call happens on a pseudo-mid below.
        diarize = qp.get("diarize") == "1" and mode != "caption"

        import recorder  # noqa: PLC0415
        t0 = time.time()
        if mode == "caption":
            # 字幕限定: fully ephemeral. A pseudo-mid (negative, never collides
            # with a real autoincrement meetings.id) satisfies every OTHER piece
            # of bookkeeping below (live_active/native_sessions/live_paused/
            # live_stop) — none of them actually require a real DB row, only a
            # hashable key. No reconnect/resume for this mode: every connection
            # mints a fresh pseudo-mid, even if a stale ?session= is present.
            mid = _next_caption_mid()
            conn_offset_ms = 0
            _caption_sessions[mid] = {"title": "字幕限定", "created_at": t0}
        else:
            # Reconnect/resume: floatpanel remembers the meeting id from the
            # {"type":"meeting"} message below and passes it back as ?session=
            # after a dropped relay socket, so the resumed audio lands in the SAME
            # meeting instead of fragmenting into a new one each reconnect. Same
            # pattern as ws_live (see there for the offset rationale).
            session = qp.get("session")
            if session and session.isdigit() and store.get_meeting(int(session)) is not None:
                mid = int(session)
                conn_offset_ms = max(0, int((t0 - store.get_meeting(mid)["created_at"]) * 1000))
            else:
                title = time.strftime("錄音 %Y-%m-%d %H:%M", time.localtime(t0))
                mid = store.create_meeting(title, t0, "zh-TW")
                conn_offset_ms = 0
        await ws.send_json({"type": "meeting", "id": mid})
        if source == "mic":
            tracks = {recorder.TRACK_MIC: ("mic", "我")}
        elif source == "system":
            tracks = {recorder.TRACK_SYSTEM: ("system", "對方")}
        else:
            tracks = {recorder.TRACK_MIC: ("mic", "我"),
                      recorder.TRACK_SYSTEM: ("system", "對方")}
        if mode in ("transcribe", "caption"):
            # 純字幕/字幕限定: no recording kept. Skip the segment row too — an
            # empty audio dir is otherwise indistinguishable from a recording
            # that failed. (caption mode also has no meeting row to attach a
            # segment to in the first place.)
            audio_files = {tag: live_session.NullSink() for tag in tracks}
        else:
            audio_dir = f"data/{mid}-{int(t0)}"
            os.makedirs(audio_dir, exist_ok=True)
            store.add_segment(mid, idx=len(store.list_segments(mid)), dir_path=audio_dir,
                              started_at=t0, duration_s=0, origin="recorded")
            audio_files = {tag: open(f"{audio_dir}/{lbl[0]}.pcm", "wb")
                           for tag, lbl in tracks.items()}
```

- [ ] **Step 6: Select the right `emit`/`flush_sessions` call and clean up `_caption_sessions`**

In the same handler's `_run()` closure (currently lines 3276-3301), change:

```python
        async def _run():
            drain = asyncio.create_task(_push_drain())
            stuck = set()
            try:
                stuck = await live_session.consume(
                    pump, sessions, tracks, rec_on=lambda: mode == "record",
                    emit=live_session.make_store_emit(mid, conn_offset_ms, store,
                                                      push=_push),
                    should_stop=_should_stop,
                    interim_lag_bytes=int(2 * live_interim_s * 16000) * 2,
                    pop_notice=_pop_notice)
            finally:
                drain.cancel()
                pump.pad_to(time.time())  # writes remaining silence to disk (before close)
                # Clear the recording state FIRST so /live/state (the panel's dot)
                # flips to idle immediately; persist the trailing utterance after —
                # a slow final ASR no longer holds the UI in "recording".
                idle["live"] = max(0, idle["live"] - 1)
                live_active.pop(mid, None)
                native_sessions.pop(mid, None)
                live_paused.discard(mid)
                _touch()
                for f in audio_files.values():
                    f.close()
                await live_session.flush_sessions(sessions, tracks, mid, conn_offset_ms,
                                                  store, skip=stuck)
```

to:

```python
        async def _run():
            drain = asyncio.create_task(_push_drain())
            stuck = set()
            try:
                stuck = await live_session.consume(
                    pump, sessions, tracks, rec_on=lambda: mode == "record",
                    emit=(live_session.make_caption_only_emit(push=_push)
                          if mode == "caption" else
                          live_session.make_store_emit(mid, conn_offset_ms, store,
                                                       push=_push)),
                    should_stop=_should_stop,
                    interim_lag_bytes=int(2 * live_interim_s * 16000) * 2,
                    pop_notice=_pop_notice)
            finally:
                drain.cancel()
                pump.pad_to(time.time())  # writes remaining silence to disk (before close)
                # Clear the recording state FIRST so /live/state (the panel's dot)
                # flips to idle immediately; persist the trailing utterance after —
                # a slow final ASR no longer holds the UI in "recording".
                idle["live"] = max(0, idle["live"] - 1)
                live_active.pop(mid, None)
                native_sessions.pop(mid, None)
                live_paused.discard(mid)
                _caption_sessions.pop(mid, None)
                _touch()
                for f in audio_files.values():
                    f.close()
                await live_session.flush_sessions(sessions, tracks, mid, conn_offset_ms,
                                                  store, skip=stuck,
                                                  persist=(mode != "caption"))
```

- [ ] **Step 7: Run the test to verify it passes**

Run: `pytest tests/test_live_modes.py -v`
Expected: all PASS, including the new `test_caption_mode_creates_no_meeting_and_no_transcripts`
and every pre-existing test in this file (default/`transcribe`/pause modes
unaffected).

- [ ] **Step 8: Run the full test suite**

Run: `pytest -q`
Expected: all PASS except the 3 pre-existing `tests/test_live_hpf.py`
failures (`ModuleNotFoundError: No module named 'scipy'` — a local
environment gap unrelated to this change; confirm the failure count and
names match exactly this pre-existing set, nothing new).

- [ ] **Step 9: Commit — CAREFULLY (see Global Constraints)**

`app.py` needs the WIP-safe staging procedure. First inspect what's yours:

```bash
git diff app.py | grep -n "^@@"
```

Identify the hunks covering the changes from Steps 3-6 above (the
`_caption_sessions`/`_caption_mid_ctr` block, the `/live/state` fallback, and
the `/ws/native-capture` changes). Stage only those, either with:

```bash
git add -p app.py   # answer y only to your hunks, n to everything else
```

or by extracting just your hunks into a patch file and `git apply --cached`
on it (same technique used earlier in this repo's history for this exact
problem — see `git log --oneline --grep="groq"` for prior art if needed).
Then verify:

```bash
git diff --cached app.py   # must show ONLY your hunks
```

If anything else appears, `git restore --staged app.py` and redo more
carefully. Once clean:

```bash
git add tests/test_live_modes.py
git commit
```

(Bare `git commit` — no pathspec — write the message in your editor or via
`-m`, but do not pass `-- app.py ...` to the commit command itself.)

---

### Task 3: Floatpanel UI toggle

**Files:**
- Modify: `swift/floatpanel/Sources/floatpanel/main.swift` — line numbers
  below are pre-Step-1; each step's insert shifts everything after it. Locate
  each edit by the exact "currently reads" quote, not the line number.

**Interfaces:**
- Consumes: `mode=caption` query param recognized by `/ws/native-capture`
  (Task 2). No Swift-side test infrastructure exists (see Global
  Constraints) — verify with `swift build -c release` from
  `swift/floatpanel/` and manual testing.

- [ ] **Step 1: Add the `captionOnly` published property**

In `swift/floatpanel/Sources/floatpanel/main.swift`, in `Model`'s property
list (currently around line 318), right after:

```swift
    @Published var showSubtitle = false   // YouTube-style caption overlay toggle
```

add:

```swift
    // 字幕限定 (caption-only, ephemeral): floatpanel-only toggle, sent as
    // ?mode=caption to /ws/native-capture. No meeting/transcript/audio is
    // ever created server-side while this is on — see
    // docs/superpowers/specs/2026-08-11-caption-only-mode-design.md. Not
    // persisted; resets to off on every launch, same as `source`.
    @Published var captionOnly = false
```

- [ ] **Step 2: Send `mode=caption` in the relay URL**

In the same file, `openRelayTask` (currently lines 614-624) currently reads:

```swift
    private func openRelayTask(session: URLSession, epoch: Int) -> URLSessionWebSocketTask? {
        var urlStr = "ws://127.0.0.1:\(port)/ws/native-capture?source=\(source.rawValue)&diarize=1"
        if let m = relayMid { urlStr += "&session=\(m)" }
```

Change to:

```swift
    private func openRelayTask(session: URLSession, epoch: Int) -> URLSessionWebSocketTask? {
        var urlStr = "ws://127.0.0.1:\(port)/ws/native-capture?source=\(source.rawValue)&diarize=1"
        if captionOnly { urlStr += "&mode=caption" }
        if let m = relayMid { urlStr += "&session=\(m)" }
```

(The server ignores `diarize=1` and any `session=` when `mode=caption` is
present — see Task 2, Step 5 — so neither needs removing here.)

- [ ] **Step 3: Skip the backfill POST in caption-only mode**

In the same file, `sendBackfill` (currently lines 738-749) currently starts:

```swift
    private func sendBackfill(gapStart: Date) {
        guard let sink = self.sink, let m = relayMid ?? mid else { return }
```

Change to:

```swift
    private func sendBackfill(gapStart: Date) {
        // No audio was ever meant to survive in caption-only mode — buffering
        // and POSTing an outage gap for a session with no meeting row to
        // attach it to would be pointless (and /native/backfill has nothing
        // sensible to do with a negative pseudo-mid).
        guard !captionOnly else { return }
        guard let sink = self.sink, let m = relayMid ?? mid else { return }
```

- [ ] **Step 4: Add the toggle to `PanelView`**

In the same file, `PanelView.body` (currently around lines 827-832) currently
reads:

```swift
            if !m.recording {
                Picker("來源", selection: $m.source) {
                    ForEach(Source.allCases) { s in Text(s.label).tag(s) }
                }
                .pickerStyle(.segmented).labelsHidden()
            }
```

Change to:

```swift
            if !m.recording {
                Picker("來源", selection: $m.source) {
                    ForEach(Source.allCases) { s in Text(s.label).tag(s) }
                }
                .pickerStyle(.segmented).labelsHidden()
                Toggle(isOn: $m.captionOnly) {
                    Text("💬 字幕限定（不錄音、不留紀錄）")
                }
                .toggleStyle(.checkbox).font(.caption)
                .help("只顯示即時字幕，不錄音、不建立會議記錄、不存逐字稿。"
                      + "效能不足時文字可能遺失（不留音檔可補救，這是預期行為）。")
            }
```

- [ ] **Step 5: Build to verify it compiles**

Run: `cd swift/floatpanel && swift build -c release`
Expected: `Build complete!` with no errors or warnings introduced by this
change.

- [ ] **Step 6: Manual verification**

Run the built binary (or `build_floatpanel_app.sh` + `open` the packaged
`.app`, matching how this repo's prior floatpanel changes were verified —
see `native-ui` project memory), toggle "💬 字幕限定" on before starting a
recording, start it, speak, and confirm:
- Captions appear in the panel as normal.
- No new row appears in the web meeting list (`/`) for this session.
- After stopping, `data/` gained no new directory for this session.

- [ ] **Step 7: Commit**

```bash
git diff swift/floatpanel/Sources/floatpanel/main.swift | grep -n "^@@"
```

This file also carries the user's own unrelated in-progress work (per this
repo's established pattern — same care as `app.py`, though historically less
severe here). Confirm which hunks are yours (the `captionOnly` property, the
`openRelayTask`/`sendBackfill`/`PanelView` changes from Steps 1-4) before
staging. If the diff shows ONLY your hunks, `git add` is fine; if not, use
the same `git add -p` / hand-built-patch technique as Task 2's Step 9.

```bash
git commit -m "feat(floatpanel): add ephemeral caption-only mode toggle"
```

---

## Note on the approved spec

The spec's "測試" section says Swift-side testing stays manual (Task 3, Step
6 here) since this repo has no XCTest infrastructure — confirmed correct by
inspecting `Package.swift` (only an `executableTarget`, no test target) and
finding no `swift test`/XCTest references anywhere in the repo's build
scripts. Nothing in the spec was left unaddressed by these three tasks.
