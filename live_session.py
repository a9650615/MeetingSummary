"""Shared plumbing for a live-transcription session, used by BOTH audio
sources MeetingSummary supports today:

  * the browser /ws/live handler — PCM arrives as websocket binary frames
    (mic via getUserMedia, system/對方 audio via getDisplayMedia)
  * /ws/native-capture — the floatpanel captures mic + system audio
    in-process and relays it framed (recorder.py's <track><len><payload>
    protocol) over the websocket, no browser involved at all

Everything after "bytes for track X arrived" — wall-clock padding so all
tracks share one clock, feeding TwoPassSession, persisting finals, the
backlog trim, the flush-on-stop — is one code path so the two entry points
in app.py don't duplicate it.
"""
import asyncio
import re
import sys
import time

from starlette.concurrency import run_in_threadpool
from starlette.websockets import WebSocketDisconnect

import recorder

TRACK_BACKLOG_MAXB = 45 * 16000 * 2  # ~45s cap; full audio is on disk for re-transcribe

_PLACEHOLDER = re.compile(r"^(我|對方|說話者|混合)\d+$")


def resolve_speaker(diar_label, side_label):
    """The speaker to DISPLAY for a live final. A recognized name (from the
    voiceprint DB) wins; an auto placeholder (說話者N / 對方N / 我N) collapses to
    the track's side label (對方 = 系統音/remote, 我 = 麥克風). So an unrecognized
    voice always shows which side spoke — never a meaningless 說話者N — and the
    output is uniform whether or not diarization named the utterance. Kept for
    display collapse only — see display_speaker for the row-level equivalent
    used once labels are STORED via store_speaker (below)."""
    name = (diar_label or side_label or "").strip()
    return side_label if _PLACEHOLDER.match(name) else (name or side_label)


def store_speaker(diar_label, side_label):
    """The value to STORE for a live final. Unlike resolve_speaker, this keeps a
    session cluster label (說話者N) or a recognized/promoted name AS-IS — never
    collapsed to 對方/我 — so a later promotion (diarize.live_speaker_labeler's
    on_promote) can target this cluster's earlier rows by name and rename them
    (store.rename_speaker). Only falls back to the side label when diarization
    produced nothing at all (e.g. the mic track, never diarized)."""
    return diar_label or side_label


_CLUSTER_LABEL = re.compile(r"^說話者\d+$")


def display_speaker(stored, track):
    """Collapse a STORED session cluster label (說話者N, not yet promoted) to its
    track's side label for DISPLAY only — 對方/我/混合. A promoted name, a
    recognized name, or an already-side label passes through unchanged. Post-
    meeting /diarize labels (對方1, 我2, …) never match here (different prefix),
    so this never touches that feature's own multi-speaker labels."""
    if stored and _CLUSTER_LABEL.match(stored):
        return {"system": "對方", "mixed": "混合"}.get(track, "我")
    return stored


class WallClockPump:
    """Keeps every track's saved PCM + ASR buffer on one shared wall clock:
    on each feed(), pads EVERY track with silence up to now-t0 first, so all
    tracks stay the same length and on one clock regardless of when each
    source starts or how bursty it is (mic vs. system audio vs. a subprocess
    that hasn't written in a while)."""

    def __init__(self, tracks, audio_files, t0):
        self.tracks = tracks
        self.audio_files = audio_files
        self.t0 = t0
        self.buffers = {tag: bytearray() for tag in tracks}
        self.written = {tag: 0 for tag in tracks}
        self.got = asyncio.Event()

    def pad_to(self, now):
        target = int((now - self.t0) * 16000) * 2
        for tag in self.tracks:
            gap = target - self.written[tag]
            gap -= gap % 2
            if gap > 0:
                sil = bytes(gap)
                self.audio_files[tag].write(sil)
                self.buffers[tag].extend(sil)
                self.written[tag] += gap

    def feed(self, tag, pcm):
        if tag not in self.buffers or not pcm:
            return
        self.pad_to(time.time())
        self.audio_files[tag].write(pcm)
        self.buffers[tag].extend(pcm)
        self.written[tag] += len(pcm)
        self.got.set()


async def pump_framed_stdout(stream_reader, pump):
    """Demux recorder.py's frame protocol from an async stream (the
    floatpanel's /ws/native-capture relay, adapted to this same
    readexactly-based interface) straight into a WallClockPump. Wire track
    ids (recorder.TRACK_SYSTEM/TRACK_MIC) double as the pump's own tag keys,
    so no remapping is needed — a native session's `tracks` dict is keyed by
    those same constants. Returns when the stream ends (floatpanel stopped
    capturing or the socket closed) so the caller can end the session."""
    async for track, payload in recorder.aiter_frames(stream_reader):
        pump.feed(track, payload)


def make_speech_fn(vad_mode=None):
    """Silero VAD per track (own RNN state); None -> energy fallback. silero
    is the default — it endpoints far better than energy RMS (which reads
    soft trailing syllables as silence and cuts words). ?vad=energy opts out;
    a missing/broken model also falls back to energy."""
    if vad_mode == "energy":
        return None
    try:
        from live import SileroVad  # noqa: PLC0415
        return SileroVad("models/silero_vad_v4.onnx")
    except Exception as e:  # noqa: BLE001
        print(f"silero vad unavailable, using energy: {e}", file=sys.stderr)
        return None


def build_track_sessions(tracks, *, live_manager, live_interim_backend, silence_ms,
                          min_speech_ms, interim_s, max_utt_s, rms_threshold,
                          interim_duty, vad_mode=None):
    """One TwoPassSession per track. ANE live: run the interim preview on the
    ANE helper too (not the MLX whisper-small interim, which would spin the
    GPU) — interim transcribes only the recent ~8s tail with adaptive duty
    pacing, so the warm ANE (RTF ~0.15) keeps up. Non-ANE keeps the
    lightweight MLX interim."""
    import backends as _bk  # noqa: PLC0415
    from live import TwoPassSession  # noqa: PLC0415

    ane_live = _bk.route(live_manager.current) == "ane"
    interim_be = live_manager if ane_live else live_interim_backend
    return {
        tag: TwoPassSession(
            backend=live_manager, interim_backend=interim_be, sample_rate=16000,
            silence_ms=silence_ms, min_speech_ms=min_speech_ms, interim_s=interim_s,
            max_utt_s=max_utt_s, rms_threshold=rms_threshold, interim_duty=interim_duty,
            track=lbl[0], speech_fn=make_speech_fn(vad_mode))
        for tag, lbl in tracks.items()
    }


def enable_diarization(sessions, tracks, store, mid=None, on_rename=None):
    """Wire live per-utterance speaker labeling onto every non-mic track's
    session (mirrors ws_live's ?diarize=1): online voiceprint clustering on
    each finalized utterance, plus recognition of voices already named in
    past meetings. Best-effort — a missing model just logs and leaves
    sessions untouched.

    mid: when given, a session cluster's later promotion to a named voice
    (diarize.live_speaker_labeler's on_promote) retroactively renames that
    cluster's already-stored rows via store.rename_speaker — so earlier lines
    stop showing 對方/我 once the voice is recognized. on_rename(old, new): also
    called on promotion (best-effort, e.g. to push a live notification)."""
    import os  # noqa: PLC0415
    try:
        import diarize as diar  # noqa: PLC0415
        extractor = diar.embedding_extractor()
        thr = float(os.environ.get("LIVE_DIAR_THRESHOLD", "0.4"))
        cont = float(os.environ.get("LIVE_DIAR_CONTINUITY", "0.5"))  # same-speaker continuity
        try:
            gthr = float(store.get_setting("speaker_threshold", "0.70"))
        except (ValueError, TypeError):
            gthr = 0.70
        rows = store.list_speakers()  # known voiceprints, read-only for live
        # One embedding per finalized utterance (省效能): label the whole
        # utterance ONCE via speaker_fn. The within-utterance windowed split
        # (split_live_utterance) embeds every ~1.5s window — N+1 embeds/utterance
        # even for the common single-speaker turn — so it's opt-in via
        # LIVE_DIAR_SPLIT=1 for paragraph mode (>1 speaker in one VAD segment, no
        # gap). Natural VAD silence gaps already split most turns into their own
        # utterances, each getting its own single embed + independent recognition;
        # a full-utterance embed is also LESS noisy than a 1.5s window, so the
        # default path is both cheaper and more accurate.
        split = os.environ.get("LIVE_DIAR_SPLIT") == "1"

        def make_on_promote(trk):
            # Per-track: a promotion renames only THIS track's earlier rows — each
            # track diarizes independently, so mic's 說話者1 ≠ system's 說話者1.
            def _promote(old, new):
                try:
                    store.rename_speaker(mid, old, new, track=trk)
                except Exception:  # noqa: BLE001  a stuck rename must not break live ASR
                    pass
                if on_rename:
                    try:
                        on_rename(old, new, trk)
                    except Exception:  # noqa: BLE001
                        pass
            return _promote

        # Diarize EVERY track, including 我/mic: the mic may capture several people
        # in the room, and named voices should be recognized there too. Unrecognized
        # mic clusters still display as 我 (display_speaker), recognized ones show
        # the real name.
        for tag, (_trk, _spk) in tracks.items():
            labeler = diar.live_speaker_labeler(
                extractor, rows, session_threshold=thr, match_threshold=gthr,
                continuity_threshold=cont,
                on_promote=make_on_promote(_trk) if mid is not None else None)
            if split:
                sessions[tag].splitter = (
                    lambda a, t, lf=labeler: diar.split_live_utterance(a, t, lf))
            else:
                sessions[tag].speaker_fn = labeler
    except Exception as e:  # noqa: BLE001
        print(f"live diarize unavailable: {e}", file=sys.stderr)


def make_store_emit(mid, conn_offset_ms, store, push=None):
    """Build an emit(ev, label) that persists finals AND (when `push` is given)
    streams interim + final events to the floatpanel over its /ws/native-capture
    socket — so the panel shows a live, updating caption while someone speaks
    (streaming), not just the finalized line after each utterance. push is an
    async callable(payload_dict). Finals are still the source of truth via the
    store (the panel's transcript poll reads them back); the pushed final only
    tells the panel to clear its tentative interim line."""
    async def emit(ev, label):
        track, speaker = label
        if ev["kind"] != "final":
            if push:
                await push({"type": "interim", "track": track,
                            "speaker": speaker, "text": ev["text"]})
            return
        ev["ts"] = time.time()
        spk = store_speaker(ev.get("speaker"), speaker)
        store.add_transcript(mid, "live", track,
                              ev["start_ms"] + conn_offset_ms,
                              ev.get("end_ms", ev["start_ms"]) + conn_offset_ms,
                              spk, ev["text"])
        if push:
            await push({"type": "final", "track": track,
                        "speaker": spk, "text": ev["text"]})
    return emit


async def consume(pump, sessions, tracks, *, rec_on, emit, should_stop,
                   interim_lag_bytes, on_notice=None, pop_notice=None, on_stop=None,
                   should_abort=None, on_rename=None, pop_rename=None):
    """The live consumer loop: wait for pump.feed(), drain each track's
    buffer through its TwoPassSession, call emit(ev, tracks[tag]) for every
    event. Trims backlog so a slow backend can't wedge forever behind a live
    source (full audio is already on disk for re-transcribe).

    Two distinct exits: should_abort() (e.g. the websocket already
    disconnected) is checked FIRST each round and breaks immediately,
    dropping whatever's buffered; should_stop() (e.g. a remote /live/stop) is
    checked AFTER draining this round's buffers, so the last bit of audio
    still gets transcribed."""
    while True:
        # Wake on new audio OR on a 0.5s tick. The tick matters: a native
        # source can start (helper alive, no error) yet stream ZERO frames --
        # a mic engine that never fires its tap, or system audio before its
        # first buffer. Without it consume() would block on pump.got forever
        # and never see should_stop(), so /live/stop couldn't end the session
        # (only killing the process would). Polling stop/abort each tick fixes
        # that regardless of whether any audio ever arrives.
        try:
            await asyncio.wait_for(pump.got.wait(), timeout=0.5)
            pump.got.clear()
        except asyncio.TimeoutError:
            pass
        if should_abort and should_abort():
            break
        for tag, buf in pump.buffers.items():
            if not buf:
                continue
            chunk = bytes(buf)
            buf.clear()
            if rec_on():
                continue  # 純錄音: PCM already saved; skip ASR (no inference)
            if len(chunk) > TRACK_BACKLOG_MAXB:
                print(f"live ASR backlog {len(chunk)//32000}s -> trim (audio saved)",
                      file=sys.stderr)
                chunk = chunk[-TRACK_BACKLOG_MAXB:]
            want_interim = len(chunk) <= interim_lag_bytes
            try:
                events = await run_in_threadpool(sessions[tag].feed, chunk, want_interim)
                for ev in events:
                    await emit(ev, tracks[tag])
            except WebSocketDisconnect:
                raise
            except Exception as e:  # transient ASR error -> keep going
                print(f"live consumer error (continuing): {e}", file=sys.stderr)
        if pop_notice and (msg := pop_notice()):
            if on_notice:
                await on_notice(msg)
        # Speaker promotions (diarize.live_speaker_labeler's on_promote, wired via
        # enable_diarization) fire from the threadpool feed() call above, not this
        # loop — pop_rename drains whatever piled up onto the (old, new) queue so
        # the caller can push a live rename notification, same pattern as notices.
        if pop_rename and on_rename:
            while (renamed := pop_rename()) is not None:
                await on_rename(*renamed)
        if should_stop():
            if on_stop:
                await on_stop()
            break


async def flush_sessions(sessions, tracks, mid, conn_offset_ms, store):
    """Final flush on stop: drain each TwoPassSession's tail (a partial
    utterance) and persist any 'final' it yields. Timed out per-track so a
    wedged backend can't hang stop forever — the backend's own watchdog
    reaps the thread."""
    for tag, s in sessions.items():
        try:
            evs = await asyncio.wait_for(run_in_threadpool(s.flush), timeout=15)
        except Exception as e:  # noqa: BLE001  (TimeoutError or backend error)
            print(f"live flush skipped ({tag}): {e}", file=sys.stderr)
            evs = []
        for ev in evs:
            if ev["kind"] == "final":  # audio-position offset, not wall-clock
                spk = store_speaker(ev.get("speaker"), tracks[tag][1])
                store.add_transcript(mid, "live", tracks[tag][0],
                                      ev["start_ms"] + conn_offset_ms,
                                      ev.get("end_ms", ev["start_ms"]) + conn_offset_ms,
                                      spk, ev["text"])
