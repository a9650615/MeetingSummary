"""Shared plumbing for a live-transcription session, used by BOTH audio
sources MeetingSummary supports today:

  * the browser /ws/live handler — PCM arrives as websocket binary frames
    (mic via getUserMedia, optionally native system audio muxed in)
  * native /live/start sessions — PCM arrives framed (recorder.py's
    <track><len><payload> protocol) from an audiocap subprocess's stdout,
    with no browser/websocket involved at all

Everything after "bytes for track X arrived" — wall-clock padding so all
tracks share one clock, feeding TwoPassSession, persisting finals, the
backlog trim, the flush-on-stop — is one code path so the two entry points
in app.py don't duplicate it.
"""
import asyncio
import sys
import time

from starlette.concurrency import run_in_threadpool
from starlette.websockets import WebSocketDisconnect

import recorder

TRACK_BACKLOG_MAXB = 45 * 16000 * 2  # ~45s cap; full audio is on disk for re-transcribe


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
    """Demux recorder.py's frame protocol from an async stream (an audiocap
    subprocess's stdout) straight into a WallClockPump. Wire track ids
    (recorder.TRACK_SYSTEM/TRACK_MIC) double as the pump's own tag keys, so
    no remapping is needed — a native session's `tracks` dict is keyed by
    those same constants. Returns when the stream ends (helper exited or
    crashed) so the caller can end the session."""
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


def enable_diarization(sessions, tracks, store):
    """Wire live per-utterance speaker labeling onto every non-mic track's
    session (mirrors ws_live's ?diarize=1): online voiceprint clustering on
    each finalized utterance, plus recognition of voices already named in
    past meetings. Best-effort — a missing model just logs and leaves
    sessions untouched."""
    import os  # noqa: PLC0415
    try:
        import diarize as diar  # noqa: PLC0415
        extractor = diar.embedding_extractor()
        thr = float(os.environ.get("LIVE_DIAR_THRESHOLD", "0.4"))
        try:
            gthr = float(store.get_setting("speaker_threshold", "0.62"))
        except (ValueError, TypeError):
            gthr = 0.62
        rows = store.list_speakers()  # known voiceprints, read-only for live
        for tag, (_trk, spk) in tracks.items():
            if spk != "我":
                labeler = diar.live_speaker_labeler(
                    extractor, rows, session_threshold=thr, match_threshold=gthr)
                sessions[tag].splitter = (
                    lambda a, t, lf=labeler: diar.split_live_utterance(a, t, lf))
    except Exception as e:  # noqa: BLE001
        print(f"live diarize unavailable: {e}", file=sys.stderr)


def make_store_emit(mid, conn_offset_ms, store):
    """Build an emit(ev, label) that only persists finals (no push channel) —
    for the native /live/start pipeline, which has no websocket to stream
    interim/final events to. /live/state reads finals back out of the store,
    so captions still show up without a live push."""
    async def emit(ev, label):
        if ev["kind"] != "final":
            return
        track, speaker = label
        ev["ts"] = time.time()
        spk = ev.get("speaker") or speaker
        store.add_transcript(mid, "live", track,
                              ev["start_ms"] + conn_offset_ms,
                              ev.get("end_ms", ev["start_ms"]) + conn_offset_ms,
                              spk, ev["text"])
    return emit


async def consume(pump, sessions, tracks, *, rec_on, emit, should_stop,
                   interim_lag_bytes, on_notice=None, pop_notice=None, on_stop=None,
                   should_abort=None):
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
        await pump.got.wait()
        pump.got.clear()
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
                spk = ev.get("speaker") or tracks[tag][1]
                store.add_transcript(mid, "live", tracks[tag][0],
                                      ev["start_ms"] + conn_offset_ms,
                                      ev.get("end_ms", ev["start_ms"]) + conn_offset_ms,
                                      spk, ev["text"])
