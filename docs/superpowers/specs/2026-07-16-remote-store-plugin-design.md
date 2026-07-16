# Remote-Store Plugin — Design

Date: 2026-07-16
Status: Approved (design), pending implementation plan

## Goal

Give MeetingSummary an **opt-in remote-storage capability**: finished meetings
(audio + transcript + summary) can be pushed from the local Mac to a shared
Azure VM that hosts a lightweight web viewer. Multiple devices on the vnet can
then browse, play back, read transcripts, and export.

Primary motivation: **offload the local storage burden** — large M4A meeting
archives live on the VM instead of the Mac. As a value-add backup, the VM also
runs a **slow, CPU-only FireRed re-correction pass** that produces a
higher-accuracy transcript, stored non-destructively alongside the Mac's
pushed transcript.

The **Mac app is unchanged** — it still runs live + accurate ASR locally
exactly as today. The VM's FireRed pass is a *backup / redundancy* layer, not a
replacement for local ASR.

The capability ships as a **plugin**: the base MeetingSummary app does not
include it. Adding the plugin folder turns it on.

## Non-goals

- No live capture, diarization, model management, or MLX/Qwen stack on the VM.
- **No voiceprint / speaker recognition on the VM.** Speaker labels come
  entirely from the Mac push; FireRed inherits them and never re-diarizes.
- The VM runs **FireRed batch correction only** (sherpa-onnx, pure CPU) — it
  never runs the real-time/Apple-only ASR stack.
- A meeting is pushed **only after it is fully processed locally** (ASR +
  speaker attribution + summary done). The VM corrects text on top of a
  complete, speaker-labeled transcript — it is never the first processor.
- No user accounts / login (vnet isolation is the security boundary).
- No automatic sync — upload is manual, per meeting.
- No editing of meetings on the VM (read-only viewer + ingest + auto FireRed).
- FireRed does NOT reduce Mac compute — Mac's own accurate pass still runs.

## Constraints discovered

- **MLX / sherpa / Qwen are Apple-silicon only** — they must never be imported
  on the x86 Linux VM. `import asr / backends / live_session` at the top of
  `app.py` are macOS-only.
- **`afconvert` is macOS-only** — the current WAV-assembly playback path
  (`_track_wav_file`, `recorder.m4a_to_pcm`) cannot run on the VM. The VM
  serves M4A **directly** to the browser `<audio>` element instead; no
  server-side decode or mixing.
- `store.py` is pure SQLite with no heavy deps → **reusable as-is** on Linux.
  It already has a `profile` column on `transcripts` (store.py:44, values like
  `"live"` / `"accurate"`) → a new `"firered"` profile stores the VM's
  correction **without touching** the Mac-pushed rows.
- **FireRed via sherpa-onnx is pure CPU onnxruntime, NOT Apple-only**
  (`firered_batch_backend`, backends.py:746, imports only numpy + sherpa_onnx)
  → it runs fine on the x86 Linux VM. This is the one ASR engine allowed on the
  VM. sherpa-onnx ships manylinux x86-64 wheels.
- MeetingSummary has **no plugin system today** — routes are hardcoded inside
  `create_app` (app.py:2252), UI is inline HTML string builders, packaged as a
  macOS `.app` launcher. The plugin seam must be built minimally (a try-import
  toggle), not as a general plugin framework.

## Architecture — three units

### ① `viewer/` — shared viewing module (Linux-safe, zero ASR deps)

The read-only rendering + store-access core, imported by **both** the base Mac
app and the VM server.

- Reuses `store.py` unchanged for all DB reads.
- Provides render/handlers for:
  - meeting **list + search**
  - meeting **detail**: playback + transcript + summary. Transcript picks the
    **best available profile** per meeting: `firered` if the VM correction is
    done, else the Mac-pushed profile (`accurate`/`live`). A small badge marks
    which is shown.
  - **export** `.md` (lift the logic behind existing `GET /meetings/{mid}/export`, app.py:3171)
- **Audio**: serves the stored `.m4a` files directly (byte range / seek), one
  route per track. No WAV assembly, no `afconvert`, no MLX.
- Hard rule: this module imports **none** of `asr`, `backends`,
  `live_session`, `live`, `diarize`, or `recorder`'s decode path.

Boundary: given a `Store` + a data dir, `viewer` renders everything the UI
needs. It knows nothing about where meetings came from.

### ② `server/` — standalone VM host (own folder, own deploy)

A thin FastAPI app that runs on the VM.

- Depends only on `fastapi + uvicorn` (+ `viewer/` + `store.py`, copied at deploy).
- Mounts the `viewer` routes (list/search, detail, playback, transcript,
  summary, export).
- Adds `POST /ingest-bundle` — accepts a finished-meeting bundle, writes it
  into the VM's own `store` + data dir. This is the **only** write path.
- Runs in its **own Docker container** on VM `10.102.0.7`, port **:5556**,
  fully isolated from acp's `docker-app-1` on :5555. Own
  `docker-compose.yml` + `deploy-vm.sh` (cloned from
  `acp_playground/scripts/deploy-vm.sh`: `az ssh` + rsync + compose up).
- No auth (vnet boundary). Reachable at `http://10.102.0.7:5556`.

### ④ FireRed correction worker (VM, background, backup)

A background worker on the VM that re-transcribes ingested meetings with
FireRedASR for higher accuracy, non-destructively.

- Engine: `sherpa-onnx` `OfflineRecognizer.from_fire_red_asr`, reusing
  `firered_batch_backend` (backends.py:746). Model: **v2 CTC int8, 740MB**
  (fastest, ~1–1.5GB loaded RAM, RTF ~0.3–0.5 on 1 core → ~20–30 min per hour
  of audio). Auto-downloaded from the k2-fsa release on first run
  (`_ensure_firered`, backends.py:699).
- **CPU isolation**: pinned to 1 core via `taskset` + `OMP_NUM_THREADS=1` so it
  never starves acp or the viewer. "Slow on purpose" is acceptable.
- **Per-segment re-transcription, speakers inherited (no diarization).** The
  worker iterates the **existing Mac-pushed segments** and, for each, feeds only
  that segment's audio span to FireRed. The output text replaces that segment's
  text while **keeping the segment's original speaker label + start/end**. This
  preserves 斷句 and speaker attribution exactly — the VM never runs voiceprint
  or diarization; it only sharpens the words inside each pre-labeled turn.
- **Long-segment chunking**: a long turn is VAD-segmented (Silero VAD) before
  FireRed so the model does not OOM / mis-decode, then the chunk texts are
  stitched back into the one segment. Decode M4A → 16kHz mono PCM with
  **ffmpeg** (not afconvert).
- **Non-destructive storage**: results written under `profile="firered"`. The
  Mac-pushed transcript set is **never cleared** — the VM must NOT reuse the
  base app's `clear_transcripts(mid)` full-replace behavior (app.py:3262/3295);
  it clears only the `firered` profile before re-writing.
- Queue: newly ingested meetings are enqueued; the worker processes one meeting
  at a time. Restart-safe (re-enqueue anything without a complete `firered`
  set).

### ③ Mac-side push plugin (opt-in)

The piece that makes the base app "remote-aware", shipped only with the plugin.

- Lives in its own folder; **absent in the base build**.
- When present: meeting **detail page gains an "上傳到 server" button**, and a
  push handler bundles the meeting and POSTs it to the VM.
- **Minimal seam in `app.py`**: at `create_app`, `try: import <plugin>` — if it
  imports, register its route + inject the button into the detail page; if not,
  the base app is unchanged and shows no button. All bundling/upload logic
  lives in the plugin folder, not in `app.py`.

## Data flow

```
Mac: record → ASR → summary → finalize → M4A compress   (unchanged base flow)
                                    │
                        [plugin] user clicks "上傳到 server"
                                    │  bundle = zip(meeting.json + tracks/*.m4a)
                                    ▼
              POST http://10.102.0.7:5556/ingest-bundle
                                    │
VM: unzip → insert into VM store (Mac profile) → save .m4a
                                    │            → viewer immediately shows pushed text
                                    ▼
   FireRed worker (1 core, VAD-chunked, slow) → store profile="firered"
                                    │
   viewer auto-upgrades to FireRed text once complete (pushed text untouched)
                                    │
   any vnet device → http://10.102.0.7:5556 → list/search/play/read/export
```

## Bundle format

A single `.zip`:

- `meeting.json` — meeting metadata + segments + transcripts + summaries
  (serialized from the Mac's SQLite rows; the VM re-inserts them into its store).
- `tracks/*.m4a` — per-track compressed audio. The **mixed** track is
  **pre-mixed on the Mac** before upload (the VM never mixes). `system` /
  `mic` tracks are uploaded as-is when present.

Ingest is idempotent by meeting id: re-uploading the same meeting replaces its
record + tracks (supports the manual "re-push" case).

## Error handling

- **Push (Mac)**: network/VM failures surface as a clear error on the detail
  page; the meeting stays local and can be re-pushed. No silent failure.
- **Ingest (VM)**: reject malformed bundles (missing `meeting.json`, bad zip)
  with a 4xx and a reason; partial writes are rolled back (write to temp dir,
  then atomic move) so a failed ingest never leaves a half-meeting.
- **Playback**: a missing/legacy track (e.g. only `.pcm` existed, no `.m4a`)
  shows as "track unavailable" rather than a broken player.

## Testing

- `viewer/`: unit tests over a temp `Store` — list, search, detail render,
  export md, M4A range serving. No network, no ASR.
- `server/`: ingest a sample bundle → assert meeting appears in list, tracks
  playable, transcript/summary present; malformed-bundle rejection; idempotent
  re-ingest.
- Plugin seam: base app with plugin absent renders no button; with plugin
  present, button appears and push posts a well-formed bundle (mock VM).
- FireRed worker: writes only `profile="firered"`, leaves pushed rows intact;
  viewer prefers `firered` when present; VAD chunking splits a long clip; a
  short real clip transcribes end-to-end (may be an opt-in/slow test).
- Follow repo convention: `pytest` + `tmp_path`, assert-based, no new frameworks.

## Deployment

- New `deploy-vm.sh` under `server/` (clone of acp's): `az ssh config` → rsync
  `server/ + viewer/ + store.py` to `10.102.0.7:~/meeting_store/` → `docker
  compose up -d --build` on port 5556 → health check `GET /`.
- `.env` + data volume live only on the VM, never overwritten by redeploy
  (same pattern as acp).
- Container deps: `fastapi + uvicorn + sherpa-onnx + onnxruntime + ffmpeg`
  (ffmpeg for M4A→PCM decode; sherpa for FireRed). FireRed model (740MB)
  auto-downloaded on first worker run into a persisted volume (not re-pulled
  each deploy).
- FireRed worker pinned to 1 core (`taskset -c 1` + `OMP_NUM_THREADS=1`) so the
  viewer + acp keep the other core responsive.
- Resource budget: acp ~2.8G + FireRed ~1.5G + viewer ≈ 4.3G / 7.7G RAM (OK,
  4G swap headroom); disk ~62G free shared with acp — M4A-only, hundreds of
  meetings viable; raw PCM never uploaded.

## Delivery order

**地端先做完** — build and verify the Mac side before the VM side:

1. `viewer/` shared module + `meeting.json` serialization (Mac can export a bundle).
2. Mac push plugin (opt-in seam + "上傳到 server" button + bundle/upload),
   verified end-to-end against a local stub server.
3. `server/` VM host: ingest + viewer routes + deploy to :5556.
4. FireRed correction worker (last — it is the value-add backup, and depends on
   ingested, speaker-labeled meetings already existing on the VM).

## Open items (defer to plan)

- Exact `meeting.json` schema (mirror `store.py` tables).
- Whether the Mac pre-mix reuses `recorder`'s mixing or a one-shot ffmpeg/afconvert call.
- Plugin folder location + how the base build excludes it (build_app.sh).
- FireRed per-segment audio slicing: cut each segment's span from the track M4A
  (ffmpeg `-ss/-to` by segment start/end) vs decode-once + slice PCM in memory.
- Silero VAD model provisioning on the VM (sherpa-onnx bundle) + threshold.
- CTC vs AED FireRed model final pick (spec defaults to v2 CTC int8; benchmark
  on the actual VM before locking, per research caveat).
