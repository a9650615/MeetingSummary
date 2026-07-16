# Remote-Store Plugin — Design

Date: 2026-07-16
Status: Approved (design), pending implementation plan

## Goal

Give MeetingSummary an **opt-in remote-storage capability**: finished meetings
(audio + transcript + summary) can be pushed from the local Mac to a shared
Azure VM that hosts a lightweight web viewer. Multiple devices on the vnet can
then browse, play back, read transcripts, and export — **without any speech
recognition on the VM**. ASR/recording/live-capture stay entirely on the Mac.

The capability ships as a **plugin**: the base MeetingSummary app does not
include it. Adding the plugin folder turns it on.

## Non-goals

- No ASR, diarization, live capture, or model management on the VM.
- No user accounts / login (vnet isolation is the security boundary).
- No automatic sync — upload is manual, per meeting.
- No editing of meetings on the VM (read-only viewer + ingest).

## Constraints discovered

- **MLX / sherpa / Qwen are Apple-silicon only** — they must never be imported
  on the x86 Linux VM. `import asr / backends / live_session` at the top of
  `app.py` are macOS-only.
- **`afconvert` is macOS-only** — the current WAV-assembly playback path
  (`_track_wav_file`, `recorder.m4a_to_pcm`) cannot run on the VM. The VM
  serves M4A **directly** to the browser `<audio>` element instead; no
  server-side decode or mixing.
- `store.py` is pure SQLite with no heavy deps → **reusable as-is** on Linux.
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
  - meeting **detail**: playback + transcript + summary
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
VM: unzip → insert into VM store → save .m4a → available in viewer
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
- Follow repo convention: `pytest` + `tmp_path`, assert-based, no new frameworks.

## Deployment

- New `deploy-vm.sh` under `server/` (clone of acp's): `az ssh config` → rsync
  `server/ + viewer/ + store.py` to `10.102.0.7:~/meeting_store/` → `docker
  compose up -d --build` on port 5556 → health check `GET /`.
- `.env` + data volume live only on the VM, never overwritten by redeploy
  (same pattern as acp).
- Disk note: VM has ~62G free, shared with acp. M4A-only storage keeps this
  viable for hundreds of meetings; raw PCM is never uploaded.

## Open items (defer to plan)

- Exact `meeting.json` schema (mirror `store.py` tables).
- Whether the Mac pre-mix reuses `recorder`'s mixing or a one-shot ffmpeg/afconvert call.
- Plugin folder location + how the base build excludes it (build_app.sh).
