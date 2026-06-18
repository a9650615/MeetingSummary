# M3 Meeting Summary — Design

**Date:** 2026-06-18
**Status:** Draft for review
**Target hardware:** Apple M3, 16 GB RAM, macOS 13+ (ScreenCaptureKit era)

> **Capture-API caveat (detail in §3, component 1):** ScreenCaptureKit captures
> *system/app audio* only. Native microphone capture inside `SCStream` exists only
> on **macOS 15+**. On macOS 13/14 the mic track comes from a **separate
> `AVAudioEngine` tap**, clock-synced to the SCK stream. The dual-track design (R8)
> therefore assumes **headphones or acoustic echo cancellation** — without them the
> mic track also records remote audio bleeding from the speakers (echo-bleed
> caveat in §2).

## 1. Goal

A local, privacy-first meeting recorder + transcriber + summarizer, similar in
spirit to [meetily](https://github.com/Zackriya-Solutions/meetily), but:

- **Optimized for Apple M-series (M3)** via the MLX runtime (Metal-native, fastest
  on Apple Silicon).
- **Highest-accuracy models** chosen by *measured* benchmark on the user's own
  audio, not assumed.
- **100% local** — no meeting content leaves the machine.

Primary language: **Mandarin (zh)** with **zh/en code-switching**. Summary
output language: **zh-TW (繁體中文)**, configurable.

## 2. Key Requirements (from brainstorm)

| # | Requirement | Source |
|---|---|---|
| R1 | Live transcript during meeting | Q1: A |
| R2 | Keep raw audio, re-run high-accuracy model after meeting | Q1 |
| R3 | Handle Mandarin + zh/en code-switching | Q2: A+C |
| R4 | Summary LLM runs **fully local** (privacy) | Q3: A |
| R5 | Speaker attribution — wanted, may phase in | Q4: A+C |
| R6 | Simple UI — local web (FastAPI + websocket) | Q5: B |
| R7 | Capture **system audio (primary)** + **mic** | Q6: B, mainly C |
| R8 | **mic and system audio kept as SEPARATE tracks** | Q7 follow-up |
| R9 | Summary = meeting minutes (overview/decisions/action items) or bullet | Q7: A/B |
| R10 | Records **mergeable**; recording may be **interrupted** — must be crash-safe | latest |

### Design consequence of R8 (separate tracks)
Separate mic + system tracks give **near-free 2-way diarization**: mic = "我",
system = "對方/others". Transcribe each track independently, interleave by
timestamp. Phase-5 pyannote sub-splits the *system* track into multiple named
speakers.

> **Echo-bleed caveat:** the clean split holds only with **headphones**. On
> open speakers the mic also picks up remote audio → double-transcription + wrong
> attribution. Mitigation, in order of preference: (1) require/assume headphones
> (documented default), (2) gate mic segments on system-track energy (suppress mic
> when system is loud), (3) full AEC. Phase 1 ships (1); (2) is a cheap Phase-3
> add. Echo-bleed is a first-class test case (§9).

## 3. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  macOS                                                        │
│                                                              │
│  ┌──────────────────────┐   PCM frames (system, mic)        │
│  │ Swift capture helper │ ──────────────┐                   │
│  │ (ScreenCaptureKit)   │               │                   │
│  │  - system audio      │               ▼                   │
│  │  - mic audio         │       ┌──────────────────┐        │
│  └──────────────────────┘       │ Recorder (Python)│        │
│                                  │  append raw PCM  │        │
│                                  │  flush ~5s       │        │
│                                  └────────┬─────────┘        │
│                                           │ segment .pcm     │
│                                           ▼                  │
│  ┌──────────────┐  live   ┌───────────────────────────┐     │
│  │  Web UI      │◄────────│ ASR engine (mlx-whisper)   │     │
│  │ FastAPI + WS │  text   │  live: large-v3-turbo      │     │
│  │  - transcript│         │  batch: large-v3 (accurate)│     │
│  │  - controls  │────────►│  + benchmark harness       │     │
│  │  - summary   │ command └───────────────────────────┘     │
│  └──────────────┘                       │ transcript        │
│         ▲                               ▼                    │
│         │ summary        ┌───────────────────────────┐      │
│         └────────────────│ Summarizer (mlx-lm)        │      │
│                          │  Qwen2.5-14B / 7B, zh-TW   │      │
│                          └───────────────────────────┘      │
│                                                              │
│  Storage: SQLite (meta) + ./data/<meeting>/<segment>/*.pcm   │
└─────────────────────────────────────────────────────────────┘
```

### Components (each independently testable)

1. **Capture helper (Swift)** — `capture/` — emits interleaved-tagged PCM
   (16 kHz mono per track, resampled from SCK's 48 kHz float) to stdout. One job:
   audio in → tagged PCM out. Replaceable by a BlackHole-based fallback without
   touching the rest.
   - **System audio:** `ScreenCaptureKit` (`SCStream` audio, macOS 13+).
   - **Mic audio:** on **macOS 15+**, the `SCStream` microphone source; on
     **macOS 13/14**, a separate `AVAudioEngine` input tap. Both paths timestamp
     frames against a shared monotonic clock so the recorder can align tracks.
   - **Permissions (TCC):** needs **Screen Recording** (system audio) +
     **Microphone** grants. Helper checks/requests both on launch and exits with a
     clear error the web UI surfaces — SCK fails silently if Screen Recording is
     not granted.

2. **Recorder (Python)** — `recorder.py` — reads helper stdout, appends each
   track's PCM to `data/<meeting_id>/<segment_id>/{system,mic}.pcm`, flushes
   every ~5 s. Writes a `manifest.json` (sample rate, channels, start ts).
   Crash-safe: append-only raw PCM, no header to corrupt.

3. **ASR engine** — `asr.py` — wraps `mlx-whisper`. Two profiles:
   `live` (large-v3-turbo, low latency, chunked) and `accurate`
   (large-v3, full re-pass on saved PCM). Returns segments with timestamps +
   track label. Pluggable backend interface so SenseVoice can slot in.

4. **Benchmark harness** — `benchmark.py` — runs candidate ASR models
   (whisper-large-v3, whisper-large-v3-turbo, FunASR SenseVoice-Small) over a
   labeled sample clip, reports CER/latency/RTF, picks the accurate-profile
   winner. This is how "highest accuracy" is *proven* for zh+code-switch.

5. **Summarizer** — `summarize.py` — wraps `mlx-lm` (Qwen2.5-14B-Instruct 4-bit,
   fallback 7B). zh-TW prompt templates: `minutes` (overview / key points /
   decisions / action items w/ owner+due) and `bullets`.
   - **Long meetings (map-reduce):** a 1–2 hr transcript is 30k–60k+ tokens and
     overflows the 32k context. When the transcript exceeds a token budget, split
     into windowed chunks → summarize each (map) → summarize the chunk-summaries
     (reduce). Short meetings take the single-pass path. This is required, not
     optional — the common case is long meetings.
   - **7B fallback trigger:** chosen by **input length / projected KV-cache size**,
     not a vague "RAM tight" — see §6 RAM note.

6. **Store** — `store.py` — SQLite: meetings, segments, transcripts, summaries.
   Files on disk under `data/`.

7. **Web app** — `app.py` — FastAPI. Routes: start/stop recording, websocket for
   live transcript, list/view meetings, trigger accurate re-transcribe,
   trigger summary, **merge meetings/segments**. One static HTML/JS page.
   - **Bind `127.0.0.1` only** (never `0.0.0.0`). Privacy-first: meeting
     transcripts must not be reachable on the LAN. No auth by design — relies on
     loopback-only binding. Surfaces capture-permission errors from the helper.

## 4. Data Model

```
meetings(id, title, created_at, lang, status)
segments(id, meeting_id, idx, dir_path, started_at, duration_s, origin)
transcripts(id, meeting_id, profile, track, start_ms, end_ms, speaker, text)
summaries(id, meeting_id, kind, lang, text, model, created_at)
```

- A meeting owns ≥1 ordered segment (`idx`). Stop/restart or crash → new segment.
  Each segment dir holds **both** tracks (`system.pcm`, `mic.pcm`).
- `segments.origin` ∈ {recorded, recovered, merged} — provenance of the segment,
  *not* an audio track (tracks live in `transcripts.track`). Renamed from `source`
  to remove the system/mic ambiguity.
- `transcripts.profile` ∈ {live, accurate}; `track` ∈ {system, mic};
  `speaker` filled by track first, refined by pyannote later.

## 5. Crash-safety & Merge (R10)

**Crash-safety**
- Capture writes append-only raw PCM, flushed every ~5 s. Max loss on hard crash
  = one flush window (~5 s).
- On launch, scan `data/` for segments whose meeting `status != finalized`;
  offer **recover** → wrap loose PCM into the meeting as a `recovered` segment.
- `manifest.json` is written at **segment start** (sample rate, channels, start
  ts) so it survives a crash. `duration_s` is **recomputed from PCM byte count**
  (`bytes / (sample_rate * 2)` for 16-bit mono), never trusted from a possibly
  unwritten end-of-segment field.

**Merge**
- *Segment merge* (same meeting): concat segments by `idx`, per track, in order.
- *Meeting merge*: user selects N meetings → new meeting with their segments
  appended in chosen order.
- After any merge: re-run accurate transcription on the combined audio (or just
  concatenate existing accurate transcripts if already done + timestamps
  re-based), then re-summarize the combined transcript.

## 6. Model Choices (initial candidates — pending benchmark)

> These are **provisional defaults**, not yet measured. Goal §1 requires accuracy
> proven by benchmark (Phase 4), so Phase 1 builds on `large-v3` as a working
> placeholder and the harness may replace it.

| Role | Model | Runtime | RAM (4-bit) | Note |
|---|---|---|---|---|
| ASR live | whisper-large-v3-turbo | mlx-whisper | ~1.5 GB | low latency |
| ASR accurate | whisper-large-v3 | mlx-whisper | ~3 GB | top multilingual, code-switch safe |
| ASR challenger | FunASR SenseVoice-Small | sherpa-onnx/funasr | ~1 GB | strong zh, fast; benchmark vs large-v3 |
| Summary | Qwen2.5-14B-Instruct (4-bit) | mlx-lm | ~8–9 GB | best zh that fits 16 GB; 7B fallback |

> Models load **sequentially** (transcribe → then summarize), so peak RAM is one
> model at a time. Newer Qwen swappable if available.

**RAM budget (16 GB, the real constraint):** 14B 4-bit weights ≈ 8–9 GB is only
part of it. Add macOS (~4–5 GB) + MLX/Python overhead + **KV cache**, which grows
with transcript length and is large at 32k context. On long meetings the 14B may
OOM/swap. Therefore: pick **14B vs 7B by projected input tokens + KV-cache size**
(map-reduce per §3 keeps each pass's context bounded), and treat 7B as the default
for very long single-pass inputs — not a rare fallback.

**Why MLX (default path):** Metal-native, fastest tokens/s and lowest ASR RTF on
M3 vs Ollama/whisper.cpp in most 2025 benchmarks. MLX runs the **default** ASR +
LLM path; the SenseVoice challenger (sherpa-onnx/FunASR) and Phase-5 pyannote
(torch/MPS) are **not** MLX. (whisper.cpp+Ollama kept as documented fallback.)

## 7. Build Phases

- **Phase 1 — Core batch loop**: capture helper (dual-track) → recorder
  (crash-safe segments) → accurate ASR (large-v3) → summary (Qwen) → web view.
  Proves end-to-end on a saved recording.
- **Phase 2 — Live**: live profile (turbo) + websocket streaming transcript.
  Note: Whisper is **not** a streaming model (30 s windows). Live = sliding-window
  chunks → latency + boundary errors, esp. on zh/en code-switch across a cut. The
  Phase-4 accurate re-pass (full-context large-v3) is what users trust; live is a
  preview. Tune window/overlap empirically.
- **Phase 3 — Merge + recovery**: segment/meeting merge, crash recovery on launch.
- **Phase 4 — Accuracy**: benchmark harness, lock accurate model for zh+code-switch.
- **Phase 5 — Diarization**: pyannote multi-speaker on system track (R5 nice-to-have).

## 8. Out of Scope (YAGNI)

- Cloud LLM / cloud ASR (R4 = local only).
- Mobile / non-macOS.
- Calendar / Zoom API integration.
- Multi-user / accounts.
- Real-time speaker labels in phase 1–2 (2-way track split only).

## 9. Testing Strategy

- Recorder: feed synthetic PCM, assert segment files + manifest, kill mid-write,
  assert recovery wraps intact PCM.
- ASR: fixed short clip → assert non-empty timestamped segments per track.
- Merge: two segments → assert concatenated transcript order + duration.
- Summarizer: stub transcript → assert sections present in output.
- Summarizer (long): synth transcript > context budget → assert map-reduce path
  taken + non-empty merged summary (B3).
- Echo-bleed: mic clip containing leaked system audio → assert mic-gating /
  dedup suppresses the duplicate, no double-attribution (B2).
- Permissions: simulate Screen Recording / Mic grant denied → assert helper exits
  with a clear error the web UI surfaces, not a silent hang (G1).
- Benchmark: labeled clip → assert CER/RTF numbers emitted per model.

## 10. Open Questions

- Mic+system simultaneous capture confirmed in design (§3): SCK system audio +
  separate `AVAudioEngine` mic tap on macOS 13/14, native `SCStream` mic on 15+.
  Open: clock-sync drift between the two sources on 13/14 — measure in Phase 1
  spike (BlackHole fallback if drift is unacceptable).
- Echo-bleed mitigation level: ship headphone-assumption (Phase 1), decide in
  Phase 3 whether system-energy mic-gating is enough or full AEC is needed (§2).
- Summary `minutes` vs `bullets` default (R9 = A or B) — default `minutes`,
  toggle in UI.
