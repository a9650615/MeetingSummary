# capture — Swift audio capture helper

System audio (ScreenCaptureKit) + mic (AVAudioEngine), both 16 kHz mono Int16
PCM, framed to **stdout** for `recorder.py`.

Frame protocol (matches `recorder.parse_frames`):

    <UInt8 track><UInt32 little-endian length><payload bytes>
    track 0 = system, 1 = mic

## Build

    cd capture
    swift build -c release       # -> .build/release/capture

## Permissions (TCC) — required, see spec G1

First run prompts for **Screen Recording** (system audio) and **Microphone**.
Grant both in System Settings → Privacy & Security. ScreenCaptureKit fails
*silently* without Screen Recording, so check stderr (`capture started` vs
`capture failed: ...`).

## Wire to the recorder

    python -m recorder data/<meeting>/<segment> --helper capture/.build/release/capture

Writes `system.pcm` + `mic.pcm` + `manifest.json` into the segment dir. Ctrl-C
to stop (recorder closes the segment).

## Not verified in CI

`// VERIFY ON DEVICE:` comments in `main.swift` mark the spots that need a real
machine: SCStream honoring 16 kHz, AVAudioConverter resampling, and the TCC
prompts. **Echo-bleed:** without headphones the mic track also records remote
audio from the speakers — see spec §2.
