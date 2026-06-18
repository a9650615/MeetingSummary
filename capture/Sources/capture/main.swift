// Capture helper: system audio (ScreenCaptureKit) + mic (AVAudioEngine),
// both as 16 kHz mono Int16 PCM, framed to stdout for recorder.py.
//
// Frame protocol (must match recorder.py parse_frames):
//   <UInt8 track><UInt32 little-endian length><length bytes payload>
//   track 0 = system, 1 = mic.
//
// VERIFY ON DEVICE: SCStream audio config, AVAudioConverter resampling, and TCC
// permission prompts (Screen Recording + Microphone) cannot be tested in CI.
// Build: `swift build -c release` in capture/. Run: `.build/release/capture`.
// The app/recorder reads this process's stdout.

import AVFoundation
import ScreenCaptureKit

let TRACK_SYSTEM: UInt8 = 0
let TRACK_MIC: UInt8 = 1
let TARGET_RATE = 16000.0

// MARK: - Framed stdout writer (thread-safe: SCK + mic run on separate queues)

final class FrameWriter {
    private let handle = FileHandle.standardOutput
    private let lock = NSLock()

    func write(track: UInt8, pcm: Data) {
        lock.lock()
        defer { lock.unlock() }
        var header = Data()
        header.append(track)
        var len = UInt32(pcm.count).littleEndian
        withUnsafeBytes(of: &len) { header.append(contentsOf: $0) }
        handle.write(header)
        handle.write(pcm)
    }
}

// Float32 [-1, 1] samples -> little-endian Int16 PCM bytes.
func floatToInt16(_ samples: UnsafeBufferPointer<Float>) -> Data {
    var out = Data(capacity: samples.count * 2)
    for s in samples {
        let clamped = max(-1.0, min(1.0, s))
        var v = Int16(clamped * 32767).littleEndian
        withUnsafeBytes(of: &v) { out.append(contentsOf: $0) }
    }
    return out
}

let writer = FrameWriter()

// MARK: - System audio via ScreenCaptureKit

final class SystemAudioCapture: NSObject, SCStreamOutput {
    private var stream: SCStream?

    func start() async throws {
        let content = try await SCShareableContent.current
        guard let display = content.displays.first else {
            FileHandle.standardError.write(Data("no display to capture\n".utf8))
            exit(1)
        }
        // Capture audio only; exclude no windows. A display filter is required
        // even when we only want system audio.
        let filter = SCContentFilter(display: display, excludingWindows: [])
        let config = SCStreamConfiguration()
        config.capturesAudio = true
        config.sampleRate = Int(TARGET_RATE)   // VERIFY ON DEVICE: SCK honoring 16 kHz
        config.channelCount = 1
        let stream = SCStream(filter: filter, configuration: config, delegate: nil)
        try stream.addStreamOutput(self, type: .audio,
                                   sampleHandlerQueue: .global(qos: .userInitiated))
        try await stream.startCapture()
        self.stream = stream
    }

    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer,
                of type: SCStreamOutputType) {
        guard type == .audio, sampleBuffer.isValid else { return }
        try? sampleBuffer.withAudioBufferList { abl, _ in
            for buffer in abl {
                guard let data = buffer.mData else { continue }
                let count = Int(buffer.mDataByteSize) / MemoryLayout<Float>.size
                let floats = UnsafeBufferPointer(
                    start: data.assumingMemoryBound(to: Float.self), count: count)
                writer.write(track: TRACK_SYSTEM, pcm: floatToInt16(floats))
            }
        }
    }
}

// MARK: - Mic via AVAudioEngine (+ resample to 16 kHz mono)

final class MicCapture {
    private let engine = AVAudioEngine()

    func start() throws {
        let input = engine.inputNode
        let inFormat = input.outputFormat(forBus: 0)
        guard let outFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32, sampleRate: TARGET_RATE,
            channels: 1, interleaved: false),
              let converter = AVAudioConverter(from: inFormat, to: outFormat) else {
            throw NSError(domain: "capture", code: 2)
        }
        // VERIFY ON DEVICE: converter ratio / buffer sizing on real hardware rate.
        input.installTap(onBus: 0, bufferSize: 4096, format: inFormat) { buf, _ in
            let ratio = TARGET_RATE / inFormat.sampleRate
            let cap = AVAudioFrameCount(Double(buf.frameLength) * ratio) + 1
            guard let out = AVAudioPCMBuffer(pcmFormat: outFormat,
                                             frameCapacity: cap) else { return }
            var err: NSError?
            converter.convert(to: out, error: &err) { _, status in
                status.pointee = .haveData
                return buf
            }
            if err != nil { return }
            guard let ch = out.floatChannelData else { return }
            let floats = UnsafeBufferPointer(start: ch[0], count: Int(out.frameLength))
            writer.write(track: TRACK_MIC, pcm: floatToInt16(floats))
        }
        try engine.start()
    }
}

// MARK: - main

let system = SystemAudioCapture()
let mic = MicCapture()

Task {
    do {
        try await system.start()   // triggers Screen Recording TCC prompt
        try mic.start()            // triggers Microphone TCC prompt
        FileHandle.standardError.write(Data("capture started\n".utf8))
    } catch {
        FileHandle.standardError.write(Data("capture failed: \(error)\n".utf8))
        exit(1)
    }
}

RunLoop.main.run()
