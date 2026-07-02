// Native audio capture helper: system audio via ScreenCaptureKit and/or the
// microphone via AVAudioEngine, both downmixed to 16 kHz mono int16-LE.
//
// Modes (CLI flags):
//   (no flags)  system audio only, RAW unframed PCM on stdout — the ORIGINAL
//               behavior, unchanged, so the browser-driven /ws/live handler
//               (which spawns this with no args) keeps working byte-for-byte.
//   --system    system audio only, FRAMED (see below)
//   --mic       microphone only, FRAMED
//   --both      system + mic, FRAMED, one process, one stdout stream
//
// Framed protocol (any explicit flag): little-endian <B track><I length>
// <payload> per chunk. track: 0 = system, 1 = mic (recorder.py: TRACK_SYSTEM/
// TRACK_MIC — see docs spec component 2 + §5). This lets ONE process feed
// MeetingSummary's /live/start endpoint both tracks over one pipe, with no
// browser/websocket involved at all.
//
// Permissions: system audio needs Screen-Recording (TCC); mic needs
// NSMicrophoneUsageDescription (Info.plist) + user consent. In --both mode a
// denial on one source doesn't kill the other — whichever source is granted
// keeps streaming solo. Prints "READY" to stderr once at least one source is
// live; "ERR <msg>" per failure (fatal only if NOTHING ended up capturing).
import AVFoundation
import CoreGraphics
import Foundation
import ScreenCaptureKit

let TARGET_SR = 16000.0
let TRACK_SYSTEM: UInt8 = 0
let TRACK_MIC: UInt8 = 1

/// Thread-safe framed stdout writer — system audio (SCStream's own queue) and
/// mic audio (AVAudioEngine's tap queue) can both write concurrently in
/// --both mode, so writes must be serialized to keep frames intact.
final class FrameWriter {
    private let out = FileHandle.standardOutput
    private let lock = NSLock()

    func write(track: UInt8, payload: Data) {
        var header = Data([track])
        var len = UInt32(payload.count).littleEndian
        withUnsafeBytes(of: &len) { header.append(contentsOf: $0) }
        lock.lock()
        out.write(header)
        out.write(payload)
        lock.unlock()
    }
}

@available(macOS 13.0, *)
final class Capturer: NSObject, SCStreamOutput, SCStreamDelegate {
    let out = FileHandle.standardOutput
    let framed: Bool
    let writer: FrameWriter?
    var stream: SCStream?
    var isRunning = false
    private var fatalOnFail = true

    init(framed: Bool, writer: FrameWriter?) {
        self.framed = framed
        self.writer = writer
    }

    func start(fatalOnFail: Bool = true) async {
        self.fatalOnFail = fatalOnFail
        // Screen-Recording permission gate. CGRequest... shows the system prompt
        // (attributed to the responsible app — the .app or the launching terminal)
        // and persists the grant. Without it SCStream just throws -3801 forever.
        if !CGPreflightScreenCaptureAccess() {
            let granted = CGRequestScreenCaptureAccess()
            if !granted {
                fail("NOPERM 需要螢幕錄製權限：系統設定 → 隱私權與安全性 → 螢幕錄製，"
                     + "勾選啟動本程式的 App，再重新開始錄音")
                return
            }
        }
        do {
            let content = try await SCShareableContent.excludingDesktopWindows(
                false, onScreenWindowsOnly: false)
            guard let display = content.displays.first else {
                fail("no display"); return
            }
            let filter = SCContentFilter(display: display, excludingWindows: [])
            let cfg = SCStreamConfiguration()
            cfg.capturesAudio = true
            cfg.sampleRate = Int(TARGET_SR)
            cfg.channelCount = 1
            cfg.excludesCurrentProcessAudio = true   // don't capture our own output
            // SCStream still requires a video config; keep it tiny + slow.
            cfg.width = 2
            cfg.height = 2
            cfg.minimumFrameInterval = CMTime(value: 1, timescale: 1)
            let s = SCStream(filter: filter, configuration: cfg, delegate: self)
            try s.addStreamOutput(self, type: .audio,
                                  sampleHandlerQueue: DispatchQueue(label: "audiocap.audio"))
            try await s.startCapture()
            self.stream = s
            self.isRunning = true
        } catch {
            fail("\(error)")
        }
    }

    func fail(_ msg: String) {
        FileHandle.standardError.write("ERR \(msg)\n".data(using: .utf8)!)
        if fatalOnFail { exit(1) }
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        // Unlike the startup preflight (fail(), gated by fatalOnFail so --both
        // can degrade to mic-only if screen-recording was never granted), a
        // stream that WAS running and then died (e.g. SCStreamErrorDomain
        // -3805 on display sleep/lock/WindowServer hiccup) always exits the
        // whole process -- even in --both mode, even if mic keeps working.
        // That keeps server-side supervision simple: any exit = respawn the
        // one helper with the same args, rather than tracking per-source
        // liveness inside a still-running process.
        isRunning = false
        FileHandle.standardError.write("ERR stopped \(error)\n".data(using: .utf8)!)
        exit(1)
    }

    func stream(_ stream: SCStream, didOutputSampleBuffer sb: CMSampleBuffer,
                of type: SCStreamOutputType) {
        guard type == .audio, CMSampleBufferDataIsReady(sb) else { return }
        guard let fmtDesc = CMSampleBufferGetFormatDescription(sb),
              let asbd = CMAudioFormatDescriptionGetStreamBasicDescription(fmtDesc)?.pointee
        else { return }
        var blockBuffer: CMBlockBuffer?
        var abl = AudioBufferList()
        let st = CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer(
            sb, bufferListSizeNeededOut: nil, bufferListOut: &abl,
            bufferListSize: MemoryLayout<AudioBufferList>.size, blockBufferAllocator: nil,
            blockBufferMemoryAllocator: nil,
            flags: kCMSampleBufferFlag_AudioBufferList_Assure16ByteAlignment,
            blockBufferOut: &blockBuffer)
        guard st == noErr else { return }
        let buffers = UnsafeMutableAudioBufferListPointer(&abl)
        let channels = max(1, Int(asbd.mChannelsPerFrame))
        let isFloat = (asbd.mFormatFlags & kAudioFormatFlagIsFloat) != 0
        guard isFloat, let first = buffers.first, let data = first.mData else { return }

        // SCStream audio is Float32. With channelCount=1 we expect one mono buffer;
        // be defensive about interleaved stereo (downmix) just in case.
        let frameCount: Int
        var out16: [Int16] = []
        if buffers.count >= channels && channels > 1 {
            // planar: one buffer per channel
            frameCount = Int(first.mDataByteSize) / MemoryLayout<Float32>.size
            out16.reserveCapacity(frameCount)
            let planes = (0..<channels).compactMap { buffers[$0].mData?.assumingMemoryBound(to: Float32.self) }
            for i in 0..<frameCount {
                var acc: Float = 0
                for p in planes { acc += p[i] }
                out16.append(f2i(acc / Float(planes.count)))
            }
        } else {
            // single buffer; interleaved if channels>1
            let total = Int(first.mDataByteSize) / MemoryLayout<Float32>.size
            let ptr = data.assumingMemoryBound(to: Float32.self)
            if channels > 1 {
                frameCount = total / channels
                out16.reserveCapacity(frameCount)
                for i in 0..<frameCount {
                    var acc: Float = 0
                    for c in 0..<channels { acc += ptr[i * channels + c] }
                    out16.append(f2i(acc / Float(channels)))
                }
            } else {
                frameCount = total
                out16.reserveCapacity(frameCount)
                for i in 0..<frameCount { out16.append(f2i(ptr[i])) }
            }
        }
        let payload = out16.withUnsafeBytes { Data($0) }
        if framed, let w = writer {
            w.write(track: TRACK_SYSTEM, payload: payload)
        } else {
            out.write(payload)
        }
    }

    @inline(__always) func f2i(_ v: Float) -> Int16 {
        let c = max(-1, min(1, v))
        return Int16(c * 32767)
    }
}

/// Microphone capture via AVAudioEngine, downsampled to 16 kHz mono int16 and
/// written as FRAMED TRACK_MIC frames only (there's no legacy unframed mic
/// mode — mic capture is new, so it always speaks the framed protocol).
final class MicCapturer {
    let engine = AVAudioEngine()
    let writer: FrameWriter

    init(writer: FrameWriter) {
        self.writer = writer
    }

    func start() throws {
        let input = engine.inputNode
        let inFormat = input.inputFormat(forBus: 0)
        guard let outFormat = AVAudioFormat(commonFormat: .pcmFormatInt16,
                                            sampleRate: TARGET_SR, channels: 1, interleaved: true),
              let converter = AVAudioConverter(from: inFormat, to: outFormat) else {
            throw NSError(domain: "audiocap.mic", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "format setup failed"])
        }
        input.installTap(onBus: 0, bufferSize: 2048, format: inFormat) { [weak self] buffer, _ in
            guard let self = self else { return }
            let ratio = TARGET_SR / inFormat.sampleRate
            let capacity = AVAudioFrameCount(Double(buffer.frameLength) * ratio) + 32
            guard let outBuf = AVAudioPCMBuffer(pcmFormat: outFormat, frameCapacity: capacity) else { return }
            var convErr: NSError?
            let status = converter.convert(to: outBuf, error: &convErr) { _, outStatus in
                outStatus.pointee = .haveData
                return buffer
            }
            guard status != .error, convErr == nil, let ch = outBuf.int16ChannelData else { return }
            let n = Int(outBuf.frameLength)
            guard n > 0 else { return }
            let payload = Data(bytes: ch[0], count: n * MemoryLayout<Int16>.size)
            self.writer.write(track: TRACK_MIC, payload: payload)
        }
        try engine.start()
    }
}

/// Non-prompting mic-permission check (authorizationStatus only — no dialog).
func micGranted() -> Bool {
    AVCaptureDevice.authorizationStatus(for: .audio) == .authorized
}

/// Prompts for mic access if undetermined; returns the outcome. Attributed to
/// this process's Info.plist (NSMicrophoneUsageDescription), same mechanism
/// the screen-recording preflight uses for its own usage string.
func requestMicAccess() async -> Bool {
    switch AVCaptureDevice.authorizationStatus(for: .audio) {
    case .authorized:
        return true
    case .notDetermined:
        return await withCheckedContinuation { cont in
            AVCaptureDevice.requestAccess(for: .audio) { cont.resume(returning: $0) }
        }
    default:
        return false
    }
}

if #available(macOS 13.0, *) {
    let args = CommandLine.arguments
    if args.contains("--check") {
        if CGPreflightScreenCaptureAccess() { print("GRANTED"); exit(0) }
        else { print("DENIED"); exit(1) }
    }
    if args.contains("--check-mic") {
        if micGranted() { print("GRANTED"); exit(0) }
        else { print("DENIED"); exit(1) }
    }

    let hasMicFlag = args.contains("--mic")
    let hasSystemFlag = args.contains("--system")
    let hasBothFlag = args.contains("--both")
    let anyModeFlag = hasMicFlag || hasSystemFlag || hasBothFlag
    let wantMic = hasMicFlag || hasBothFlag
    let wantSystem = hasSystemFlag || hasBothFlag || !anyModeFlag  // no flags = legacy default
    let framed = anyModeFlag

    Task {
        let writer = framed ? FrameWriter() : nil
        var started = false

        if wantMic {
            if await requestMicAccess() {
                let mic = MicCapturer(writer: writer!)
                do {
                    try mic.start()
                    started = true
                } catch {
                    FileHandle.standardError.write("ERR mic \(error)\n".data(using: .utf8)!)
                }
            } else {
                FileHandle.standardError.write(
                    "ERR NOPERM 需要麥克風權限：系統設定 → 隱私權與安全性 → 麥克風\n".data(using: .utf8)!)
            }
        }
        if wantSystem {
            // In --both mode, degrade instead of exiting if mic is already streaming.
            let cap = Capturer(framed: framed, writer: writer)
            await cap.start(fatalOnFail: !started)
            if cap.isRunning { started = true }
        }
        guard started else {
            FileHandle.standardError.write("ERR 沒有可用的音訊來源\n".data(using: .utf8)!)
            exit(1)
        }
        FileHandle.standardError.write("READY\n".data(using: .utf8)!)
    }
    RunLoop.main.run()
} else {
    FileHandle.standardError.write("ERR macOS 13+ required\n".data(using: .utf8)!)
    exit(1)
}
