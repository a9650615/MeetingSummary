// Native system-audio capture via ScreenCaptureKit. Captures the whole-display
// audio (what you hear — the "other side" of a call), downmixes to 16 kHz mono,
// and writes raw int16-LE PCM to stdout continuously. The MeetingSummary server
// spawns this and feeds the PCM into the live pipeline as the system/對方 track,
// so system audio needs NO per-session browser "share screen + audio" dialog.
//
// Permission: ScreenCaptureKit needs Screen-Recording permission (TCC, per-app).
// Run from a granted app (the packaged .app) or grant the launching process once.
// Prints "READY" to stderr once the stream starts; "ERR <msg>" on failure.
import AVFoundation
import Foundation
import ScreenCaptureKit

let TARGET_SR = 16000.0

@available(macOS 13.0, *)
final class Capturer: NSObject, SCStreamOutput, SCStreamDelegate {
    let out = FileHandle.standardOutput
    var stream: SCStream?

    func start() async {
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
            FileHandle.standardError.write("READY\n".data(using: .utf8)!)
        } catch {
            fail("\(error)")
        }
    }

    func fail(_ msg: String) {
        FileHandle.standardError.write("ERR \(msg)\n".data(using: .utf8)!)
        exit(1)
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        fail("stopped \(error)")
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
        out16.withUnsafeBytes { out.write(Data($0)) }
    }

    @inline(__always) func f2i(_ v: Float) -> Int16 {
        let c = max(-1, min(1, v))
        return Int16(c * 32767)
    }
}

if #available(macOS 13.0, *) {
    let cap = Capturer()
    Task { await cap.start() }
    RunLoop.main.run()
} else {
    FileHandle.standardError.write("ERR macOS 13+ required\n".data(using: .utf8)!)
    exit(1)
}
