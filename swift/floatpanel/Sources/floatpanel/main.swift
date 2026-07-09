// Floating control panel for MeetingSummary — native SwiftUI in a translucent
// always-on-top NSPanel (HUD material). Shows recording state, meeting title,
// a source picker, the last few live-caption lines, elapsed time, and a
// quick-note field that appends to the meeting notes. Talks to the local
// server; honors MEETING_PORT (default 8765).
//
// Browserless start: 開始錄音 always captures in-process (AVCaptureSession for
// mic, a Core Audio process tap for system audio — both defined below) and
// relays framed PCM to /ws/native-capture. No /live page, no getUserMedia,
// no separate helper binary — this IS the native front-end. macOS TCC then
// attributes mic/screen-recording access to THIS app, not the detached
// python server, since floatpanel opens the mic/tap itself.
import AVFoundation
import AppKit
import Combine
import CoreAudio
import CoreGraphics
import Foundation
import SwiftUI

let port = ProcessInfo.processInfo.environment["MEETING_PORT"] ?? "8765"
let base = "http://127.0.0.1:\(port)"

// Single-instance guard: the server's in-memory Popen handle forgets us across
// its restarts (we're detached via start_new_session), and nothing stops a
// manual second launch — so hold an flock on a per-port lock file and quietly
// exit if another panel already owns it. Lock auto-releases when we die.
let lockPath = NSTemporaryDirectory() + "meetingsummary-floatpanel-\(port).lock"
let lockFd = open(lockPath, O_CREAT | O_RDWR, 0o644)
if lockFd < 0 || flock(lockFd, LOCK_EX | LOCK_NB) != 0 {
    exit(0)  // another instance is running; it already shows the window
}

struct Row: Identifiable {
    let id: Int
    let speaker: String
    let text: String
}

// ── Native capture, in-process (single App identity) ───────────────────────────
// Mic via AVCaptureSession (+AGC), system audio via a Core Audio process tap.
// Doing it HERE (not a separate helper binary) means the screen-recording/mic
// TCC grant belongs to THIS app — "透過 App 授權" — instead of a separate bundle
// id. Frames go straight out the relay socket in the same
// <track:UInt8><len:UInt32LE><payload> format /ws/native-capture expects.
private let CAP_TARGET_SR = 16000.0
private let CAP_TRACK_SYSTEM: UInt8 = 0
private let CAP_TRACK_MIC: UInt8 = 1

/// Builds framed messages and sends them over the relay WebSocket. send() is
/// enqueued per whole message, so concurrent mic+system writes never interleave.
/// The task is swappable (setTask) so a reconnect can swap in a fresh
/// URLSessionWebSocketTask under the SAME sink — the mic/system capturers hold
/// one sink for their whole lifetime and must keep writing across reconnects.
final class WSFrameSink {
    private var task: URLSessionWebSocketTask?
    init(_ task: URLSessionWebSocketTask?) { self.task = task }
    func setTask(_ t: URLSessionWebSocketTask?) { task = t }
    func write(track: UInt8, payload: Data) {
        guard let task = task else { return }  // no live socket (e.g. reconnecting) — drop
        var frame = Data([track])
        var len = UInt32(payload.count).littleEndian
        withUnsafeBytes(of: &len) { frame.append(contentsOf: $0) }
        frame.append(payload)
        task.send(.data(frame)) { _ in }
    }
}

/// Mic capture via AVCaptureSession -> 16 kHz mono int16, with peak-normalize AGC
/// (raw capture has no auto-gain, unlike browser getUserMedia).
final class PanelMicCapturer: NSObject, AVCaptureAudioDataOutputSampleBufferDelegate {
    let session = AVCaptureSession()
    let sink: WSFrameSink
    private var agcEnv: Float = 200

    init(sink: WSFrameSink) { self.sink = sink; super.init() }

    func start() throws {
        guard let dev = AVCaptureDevice.default(for: .audio) else {
            throw NSError(domain: "panel.mic", code: 3,
                          userInfo: [NSLocalizedDescriptionKey: "找不到麥克風裝置"])
        }
        let devInput = try AVCaptureDeviceInput(device: dev)
        guard session.canAddInput(devInput) else {
            throw NSError(domain: "panel.mic", code: 4,
                          userInfo: [NSLocalizedDescriptionKey: "無法加入麥克風輸入"])
        }
        session.addInput(devInput)
        let out = AVCaptureAudioDataOutput()
        out.audioSettings = [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: CAP_TARGET_SR,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsFloatKey: false,
            AVLinearPCMIsBigEndianKey: false,
            AVLinearPCMIsNonInterleaved: false,
        ]
        out.setSampleBufferDelegate(self, queue: DispatchQueue(label: "panel.mic"))
        guard session.canAddOutput(out) else {
            throw NSError(domain: "panel.mic", code: 5,
                          userInfo: [NSLocalizedDescriptionKey: "無法加入音訊輸出"])
        }
        session.addOutput(out)
        session.startRunning()
    }

    func stop() { if session.isRunning { session.stopRunning() } }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard CMSampleBufferDataIsReady(sampleBuffer),
              let bb = CMSampleBufferGetDataBuffer(sampleBuffer) else { return }
        var len = 0
        var ptr: UnsafeMutablePointer<Int8>?
        guard CMBlockBufferGetDataPointer(bb, atOffset: 0, lengthAtOffsetOut: nil,
                                          totalLengthOut: &len, dataPointerOut: &ptr) == noErr,
              let p = ptr, len > 0 else { return }
        let count = len / MemoryLayout<Int16>.size
        p.withMemoryRebound(to: Int16.self, capacity: count) { s in
            var peak: Float = 0
            for i in 0..<count { let a = abs(Float(s[i])); if a > peak { peak = a } }
            agcEnv += (peak > agcEnv ? 0.6 : 0.05) * (peak - agcEnv)
            let gain: Float = agcEnv > 40 ? min(30.0, max(1.0, 22000.0 / agcEnv)) : 1.0
            if gain > 1.01 {
                for i in 0..<count {
                    s[i] = Int16(max(-32767.0, min(32767.0, Float(s[i]) * gain)))
                }
            }
        }
        sink.write(track: CAP_TRACK_MIC, payload: Data(bytes: p, count: len))
    }
}

/// System audio via a Core Audio process tap (macOS 14.2+). Headless, drift
/// compensation OFF (tapping the global mix shares the output clock — enabling it
/// resamples the live output and garbles what the user hears).
@available(macOS 14.2, *)
final class PanelSystemTapCapturer {
    let sink: WSFrameSink
    private var tapID = AudioObjectID(kAudioObjectUnknown)
    private var aggID = AudioObjectID(kAudioObjectUnknown)
    private var procID: AudioDeviceIOProcID?
    private var converter: AVAudioConverter?
    private var inFormat: AVAudioFormat?
    private let outFormat = AVAudioFormat(commonFormat: .pcmFormatInt16,
                                          sampleRate: CAP_TARGET_SR, channels: 1, interleaved: true)!
    // A tap started WITHOUT Screen & System Audio Recording permission streams
    // buffers of exact zeros (no error) — indistinguishable from success unless
    // we look at the samples. sawSignal flips true on the first non-zero sample;
    // onFirstAudio fires once then, so the UI can say "真的在錄" vs "還沒收到音訊".
    private(set) var sawSignal = false
    var onFirstAudio: (() -> Void)?

    init(sink: WSFrameSink) { self.sink = sink }

    func start() -> Bool {
        let desc = CATapDescription(monoGlobalTapButExcludeProcesses: [])
        desc.isPrivate = true
        desc.muteBehavior = .unmuted
        var tap = AudioObjectID(kAudioObjectUnknown)
        var st = AudioHardwareCreateProcessTap(desc, &tap)
        guard st == noErr, tap != kAudioObjectUnknown else { return false }
        tapID = tap

        var fmt = AudioStreamBasicDescription()
        var sz = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
        var fmtAddr = AudioObjectPropertyAddress(
            mSelector: kAudioTapPropertyFormat, mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain)
        st = AudioObjectGetPropertyData(tapID, &fmtAddr, 0, nil, &sz, &fmt)
        guard st == noErr, let inFmt = AVAudioFormat(streamDescription: &fmt),
              let conv = AVAudioConverter(from: inFmt, to: outFormat) else { return false }
        inFormat = inFmt
        converter = conv

        let aggDesc: [String: Any] = [
            kAudioAggregateDeviceNameKey: "MeetingSummary Panel Tap",
            kAudioAggregateDeviceUIDKey: "io.meetingsummary.panel.aggregate",
            kAudioAggregateDeviceIsPrivateKey: true,
            kAudioAggregateDeviceIsStackedKey: false,
            kAudioAggregateDeviceTapAutoStartKey: true,
            kAudioAggregateDeviceTapListKey: [[
                kAudioSubTapUIDKey: desc.uuid.uuidString,
                kAudioSubTapDriftCompensationKey: false,
            ]],
        ]
        var agg = AudioObjectID(kAudioObjectUnknown)
        st = AudioHardwareCreateAggregateDevice(aggDesc as CFDictionary, &agg)
        guard st == noErr, agg != kAudioObjectUnknown else { return false }
        aggID = agg

        let queue = DispatchQueue(label: "panel.systemtap")
        st = AudioDeviceCreateIOProcIDWithBlock(&procID, aggID, queue) { [weak self] _, inData, _, _, _ in
            self?.handle(inData)
        }
        guard st == noErr, let proc = procID else { return false }
        st = AudioDeviceStart(aggID, proc)
        return st == noErr
    }

    private func handle(_ inData: UnsafePointer<AudioBufferList>) {
        guard let conv = converter, let inFmt = inFormat else { return }
        let bytesPerFrame = inFmt.streamDescription.pointee.mBytesPerFrame
        guard bytesPerFrame > 0 else { return }
        let abl = UnsafeMutableAudioBufferListPointer(UnsafeMutablePointer(mutating: inData))
        guard let first = abl.first, let mData = first.mData else { return }
        let inFrames = first.mDataByteSize / bytesPerFrame
        guard inFrames > 0, let inBuf = AVAudioPCMBuffer(pcmFormat: inFmt, frameCapacity: inFrames) else { return }
        inBuf.frameLength = inFrames
        memcpy(inBuf.audioBufferList.pointee.mBuffers.mData, mData, Int(first.mDataByteSize))
        let ratio = CAP_TARGET_SR / inFmt.sampleRate
        let cap = AVAudioFrameCount(Double(inFrames) * ratio) + 32
        guard let outBuf = AVAudioPCMBuffer(pcmFormat: outFormat, frameCapacity: cap) else { return }
        var convErr: NSError?
        var supplied = false
        let status = conv.convert(to: outBuf, error: &convErr) { _, s in
            if supplied { s.pointee = .noDataNow; return nil }
            supplied = true
            s.pointee = .haveData
            return inBuf
        }
        guard status != .error, convErr == nil, let ch = outBuf.int16ChannelData else { return }
        let n = Int(outBuf.frameLength)
        guard n > 0 else { return }
        if !sawSignal {  // silent (unpermitted) tap = exact zeros; any non-zero = real audio
            for i in 0..<n where ch[0][i] != 0 { sawSignal = true; break }
            if sawSignal { let cb = onFirstAudio; DispatchQueue.main.async { cb?() } }
        }
        sink.write(track: CAP_TRACK_SYSTEM, payload: Data(bytes: ch[0], count: n * MemoryLayout<Int16>.size))
    }

    func stop() {
        if let proc = procID, aggID != kAudioObjectUnknown {
            AudioDeviceStop(aggID, proc)
            AudioDeviceDestroyIOProcID(aggID, proc)
        }
        if aggID != kAudioObjectUnknown { AudioHardwareDestroyAggregateDevice(aggID) }
        if tapID != kAudioObjectUnknown { AudioHardwareDestroyProcessTap(tapID) }
    }
}

enum Source: String, CaseIterable, Identifiable {
    case mic, system, both
    var id: String { rawValue }
    var label: String {
        switch self {
        case .mic: return "麥克風(我)"
        case .system: return "系統音(對方)"
        case .both: return "兩者(混合)"
        }
    }
}

final class Model: ObservableObject {
    @Published var connected = false
    @Published var recording = false
    @Published var title = "待機"
    @Published var captions: [String] = []
    @Published var elapsed = ""
    @Published var note = ""
    @Published var hint = ""
    // Top-dot state: starting = capture began but no real audio yet (grey
    // breathing); audioLive = first non-zero sample arrived (red). Idle = neither.
    @Published var starting = false
    @Published var audioLive = false
    @Published var source: Source = .mic
    // A native session can fail AFTER capture already started (e.g. mic/screen-
    // recording access denied when the capturer actually opens the device) —
    // /live/state carries a best-effort notice for whatever's currently recording.
    @Published var liveNotice = ""
    private var wsTask: URLSessionWebSocketTask?
    private var relaySession: URLSession?  // dedicated session for the relay socket (not .shared)
    private var pingTimer: Timer?
    // Bumped on every real start()/stop() (NOT on an internal reconnect) — lets
    // async closures (permission callbacks, receive/ping/reconnect) tell "this
    // recording session ended/restarted" apart from "the relay socket dropped
    // and got swapped under us", which must NOT reset audioLive or re-prompt.
    private var relayEpoch = 0
    private var userStopped = false
    private var reconnectAttempt = 0
    private var relayMid: Int?          // learned from the server's {"type":"meeting"} message
    private var micCap: PanelMicCapturer?
    private var sysCap: AnyObject?   // PanelSystemTapCapturer (macOS 14.2+); retained
    private var sink: WSFrameSink?
    // Full live transcript, polled incrementally from /meetings/{mid}/transcripts
    // (?after=lastId) — same endpoint /live attach-on-load uses. Poll not WS: the
    // page already polls, no reason to add a Swift WebSocket client.
    @Published var transcripts: [Row] = []
    private var lastId = 0
    private var trackedMid: Int?
    var mid: Int?
    var startedAt: Date?
    private var lastShowSeq = -1        // last /floatpanel/open counter seen (-1 = not yet polled)
    var onShowRequest: (() -> Void)?    // set by AppDelegate: bring the panel forward

    private func req(_ path: String, method: String = "GET", json: [String: Any]? = nil,
                     done: ((Data?, Int) -> Void)? = nil) {
        guard let url = URL(string: base + path) else { return }
        var r = URLRequest(url: url); r.httpMethod = method; r.timeoutInterval = 4
        if let j = json {
            r.setValue("application/json", forHTTPHeaderField: "Content-Type")
            r.httpBody = try? JSONSerialization.data(withJSONObject: j)
        }
        URLSession.shared.dataTask(with: r) { d, resp, _ in
            let code = (resp as? HTTPURLResponse)?.statusCode ?? 0
            DispatchQueue.main.async { done?(d, code) }
        }.resume()
    }

    func poll() {
        req("/live/state") { [weak self] data, _ in
            guard let self = self else { return }
            guard let d = data,
                  let o = try? JSONSerialization.jsonObject(with: d) as? [String: Any] else {
                self.connected = false; self.title = "伺服器未連線"; return
            }
            self.connected = true
            let rec = (o["recording"] as? Bool) ?? false
            if rec && !self.recording { self.startedAt = Date() }
            if !rec { self.startedAt = nil; self.elapsed = "" }
            // A live session that ends (stop OR the relay dying) returns the top
            // dot to static idle — clear preparing/live so it stops breathing.
            // The initial preparing window (started, not yet server-confirmed) is
            // preserved: reset only on the recording -> not-recording transition.
            if self.recording && !rec { self.starting = false; self.audioLive = false; self.liveNotice = "" }
            self.recording = rec
            let newMid = o["mid"] as? Int
            if newMid != self.trackedMid {  // new session (or ended) -> reset transcript cursor
                self.trackedMid = newMid
                self.transcripts = []
                self.lastId = 0
            }
            self.mid = newMid
            self.title = rec ? ((o["title"] as? String) ?? "錄音中") : "待機"
            // Only take a server notice when present — don't clobber a locally-set
            // capture warning (e.g. the silent-tap / permission notice) every poll.
            if let n = o["notice"] as? String, !n.isEmpty { self.liveNotice = n }
            if let lines = o["captions"] as? [String] {
                self.captions = lines
            } else if let one = o["caption"] as? String, !one.isEmpty {
                self.captions = [one]
            } else {
                self.captions = []
            }
            // Server-driven re-show: /floatpanel/open bumps show_seq. Seeing it
            // increase means "bring the panel back" — works even when the
            // menu-bar light isn't visible (dev raw binary / full menu bar).
            if let seq = o["show_seq"] as? Int {
                if self.lastShowSeq >= 0 && seq > self.lastShowSeq { self.onShowRequest?() }
                self.lastShowSeq = seq
            }
        }
    }

    func pollTranscripts() {
        guard recording, let m = mid else { return }
        req("/meetings/\(m)/transcripts?after=\(lastId)") { [weak self] data, _ in
            guard let self = self, let d = data,
                  let o = try? JSONSerialization.jsonObject(with: d) as? [String: Any],
                  let rows = o["rows"] as? [[String: Any]] else { return }
            for r in rows {
                guard let id = r["id"] as? Int else { continue }
                let sp = (r["speaker"] as? String) ?? ""
                let tx = (r["text"] as? String) ?? ""
                self.transcripts.append(Row(id: id, speaker: sp, text: tx))
                if id > self.lastId { self.lastId = id }
            }
        }
    }

    func tick() {
        guard recording, let s = startedAt else { return }
        let e = Int(Date().timeIntervalSince(s))
        elapsed = String(format: "%d:%02d", e / 60, e % 60)
    }

    func start() {
        // Always capture natively, in this app (single TCC identity) — no
        // browser fallback, no server-spawned helper. Permission prompts
        // (mic / screen-recording) fire inline in startNativeRelay(); any
        // denial surfaces via liveNotice instead of falling back elsewhere.
        startNativeRelay()
    }

    // Capture natively IN THIS APP and stream frames to /ws/native-capture. The mic
    // prompt (AVCaptureDevice.requestAccess) and the system tap both run under the
    // App's identity, so the TCC grant is "透過 App 授權" — one entry, MeetingSummary.
    private func startNativeRelay() {
        relayEpoch += 1
        let epoch = relayEpoch
        userStopped = false
        reconnectAttempt = 0
        relayMid = nil  // a fresh start() always begins a NEW meeting, never resumes the last one

        let cfg = URLSessionConfiguration.default
        cfg.timeoutIntervalForRequest = 3600   // long-lived socket, not a short HTTP request
        cfg.waitsForConnectivity = true
        let session = URLSession(configuration: cfg)
        relaySession = session

        guard let task = openRelayTask(session: session, epoch: epoch) else { return }
        let sink = WSFrameSink(task)
        self.sink = sink
        // Indicator: starting = socket up + asking permission + warming the
        // capturers, but no real audio yet (grey breathing dot); audioLive flips
        // on the first non-zero sample from any track (red dot).
        self.starting = true; self.audioLive = false; self.liveNotice = ""

        let wantMic = (source == .mic || source == .both)
        let wantSystem = (source == .system || source == .both)

        if wantMic {
            AVCaptureDevice.requestAccess(for: .audio) { [weak self] granted in
                DispatchQueue.main.async {
                    guard let self = self, self.relayEpoch == epoch else { return }  // not stopped/restarted meanwhile
                    if granted {
                        let mic = PanelMicCapturer(sink: sink)
                        do { try mic.start(); self.micCap = mic; self.audioLive = true }
                        catch { self.liveNotice = "麥克風擷取失敗: \(error.localizedDescription)" }
                    } else {
                        self.liveNotice = "需要麥克風權限：系統設定 → 隱私權與安全性 → 麥克風"
                    }
                }
            }
        }
        if wantSystem {
            if #available(macOS 14.2, *) {
                // A Core Audio system-audio tap started WITHOUT Screen & System
                // Audio Recording permission streams pure silence (zeros, no
                // error). Taps have no dedicated requestAccess, so trigger the
                // Screen Recording prompt (same TCC gate) — it registers the app
                // in the privacy pane and asks for consent. First grant needs a
                // fresh tap, so tell the user to re-start after enabling.
                if !CGPreflightScreenCaptureAccess() {
                    _ = CGRequestScreenCaptureAccess()
                    self.liveNotice = "請在 系統設定 → 隱私權與安全性 → 螢幕與系統音訊錄製 開啟 MeetingSummary，再重新開始錄音"
                }
                let sys = PanelSystemTapCapturer(sink: sink)
                sys.onFirstAudio = { [weak self] in
                    guard let self = self, self.relayEpoch == epoch else { return }
                    self.audioLive = true
                    if self.liveNotice.contains("系統音") || self.liveNotice.contains("螢幕") {
                        self.liveNotice = ""  // audio flowing -> clear the permission nag
                    }
                }
                if sys.start() {
                    self.sysCap = sys
                    // Started ≠ capturing: an unpermitted tap yields only zeros.
                    // Check a few seconds in and tell the user instead of silently
                    // recording silence (this is the "真的在錄嗎" signal).
                    DispatchQueue.main.asyncAfter(deadline: .now() + 3) { [weak self] in
                        guard let self = self, self.relayEpoch == epoch else { return }
                        if (self.sysCap as? PanelSystemTapCapturer)?.sawSignal == false {
                            self.liveNotice = "⚠️ 未擷取到系統音（可能未授權）：系統設定 → 隱私權與安全性 → 螢幕與系統音訊錄製 開啟 MeetingSummary 後重新開始"
                        }
                    }
                } else {
                    self.liveNotice = "系統音擷取啟動失敗（需授權／macOS 14.2+）"
                }
            } else {
                self.liveNotice = "系統音原生擷取需 macOS 14.2 以上"
            }
        }
    }

    // Opens one WS attempt (initial connect OR reconnect) on the dedicated
    // relay session: builds the URL (resuming into relayMid if known), starts
    // the task, and kicks off its receive pump + keepalive ping timer. Does
    // NOT touch the sink/capturers — callers wire/rewire those as needed.
    private func openRelayTask(session: URLSession, epoch: Int) -> URLSessionWebSocketTask? {
        var urlStr = "ws://127.0.0.1:\(port)/ws/native-capture?source=\(source.rawValue)&diarize=1"
        if let m = relayMid { urlStr += "&session=\(m)" }
        guard let wsURL = URL(string: urlStr) else { return nil }
        let task = session.webSocketTask(with: wsURL)
        wsTask = task
        task.resume()
        pumpReceive(task: task, epoch: epoch)
        restartPingTimer(epoch: epoch)
        return task
    }

    // Recursive receive loop — REQUIRED so URLSessionWebSocketTask doesn't stall/
    // get reclaimed for having its read side never pumped. Only text message we
    // care about is the server's {"type":"meeting","id":...} (remembered for
    // resume-on-reconnect); everything else is ignored. A failure here means the
    // socket died — reconnect if we're still meant to be recording.
    private func pumpReceive(task: URLSessionWebSocketTask, epoch: Int) {
        task.receive { [weak self] result in
            guard let self = self else { return }
            switch result {
            case .success(let message):
                var newMid: Int?
                if case .string(let text) = message,
                   let data = text.data(using: .utf8),
                   let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                   obj["type"] as? String == "meeting", let mid = obj["id"] as? Int {
                    newMid = mid
                }
                DispatchQueue.main.async {
                    guard self.relayEpoch == epoch else { return }
                    if let mid = newMid { self.relayMid = mid }
                    self.reconnectAttempt = 0  // a successful read = the socket is healthy again
                    if self.liveNotice == "重新連線中…" { self.liveNotice = "" }
                }
                self.pumpReceive(task: task, epoch: epoch)
            case .failure:
                DispatchQueue.main.async {
                    guard self.relayEpoch == epoch, !self.userStopped else { return }
                    self.scheduleReconnect(epoch: epoch)
                }
            }
        }
    }

    // ~10s keepalive: a plain send-only socket with no ping/pong and no reads
    // is exactly what stalls/gets silently reclaimed. A ping failure means the
    // connection is dead even though writes haven't errored yet.
    private func restartPingTimer(epoch: Int) {
        pingTimer?.invalidate()
        pingTimer = Timer.scheduledTimer(withTimeInterval: 10, repeats: true) { [weak self] _ in
            guard let self = self, self.relayEpoch == epoch, let task = self.wsTask else { return }
            task.sendPing { [weak self] error in
                guard let self = self, error != nil else { return }
                DispatchQueue.main.async {
                    guard self.relayEpoch == epoch, !self.userStopped else { return }
                    self.scheduleReconnect(epoch: epoch)
                }
            }
        }
    }

    // Bounded backoff reconnect: 0.5s, 1s, 2s, 4s, 5s(cap), 5s — gives up after
    // ~6 tries. Does NOT touch mic/system capturers (they keep running, still
    // writing to the same `sink`) — only swaps a fresh task under it.
    private func scheduleReconnect(epoch: Int) {
        guard relayEpoch == epoch, !userStopped else { return }
        guard reconnectAttempt < 6 else {
            liveNotice = "重新連線失敗，請手動重新開始錄音"
            return
        }
        let delay = min(5.0, 0.5 * pow(2.0, Double(reconnectAttempt)))
        reconnectAttempt += 1
        liveNotice = "重新連線中…"
        DispatchQueue.main.asyncAfter(deadline: .now() + delay) { [weak self] in
            guard let self = self, self.relayEpoch == epoch, !self.userStopped,
                  let session = self.relaySession else { return }
            guard let task = self.openRelayTask(session: session, epoch: epoch) else {
                self.scheduleReconnect(epoch: epoch)
                return
            }
            self.sink?.setTask(task)
        }
    }

    private func stopNativeRelay() {
        relayEpoch += 1   // invalidate any in-flight permission/receive/ping/reconnect callback
        userStopped = true
        self.starting = false; self.audioLive = false
        micCap?.stop(); micCap = nil
        if #available(macOS 14.2, *) { (sysCap as? PanelSystemTapCapturer)?.stop() }
        sysCap = nil
        pingTimer?.invalidate(); pingTimer = nil
        sink?.setTask(nil); sink = nil
        relayMid = nil
        // Closing the socket ends the /ws/native-capture session server-side.
        wsTask?.cancel(with: .goingAway, reason: nil); wsTask = nil
        relaySession?.invalidateAndCancel(); relaySession = nil
    }

    // 主控台: the review/list/settings UI still lives in the browser (native entry
    // is /live only for now) — this is the one door back to it.
    func openConsole() {
        if let u = URL(string: base) { NSWorkspace.shared.open(u) }
    }

    func stop() {
        stopNativeRelay()                  // stop our in-process capture + close the socket
        req("/live/stop", method: "POST")  // also covers any browser /ws/live session
    }

    func saveNote() {
        let line = note.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let m = mid, !line.isEmpty else { return }
        req("/meetings/\(m)/notes/append", method: "POST", json: ["value": line])
        note = ""; hint = "已記下筆記"
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.6) { self.hint = "" }
    }
}

struct PanelView: View {
    @ObservedObject var m: Model

    // Session active (started or server-confirmed) — drives the top dot.
    private var dotActive: Bool { m.recording || m.starting }

    var body: some View {
        VStack(alignment: .leading, spacing: 11) {
            HStack(spacing: 8) {
                // 3-state: idle = grey static; 準備中 (started, no audio yet) = grey
                // breathing; 錄音中 (real audio flowing) = red breathing.
                Image(systemName: dotActive ? "record.circle.fill" : "circle")
                    .foregroundStyle(dotActive ? (m.audioLive ? Color.red : Color.gray) : Color.secondary)
                    .opacity(dotActive ? pulse : 1)
                Text(m.title).font(.headline).lineLimit(1)
                Spacer()
                if !m.elapsed.isEmpty {
                    Text(m.elapsed).font(.system(.subheadline, design: .monospaced))
                        .foregroundStyle(.secondary)
                }
                Button { m.openConsole() } label: {
                    Image(systemName: "list.bullet.rectangle")
                }.buttonStyle(.borderless).help("開啟主控台（會議清單／摘要／設定）")
            }
            if !m.recording {
                Picker("來源", selection: $m.source) {
                    ForEach(Source.allCases) { s in Text(s.label).tag(s) }
                }
                .pickerStyle(.segmented).labelsHidden()
            }
            if !m.liveNotice.isEmpty {
                // e.g. mic/screen-recording denied when capture actually starts —
                // this is the only place that shows up.
                Text("⚠️ " + m.liveNotice).font(.caption2).foregroundStyle(.orange).lineLimit(2)
            }
            transcriptView
            TextField("現場筆記… (Enter 記下)", text: $m.note)
                .textFieldStyle(.roundedBorder).disabled(!m.recording)
                .onSubmit { m.saveNote() }
            if !m.hint.isEmpty {
                Text(m.hint).font(.caption).foregroundStyle(.green)
            }
            HStack(spacing: 8) {
                Button { m.start() } label: {
                    Label("開始錄音", systemImage: "record.circle").frame(maxWidth: .infinity)
                }.buttonStyle(.borderedProminent).disabled(m.recording)
                Button { m.stop() } label: {
                    Label("停止", systemImage: "stop.fill").frame(maxWidth: .infinity)
                }.tint(.red).buttonStyle(.bordered).disabled(!m.recording)
            }.controlSize(.large)
        }
        .padding(16).frame(width: 320)
        .onReceive(Timer.publish(every: 0.7, on: .main, in: .common).autoconnect()) { _ in
            withAnimation(.easeInOut(duration: 0.6)) { pulse = pulse == 1 ? 0.35 : 1 }
        }
    }

    // Fixed-height scrolling transcript, auto-scrolls to the newest line.
    // ponytail: fixed 200pt box, always visible so the panel is sized once at
    // launch. Add a collapse toggle (recompute host.fittingSize + setContentSize)
    // only if the always-open panel proves too tall in use.
    @ViewBuilder private var transcriptView: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 5) {
                    if m.transcripts.isEmpty {
                        Text(m.recording ? "（聆聽中…）" : "尚未開始錄音")
                            .font(.callout).foregroundStyle(.secondary)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    } else {
                        ForEach(m.transcripts) { r in
                            (Text(r.speaker.isEmpty ? "" : r.speaker + "  ")
                                .font(.caption).foregroundColor(.secondary)
                             + Text(r.text).font(.callout))
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .id(r.id)
                        }
                    }
                }
                .frame(maxWidth: .infinity, alignment: .topLeading)
            }
            .frame(height: 200)
            .onChange(of: m.transcripts.count) { _ in
                if let last = m.transcripts.last {
                    withAnimation { proxy.scrollTo(last.id, anchor: .bottom) }
                }
            }
        }
    }

    @State private var pulse: Double = 1
}

final class PanelDelegate: NSObject, NSWindowDelegate {
    // Close = hide to the menu-bar light, not quit. The status item is the app's
    // persistent native presence now; quitting is the 結束 menu item.
    func windowShouldClose(_ sender: NSWindow) -> Bool {
        sender.orderOut(nil)
        return false
    }
}

// Menu-bar 指示燈: always-present status item — red dot while recording, click to
// show/hide the panel. Makes the app a proper background native presence (no dock,
// no browser) once launched.
final class StatusController: NSObject {
    let item = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
    let m: Model
    let panel: NSPanel
    private var bag: Set<AnyCancellable> = []

    init(model: Model, panel: NSPanel) {
        self.m = model
        self.panel = panel
        super.init()
        refresh(recording: m.recording)
        let menu = NSMenu()
        let toggle = NSMenuItem(title: "顯示／隱藏面板", action: #selector(togglePanel), keyEquivalent: "")
        toggle.target = self; menu.addItem(toggle)
        let console = NSMenuItem(title: "主控台（清單／摘要／設定）", action: #selector(openConsole), keyEquivalent: "")
        console.target = self; menu.addItem(console)
        menu.addItem(.separator())
        let quit = NSMenuItem(title: "結束 MeetingSummary", action: #selector(quit), keyEquivalent: "q")
        quit.target = self; menu.addItem(quit)
        item.menu = menu
        // Light follows recording state live.
        m.$recording.receive(on: RunLoop.main)
            .sink { [weak self] rec in self?.refresh(recording: rec) }
            .store(in: &bag)
    }

    func refresh(recording: Bool) {
        guard let b = item.button else {
            FileHandle.standardError.write("floatpanel: refresh but item.button==nil\n".data(using: .utf8)!)
            return
        }
        let name = recording ? "record.circle.fill" : "waveform.circle"
        if let img = NSImage(systemSymbolName: name, accessibilityDescription: "MeetingSummary") {
            img.isTemplate = true
            b.image = img
            b.title = ""
        } else {
            // SF Symbol unavailable -> a text title so the light is never invisible.
            b.image = nil
            b.title = recording ? "● REC" : "◉ MS"
        }
        b.contentTintColor = recording ? .systemRed : nil
    }

    @objc func togglePanel() {
        if panel.isVisible {
            panel.orderOut(nil)
        } else {
            panel.makeKeyAndOrderFront(nil)
            NSApp.activate(ignoringOtherApps: true)
        }
    }

    @objc func openConsole() { m.openConsole() }
    @objc func quit() { NSApp.terminate(nil) }
}

let model = Model()
let delegate = PanelDelegate()
let app = NSApplication.shared
// .regular = a Dock icon (reliable in every launch mode, incl. a dev raw binary —
// unlike the menu-bar status item, which only renders dependably from a bundled
// .app). The Dock icon is the app's persistent handle: closing the panel only
// hides it (windowShouldClose), so clicking the Dock icon re-shows it.
app.setActivationPolicy(.regular)

let panel = NSPanel(contentRect: NSRect(x: 0, y: 0, width: 300, height: 200),
                    styleMask: [.titled, .closable, .utilityWindow, .nonactivatingPanel],
                    backing: .buffered, defer: false)
panel.title = "MeetingSummary"
panel.level = .floating
panel.isFloatingPanel = true
panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
panel.hidesOnDeactivate = false
panel.titlebarAppearsTransparent = true
panel.isMovableByWindowBackground = true
panel.delegate = delegate

let vev = NSVisualEffectView()
vev.material = .hudWindow
vev.blendingMode = .behindWindow
vev.state = .active
let host = NSHostingView(rootView: PanelView(m: model))
host.translatesAutoresizingMaskIntoConstraints = false
vev.addSubview(host)
NSLayoutConstraint.activate([
    host.leadingAnchor.constraint(equalTo: vev.leadingAnchor),
    host.trailingAnchor.constraint(equalTo: vev.trailingAnchor),
    host.topAnchor.constraint(equalTo: vev.topAnchor),
    host.bottomAnchor.constraint(equalTo: vev.bottomAnchor),
])
panel.contentView = vev
panel.setContentSize(host.fittingSize)
panel.center()

// Create the menu-bar light, show the panel, and start polling only AFTER the
// app finishes launching. Creating an NSStatusItem before the run loop leaves
// item.button == nil, so refresh() bails and the image is never set -> a
// zero-width (invisible) status item. With no visible light, once the panel is
// closed (= hidden) there's no way to re-summon it. didFinishLaunching
// guarantees the status bar + its button exist, so the light actually appears
// and stays as the app's persistent wake handle.
final class AppDelegate: NSObject, NSApplicationDelegate {
    let m: Model
    let panel: NSPanel
    var status: StatusController?  // retains the status item for the app's life
    init(m: Model, panel: NSPanel) { self.m = m; self.panel = panel; super.init() }
    func applicationDidFinishLaunching(_ note: Notification) {
        panel.makeKeyAndOrderFront(nil)
        status = StatusController(model: m, panel: panel)
        // Server-driven re-show handle (menu-bar-independent): /floatpanel/open
        // bumps show_seq; the model sees it on poll and calls this to surface us.
        m.onShowRequest = { [weak self] in
            guard let self = self else { return }
            self.panel.makeKeyAndOrderFront(nil)
            NSApp.activate(ignoringOtherApps: true)
        }
        m.poll()
        let mm = m
        Timer.scheduledTimer(withTimeInterval: 1.5, repeats: true) { _ in mm.poll() }
        Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { _ in mm.tick() }
        Timer.scheduledTimer(withTimeInterval: 1.2, repeats: true) { _ in mm.pollTranscripts() }
    }

    // Closing the panel only hides it (windowShouldClose), so never quit on last
    // window closed — the app lives on in the Dock as the persistent handle.
    func applicationShouldTerminateAfterLastWindowClosed(_ app: NSApplication) -> Bool { false }

    // Clicking the Dock icon (app already running, panel hidden) re-shows the panel.
    func applicationShouldHandleReopen(_ app: NSApplication, hasVisibleWindows flag: Bool) -> Bool {
        if !flag { panel.makeKeyAndOrderFront(nil); NSApp.activate(ignoringOtherApps: true) }
        return true
    }
}
let appDelegate = AppDelegate(m: model, panel: panel)
app.delegate = appDelegate
app.run()
