// Floating control panel for MeetingSummary — native SwiftUI in a translucent
// always-on-top NSPanel (HUD material). Shows recording state, meeting title,
// a source picker, the last few live-caption lines, elapsed time, and a
// quick-note field that appends to the meeting notes. Talks to the local
// server; honors MEETING_PORT (default 8765).
//
// Browserless start: 開始錄音 calls POST /live/start directly when the server
// reports native_start[source] == true (audiocap installed + the relevant
// permission already granted — see /native/capability). The server spawns
// audiocap itself and feeds it straight into the live pipeline, so no /live
// page or getUserMedia is involved. If native start isn't available for the
// chosen source (e.g. mic permission not granted yet, or audiocap missing),
// this falls back to opening /live in the browser like before — the first
// browser mic use, or a manual "螢幕錄製" grant + retry, is what flips
// native_start to true afterwards.
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
    // Auto placeholder (說話者1 / 對方2 / 我3) = not yet recognized -> no name shown.
    var displaySpeaker: String {
        let s = speaker.trimmingCharacters(in: .whitespaces)
        if s.range(of: "^(我|對方|說話者)[0-9]+$", options: .regularExpression) != nil { return "" }
        return s
    }
}

// ── Native capture, in-process (approach B, single App identity) ──────────────
// Ported from swift/audiocap: mic via AVCaptureSession (+AGC), system audio via a
// Core Audio process tap. Doing it HERE (not a separate audiocap binary) means the
// screen-recording/mic TCC grant belongs to THIS app — "透過 App 授權" — instead of
// being split across a second bundle id. Frames go straight out the relay socket in
// the same <track:UInt8><len:UInt32LE><payload> format /ws/native-capture expects.
private let CAP_TARGET_SR = 16000.0
private let CAP_TRACK_SYSTEM: UInt8 = 0
private let CAP_TRACK_MIC: UInt8 = 1

/// Builds framed messages and sends them over the relay WebSocket. send() is
/// enqueued per whole message, so concurrent mic+system writes never interleave.
final class WSFrameSink {
    private let task: URLSessionWebSocketTask
    init(_ task: URLSessionWebSocketTask) { self.task = task }
    func write(track: UInt8, payload: Data) {
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
    @Published var source: Source = .mic
    // Additive /native/capability field: which sources /live/start can drive
    // right now (audiocap installed + the relevant permission already granted).
    @Published var nativeStart: [String: Bool] = ["mic": false, "system": false, "both": false]
    // A native session can fail AFTER /live/start already returned 200 (e.g.
    // mic access denied when audiocap actually opens the input stream) — the
    // HTTP call alone can't tell us that, so /live/state carries a best-effort
    // notice for whatever's currently recording.
    @Published var liveNotice = ""
    // Approach B: when the server reports relay==true, THIS app spawns audiocap
    // (audiocapPath) and streams its framed stdout to /ws/native-capture, so macOS
    // TCC attributes screen-recording/mic to the native app instead of the detached
    // python server. When false, fall back to POST /live/start (python spawns).
    @Published var relay = false
    private var wsTask: URLSessionWebSocketTask?
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
            self.recording = rec
            let newMid = o["mid"] as? Int
            if newMid != self.trackedMid {  // new session (or ended) -> reset transcript cursor
                self.trackedMid = newMid
                self.transcripts = []
                self.lastId = 0
            }
            self.mid = newMid
            self.title = rec ? ((o["title"] as? String) ?? "錄音中") : "待機"
            self.liveNotice = (o["notice"] as? String) ?? ""
            if let lines = o["captions"] as? [String] {
                self.captions = lines
            } else if let one = o["caption"] as? String, !one.isEmpty {
                self.captions = [one]
            } else {
                self.captions = []
            }
        }
    }

    func pollCapability() {
        req("/native/capability") { [weak self] data, _ in
            guard let self = self, let d = data,
                  let o = try? JSONSerialization.jsonObject(with: d) as? [String: Any] else { return }
            if let ns = o["native_start"] as? [String: Bool] {
                self.nativeStart = ns  // absent on an old server -> stays all-false (safe default)
            }
            self.relay = (o["relay"] as? Bool) ?? false
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
        // Approach B: the App captures natively (single TCC identity) + relays.
        if relay {
            startNativeRelay()
            return
        }
        guard nativeStart[source.rawValue] ?? false else {
            openInBrowser()
            return
        }
        req("/live/start", method: "POST", json: ["source": source.rawValue]) { [weak self] _, code in
            guard let self = self else { return }
            if code != 200 {
                self.openInBrowser()  // server declined (e.g. lost permission mid-flight) -> fall back
            }
            // success: no local state flip needed -- the next poll() picks up
            // recording=true from /live/state, same as a browser session.
        }
    }

    // Capture natively IN THIS APP and stream frames to /ws/native-capture. The mic
    // prompt (AVCaptureDevice.requestAccess) and the system tap both run under the
    // App's identity, so the TCC grant is "透過 App 授權" — one entry, MeetingSummary.
    private func startNativeRelay() {
        guard let wsURL = URL(string:
            "ws://127.0.0.1:\(port)/ws/native-capture?source=\(source.rawValue)&diarize=1") else { return }
        let task = URLSession.shared.webSocketTask(with: wsURL)
        wsTask = task
        task.resume()
        let sink = WSFrameSink(task)
        self.sink = sink

        let wantMic = (source == .mic || source == .both)
        let wantSystem = (source == .system || source == .both)

        if wantMic {
            AVCaptureDevice.requestAccess(for: .audio) { [weak self] granted in
                DispatchQueue.main.async {
                    guard let self = self, self.wsTask === task else { return }  // not stopped meanwhile
                    if granted {
                        let mic = PanelMicCapturer(sink: sink)
                        do { try mic.start(); self.micCap = mic }
                        catch { self.liveNotice = "麥克風擷取失敗: \(error.localizedDescription)" }
                    } else {
                        self.liveNotice = "需要麥克風權限：系統設定 → 隱私權與安全性 → 麥克風"
                    }
                }
            }
        }
        if wantSystem {
            if #available(macOS 14.2, *) {
                let sys = PanelSystemTapCapturer(sink: sink)
                if sys.start() { self.sysCap = sys }
                else { self.liveNotice = "系統音擷取啟動失敗（需授權／macOS 14.2+）" }
            } else {
                self.liveNotice = "系統音原生擷取需 macOS 14.2 以上"
            }
        }
    }

    private func stopNativeRelay() {
        micCap?.stop(); micCap = nil
        if #available(macOS 14.2, *) { (sysCap as? PanelSystemTapCapturer)?.stop() }
        sysCap = nil
        sink = nil
        // Closing the socket ends the /ws/native-capture session server-side.
        wsTask?.cancel(with: .goingAway, reason: nil); wsTask = nil
    }

    private func openInBrowser() {
        if let u = URL(string: base + "/live?source=\(source.rawValue)") { NSWorkspace.shared.open(u) }
    }

    // 主控台: the review/list/settings UI still lives in the browser (native entry
    // is /live only for now) — this is the one door back to it.
    func openConsole() {
        if let u = URL(string: base) { NSWorkspace.shared.open(u) }
    }

    func stop() {
        stopNativeRelay()               // relay session: kill our audiocap + close socket
        req("/live/stop", method: "POST")  // also covers legacy /live/start + browser sessions
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

    var body: some View {
        VStack(alignment: .leading, spacing: 11) {
            HStack(spacing: 8) {
                Image(systemName: m.recording ? "record.circle.fill" : "circle")
                    .foregroundStyle(m.recording ? .red : .secondary)
                    .opacity(m.recording ? pulse : 1)
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
                if !m.relay && !(m.nativeStart[m.source.rawValue] ?? false) {
                    Text(m.source == .mic
                        ? "尚未偵測到麥克風權限，將於瀏覽器開啟（首次使用會請求授權）"
                        : "尚未偵測到原生系統音擷取權限，將於瀏覽器分享畫面時勾選「分享音訊」")
                        .font(.caption2).foregroundStyle(.orange)
                }
            }
            if !m.liveNotice.isEmpty {
                // e.g. mic/screen-recording denied AFTER /live/start already
                // returned success — this is the only place that shows up.
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
                            (Text(r.displaySpeaker.isEmpty ? "" : r.displaySpeaker + "  ")
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
        guard let b = item.button else { return }
        let name = recording ? "record.circle.fill" : "waveform.circle"
        b.image = NSImage(systemSymbolName: name, accessibilityDescription: "MeetingSummary")
        b.image?.isTemplate = true
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
app.setActivationPolicy(.accessory)

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
panel.makeKeyAndOrderFront(nil)

let statusController = StatusController(model: model, panel: panel)

model.poll()
model.pollCapability()
Timer.scheduledTimer(withTimeInterval: 1.5, repeats: true) { _ in model.poll() }
Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { _ in model.tick() }
Timer.scheduledTimer(withTimeInterval: 1.2, repeats: true) { _ in model.pollTranscripts() }
Timer.scheduledTimer(withTimeInterval: 10.0, repeats: true) { _ in model.pollCapability() }
app.run()
