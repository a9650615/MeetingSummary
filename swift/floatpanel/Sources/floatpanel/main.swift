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
import AppKit
import Combine
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
            self.mid = o["mid"] as? Int
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
        }
    }

    func tick() {
        guard recording, let s = startedAt else { return }
        let e = Int(Date().timeIntervalSince(s))
        elapsed = String(format: "%d:%02d", e / 60, e % 60)
    }

    func start() {
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

    private func openInBrowser() {
        if let u = URL(string: base + "/live?source=\(source.rawValue)") { NSWorkspace.shared.open(u) }
    }

    func stop() { req("/live/stop", method: "POST") }

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
            }
            if !m.recording {
                Picker("來源", selection: $m.source) {
                    ForEach(Source.allCases) { s in Text(s.label).tag(s) }
                }
                .pickerStyle(.segmented).labelsHidden()
                if !(m.nativeStart[m.source.rawValue] ?? false) {
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
            captionList
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

    @ViewBuilder private var captionList: some View {
        VStack(alignment: .leading, spacing: 3) {
            if m.captions.isEmpty {
                Text(m.recording ? "（聆聽中…）" : "尚未開始錄音")
                    .font(.callout).foregroundStyle(.secondary)
            } else {
                ForEach(Array(m.captions.suffix(3).enumerated()), id: \.offset) { i, line in
                    Text(line)
                        .font(.callout)
                        .foregroundStyle(i == m.captions.suffix(3).count - 1 ? .primary : .secondary)
                        .lineLimit(1)
                }
            }
        }
        .frame(maxWidth: .infinity, minHeight: 56, alignment: .topLeading)
    }

    @State private var pulse: Double = 1
}

final class PanelDelegate: NSObject, NSWindowDelegate {
    func windowWillClose(_ n: Notification) { NSApplication.shared.terminate(nil) }
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

model.poll()
model.pollCapability()
Timer.scheduledTimer(withTimeInterval: 1.5, repeats: true) { _ in model.poll() }
Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { _ in model.tick() }
Timer.scheduledTimer(withTimeInterval: 10.0, repeats: true) { _ in model.pollCapability() }
app.run()
