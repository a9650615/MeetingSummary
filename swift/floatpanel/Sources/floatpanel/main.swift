// Floating control panel for MeetingSummary — native SwiftUI in a translucent
// always-on-top NSPanel (HUD material). Shows recording state, meeting title,
// a source picker, the last few live-caption lines, elapsed time, and a
// quick-note field that appends to the meeting notes. Talks to the local
// server; honors MEETING_PORT (default 8765).
//
// Native-start gap: mic capture uses the browser's getUserMedia and system
// audio (even with the native ScreenCaptureKit helper) is driven by a
// WebSocket the /live page's JS opens — there is no server endpoint today
// that starts a recording without that page. So 開始錄音 here still opens
// /live in the browser; the source picker's value is passed as a `source`
// query param so the page can preselect it once it reads that param (it
// does not yet — see app.py's _LIVE_JS). Until then this is a same-value
// convenience, not a behavior change.
import AppKit
import Combine
import Foundation
import SwiftUI

let port = ProcessInfo.processInfo.environment["MEETING_PORT"] ?? "8765"
let base = "http://127.0.0.1:\(port)"

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
    @Published var systemAudioReady = false  // audiocap installed + permission granted
    var mid: Int?
    var startedAt: Date?

    private func req(_ path: String, method: String = "GET", json: [String: Any]? = nil,
                     done: ((Data?) -> Void)? = nil) {
        guard let url = URL(string: base + path) else { return }
        var r = URLRequest(url: url); r.httpMethod = method; r.timeoutInterval = 4
        if let j = json {
            r.setValue("application/json", forHTTPHeaderField: "Content-Type")
            r.httpBody = try? JSONSerialization.data(withJSONObject: j)
        }
        URLSession.shared.dataTask(with: r) { d, _, _ in
            DispatchQueue.main.async { done?(d) }
        }.resume()
    }

    func poll() {
        req("/live/state") { [weak self] data in
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
        req("/native/capability") { [weak self] data in
            guard let self = self, let d = data,
                  let o = try? JSONSerialization.jsonObject(with: d) as? [String: Any] else { return }
            self.systemAudioReady = (o["audiocap"] as? Bool ?? false) && (o["granted"] as? Bool ?? false)
        }
    }

    func tick() {
        guard recording, let s = startedAt else { return }
        let e = Int(Date().timeIntervalSince(s))
        elapsed = String(format: "%d:%02d", e / 60, e % 60)
    }

    func start() {
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
                if m.source != .mic && !m.systemAudioReady {
                    Text("系統音需在瀏覽器分享畫面時勾選「分享音訊」（未偵測到原生擷取權限）")
                        .font(.caption2).foregroundStyle(.orange)
                }
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
