// Floating control panel for MeetingSummary — native SwiftUI in a translucent
// always-on-top NSPanel (HUD material). Shows recording state, meeting title,
// live caption, elapsed time, and a quick-note field that appends to the meeting
// notes. Talks to the local server; honors MEETING_PORT (default 8765).
import AppKit
import Combine
import Foundation
import SwiftUI

let port = ProcessInfo.processInfo.environment["MEETING_PORT"] ?? "8765"
let base = "http://127.0.0.1:\(port)"

final class Model: ObservableObject {
    @Published var connected = false
    @Published var recording = false
    @Published var title = "待機"
    @Published var caption = ""
    @Published var elapsed = ""
    @Published var note = ""
    @Published var hint = ""
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
            self.caption = (o["caption"] as? String) ?? ""
        }
    }

    func tick() {
        guard recording, let s = startedAt else { return }
        let e = Int(Date().timeIntervalSince(s))
        elapsed = String(format: "%d:%02d", e / 60, e % 60)
    }

    func start() { if let u = URL(string: base + "/live") { NSWorkspace.shared.open(u) } }
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
            Text(m.caption.isEmpty ? (m.recording ? "（聆聽中…）" : "尚未開始錄音") : m.caption)
                .font(.callout).foregroundStyle(.secondary)
                .lineLimit(2).frame(maxWidth: .infinity, minHeight: 34, alignment: .topLeading)
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
        .padding(16).frame(width: 300)
        .onReceive(Timer.publish(every: 0.7, on: .main, in: .common).autoconnect()) { _ in
            withAnimation(.easeInOut(duration: 0.6)) { pulse = pulse == 1 ? 0.35 : 1 }
        }
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
Timer.scheduledTimer(withTimeInterval: 1.5, repeats: true) { _ in model.poll() }
Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { _ in model.tick() }
app.run()
