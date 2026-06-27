// Lightweight floating control panel for MeetingSummary. An always-on-top NSPanel
// that stays visible over any app (Teams/Zoom/etc.) so you can see recording state
// + elapsed time and start/stop without alt-tabbing to the browser. It talks to the
// local server: polls GET /live/state, posts /live/stop, opens /live to start.
// Run alongside the server; honors MEETING_PORT (default 8765).
import AppKit
import Foundation

let port = ProcessInfo.processInfo.environment["MEETING_PORT"] ?? "8765"
let base = "http://127.0.0.1:\(port)"

final class Panel: NSObject {
    let dot = NSTextField(labelWithString: "●")
    let status = NSTextField(labelWithString: "連線中…")
    let timer = NSTextField(labelWithString: "")
    let startBtn = NSButton(title: "開始", target: nil, action: nil)
    let stopBtn = NSButton(title: "停止", target: nil, action: nil)
    var recording = false
    var startedAt: Date?

    func req(_ path: String, method: String = "GET", done: ((Data?) -> Void)? = nil) {
        guard let url = URL(string: base + path) else { return }
        var r = URLRequest(url: url)
        r.httpMethod = method
        r.timeoutInterval = 3
        URLSession.shared.dataTask(with: r) { d, _, _ in
            DispatchQueue.main.async { done?(d) }
        }.resume()
    }

    @objc func onStart() { NSWorkspace.shared.open(URL(string: base + "/live")!) }
    @objc func onStop() { req("/live/stop", method: "POST") }

    func pollState() {
        req("/live/state") { [weak self] data in
            guard let self = self else { return }
            var rec = false
            if let d = data, let o = try? JSONSerialization.jsonObject(with: d) as? [String: Any] {
                rec = (o["recording"] as? Bool) ?? false
            } else {
                self.status.stringValue = "伺服器未連線"
                self.dot.textColor = .systemGray
                return
            }
            if rec && !self.recording { self.startedAt = Date() }
            if !rec { self.startedAt = nil; self.timer.stringValue = "" }
            self.recording = rec
            self.dot.textColor = rec ? .systemRed : .systemGray
            self.status.stringValue = rec ? "錄音中" : "待機"
            self.stopBtn.isEnabled = rec
            self.startBtn.isEnabled = !rec
        }
    }

    func tickTimer() {
        guard recording, let s = startedAt else { return }
        let e = Int(Date().timeIntervalSince(s))
        timer.stringValue = String(format: "%d:%02d", e / 60, e % 60)
    }

    func makePanel() -> NSPanel {
        let panel = NSPanel(
            contentRect: NSRect(x: 0, y: 0, width: 240, height: 92),
            styleMask: [.titled, .closable, .utilityWindow, .nonactivatingPanel],
            backing: .buffered, defer: false)
        panel.title = "MeetingSummary"
        panel.level = .floating
        panel.isFloatingPanel = true
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        panel.hidesOnDeactivate = false

        dot.font = .systemFont(ofSize: 13)
        status.font = .boldSystemFont(ofSize: 15)
        timer.font = .monospacedDigitSystemFont(ofSize: 14, weight: .regular)
        timer.textColor = .secondaryLabelColor
        for b in [startBtn, stopBtn] { b.bezelStyle = .rounded; b.controlSize = .large }
        startBtn.target = self; startBtn.action = #selector(onStart)
        stopBtn.target = self; stopBtn.action = #selector(onStop)
        stopBtn.isEnabled = false

        let row = NSStackView(views: [dot, status, NSView(), timer])
        row.orientation = .horizontal
        row.spacing = 8
        let btns = NSStackView(views: [startBtn, stopBtn])
        btns.orientation = .horizontal
        btns.distribution = .fillEqually
        btns.spacing = 8
        let stack = NSStackView(views: [row, btns])
        stack.orientation = .vertical
        stack.spacing = 12
        stack.edgeInsets = NSEdgeInsets(top: 14, left: 16, bottom: 14, right: 16)
        stack.translatesAutoresizingMaskIntoConstraints = false
        let content = NSView()
        content.addSubview(stack)
        NSLayoutConstraint.activate([
            stack.leadingAnchor.constraint(equalTo: content.leadingAnchor),
            stack.trailingAnchor.constraint(equalTo: content.trailingAnchor),
            stack.topAnchor.constraint(equalTo: content.topAnchor),
            stack.bottomAnchor.constraint(equalTo: content.bottomAnchor),
        ])
        panel.contentView = content
        panel.center()
        panel.makeKeyAndOrderFront(nil)
        return panel
    }
}

let appDelegatePanel = Panel()
let app = NSApplication.shared
app.setActivationPolicy(.accessory)
let panel = appDelegatePanel.makePanel()
appDelegatePanel.pollState()
Timer.scheduledTimer(withTimeInterval: 2.0, repeats: true) { _ in appDelegatePanel.pollState() }
Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { _ in appDelegatePanel.tickTimer() }
app.run()
