// Floating control panel for MeetingSummary. Always-on-top NSPanel visible over
// any app (Teams/Zoom/…): shows recording state, meeting title, elapsed time, the
// latest live caption, and a quick-note field that appends to the meeting notes.
// Talks to the local server (GET /live/state, POST /live/stop, /meetings/{id}/
// notes/append, opens /live to start). Honors MEETING_PORT (default 8765).
import AppKit
import Foundation

let port = ProcessInfo.processInfo.environment["MEETING_PORT"] ?? "8765"
let base = "http://127.0.0.1:\(port)"

final class Panel: NSObject, NSWindowDelegate {
    func windowWillClose(_ notification: Notification) {
        NSApplication.shared.terminate(nil)  // close = exit so the panel can be reopened
    }

    let dot = NSTextField(labelWithString: "●")
    let title = NSTextField(labelWithString: "待機")
    let timer = NSTextField(labelWithString: "")
    let caption = NSTextField(wrappingLabelWithString: "")
    let note = NSTextField()
    let hint = NSTextField(labelWithString: "")
    let startBtn = NSButton(title: "開始錄音", target: nil, action: nil)
    let stopBtn = NSButton(title: "停止", target: nil, action: nil)
    var recording = false
    var startedAt: Date?
    var mid: Int?

    func req(_ path: String, method: String = "GET", json: [String: Any]? = nil,
             done: ((Data?) -> Void)? = nil) {
        guard let url = URL(string: base + path) else { return }
        var r = URLRequest(url: url)
        r.httpMethod = method
        r.timeoutInterval = 4
        if let j = json {
            r.setValue("application/json", forHTTPHeaderField: "Content-Type")
            r.httpBody = try? JSONSerialization.data(withJSONObject: j)
        }
        URLSession.shared.dataTask(with: r) { d, _, _ in
            DispatchQueue.main.async { done?(d) }
        }.resume()
    }

    @objc func onStart() {
        if let url = URL(string: base + "/live") { NSWorkspace.shared.open(url) }
    }
    @objc func onStop() { req("/live/stop", method: "POST") }

    @objc func onNote() {
        let line = note.stringValue.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let m = mid, !line.isEmpty else { return }
        req("/meetings/\(m)/notes/append", method: "POST", json: ["value": line])
        note.stringValue = ""
        hint.stringValue = "已記下筆記"
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.6) { self.hint.stringValue = "" }
    }

    func pollState() {
        req("/live/state") { [weak self] data in
            guard let self = self else { return }
            guard let d = data,
                  let o = try? JSONSerialization.jsonObject(with: d) as? [String: Any] else {
                self.title.stringValue = "伺服器未連線"; self.dot.textColor = .systemGray
                return
            }
            let rec = (o["recording"] as? Bool) ?? false
            if rec && !self.recording { self.startedAt = Date() }
            if !rec { self.startedAt = nil; self.timer.stringValue = "" }
            self.recording = rec
            self.mid = o["mid"] as? Int
            self.dot.textColor = rec ? .systemRed : .systemGray
            self.title.stringValue = rec ? ((o["title"] as? String) ?? "錄音中") : "待機"
            self.caption.stringValue = (o["caption"] as? String) ?? ""
            self.stopBtn.isEnabled = rec
            self.startBtn.isEnabled = !rec
            self.note.isEnabled = rec
        }
    }

    func tickTimer() {
        guard recording, let s = startedAt else { return }
        let e = Int(Date().timeIntervalSince(s))
        timer.stringValue = String(format: "%d:%02d", e / 60, e % 60)
    }

    func makePanel() -> NSPanel {
        let panel = NSPanel(
            contentRect: NSRect(x: 0, y: 0, width: 300, height: 180),
            styleMask: [.titled, .closable, .utilityWindow, .nonactivatingPanel],
            backing: .buffered, defer: false)
        panel.title = "MeetingSummary"
        panel.level = .floating
        panel.isFloatingPanel = true
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        panel.hidesOnDeactivate = false

        dot.font = .systemFont(ofSize: 12)
        title.font = .boldSystemFont(ofSize: 14)
        title.lineBreakMode = .byTruncatingTail
        timer.font = .monospacedDigitSystemFont(ofSize: 13, weight: .regular)
        timer.textColor = .secondaryLabelColor
        caption.font = .systemFont(ofSize: 12)
        caption.textColor = .secondaryLabelColor
        caption.maximumNumberOfLines = 2
        caption.setContentHuggingPriority(.defaultLow, for: .horizontal)
        note.placeholderString = "現場筆記… (Enter 記下)"
        note.font = .systemFont(ofSize: 12)
        note.target = self; note.action = #selector(onNote)
        note.isEnabled = false
        hint.font = .systemFont(ofSize: 11); hint.textColor = .systemGreen
        for b in [startBtn, stopBtn] { b.bezelStyle = .rounded; b.controlSize = .large }
        startBtn.target = self; startBtn.action = #selector(onStart)
        stopBtn.target = self; stopBtn.action = #selector(onStop)
        stopBtn.isEnabled = false

        let head = NSStackView(views: [dot, title, NSView(), timer])
        head.orientation = .horizontal; head.spacing = 8
        let btns = NSStackView(views: [startBtn, stopBtn])
        btns.orientation = .horizontal; btns.distribution = .fillEqually; btns.spacing = 8
        let stack = NSStackView(views: [head, caption, note, hint, btns])
        stack.orientation = .vertical; stack.spacing = 9
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
        panel.delegate = self
        panel.makeKeyAndOrderFront(nil)
        return panel
    }
}

let appDelegatePanel = Panel()
let app = NSApplication.shared
app.setActivationPolicy(.accessory)
let panel = appDelegatePanel.makePanel()
appDelegatePanel.pollState()
Timer.scheduledTimer(withTimeInterval: 1.5, repeats: true) { _ in appDelegatePanel.pollState() }
Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { _ in appDelegatePanel.tickTimer() }
app.run()
