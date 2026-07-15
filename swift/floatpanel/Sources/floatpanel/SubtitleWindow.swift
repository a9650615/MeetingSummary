// YouTube-style live-caption overlay: a borderless, click-through, always-on-top
// panel pinned near the bottom of the main screen showing the latest caption line.
// Toggled from the main panel (Model.showSubtitle). Observes Model.captions —
// the last finalized line — and re-renders as new finals arrive (poll cadence).
import AppKit
import Combine
import SwiftUI

struct SubtitleOverlayView: View {
    @ObservedObject var m: Model
    var body: some View {
        // Prefer the tentative interim (updates live while speaking) over the last
        // finalized caption, so the overlay streams instead of jumping per utterance.
        let line = m.interim.isEmpty ? (m.captions.last ?? "") : m.interim
        Text(line.isEmpty ? "…" : line)
            .font(.system(size: 28, weight: .semibold))
            .foregroundStyle(.white)
            .multilineTextAlignment(.center)
            .lineLimit(3)
            .fixedSize(horizontal: false, vertical: true)
            .padding(.horizontal, 22).padding(.vertical, 14)
            .background(Color.black.opacity(0.6), in: RoundedRectangle(cornerRadius: 14))
            .shadow(color: .black.opacity(0.5), radius: 10)
            .frame(maxWidth: 960)
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottom)
            .padding(.bottom, 90)
            .allowsHitTesting(false)
    }
}

/// Owns the overlay NSPanel. setVisible(true) shows a full-screen transparent,
/// click-through floating panel (the caption sits at the bottom); false hides it.
final class SubtitleController {
    private var panel: NSPanel?
    private let model: Model
    init(model: Model) { self.model = model }

    func setVisible(_ on: Bool) { on ? show() : panel?.orderOut(nil) }

    private func show() {
        if panel == nil {
            let frame = NSScreen.main?.frame ?? NSRect(x: 0, y: 0, width: 1440, height: 900)
            let p = NSPanel(contentRect: frame,
                            styleMask: [.borderless, .nonactivatingPanel],
                            backing: .buffered, defer: false)
            p.level = .floating                 // above normal windows
            p.isOpaque = false
            p.backgroundColor = .clear
            p.hasShadow = false
            p.ignoresMouseEvents = true         // click-through: never steals focus/clicks
            p.collectionBehavior = [.canJoinAllSpaces, .stationary, .fullScreenAuxiliary]
            p.contentView = NSHostingView(rootView: SubtitleOverlayView(m: model))
            p.setFrame(frame, display: true)
            panel = p
        }
        panel?.orderFrontRegardless()
    }
}
