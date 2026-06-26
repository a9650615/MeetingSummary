// swift-tools-version: 5.10
import PackageDescription

// Persistent ANE (Neural Engine) ASR helper for live transcription. The Python
// server keeps its own VAD/endpointing and pipes each utterance's PCM here; this
// process loads the Qwen3-ASR CoreML model ONCE and transcribes on the ANE — so
// live runs off the GPU (power). Bundled into the Mac .app/.dmg.
let package = Package(
    name: "Qwen3AneServer",
    platforms: [.macOS("15.0")],
    dependencies: [
        .package(url: "https://github.com/soniqo/speech-swift", branch: "main"),
    ],
    targets: [
        .executableTarget(
            name: "qwen3-ane",
            dependencies: [.product(name: "Qwen3ASR", package: "speech-swift")]),
    ]
)
