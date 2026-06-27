// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "audiocap",
    platforms: [.macOS(.v13)],
    targets: [
        .executableTarget(name: "audiocap"),
    ]
)
