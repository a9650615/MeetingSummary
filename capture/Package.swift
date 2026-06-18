// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "capture",
    platforms: [.macOS(.v13)],
    targets: [
        .executableTarget(name: "capture", path: "Sources/capture")
    ]
)
