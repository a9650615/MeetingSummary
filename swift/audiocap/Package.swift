// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "audiocap",
    platforms: [.macOS(.v13)],
    targets: [
        // Embed Info.plist into the Mach-O so macOS TCC treats audiocap as its OWN
        // Screen-Recording subject (named entry) instead of blaming the parent
        // python process ("python3.10" in the permission list).
        .executableTarget(name: "audiocap", linkerSettings: [
            .unsafeFlags(["-Xlinker", "-sectcreate", "-Xlinker", "__TEXT",
                          "-Xlinker", "__info_plist", "-Xlinker", "Info.plist"]),
        ]),
    ]
)
