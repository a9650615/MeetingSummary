// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "floatpanel",
    platforms: [.macOS(.v13)],
    targets: [
        // Embed Info.plist into the Mach-O so macOS TCC treats floatpanel as its
        // OWN app identity (io.meetingsummary.floatpanel) — the native front that
        // spawns audiocap, so recording grants attribute to the app, not python.
        .executableTarget(name: "floatpanel", linkerSettings: [
            .unsafeFlags(["-Xlinker", "-sectcreate", "-Xlinker", "__TEXT",
                          "-Xlinker", "__info_plist", "-Xlinker", "Info.plist"]),
        ]),
    ]
)
