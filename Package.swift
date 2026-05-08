// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "TMSileroVAD",
    platforms: [
        .iOS(.v14),
        .macOS(.v11)
    ],
    products: [
        .library(name: "TMSileroVAD", targets: ["TMSileroVAD"])
    ],
    targets: [
        .target(
            name: "TMSileroVAD",
            path: "Sources/TMSileroVAD",
            resources: [
                .copy("Resources/silero-vad-unified-v6.0.0.mlmodelc"),
                .copy("Resources/silero-vad-unified-256ms-v6.0.0.mlmodelc")
            ]
        ),
        .testTarget(
            name: "TMSileroVADTests",
            dependencies: ["TMSileroVAD"],
            path: "Tests/TMSileroVADTests",
            exclude: [
                "Helpers/.gitkeep"
            ]
        )
    ]
)
