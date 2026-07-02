// swift-tools-version: 5.9
import PackageDescription

let package = Package(
  name: "comfybox",
  platforms: [.macOS(.v14), .iOS(.v16)],
  products: [
    .library(name: "ZImage", targets: ["ZImage"]),
    .executable(name: "ComfyBox", targets: ["ComfyBox"]),
    .executable(name: "ComfyBoxDesktop", targets: ["ComfyBoxDesktop"])
  ],
  dependencies: [
    .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.29.1")),
    .package(
      url: "https://github.com/huggingface/swift-transformers",
      from: "1.1.6"
    ),
    .package(url: "https://github.com/apple/swift-log.git", from: "1.6.4")
  ],
  targets: [
    .target(
      name: "ZImage",
      dependencies: [
        .product(name: "MLX", package: "mlx-swift"),
        .product(name: "MLXFast", package: "mlx-swift"),
        .product(name: "MLXNN", package: "mlx-swift"),
        .product(name: "MLXRandom", package: "mlx-swift"),
        .product(name: "Transformers", package: "swift-transformers"),
        .product(name: "Logging", package: "swift-log")
      ],
      path: "Sources/ZImage"
    ),
    .executableTarget(
      name: "ComfyBox",
      dependencies: ["ZImage"],
      path: "Sources/ComfyBox"
    ),
    .executableTarget(
      name: "ComfyBoxDesktop",
      dependencies: ["ZImage"],
      path: "Sources/ComfyBoxDesktop"
    ),
    .testTarget(
      name: "ZImageTests",
      dependencies: [
        "ZImage",
        .product(name: "MLX", package: "mlx-swift"),
        .product(name: "MLXRandom", package: "mlx-swift")
      ],
      path: "Tests/ZImageTests"
    ),
    .testTarget(
      name: "ZImageIntegrationTests",
      dependencies: [
        "ZImage",
        .product(name: "MLX", package: "mlx-swift"),
        .product(name: "MLXNN", package: "mlx-swift"),
        .product(name: "MLXRandom", package: "mlx-swift")
      ],
      path: "Tests/ZImageIntegrationTests",
      resources: [
        .copy("Resources")
      ]
    ),
    .testTarget(
      name: "ZImageE2ETests",
      dependencies: ["ZImage"],
      path: "Tests/ZImageE2ETests"
    ),
    .testTarget(
      name: "ComfyBoxDesktopTests",
      dependencies: ["ComfyBoxDesktop"],
      path: "Tests/ComfyBoxDesktopTests"
    )
  ]
)
