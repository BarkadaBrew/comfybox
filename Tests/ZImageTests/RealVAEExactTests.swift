import XCTest
import MLX
import MLXNN
import Logging
@testable import ZImage

final class RealVAEExactTests: XCTestCase {
  func testRealDataExactParity() throws {
    let env = ProcessInfo.processInfo.environment
    guard let vaePath = env["REAL_VAE_PATH"], let latPath = env["REAL_LATENT_PATH"],
          FileManager.default.fileExists(atPath: vaePath) else { throw XCTSkip("no assets") }
    let vae = LTX2VAE(config: .v23)
    let raw = try MLX.loadArrays(url: URL(fileURLWithPath: vaePath))
    try LTX2WeightLoader.loadVAEWeightsFromTensors(
      into: vae, tensors: LTX2EchoCheckpoint.videoVAETensors(from: raw), logger: Logger(label: "t"))
    MLX.eval(vae.parameters())
    // Spatial crop 20x12: at the final conv level (160x96 = 15360 rows/frame)
    // even the full-frame plain reference stays far below the Metal int32
    // corruption onset — an overflow-SAFE exact reference. (The full 40x24
    // latent's plain decode is itself corrupt at the boundary frames — see
    // ConvGroundTruthTests — and must not be used as a reference.)
    let full = try MLX.loadArrays(url: URL(fileURLWithPath: latPath))["latent"]!.asType(.bfloat16)
    let latent = full[0..., 0..., 0..., 0..<20, 0..<12]

    let plain = vae.decode(latent)
    MLX.eval(plain)
    let streamed = vae.decodeStreamed(latent)
    MLX.eval(streamed)
    XCTAssertEqual(plain.shape, streamed.shape, "shape mismatch")

    // Per-frame relative error against plain (the ground truth here).
    var worst: Float = 0
    for i in 0..<plain.dim(2) {
      let p = plain[0..., 0..., i..<(i + 1), 0..., 0...].asType(.float32)
      let s = streamed[0..., 0..., i..<(i + 1), 0..., 0...].asType(.float32)
      let r = MLX.abs(p - s).max().item(Float.self) / max(MLX.abs(p).max().item(Float.self), 1e-6)
      if r > 0.02 { print("EXACT frame \(i): rel \(r) MISMATCH") }
      worst = max(worst, r)
    }
    print("EXACT worst frame rel: \(worst)")
    XCTAssertLessThan(worst, 0.02, "streamed != plain on real data")
  }
}
