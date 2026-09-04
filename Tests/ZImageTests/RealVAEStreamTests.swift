// Offline real-data reproduction harness for the streamed-decode corruption
// (#36). Gated on env so CI skips: needs the extracted VAE weights and a
// dumped production latent.
import XCTest
import MLX
import MLXNN
import Logging
@testable import ZImage

final class RealVAEStreamTests: XCTestCase {

  func testRealWeightsRealLatent() throws {
    let env = ProcessInfo.processInfo.environment
    guard let vaePath = env["REAL_VAE_PATH"], let latPath = env["REAL_LATENT_PATH"],
          FileManager.default.fileExists(atPath: vaePath),
          FileManager.default.fileExists(atPath: latPath) else {
      throw XCTSkip("REAL_VAE_PATH / REAL_LATENT_PATH not set")
    }
    let vae = LTX2VAE(config: .v23)
    let raw = try MLX.loadArrays(url: URL(fileURLWithPath: vaePath))
    let tensors = LTX2EchoCheckpoint.videoVAETensors(from: raw)
    try LTX2WeightLoader.loadVAEWeightsFromTensors(into: vae, tensors: tensors, logger: Logger(label: "test"))
    MLX.eval(vae.parameters())
    print("REAL vae loaded")

    let latent = try MLX.loadArrays(url: URL(fileURLWithPath: latPath))["latent"]!.asType(.bfloat16)
    print("REAL latent \(latent.shape)")

    func frameStats(_ t: MLXArray, tag: String) -> Int {
      var bad = 0
      for i in stride(from: 0, to: t.dim(2), by: 4) {
        let fr = t[0..., 0..., i..<(i + 1), 0..., 0...].asType(.float32)
        let sd = MLX.sqrt(((fr - fr.mean()) * (fr - fr.mean())).mean()).item(Float.self)
        let mn = fr.mean().item(Float.self)
        if !(sd.isFinite && sd > 0.01) { bad += 1; print("REAL \(tag) frame \(i): mean \(mn) std \(sd) DEGENERATE") }
      }
      print("REAL \(tag): degenerate \(bad)")
      return bad
    }

    // Streamed decode of the REAL latent with REAL weights.
    let streamed = try vae.decodeStreamed(latent)
    MLX.eval(streamed)
    let badStreamed = frameStats(streamed, tag: "streamed")

    // Tiled reference (known-valid path).
    let tiled = try vae.decodeTiled(latent)
    MLX.eval(tiled)
    let badTiled = frameStats(tiled, tag: "tiled")

    XCTAssertEqual(badTiled, 0, "tiled reference degenerate — latent itself bad")
    XCTAssertEqual(badStreamed, 0, "streamed decode degenerate on real data")
  }
}
