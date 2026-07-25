import XCTest
import MLX
import MLXNN
import MLXRandom
@testable import ZImage

final class BisectScaleTests: XCTestCase {

  private func makeDecoder() -> LTX2Decoder3D {
    let decoder = LTX2Decoder3D(config: .v23)
    MLXRandom.seed(11)
    for m in decoder.modules() {
      if let conv = m as? CausalConv3d {
        conv.weight = MLXRandom.normal(conv.weight.shape) * 0.05
        conv.bias = MLXRandom.normal(conv.bias.shape) * 0.01
      }
    }
    MLX.eval(decoder.parameters())
    return decoder
  }

  private func rel(_ a: MLXArray, _ b: MLXArray) -> Float {
    MLX.abs(a - b).max().item(Float.self) / max(MLX.abs(b).max().item(Float.self), 1e-6)
  }

  func testMidScaleBF16Parity() throws {
    let decoder = makeDecoder()
    MLXRandom.seed(41)
    // Production-like dims at plain-safe volume (7x20x12 = 1680), bf16.
    let latent = MLXRandom.normal([1, 128, 7, 20, 12]).asType(.bfloat16)
    let plain = decoder(latent)
    let streamed = decoder.decodeStreamed(latent)
    MLX.eval(plain, streamed)
    XCTAssertEqual(plain.shape, streamed.shape)
    let r = rel(streamed, plain)
    print("SCALE bf16 7x20x12 chunk1: rel \(r)")
    XCTAssertLessThan(r, 2e-2, "bf16 mid-scale parity broken")
  }

  func testMidScaleF32Parity() throws {
    let decoder = makeDecoder()
    MLXRandom.seed(43)
    let latent = MLXRandom.normal([1, 128, 7, 20, 12]).asType(.float32)
    let plain = decoder(latent)
    let streamed = decoder.decodeStreamed(latent)
    MLX.eval(plain, streamed)
    let r = rel(streamed, plain)
    print("SCALE f32 7x20x12 chunk1: rel \(r)")
    XCTAssertLessThan(r, 1e-4)
  }

  func testFullScaleStreamedSanity() throws {
    // 7x40x24 (the corrupting production size). Plain corrupts here, so
    // compare streamed against per-frame statistics of the tiled decode
    // (seams make it approximate — check gross sanity per frame, not bits).
    let decoder = makeDecoder()
    MLXRandom.seed(47)
    let latent = MLXRandom.normal([1, 128, 7, 40, 24]).asType(.bfloat16)
    let streamed = decoder.decodeStreamed(latent)
    MLX.eval(streamed)
    print("SCALE full 7x40x24 streamed shape \(streamed.shape)")
    let f = streamed.dim(2)
    var bad = 0
    for i in stride(from: 0, to: f, by: 4) {
      let fr = streamed[0..., 0..., i..<(i + 1), 0..., 0...].asType(.float32)
      let sd = MLX.sqrt(((fr - fr.mean()) * (fr - fr.mean())).mean()).item(Float.self)
      if !(sd.isFinite && sd > 1e-4) { bad += 1; print("SCALE frame \(i) degenerate std \(sd)") }
    }
    print("SCALE degenerate sampled frames: \(bad)")
    XCTAssertEqual(bad, 0, "streamed decode produced degenerate frames at 7x40x24")
  }
}

extension BisectScaleTests {

  func testFullScaleWithCacheLimit() throws {
    // Reproduce the server's GPU cache limit (#34 plist: 8192MB): buffer
    // recycling under a bounded pool is the env difference vs clean tests.
    MLX.GPU.set(cacheLimit: 8192 * 1024 * 1024)
    defer { MLX.GPU.set(cacheLimit: .max) }
    let decoder = makeDecoder()
    MLXRandom.seed(53)
    let latent = MLXRandom.normal([1, 128, 7, 40, 24]).asType(.bfloat16)
    let streamed = decoder.decodeStreamed(latent)
    MLX.eval(streamed)
    var bad = 0
    for i in stride(from: 0, to: streamed.dim(2), by: 4) {
      let fr = streamed[0..., 0..., i..<(i + 1), 0..., 0...].asType(.float32)
      let sd = MLX.sqrt(((fr - fr.mean()) * (fr - fr.mean())).mean()).item(Float.self)
      if !(sd.isFinite && sd > 1e-4) { bad += 1; print("SCALE cachelim frame \(i) degenerate \(sd)") }
    }
    print("SCALE cache-limit degenerate frames: \(bad)")
    XCTAssertEqual(bad, 0)
  }

  func testFullScaleWithBallast() throws {
    // Reproduce the server's resident-weights Metal heap (~70GB): allocate
    // ballast tensors, then run the full-scale streamed decode.
    var ballast: [MLXArray] = []
    for _ in 0..<15 {  // 15 x 4GB = 60GB
      let b = MLXArray.zeros([1024, 1024, 1024], dtype: .float32)
      MLX.eval(b)
      ballast.append(b)
    }
    print("SCALE ballast allocated: \(ballast.count * 4)GB")
    let decoder = makeDecoder()
    MLXRandom.seed(59)
    let latent = MLXRandom.normal([1, 128, 7, 40, 24]).asType(.bfloat16)
    let streamed = decoder.decodeStreamed(latent)
    MLX.eval(streamed)
    var bad = 0
    for i in stride(from: 0, to: streamed.dim(2), by: 4) {
      let fr = streamed[0..., 0..., i..<(i + 1), 0..., 0...].asType(.float32)
      let sd = MLX.sqrt(((fr - fr.mean()) * (fr - fr.mean())).mean()).item(Float.self)
      if !(sd.isFinite && sd > 1e-4) { bad += 1; print("SCALE ballast frame \(i) degenerate \(sd)") }
    }
    print("SCALE ballast degenerate frames: \(bad)")
    ballast.removeAll()
    XCTAssertEqual(bad, 0)
  }
}
