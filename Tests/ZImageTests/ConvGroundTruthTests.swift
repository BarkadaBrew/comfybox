import XCTest
import MLX
import MLXNN
import Logging
@testable import ZImage

final class ConvGroundTruthTests: XCTestCase {
  func testWhichCallIsCorrupt() throws {
    let env = ProcessInfo.processInfo.environment
    guard let vaePath = env["REAL_VAE_PATH"], let latPath = env["REAL_LATENT_PATH"],
          FileManager.default.fileExists(atPath: vaePath) else { throw XCTSkip("no assets") }
    let vae = LTX2VAE(config: .v23)
    let raw = try MLX.loadArrays(url: URL(fileURLWithPath: vaePath))
    try LTX2WeightLoader.loadVAEWeightsFromTensors(
      into: vae, tensors: LTX2EchoCheckpoint.videoVAETensors(from: raw), logger: Logger(label: "t"))
    MLX.eval(vae.parameters())
    let decoder = vae.decoder
    let full = try MLX.loadArrays(url: URL(fileURLWithPath: latPath))["latent"]!.asType(.bfloat16)
    let latent = full[0..., 0..., 0..<4, 0..., 0...]
    var x = decoder.perChannelStatistics.unNormalize(latent)
    x = decoder.convIn(x)
    let keys = decoder.upBlocks.keys.sorted { Int($0)! < Int($1)! }
    for key in keys.dropLast() {
      let block = decoder.upBlocks[key]!
      if let g = block as? LTX2DecoderResBlockGroup { x = g(x, timestep: nil) }
      else if let u = block as? LTX2DepthToSpaceUpsample { x = u(x) }
    }
    MLX.eval(x)  // (1,128,25,320,192)
    let group = decoder.upBlocks[keys.last!] as! LTX2DecoderResBlockGroup
    let conv = group.resBlocks[group.resBlocks.keys.sorted { Int($0)! < Int($1)! }[0]]!
      .modules().compactMap { $0 as? CausalConv3d }.first!

    // Overflow-proof ground truth: 4 spatial quadrants with 1-px halo,
    // trimmed and reassembled. Each call: 25f x <=162x98 = tiny rows.
    let H = x.dim(3), W = x.dim(4)
    let hMid = H / 2, wMid = W / 2
    func quadrant(_ h0: Int, _ h1: Int, _ w0: Int, _ w1: Int) -> MLXArray {
      let hLo = max(0, h0 - 1), hHi = min(H, h1 + 1)
      let wLo = max(0, w0 - 1), wHi = min(W, w1 + 1)
      let sub = x[0..., 0..., 0..., hLo..<hHi, wLo..<wHi]
      let out = conv(sub)  // symmetric temporal pad internally; spatial pad 1
      // Trim halo: the conv preserves spatial size (pad 1), so output coords
      // match input coords; take [h0-hLo, ...).
      return out[0..., 0..., 0..., (h0 - hLo)..<(h1 - hLo), (w0 - wLo)..<(w1 - wLo)]
    }
    let top = MLX.concatenated([quadrant(0, hMid, 0, wMid), quadrant(0, hMid, wMid, W)], axis: 4)
    let bot = MLX.concatenated([quadrant(hMid, H, 0, wMid), quadrant(hMid, H, wMid, W)], axis: 4)
    let truth = MLX.concatenated([top, bot], axis: 3)
    MLX.eval(truth)

    let plainOut = conv(x)
    MLX.eval(plainOut)
    conv.resetStream(active: true)
    let s1 = conv(x[0..., 0..., 0..<7, 0..., 0...])
    conv.resetStream(active: false)
    MLX.eval(s1)

    let scale = max(MLX.abs(truth.asType(.float32)).max().item(Float.self), 1e-6)
    for i in 0..<3 {
      let rp = MLX.abs(plainOut[0..., 0..., i..<(i+1), 0..., 0...].asType(.float32)
        - truth[0..., 0..., i..<(i+1), 0..., 0...].asType(.float32)).max().item(Float.self) / scale
      let rs = MLX.abs(s1[0..., 0..., i..<(i+1), 0..., 0...].asType(.float32)
        - truth[0..., 0..., i..<(i+1), 0..., 0...].asType(.float32)).max().item(Float.self) / scale
      print("TRUTH frame \(i): plain rel \(rp)  streamed rel \(rs)")
    }
    for i in [12, 20, 24] {
      let rp = MLX.abs(plainOut[0..., 0..., i..<(i+1), 0..., 0...].asType(.float32)
        - truth[0..., 0..., i..<(i+1), 0..., 0...].asType(.float32)).max().item(Float.self) / scale
      print("TRUTH frame \(i): plain rel \(rp)")
    }
  }
}
