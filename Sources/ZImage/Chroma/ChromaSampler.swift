import Foundation
import MLX
import MLXRandom

/// Chroma-specific flow matching sampler.
///
/// Uses time-shifted sigmas with linear interpolation for mu based on
/// image sequence length. Different from Flux 1/2 scheduling.
public struct ChromaSampler {
  public let baseShift: Float
  public let maxShift: Float

  public init(baseShift: Float = 0.5, maxShift: Float = 1.15) {
    self.baseShift = baseShift
    self.maxShift = maxShift
  }

  /// Compute time-shifted timesteps for Chroma.
  public func timesteps(numSteps: Int, imageSequenceLength: Int, start: Float = 1.0, stop: Float = 0.0) -> [Float] {
    let x1: Float = 256.0, y1 = baseShift
    let x2: Float = 4096.0, y2 = maxShift
    let m = (y2 - y1) / (x2 - x1)
    let b = y1 - m * x1
    let mu = m * Float(imageSequenceLength) + b

    var t: [Float] = []
    for i in 0...numSteps {
      let val = start + (stop - start) * Float(i) / Float(numSteps)
      t.append(val)
    }

    return t.map { tVal in
      if tVal <= 0.0 { return 0.0 }
      if tVal >= 1.0 { return 1.0 }
      return exp(mu) / (exp(mu) + pow(1.0 / tVal - 1.0, 1.0))
    }
  }

  /// Euler step: x_t + (t_prev - t) * pred
  public func step(pred: MLXArray, xT: MLXArray, t: Float, tPrev: Float) -> MLXArray {
    xT.asType(pred.dtype) + MLXArray(tPrev - t) * pred
  }

  /// Sample from prior (standard normal)
  public func samplePrior(shape: [Int], seed: UInt64? = nil) -> MLXArray {
    let key = seed.map { MLXRandom.key($0) }
    if let key = key {
      return MLXRandom.normal(shape, key: key).asType(.bfloat16)
    }
    return MLXRandom.normal(shape).asType(.bfloat16)
  }
}
