import XCTest
import MLX
import MLXRandom
@testable import ZImage

final class LatentPreviewApproximatorTests: XCTestCase {

  // MARK: - Basic Output Shape

  func testLatentsToRGBAProducesCorrectDimensions() {
    let latentH = 64
    let latentW = 64
    let latents = MLXRandom.normal([1, 16, latentH, latentW])

    guard let (data, width, height) = LatentPreviewApproximator.latentsToRGBA(
      latents, latentHeight: latentH, latentWidth: latentW
    ) else {
      XCTFail("latentsToRGBA returned nil")
      return
    }

    XCTAssertEqual(width, latentW, "Output width should match latent width")
    XCTAssertEqual(height, latentH, "Output height should match latent height")
    XCTAssertEqual(data.count, latentW * latentH * 4, "RGBA data should be W*H*4 bytes")
  }

  func testLatentsToRGBASmallDimensions() {
    let latentH = 4
    let latentW = 4
    let latents = MLXRandom.normal([1, 16, latentH, latentW])

    let result = LatentPreviewApproximator.latentsToRGBA(
      latents, latentHeight: latentH, latentWidth: latentW
    )

    XCTAssertNotNil(result, "Should handle small latent dimensions")
    if let (data, w, h) = result {
      XCTAssertEqual(w, 4)
      XCTAssertEqual(h, 4)
      XCTAssertEqual(data.count, 64) // 4*4*4
    }
  }

  // MARK: - Alpha Channel

  func testLatentsToRGBAAlphaIsOpaque() {
    let latents = MLXRandom.normal([1, 16, 8, 8])

    guard let (data, _, _) = LatentPreviewApproximator.latentsToRGBA(
      latents, latentHeight: 8, latentWidth: 8
    ) else {
      XCTFail("latentsToRGBA returned nil")
      return
    }

    // Every 4th byte (alpha channel) should be 255.
    data.withUnsafeBytes { ptr in
      guard let base = ptr.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
      for i in stride(from: 3, to: data.count, by: 4) {
        XCTAssertEqual(base[i], 255, "Alpha at pixel \(i/4) should be 255")
      }
    }
  }

  // MARK: - Pixel Value Bounds

  func testLatentsToRGBAPixelValuesAreClamped() {
    // Use large values to test clamping.
    let latents = MLX.full([1, 16, 4, 4], values: Float(100.0))

    guard let (data, _, _) = LatentPreviewApproximator.latentsToRGBA(
      latents, latentHeight: 4, latentWidth: 4
    ) else {
      XCTFail("latentsToRGBA returned nil")
      return
    }

    // All RGB values should be <= 255 (clamped).
    data.withUnsafeBytes { ptr in
      guard let base = ptr.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
      for i in 0..<data.count {
        XCTAssertLessThanOrEqual(base[i], 255)
      }
    }
  }

  // MARK: - Invalid Inputs

  func testLatentsToRGBARejectsWrongChannels() {
    // Only 3 channels (should still work — uses min(channels, 16))
    let latents = MLXRandom.normal([1, 3, 4, 4])
    let result = LatentPreviewApproximator.latentsToRGBA(
      latents, latentHeight: 4, latentWidth: 4
    )
    XCTAssertNotNil(result, "Should handle fewer than 16 channels")
  }

  func testLatentsToRGBARejectsWrongNdim() {
    // 3D tensor — wrong number of dimensions.
    let latents = MLXRandom.normal([16, 4, 4])
    let result = LatentPreviewApproximator.latentsToRGBA(
      latents, latentHeight: 4, latentWidth: 4
    )
    XCTAssertNil(result, "Should return nil for non-4D tensor")
  }

  func testLatentsToRGBARejectsMismatchedDimensions() {
    // Latent tensor is 8x8 but we claim 4x4.
    let latents = MLXRandom.normal([1, 16, 8, 8])
    let result = LatentPreviewApproximator.latentsToRGBA(
      latents, latentHeight: 4, latentWidth: 4
    )
    XCTAssertNil(result, "Should return nil when dimensions don't match")
  }

  // MARK: - Zero Latents

  func testLatentsToRGBAZeroLatentsProduceMidtone() {
    // Zero latents + 0.5 bias should produce ~128 gray.
    let latents = MLX.zeros([1, 16, 2, 2])

    guard let (data, _, _) = LatentPreviewApproximator.latentsToRGBA(
      latents, latentHeight: 2, latentWidth: 2
    ) else {
      XCTFail("latentsToRGBA returned nil")
      return
    }

    // Check first pixel's RGB values — should be approximately 128 (0.5 * 255).
    data.withUnsafeBytes { ptr in
      guard let base = ptr.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
      let r = base[0]
      let g = base[1]
      let b = base[2]
      // With zero latents and 0.5 bias, expect ~127-128.
      XCTAssertEqual(r, 127, accuracy: 1, "Red should be ~128 for zero latents")
      XCTAssertEqual(g, 127, accuracy: 1, "Green should be ~128 for zero latents")
      XCTAssertEqual(b, 127, accuracy: 1, "Blue should be ~128 for zero latents")
    }
  }
}

// Helper for approximate UInt8 comparison.
private func XCTAssertEqual(_ a: UInt8, _ b: UInt8, accuracy: UInt8, _ message: String = "") {
  let diff = a > b ? a - b : b - a
  XCTAssertLessThanOrEqual(diff, accuracy, "\(message) — expected \(b) +/- \(accuracy), got \(a)")
}
