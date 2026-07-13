// VideoGeneratorHolder.swift — lock-based shared owner of the LTX-2 generator.
//
// The LTX-2 video stack (~65GB) used to live in a lone `WarmServer.ltx2Generator`
// property, invisible to the ModelPool and never released — so a resident image
// model plus LTX-2 blew past physical RAM and tripped OS_REASON_JETSAM (#218).
//
// This holder makes the generator a *shared, lock-protected* resource that both
// WarmServer (which creates/uses it on the video path) and WarmServerCoordinator
// (which must evict it before loading an image model) can reach without an actor
// hop. The render flag lets the memory-pressure guard release an idle generator
// while never yanking memory out from under an in-flight video render.
//
// Issue: #218

import Foundation
import MLX

final class VideoGeneratorHolder: @unchecked Sendable {
  private let lock = NSLock()
  private var generator: LTX2VideoGenerator?
  private var rendering = false

  /// The current generator, if resident.
  func get() -> LTX2VideoGenerator? {
    lock.lock(); defer { lock.unlock() }
    return generator
  }

  /// Install (or clear) the generator.
  func set(_ g: LTX2VideoGenerator?) {
    lock.lock(); defer { lock.unlock() }
    generator = g
  }

  func isResident() -> Bool {
    lock.lock(); defer { lock.unlock() }
    return generator != nil
  }

  func beginRender() { lock.lock(); rendering = true; lock.unlock() }
  func endRender() { lock.lock(); rendering = false; lock.unlock() }

  func isRendering() -> Bool {
    lock.lock(); defer { lock.unlock() }
    return rendering
  }

  /// Force-release the generator and free its GPU memory. The caller must
  /// guarantee no render is in flight (e.g. running on the serial render queue,
  /// where a video render and this call are mutually exclusive). Returns true
  /// if a generator was actually released.
  @discardableResult
  func release() -> Bool {
    lock.lock(); defer { lock.unlock() }
    let had = generator != nil
    generator?.unload()
    generator = nil
    if had { GPU.clearCache() }
    return had
  }

  /// Release the generator ONLY if it is resident and not mid-render. Safe to
  /// call from the memory-pressure guard on an arbitrary thread. Returns true
  /// if released.
  @discardableResult
  func releaseIfIdle() -> Bool {
    lock.lock(); defer { lock.unlock() }
    guard generator != nil, !rendering else { return false }
    generator?.unload()
    generator = nil
    GPU.clearCache()
    return true
  }
}
