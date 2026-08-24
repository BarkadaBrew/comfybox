import Foundation

/// Named, deterministic, engine-applied post-process looks (Todd 2026-08-24
/// "I prefer style pack"). A style is selected by NAME — on the preset
/// (`ImagePreset.style`) or the request (`style` wire field) — resolved at
/// the route (unknown names fail loud there, never a silent no-op) and
/// applied after VAE decode, before save.
///
/// v1 pack:
///   - `phone`   — the PhoneLook recipe (accel-distill color correction:
///                 levels / contrast / adaptive saturation / unsharp).
///   - `trix-bw` — guaranteed monochrome pushed-film look. Prompts alone
///                 rendered "Kodak Tri-X" accents in color; the style makes
///                 B&W deterministic: pan-film channel mix, percentile
///                 levels, pushed S-curve with deep blacks and a soft
///                 highlight shoulder. Grain stays the model's job.
///
/// Every style is a pure function over an interleaved HWC RGB [0,1] buffer —
/// no server, no weights — tested in `StylePackTests`.
public enum StylePack: String, CaseIterable, Sendable {
  case phone = "phone"
  case trixBW = "trix-bw"

  /// Registry lookup. nil for unknown names — the caller decides whether
  /// that is a 400 (the generate decode) or a skip.
  ///
  /// Names are matched after trimming and lowercasing so `"Trix-BW "` and
  /// `"trix-bw"` are the same style; an empty/whitespace name is "no style",
  /// not an unknown one.
  public static func named(_ name: String) -> StylePack? {
    let key = name.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    guard !key.isEmpty else { return nil }
    return StylePack(rawValue: key)
  }

  /// Every name this engine answers to, for error messages and `/v1/styles`.
  public static var knownNames: [String] { allCases.map(\.rawValue) }

  /// #399 — resolve ONE style name from the request and the named preset.
  ///
  /// Precedence is main's rule for every other preset-contributed field
  /// (#286 steps/guidance, #285 vae, #154 shift): **explicit request wins,
  /// then the preset, then nothing**. `phoneLook` is the legacy alias for
  /// `"phone"` on both sides, and it is only consulted when that side named
  /// no `style` — so a preset that says `style: "trix-bw"` is not overruled
  /// by its own historical `phone_look: true`.
  ///
  /// Returns nil when neither side asked for a look. nil is load-bearing:
  /// it is what makes a render without a style pack byte-identical to the
  /// pre-#399 engine (`StylePackParityTests`).
  public static func resolveName(
    requestStyle: String?, requestPhoneLook: Bool?,
    presetStyle: String?, presetPhoneLook: Bool?
  ) -> String? {
    func named(_ s: String?) -> String? {
      guard let t = s?.trimmingCharacters(in: .whitespacesAndNewlines), !t.isEmpty else { return nil }
      return t
    }
    if let s = named(requestStyle) { return s }
    if requestPhoneLook == true { return StylePack.phone.rawValue }
    if let s = named(presetStyle) { return s }
    if presetPhoneLook == true { return StylePack.phone.rawValue }
    return nil
  }

  /// Fail loud on a name this engine does not know. nil (no style asked for)
  /// is fine; an unknown name is a 400 at the generate decode, never a
  /// silent no-op at save time.
  public static func validate(_ name: String?) throws {
    guard let name, !name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
    guard named(name) == nil else { return }
    throw WarmServerError.unknownStyle(name: name, valid: knownNames)
  }

  public func apply(pixels: inout [Float], width: Int, height: Int) {
    switch self {
    case .phone:
      PhoneLook.apply(pixels: &pixels, width: width, height: height)
    case .trixBW:
      Self.applyTrixBW(pixels: &pixels, width: width, height: height)
    }
  }

  // MARK: - trix-bw

  /// Pan-film channel mix: mild red emphasis over rec709 — the classic
  /// filterless panchromatic response (skies darken a touch, skin lifts).
  static let panMix: (r: Float, g: Float, b: Float) = (0.35, 0.55, 0.10)
  static let pushContrast: Float = 1.3
  /// Shoulder start: highlights above this compress instead of clipping —
  /// "held highlights", the Diafine signature.
  static let shoulderStart: Float = 0.82
  static let shoulderStrength: Float = 0.55
  static let minLevelsWindow: Float = 0.05

  static func applyTrixBW(pixels: inout [Float], width: Int, height: Int) {
    let n = width * height
    guard n > 0, pixels.count == n * 3 else { return }

    // ── monochrome via pan mix ──
    var mono = [Float](repeating: 0, count: n)
    for i in 0..<n {
      mono[i] = panMix.r * pixels[i * 3] + panMix.g * pixels[i * 3 + 1] + panMix.b * pixels[i * 3 + 2]
    }

    // ── percentile levels (guarded, as PhoneLook) ──
    let sorted = mono.sorted()
    let lo = sorted[min(n - 1, Int(0.005 * Float(n)))]
    let hi = sorted[min(n - 1, Int(0.995 * Float(n)))]
    let window = hi - lo
    let stretch = window >= minLevelsWindow
    let inv = stretch ? 1 / window : 1

    for i in 0..<n {
      var v = mono[i]
      if stretch { v = (v - lo) * inv }
      // pushed contrast around mid
      v = 0.5 + (v - 0.5) * pushContrast
      // soft shoulder: compress above shoulderStart instead of clipping
      if v > shoulderStart {
        v = shoulderStart + (v - shoulderStart) * shoulderStrength
      }
      v = min(1, max(0, v))
      pixels[i * 3] = v
      pixels[i * 3 + 1] = v
      pixels[i * 3 + 2] = v
    }
  }
}
