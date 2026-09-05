// Krea2VAESelection.swift — VAE selection and in-place reload (WP-E9, FDD §3.9, D16, D17).
//
// Precedence, each recorded: `payload.vae` → preset `vae` (resolved daemon-side
// into `payload.vae`; the engine never expands image presets) → the model
// directory (`model_index.json` `"vae_file"`, else
// `vae/diffusion_pytorch_model.safetensors`). A named VAE that does not exist
// FAILS the render; it never falls back.
//
// The VAE does not join the pool key (D17): a `.krea2` entry is ~22.5 GB
// against a 40 GB budget, so a pool-keyed VAE would turn a 508 MB decoder swap
// into a ~67 s full reload. Instead `Krea2VAESlot` reloads the decoder IN
// PLACE on the one resident `Krea2VAE` — fail-closed (a missing file or an
// unrecognised layout leaves the resident decoder untouched), counted, and
// recorded. One instance serves both `encode` and `decode`, so encoder-side
// selection follows decoder-side automatically (AC-57).

import Foundation
import MLX

/// What decoded (and encoded) a render, and how it was chosen.
public struct Krea2VAESelection: Equatable, Sendable {
  public enum Source: String, Sendable, Codable, Equatable {
    /// `payload.vae` on the request.
    case payload
    /// A preset's `vae` field (daemon-resolved into the payload; reserved for
    /// the record when the daemon reports it, WP-E10/E20).
    case preset
    /// The model directory's VAE — the no-regression default (D16).
    case modelDir = "model_dir"
  }

  public let file: URL
  public let layout: VAELayout
  public let source: Source

  public init(file: URL, layout: VAELayout, source: Source) {
    self.file = file
    self.layout = layout
    self.source = source
  }
}

public enum Krea2VAESelectionError: Error, Equatable, LocalizedError {
  /// The selected VAE file is not on disk. The path is the tilde-expanded
  /// path that was looked at; the source says who named it.
  case vaeNotFound(path: String, source: Krea2VAESelection.Source)

  public var errorDescription: String? {
    switch self {
    case .vaeNotFound(let path, let source):
      return "Krea2 VAE: '\(path)' (selected by \(source.rawValue)) does not exist — the render fails; the model directory's VAE is never substituted"
    }
  }
}

/// The precedence rule as a pure, weight-free function.
public enum Krea2VAESelector {
  /// `requested` is `payload.vae` — either sent directly on the request, or
  /// filled in by `PresetLoRAStack`'s expansion of a named `preset` (#285:
  /// request > preset > model dir, the same precedence `RequestStackResolver`
  /// uses for LoRAs). Absent → the model dir's file (`paths.vaeFile`, which
  /// already honours `model_index.json` `"vae_file"`). Existence is checked
  /// here so a bad name fails BEFORE any pipeline work; an empty string is a
  /// bad selection, not "no selection".
  ///
  /// `fromPreset` says which of the two non-default sources filled
  /// `requested` — by the time it reaches here both have collapsed onto the
  /// same field, so the caller (`WarmServer`, from
  /// `payload.presetVAEApplied`) is what still knows. It only changes the
  /// recorded ``Krea2VAESelection/Source``, never the resolution itself.
  public static func resolve(
    requested: String?, paths: Krea2ModelPaths, fromPreset: Bool = false
  ) throws -> (file: URL, source: Krea2VAESelection.Source) {
    let file: URL
    let source: Krea2VAESelection.Source
    if let requested {
      source = fromPreset ? .preset : .payload
      let trimmed = requested.trimmingCharacters(in: .whitespacesAndNewlines)
      // "" would resolve to the working directory — a bad selection, not "none".
      guard !trimmed.isEmpty else {
        throw Krea2VAESelectionError.vaeNotFound(path: requested, source: source)
      }
      file = URL(fileURLWithPath: (trimmed as NSString).expandingTildeInPath)
    } else {
      file = paths.vaeFile
      source = .modelDir
    }
    try requireRegularFile(file, source: source)
    return (file, source)
  }

  /// A VAE is a regular file on disk — a directory or a missing path is
  /// `vaeNotFound`, naming the path that was looked at.
  static func requireRegularFile(_ file: URL, source: Krea2VAESelection.Source) throws {
    var isDir: ObjCBool = false
    guard FileManager.default.fileExists(atPath: file.path, isDirectory: &isDir), !isDir.boolValue else {
      throw Krea2VAESelectionError.vaeNotFound(path: file.path, source: source)
    }
  }
}

/// Owns the one `Krea2VAE` instance of a pipeline and reloads its weights in
/// place. Never replaces the instance — `vae` is a `let`.
public final class Krea2VAESlot {
  public let vae: Krea2VAE
  /// What is resident now. Always names what decoded (AC-59).
  public private(set) var current: Krea2VAESelection
  /// Incremented on every in-place reload (AC-59).
  public private(set) var reloadCount = 0

  /// The encode path and the decode path — the same object, by construction.
  public var encoder: Krea2VAE { vae }
  public var decoder: Krea2VAE { vae }

  /// Load `file` into a fresh `Krea2VAE`. The layout is sniffed from the keys.
  public init(loading file: URL, layout: VAELayout? = nil, source: Krea2VAESelection.Source) throws {
    try Krea2VAESelector.requireRegularFile(file, source: source)
    let vae = Krea2VAE()
    let used = try Krea2WeightLoader.loadVAE(vae, from: file, layout: layout)
    self.vae = vae
    self.current = Krea2VAESelection(file: file, layout: used, source: source)
  }

  /// Adopt an already-built instance without loading (tests; the pipeline's
  /// own init goes through `init(loading:)`).
  public init(unloaded vae: Krea2VAE, file: URL, layout: VAELayout, source: Krea2VAESelection.Source) {
    self.vae = vae
    self.current = Krea2VAESelection(file: file, layout: layout, source: source)
  }

  /// Make `file` the resident decoder. Returns `true` when weights were
  /// reloaded, `false` when `file` was already resident (no work, but the
  /// recorded `source` is updated to say who asked). Fail-closed: a missing
  /// file, an unrecognised layout, a layout that contradicts `layout:`, an
  /// unmapped key or a shape mismatch all throw BEFORE any weight is touched.
  @discardableResult
  public func ensure(
    file: URL, layout: VAELayout? = nil, source: Krea2VAESelection.Source
  ) throws -> Bool {
    try Krea2VAESelector.requireRegularFile(file, source: source)
    let same = file.standardizedFileURL.path == current.file.standardizedFileURL.path
    if same, layout == nil || layout == current.layout {
      current = Krea2VAESelection(file: current.file, layout: current.layout, source: source)
      return false
    }
    let used = try Krea2WeightLoader.loadVAE(vae, from: file, layout: layout)
    MLX.eval(vae.parameters())
    reloadCount += 1
    current = Krea2VAESelection(file: file, layout: used, source: source)
    return true
  }
}
