// Krea2ModelDetection.swift — Fail-closed detection of Krea-2 model specs and
// directories (WP-E5, FDD §3.5).
//
// F3 (the live silent-substitution bug): `detect(at:)` used to require
// `turbo.safetensors` by name and `resolve(spec:)` fell through to the HF
// Krea-2-Turbo snapshot with no error and no log, so pointing the engine at
// `~/LocalModels/krea2-raw` rendered Turbo under the name Raw. Resolution is
// now: explicit path → declared alias (ONE table, shared with
// `WarmServer.parseModelSpec`) → the four Turbo HF aliases → throw.

import Foundation

public enum Krea2ModelDetection {
  /// Aliases that mean the HF Krea-2-Turbo snapshot, and nothing else. These
  /// four — and only these four — reach `Krea2ModelPaths.turboSnapshot()`.
  public static let turboAliases: Set<String> = ["krea2", "krea-2", "krea-2-turbo", "krea/krea-2-turbo"]

  /// Built-in declared spec → directory table (the installs on this machine).
  /// `~/.comfybox/config.json` `krea2Models` is merged OVER these at server
  /// start (`configureSpecDirectories`).
  public static let defaultSpecDirectories: [String: String] = [
    "krea2-raw": "~/LocalModels/krea2-raw",
    // Kroma v0.2 (lodestones) — a Krea-2 fine-tune shipped as a full turbo
    // checkpoint. The dir holds the Kroma transformer as turbo.safetensors
    // with text_encoder/vae/tokenizer symlinked from the Krea-2 snapshot.
    "kroma-v0.2-turbo": "~/LocalModels/kroma-v0.2",
  ]

  private static let tableLock = NSLock()
  private static var configuredSpecDirectories: [String: String] = defaultSpecDirectories

  /// THE single spec→directory table. `WarmServer.parseModelSpec` consults it
  /// instead of growing a second one.
  public static var specDirectories: [String: String] {
    tableLock.lock(); defer { tableLock.unlock() }
    return configuredSpecDirectories
  }

  /// Seed the table from config. `overrides` are merged over the built-in
  /// defaults (`replace: true` installs exactly `overrides` — used by tests to
  /// restore state).
  public static func configureSpecDirectories(_ overrides: [String: String], replace: Bool = false) {
    tableLock.lock(); defer { tableLock.unlock() }
    if replace {
      configuredSpecDirectories = overrides
    } else {
      configuredSpecDirectories = defaultSpecDirectories.merging(overrides) { _, override in override }
    }
  }

  /// Declared spec → directory (tilde-expanded), or nil when the spec is not
  /// in the table. Case-insensitive on the spec.
  public static func specDirectory(_ spec: String, table: [String: String]? = nil) -> URL? {
    let t = table ?? specDirectories
    let lower = spec.lowercased()
    guard let path = t.first(where: { $0.key.lowercased() == lower })?.value else { return nil }
    return URL(fileURLWithPath: (path as NSString).expandingTildeInPath, isDirectory: true)
  }

  /// Family hint ONLY: routes a spec to `.krea2` in `ModelPool.detectFamily` /
  /// `WarmServer.prepare`. The directory comes from `resolve(spec:)`, which
  /// throws for an alias nobody has declared instead of loading Turbo.
  public static func isKnownKrea2Model(_ spec: String) -> Bool {
    let lower = spec.lowercased()
    if turboAliases.contains(lower) { return true }
    if lower == "krea2-raw" || lower == "krea-2-raw" { return true }
    if lower.contains("krea-2-turbo") { return true }
    if specDirectory(lower) != nil { return true }
    return false
  }

  /// Non-throwing probe for family detection on an already-resolved snapshot.
  public static func isKrea2ModelDirectory(_ url: URL) -> Bool {
    (try? detect(at: url)) != nil
  }

  /// Detect a Krea-2 model root and its variant. Fail-closed:
  /// 1. `raw.safetensors` → `.raw`; `turbo.safetensors` → `.turbo`. Both →
  ///    `ambiguousVariant`. Never guess.
  /// 2. Neither → `model_index.json` `"krea2_variant"` + `"transformer_file"`
  ///    (escape hatch for a third filename, e.g. Comfy-Org's
  ///    `krea2_raw_bf16.safetensors`).
  /// 3. Still nothing → `notAKrea2ModelDirectory`.
  /// The Qwen text-encoder / VAE files must be present as well.
  public static func detect(at url: URL) throws -> Krea2ModelPaths {
    let fm = FileManager.default
    var isDir: ObjCBool = false
    guard fm.fileExists(atPath: url.path, isDirectory: &isDir), isDir.boolValue else {
      throw Krea2ModelPathsError.notAKrea2ModelDirectory(url.path, reason: .notADirectory)
    }

    let rawFile = url.appending(path: Krea2Variant.raw.transformerFilename)
    let turboFile = url.appending(path: Krea2Variant.turbo.transformerFilename)
    let hasRaw = fm.fileExists(atPath: rawFile.path)
    let hasTurbo = fm.fileExists(atPath: turboFile.path)

    let variant: Krea2Variant
    let transformerFile: URL
    switch (hasRaw, hasTurbo) {
    case (true, true):
      throw Krea2ModelPathsError.ambiguousVariant(url)
    case (true, false):
      variant = .raw
      transformerFile = rawFile
    case (false, true):
      variant = .turbo
      transformerFile = turboFile
    case (false, false):
      let indexURL = url.appending(path: "model_index.json")
      guard fm.fileExists(atPath: indexURL.path) else {
        throw Krea2ModelPathsError.notAKrea2ModelDirectory(url.path, reason: .noTransformer)
      }
      (variant, transformerFile) = try readModelIndex(indexURL, root: url)
    }

    // WP-E9: `model_index.json` may name the model dir's VAE (`"vae_file"`);
    // absent → `vae/diffusion_pytorch_model.safetensors`. A declared file
    // that is not there is an invalid index, never a fallback.
    let vaeFile = try readModelIndexVAEFile(url.appending(path: "model_index.json"), root: url)
    let paths = Krea2ModelPaths(root: url, variant: variant, transformerFile: transformerFile, vaeFile: vaeFile)
    guard fm.fileExists(atPath: paths.textEncoderFile.path) else {
      throw Krea2ModelPathsError.notAKrea2ModelDirectory(url.path, reason: .missingTextEncoder)
    }
    guard fm.fileExists(atPath: paths.vaeFile.path) else {
      throw Krea2ModelPathsError.notAKrea2ModelDirectory(url.path, reason: .missingVAE)
    }
    return paths
  }

  /// Optional `"vae_file"` in `model_index.json` (relative to `root`). nil
  /// when there is no index or it does not declare one.
  private static func readModelIndexVAEFile(_ indexURL: URL, root: URL) throws -> URL? {
    guard FileManager.default.fileExists(atPath: indexURL.path),
          let data = try? Data(contentsOf: indexURL),
          let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
          let file = json["vae_file"] as? String
    else { return nil }
    guard !file.isEmpty else {
      throw Krea2ModelPathsError.notAKrea2ModelDirectory(
        root.path, reason: .invalidModelIndex("\"vae_file\" is empty"))
    }
    let url = root.appending(path: file)
    guard FileManager.default.fileExists(atPath: url.path) else {
      throw Krea2ModelPathsError.notAKrea2ModelDirectory(
        root.path, reason: .invalidModelIndex("vae_file \"\(file)\" does not exist in \(root.path)"))
    }
    return url
  }

  private static func readModelIndex(_ indexURL: URL, root: URL) throws -> (Krea2Variant, URL) {
    func invalid(_ detail: String) -> Krea2ModelPathsError {
      .notAKrea2ModelDirectory(root.path, reason: .invalidModelIndex(detail))
    }
    guard let data = try? Data(contentsOf: indexURL),
          let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    else { throw invalid("not a JSON object") }
    guard let variantName = json["krea2_variant"] as? String else {
      throw invalid("missing \"krea2_variant\"")
    }
    guard let variant = Krea2Variant(rawValue: variantName.lowercased()) else {
      throw invalid("unknown krea2_variant \"\(variantName)\" (expected raw|turbo)")
    }
    guard let file = json["transformer_file"] as? String, !file.isEmpty else {
      throw invalid("missing \"transformer_file\"")
    }
    let transformerFile = root.appending(path: file)
    guard FileManager.default.fileExists(atPath: transformerFile.path) else {
      throw invalid("transformer_file \"\(file)\" does not exist in \(root.path)")
    }
    return (variant, transformerFile)
  }

  /// Resolve a spec to model paths. Order:
  /// 1. an existing directory path → `detect(at:)`;
  /// 2. a declared alias in the spec→directory table → `detect(at:)`;
  /// 3. one of the four Turbo aliases → the HF Krea-2-Turbo snapshot (the
  ///    ONLY fallback);
  /// 4. anything else throws `notAKrea2ModelDirectory(reason: .unmappedSpec)`.
  ///
  /// `specDirectories` / `turboSnapshot` are injectable so the resolution
  /// order is testable without `~/.cache` or the live installs.
  public static func resolve(
    spec: String,
    specDirectories table: [String: String]? = nil,
    turboSnapshot: () throws -> Krea2ModelPaths = Krea2ModelPaths.turboSnapshot
  ) throws -> Krea2ModelPaths {
    let expanded = (spec as NSString).expandingTildeInPath
    var isDir: ObjCBool = false
    if FileManager.default.fileExists(atPath: expanded, isDirectory: &isDir), isDir.boolValue {
      return try detect(at: URL(fileURLWithPath: expanded, isDirectory: true))
    }
    if let dir = specDirectory(spec, table: table) {
      return try detect(at: dir)
    }
    if turboAliases.contains(spec.lowercased()) {
      return try turboSnapshot()
    }
    throw Krea2ModelPathsError.notAKrea2ModelDirectory(spec, reason: .unmappedSpec)
  }
}
