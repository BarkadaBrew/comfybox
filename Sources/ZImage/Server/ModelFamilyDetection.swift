// ModelFamilyDetection.swift — GET /v1/model/family (comfybox#359).
//
// The desktop's preset editor can save `custom_model_path` (an on-disk
// directory) without ever naming an engine `model` spec or a
// `checkpoint_family` — `PresetLoRAStack.declaredFamily` then has neither to
// go on, so `/v1/generate {"preset": id}` can never expand the preset (it
// stays a `preset_unresolved_reason: "no_model"` label forever, even though
// the engine could load the checkpoint just fine via Apply/Generate).
//
// Rather than have the desktop re-implement `Krea2ModelDetection`'s
// fail-closed resolution client-side (or worse, guess a family from a
// filename — exactly the F3 bug that detection exists to prevent), this
// route lets it ask the engine what a model spec resolves to.
//
// The route answers TWO questions, and the second is the load-bearing one:
//
//   * `family` / `variant` — the broad, engine-detectable half. The desktop
//     combines it with the PRESET's own declared LoRA roles (the accel/stock
//     split within Krea-2 "raw" depends on whether `loras[]` declares
//     `role: "accel"`, which only the preset document knows) into one of the
//     five `checkpoint_family` policy labels `PresetStore.validate` accepts.
//   * `spec` / `loadable` / `reason` — the canonical engine model spec for
//     the probed value, and whether `/v1/generate` would accept it as
//     `model`. This is what actually fixes the preset: `PresetLoRAStack
//     .decide` returns `no_model` whenever the preset's `model` is empty and
//     the request names none, BEFORE `checkpoint_family` is read, so a
//     backfill that writes only `checkpoint_family` changes nothing at all
//     for a `custom_model_path`-only preset. The desktop writes `model =
//     spec` (and keeps `custom_model_path` for its own Apply path), which
//     takes `decide`'s `asked.isEmpty → expansion.model = presetModel`
//     branch. When `loadable` is false the desktop must FAIL the backfill
//     with `reason` rather than write a label that changes nothing.
//
// The engine deliberately keeps no opinion on `checkpoint_family` here.
//
// SECURITY/SCOPE NOTE (deliberate): `model` may be ANY local path, and the
// route will stat it. That is intentional — the warm server is a
// localhost-trusted process on Todd's Mac, the probe is file existence plus
// a `model_index.json` read inside a model root, it never loads weights,
// never mutates the pool, and never returns file contents. It is safe to
// call once per preset in a "Backfill all" batch without disturbing whatever
// is resident.

import Foundation

/// `GET /v1/model/family?model=<spec>` response body.
struct ModelFamilyDetectionResponse: Encodable, Sendable, Equatable {
  /// Echoed verbatim (untrimmed) — the caller's own request value.
  let model: String
  /// Broad family the engine classifies this spec as: `"krea2"` | `"z-image"`.
  /// nil when unclassifiable.
  let family: String?
  /// The physical/declared variant within that family: krea2 `"turbo"` |
  /// `"raw"`; z-image `"turbo"` | `"base"`. nil when `family` is nil, or the
  /// spec resolves to the family but not to a determinable variant.
  let variant: String?
  /// FIX ROUND 1 — the field that makes this route useful at all.
  ///
  /// The CANONICAL engine spec for the probed value: the declared alias when
  /// the path matches a `Krea2ModelDetection.specDirectory` entry (built-in
  /// or `config.json krea2Models`), otherwise the spec/path itself,
  /// tilde-expanded and standardized. This is what a caller writes into a
  /// preset's `model`.
  ///
  /// Why it matters: `PresetLoRAStack.decide` returns `no_model` whenever the
  /// preset's `model` is empty and the request names none — BEFORE
  /// `checkpoint_family` is consulted. Writing only `checkpoint_family` on a
  /// `custom_model_path`-only preset therefore changes NOTHING. `model` is
  /// the field that flips the preset to expandable.
  let spec: String
  /// Would `POST /v1/generate {"model": spec}` be accepted by the loader?
  /// Determined against `WarmServer.parseModelSpec` + `ModelPool.detectFamily`
  /// / `Krea2ModelDetection.resolve` / `ModelResolution.resolve`, by file
  /// existence only — never a weight load, so a truncated or corrupt
  /// checkpoint still reads as loadable here.
  let loadable: Bool
  /// Why `loadable` is false. nil when it is true.
  let reason: String?
}

enum ModelFamilyDetector {
  /// z-image alias/variant classification. Deliberately independent of both
  /// `ComfyBox/main.swift`'s CLI alias resolution and the
  /// `ComfyBoxModelRegistry` catalog — this route needs neither the CLI
  /// target nor a catalog lookup, just the same aliases they already use.
  private static let baseAliases: Set<String> = ["z-image-base", "zimage-base"]
  private static let turboAliases: Set<String> = ["z-image-turbo", "zimage-turbo", "z-image"]

  /// nil when `spec` does not name z-image at all.
  static func zImageVariant(for spec: String) -> String? {
    let lower = spec.lowercased()
    if baseAliases.contains(lower) { return "base" }
    if turboAliases.contains(lower) { return "turbo" }
    guard lower.contains("z-image") || lower.contains("zimage") else { return nil }
    // An id that names z-image but neither alias exactly (e.g. a HF id like
    // "Tongyi-MAI/Z-Image-Turbo-BF16", or a catalog id like
    // "z-image-base-bf16") — read "base" vs "turbo" off the text itself,
    // same as the alias table above, just not an exact match.
    return lower.contains("base") ? "base" : "turbo"
  }

  /// Detect a spec's family + variant without loading it. Deliberately does
  /// NOT delegate to `PresetLoRAStack.modelFamily`: that function classifies
  /// a `/v1/generate` request's own `model` field, which is always an engine
  /// spec (alias, catalog id, HF id) — never a raw `custom_model_path`
  /// directory, because "the engine never loads from it" there. This route
  /// exists specifically to answer for THAT shape too, so `detectVariant`'s
  /// existing-directory branch (mirroring `resolve(spec:)`'s own precedence)
  /// runs unconditionally, not gated behind a family guess first.
  static func detect(spec: String) -> ModelFamilyDetectionResponse {
    let trimmed = spec.trimmingCharacters(in: .whitespacesAndNewlines)
    let canonical = canonicalize(trimmed)

    if let variant = Krea2ModelDetection.detectVariant(spec: trimmed) {
      return ModelFamilyDetectionResponse(
        model: spec, family: "krea2", variant: variant.rawValue,
        spec: canonical.spec, loadable: canonical.loadable, reason: canonical.reason)
    }
    if Krea2ModelDetection.isKnownKrea2Model(trimmed) {
      // Known by alias/table (or a "krea-2-turbo" substring) but its
      // directory could not be read (e.g. a declared alias whose directory
      // is missing) — still krea2, variant unknown rather than a guess.
      return ModelFamilyDetectionResponse(
        model: spec, family: "krea2", variant: nil,
        spec: canonical.spec, loadable: canonical.loadable, reason: canonical.reason)
    }
    if let variant = zImageVariant(for: trimmed) {
      return ModelFamilyDetectionResponse(
        model: spec, family: "z-image", variant: variant,
        spec: canonical.spec, loadable: canonical.loadable, reason: canonical.reason)
    }
    return ModelFamilyDetectionResponse(
      model: spec, family: nil, variant: nil,
      spec: canonical.spec, loadable: canonical.loadable, reason: canonical.reason)
  }

  /// The canonical engine spec for `trimmed`, and whether the loader would
  /// accept it as `model` on `/v1/generate`.
  ///
  /// This mirrors the real acceptance path — deliberately, step for step, so
  /// it can be checked against it:
  ///
  /// 1. `WarmServer.parseModelSpec` maps a declared krea2 alias to its
  ///    directory and a CivitAI id to its checkpoint file, and passes
  ///    everything else (including a raw path) through untouched.
  /// 2. `ModelPool.detectFamily` routes a krea2 spec to
  ///    `Krea2ModelDetection.resolve` (existing path → declared alias → the
  ///    four Turbo HF aliases → throw), a `.safetensors` file to flux1, and
  ///    anything else through `ModelResolution.resolveOrDefault`, which
  ///    returns ANY existing path as-is and otherwise needs a Hugging Face
  ///    id.
  ///
  /// File existence only. A weight load can still fail afterwards (a
  /// truncated safetensors, a directory missing its text encoder) — this
  /// answers "the engine would accept this spec", not "this render will
  /// succeed".
  static func canonicalize(_ trimmed: String) -> (spec: String, loadable: Bool, reason: String?) {
    guard !trimmed.isEmpty else {
      return ("", false, "no model spec given")
    }

    // 1. Krea-2 resolution — the case the desktop backfill actually hits.
    //    A path that IS a declared alias's directory comes back as the
    //    alias: portable across machines, and what `/health.model_alias`
    //    reports.
    if let alias = Krea2ModelDetection.alias(forSpec: trimmed),
       Krea2ModelDetection.detectVariant(spec: alias) != nil {
      return (alias, true, nil)
    }
    if Krea2ModelDetection.detectVariant(spec: trimmed) != nil {
      // An existing, detectable Krea-2 directory nobody declared an alias
      // for. `Krea2ModelDetection.resolve`'s first branch takes it.
      return (standardizedPath(trimmed) ?? trimmed, true, nil)
    }
    if Krea2ModelDetection.specDirectory(trimmed) != nil {
      return (trimmed, false,
        "'\(trimmed)' is a declared krea2 alias but its directory is not a readable Krea-2 model "
          + "root — the engine would refuse to load it")
    }

    // 2. Path-shaped specs. `ModelResolution.resolve` returns any existing
    //    path unchanged; a missing one throws `modelNotFound` because a path
    //    is never a Hugging Face id.
    if isPathShaped(trimmed) {
      guard let path = standardizedPath(trimmed) else {
        return (trimmed, false, "'\(trimmed)' is not a usable filesystem path")
      }
      var isDir: ObjCBool = false
      guard FileManager.default.fileExists(atPath: path, isDirectory: &isDir) else {
        return (path, false, "'\(path)' does not exist on this machine")
      }
      if isDir.boolValue { return (path, true, nil) }
      guard (path as NSString).pathExtension.lowercased() == "safetensors" else {
        return (path, false,
          "'\(path)' is a file but not a .safetensors checkpoint, and not a model directory")
      }
      return (path, true, nil)
    }

    // 3. Bare specs the engine already knows by name.
    if WarmServer.knownModelSpecs.contains(trimmed) { return (trimmed, true, nil) }
    if let mapped = WarmServer.civitaiCheckpointPaths[trimmed] {
      let expanded = (mapped as NSString).expandingTildeInPath
      guard FileManager.default.fileExists(atPath: expanded) else {
        return (trimmed, false,
          "'\(trimmed)' maps to \(expanded), which does not exist on this machine")
      }
      return (trimmed, true, nil)
    }
    if zImageVariant(for: trimmed) != nil {
      // A z-image alias or an id that names z-image — resolved from the HF
      // cache (or downloaded) by `ModelResolution`, which the engine accepts.
      return (trimmed, true, nil)
    }

    return (trimmed, false,
      "'\(trimmed)' is neither an engine-known model spec nor an existing local path — "
        + "/v1/generate would refuse it")
  }

  /// `~/…`, `/…`, `./…`, `../…` — the shapes `ModelResolution` treats as a
  /// filesystem path rather than a Hugging Face id.
  private static func isPathShaped(_ spec: String) -> Bool {
    spec.hasPrefix("/") || spec.hasPrefix("~") || spec.hasPrefix("./") || spec.hasPrefix("../")
  }

  /// Tilde-expanded, standardized, trailing-slash-free absolute path — the
  /// only spelling `ModelResolution.resolve` can open (it does NOT expand
  /// `~`). nil when the spec does not name an absolute path once expanded.
  private static func standardizedPath(_ spec: String) -> String? {
    let expanded = (spec as NSString).expandingTildeInPath
    guard expanded.hasPrefix("/") else { return nil }
    var out = URL(fileURLWithPath: expanded).standardizedFileURL.path
    while out.count > 1, out.hasSuffix("/") { out.removeLast() }
    return out
  }
}
