import Foundation

// MARK: - DyPE Configuration

/// DyPE (Dynamic Position Extrapolation) method for high-resolution generation.
/// Modifies RoPE frequency bases at inference time to handle resolutions beyond training scale.
public enum DyPEMethod: String, Codable, Sendable {
  /// No extrapolation — use base RoPE frequencies (default behavior).
  case none
  /// NTK-aware frequency scaling — adjusts theta to spread frequencies.
  case ntk
  /// YaRN — NTK + linear interpolation blend with damping. Full DyPE pipeline.
  case yarn
}

/// Configuration for DyPE high-resolution generation.
public struct DyPEConfig: Codable, Sendable {
  /// Whether DyPE is enabled. When false, all RoPE behavior is vanilla.
  public var enabled: Bool

  /// The extrapolation method. `.ntk` is simpler and faster; `.yarn` adds damping for better quality.
  public var method: DyPEMethod

  /// Base resolution the model was trained at (pixels). Used to compute the scale factor.
  /// Default: 1024 (Z-Image-Turbo training resolution).
  public var baseResolution: Int

  /// YaRN damping range lower bound. Controls which frequency bands get interpolated vs extrapolated.
  /// Only used when method == .yarn. Default: 1.0
  public var beta0: Float

  /// YaRN damping range upper bound. Default: 32.0
  public var beta1: Float

  /// YaRN timestep damping range lower bound. Default: 1.0
  public var gamma0: Float

  /// YaRN timestep damping range upper bound. Default: 32.0
  public var gamma1: Float

  public init(
    enabled: Bool = false,
    method: DyPEMethod = .ntk,
    baseResolution: Int = 1024,
    beta0: Float = 1.0,
    beta1: Float = 32.0,
    gamma0: Float = 1.0,
    gamma1: Float = 32.0
  ) {
    self.enabled = enabled
    self.method = method
    self.baseResolution = baseResolution
    self.beta0 = beta0
    self.beta1 = beta1
    self.gamma0 = gamma0
    self.gamma1 = gamma1
  }

  /// Default config with DyPE disabled.
  public static let disabled = DyPEConfig()

  /// Default NTK-only config.
  public static let ntk = DyPEConfig(enabled: true, method: .ntk)

  /// Default YaRN config (full DyPE pipeline).
  public static let yarn = DyPEConfig(enabled: true, method: .yarn)
}

// MARK: - Model Configs

public struct ZImageTransformerConfig: Decodable {
  public let inChannels: Int
  public let dim: Int
  public let nLayers: Int
  public let nRefinerLayers: Int
  public let nHeads: Int
  public let nKVHeads: Int
  public let normEps: Float
  public let qkNorm: Bool
  public let capFeatDim: Int
  public let ropeTheta: Float
  public let tScale: Float
  public let axesDims: [Int]
  public let axesLens: [Int]

  enum CodingKeys: String, CodingKey {
    case inChannels = "in_channels"
    case dim
    case nLayers = "n_layers"
    case nRefinerLayers = "n_refiner_layers"
    case nHeads = "n_heads"
    case nKVHeads = "n_kv_heads"
    case normEps = "norm_eps"
    case qkNorm = "qk_norm"
    case capFeatDim = "cap_feat_dim"
    case ropeTheta = "rope_theta"
    case tScale = "t_scale"
    case axesDims = "axes_dims"
    case axesLens = "axes_lens"
  }
}

public struct ZImageVAEConfig: Decodable {
  public let blockOutChannels: [Int]
  public let latentChannels: Int
  public let scalingFactor: Float
  public let shiftFactor: Float
  public let sampleSize: Int
  public let inChannels: Int
  public let outChannels: Int
  public let layersPerBlock: Int
  public let normNumGroups: Int
  public let midBlockAddAttention: Bool
  public let usePostQuantConv: Bool?
  public let useQuantConv: Bool?

  enum CodingKeys: String, CodingKey {
    case blockOutChannels = "block_out_channels"
    case latentChannels = "latent_channels"
    case scalingFactor = "scaling_factor"
    case shiftFactor = "shift_factor"
    case sampleSize = "sample_size"
    case inChannels = "in_channels"
    case outChannels = "out_channels"
    case layersPerBlock = "layers_per_block"
    case normNumGroups = "norm_num_groups"
    case midBlockAddAttention = "mid_block_add_attention"
    case usePostQuantConv = "use_post_quant_conv"
    case useQuantConv = "use_quant_conv"
  }

  public var vaeScaleFactor: Int {
    max(1, 1 << max(0, blockOutChannels.count - 1))
  }

  public var latentDivisor: Int {
    vaeScaleFactor  // 8 for Z-Image-Turbo (4 downsampling stages with factor 2 each)
  }
}

public struct ZImageSchedulerConfig: Decodable {
  /// Which discrete sigma table the model is sampled on — ComfyUI's
  /// `ModelSampling*` class, which decides what the table-backed schedules
  /// (`beta`, `beta57`) index and where `karras`/`exponential` take their bounds.
  ///
  /// - `.discreteFlow`: `ModelSamplingDiscreteFlow` — `numTrainTimesteps` entries
  ///   of `σ = shift·t / (1 + (shift − 1)·t)`, built from this config's **linear**
  ///   `shift`. Every family that decodes a `scheduler_config.json` (Z-Image)
  ///   is this, and needs no `mu` for those schedules.
  /// - `.flux(tableSize:)`: `ModelSamplingFlux` — `tableSize` entries of
  ///   `σ = e^mu / (e^mu + 1/t − 1)`, built from the render's **`mu`** (a
  ///   log-shift), which `SchedulerFactory` therefore requires. Krea 2 is this
  ///   (FDD-krea2-raw-recipe Addendum A.1: ComfyUI registers it
  ///   `ModelType.FLUX` → `ModelSamplingFlux(shift=1.15, timesteps=10000)`, and
  ///   `shift` on the wire IS mu). `config.shift` is not consulted for the table.
  public enum ModelSampling: Equatable, Sendable {
    case discreteFlow
    case flux(tableSize: Int)
  }

  public let numTrainTimesteps: Int
  public let shift: Float
  public let useDynamicShifting: Bool
  public let baseShift: Float?
  public let maxShift: Float?
  public let baseImageSeqLen: Int?
  public let maxImageSeqLen: Int?
  /// Not a `scheduler_config.json` key: decoded configs are always
  /// `.discreteFlow`; only `Krea2Sampling.schedulerConfig()` sets `.flux`.
  public let modelSampling: ModelSampling

  /// Memberwise init. Model families that ship a `scheduler_config.json`
  /// decode it; families without one (Krea 2) construct it directly —
  /// see `Krea2Sampling.schedulerConfig()`.
  public init(
    numTrainTimesteps: Int,
    shift: Float,
    useDynamicShifting: Bool,
    baseShift: Float? = nil,
    maxShift: Float? = nil,
    baseImageSeqLen: Int? = nil,
    maxImageSeqLen: Int? = nil,
    modelSampling: ModelSampling = .discreteFlow
  ) {
    self.numTrainTimesteps = numTrainTimesteps
    self.shift = shift
    self.useDynamicShifting = useDynamicShifting
    self.baseShift = baseShift
    self.maxShift = maxShift
    self.baseImageSeqLen = baseImageSeqLen
    self.maxImageSeqLen = maxImageSeqLen
    self.modelSampling = modelSampling
  }

  /// comfybox#154 — ComfyUI's `ModelSamplingAuraFlow` node, as a value.
  ///
  /// The node (`comfy_extras/nodes_model_advanced.py:148`, a `ModelSamplingSD3`
  /// subclass) patches the model's whole `model_sampling` object with
  /// `ModelSamplingDiscreteFlow.set_parameters(shift:)`, so from the sampler's
  /// point of view the model simply HAS a different shift for that run. This is
  /// that patch: `nil` returns `self` untouched — the caller's grid is
  /// byte-identical to what it was before #154 — and a value returns a copy
  /// whose `shift` is the caller's and whose `useDynamicShifting` is **false**.
  ///
  /// **Precedence (documented, and pinned by `ModelSamplingShiftTests`):** an
  /// explicit shift REPLACES the resolution-dependent `mu` dynamic shift for
  /// that request; the two are never composed. That is what the node does —
  /// ComfyUI has no path where `ModelSamplingAuraFlow` and `ModelSamplingFlux`
  /// both apply — and composing them would silently double-warp a grid whose
  /// mu the caller cannot see.
  ///
  /// Everything else (`numTrainTimesteps`, the `base/max_shift` seq-len ramp
  /// that produced `mu`, `modelSampling`) is carried over, so the schedules
  /// that index the model's discrete sigma table (`simple`, `beta`, `beta57`,
  /// and the `karras`/`exponential` bounds) pick the new shift up for free —
  /// as they do in ComfyUI, where they read the patched `model_sampling`.
  ///
  /// Krea 2 is NOT on this path: its `modelSampling` is `.flux`, whose table is
  /// built from `mu`, and on that family the wire's `shift` IS mu
  /// (FDD-krea2-raw-recipe Addendum A.1). `Krea2Pipeline` never calls this.
  public func applyingExplicitShift(_ explicitShift: Float?) -> ZImageSchedulerConfig {
    guard let explicitShift else { return self }
    return ZImageSchedulerConfig(
      numTrainTimesteps: numTrainTimesteps,
      shift: explicitShift,
      useDynamicShifting: false,
      baseShift: baseShift,
      maxShift: maxShift,
      baseImageSeqLen: baseImageSeqLen,
      maxImageSeqLen: maxImageSeqLen,
      modelSampling: modelSampling
    )
  }

  public init(from decoder: Decoder) throws {
    let c = try decoder.container(keyedBy: CodingKeys.self)
    self.init(
      numTrainTimesteps: try c.decode(Int.self, forKey: .numTrainTimesteps),
      shift: try c.decode(Float.self, forKey: .shift),
      useDynamicShifting: try c.decode(Bool.self, forKey: .useDynamicShifting),
      baseShift: try c.decodeIfPresent(Float.self, forKey: .baseShift),
      maxShift: try c.decodeIfPresent(Float.self, forKey: .maxShift),
      baseImageSeqLen: try c.decodeIfPresent(Int.self, forKey: .baseImageSeqLen),
      maxImageSeqLen: try c.decodeIfPresent(Int.self, forKey: .maxImageSeqLen),
      modelSampling: .discreteFlow
    )
  }

  enum CodingKeys: String, CodingKey {
    case numTrainTimesteps = "num_train_timesteps"
    case shift
    case useDynamicShifting = "use_dynamic_shifting"
    case baseShift = "base_shift"
    case maxShift = "max_shift"
    case baseImageSeqLen = "base_image_seq_len"
    case maxImageSeqLen = "max_image_seq_len"
  }
}

public struct ZImageTextEncoderConfig: Decodable {
  public let hiddenSize: Int
  public let numHiddenLayers: Int
  public let numAttentionHeads: Int
  public let numKeyValueHeads: Int
  public let intermediateSize: Int
  public let maxPositionEmbeddings: Int
  public let ropeTheta: Float
  public let vocabSize: Int
  public let rmsNormEps: Float
  public let headDim: Int

  enum CodingKeys: String, CodingKey {
    case hiddenSize = "hidden_size"
    case numHiddenLayers = "num_hidden_layers"
    case numAttentionHeads = "num_attention_heads"
    case numKeyValueHeads = "num_key_value_heads"
    case intermediateSize = "intermediate_size"
    case maxPositionEmbeddings = "max_position_embeddings"
    case ropeTheta = "rope_theta"
    case vocabSize = "vocab_size"
    case rmsNormEps = "rms_norm_eps"
    case headDim = "head_dim"
  }
}

public struct ZImageModelConfigs {
  public let transformer: ZImageTransformerConfig
  public let vae: ZImageVAEConfig
  public let scheduler: ZImageSchedulerConfig
  public let textEncoder: ZImageTextEncoderConfig

  public static func load(from snapshot: URL, textEncoderDirectory: URL? = nil) throws -> ZImageModelConfigs {
    let decoder = JSONDecoder()
    func loadJSON<T: Decodable>(_ relativePath: String, as type: T.Type) throws -> T {
      let url = snapshot.appending(path: relativePath)
      let data = try Data(contentsOf: url)
      return try decoder.decode(T.self, from: data)
    }

    let selectedTextEncoderDirectory = textEncoderDirectory
      ?? ZImageFiles.resolveTextEncoderSelection(at: snapshot, overridePath: nil, environment: [:]).directory

    let transformer: ZImageTransformerConfig
    if FileManager.default.fileExists(atPath: snapshot.appending(path: ZImageFiles.transformerConfig).path) {
      transformer = try loadJSON(ZImageFiles.transformerConfig, as: ZImageTransformerConfig.self)
    } else {
      transformer = try inferTransformerConfig(from: snapshot)
    }

    let vae: ZImageVAEConfig
    if FileManager.default.fileExists(atPath: snapshot.appending(path: ZImageFiles.vaeConfig).path) {
      vae = try loadJSON(ZImageFiles.vaeConfig, as: ZImageVAEConfig.self)
    } else {
      vae = try inferVAEConfig(from: snapshot)
    }

    let scheduler: ZImageSchedulerConfig
    if FileManager.default.fileExists(atPath: snapshot.appending(path: ZImageFiles.schedulerConfig).path) {
      scheduler = try loadJSON(ZImageFiles.schedulerConfig, as: ZImageSchedulerConfig.self)
    } else {
      scheduler = defaultSchedulerConfig
    }

    let textEncoderConfigURL = selectedTextEncoderDirectory.appendingPathComponent("config.json")
    let textEncoder: ZImageTextEncoderConfig
    if FileManager.default.fileExists(atPath: textEncoderConfigURL.path) {
      let data = try Data(contentsOf: textEncoderConfigURL)
      textEncoder = try decoder.decode(ZImageTextEncoderConfig.self, from: data)
    } else {
      textEncoder = try inferTextEncoderConfig(from: selectedTextEncoderDirectory)
    }

    return ZImageModelConfigs(transformer: transformer, vae: vae, scheduler: scheduler, textEncoder: textEncoder)
  }

  static var defaultSchedulerConfig: ZImageSchedulerConfig {
    ZImageSchedulerConfig(
      numTrainTimesteps: 1000,
      shift: 3.0,
      useDynamicShifting: false,
      baseShift: nil,
      maxShift: nil,
      baseImageSeqLen: nil,
      maxImageSeqLen: nil
    )
  }

  static var defaultVAEConfig: ZImageVAEConfig {
    ZImageVAEConfig(
      blockOutChannels: [128, 256, 512, 512],
      latentChannels: 16,
      scalingFactor: 0.3611,
      shiftFactor: 0.1159,
      sampleSize: 1024,
      inChannels: 3,
      outChannels: 3,
      layersPerBlock: 2,
      normNumGroups: 32,
      midBlockAddAttention: true,
      usePostQuantConv: false,
      useQuantConv: false
    )
  }

  static func inferTransformerConfig(from snapshot: URL) throws -> ZImageTransformerConfig {
    let shapes = try loadTensorShapes(from: ZImageFiles.resolveWeightFiles(in: snapshot.appendingPathComponent("transformer"), componentName: "transformer"))
    guard let config = inferTransformerConfig(fromTensorShapes: shapes) else {
      throw CocoaError(.fileReadCorruptFile)
    }
    return config
  }

  static func inferTransformerConfig(fromTensorShapes shapes: [String: [Int]]) -> ZImageTransformerConfig? {
    let layerCount = maxIndex(prefix: "layers", in: shapes.keys).map { $0 + 1 } ?? 0
    let noiseRefinerCount = maxIndex(prefix: "noise_refiner", in: shapes.keys).map { $0 + 1 } ?? 0
    let contextRefinerCount = maxIndex(prefix: "context_refiner", in: shapes.keys).map { $0 + 1 } ?? 0
    let refinerCount = max(noiseRefinerCount, contextRefinerCount)

    guard let qShape = shapes["layers.0.attention.to_q.weight"] ?? shapes["noise_refiner.0.attention.to_q.weight"],
          qShape.count == 2,
          let normShape = shapes["layers.0.attention.norm_q.weight"] ?? shapes["noise_refiner.0.attention.norm_q.weight"],
          let headDim = normShape.first,
          headDim > 0 else {
      return nil
    }

    let dim = qShape[0]
    let nHeads = max(1, dim / headDim)
    let nKVHeads: Int
    if let kShape = shapes["layers.0.attention.to_k.weight"] ?? shapes["noise_refiner.0.attention.to_k.weight"], kShape.count == 2 {
      nKVHeads = max(1, kShape[0] / headDim)
    } else {
      nKVHeads = nHeads
    }

    let capFeatDim = (shapes["cap_embedder.1.weight"]?.count == 2 ? shapes["cap_embedder.1.weight"]?[1] : nil) ?? 2560
    let patchVolume = (shapes["all_x_embedder.2-1.weight"]?.count == 2 ? shapes["all_x_embedder.2-1.weight"]?[1] : nil) ?? 64
    let inChannels = max(1, patchVolume / 4)

    return ZImageTransformerConfig(
      inChannels: inChannels,
      dim: dim,
      nLayers: layerCount,
      nRefinerLayers: refinerCount,
      nHeads: nHeads,
      nKVHeads: nKVHeads,
      normEps: 1e-5,
      qkNorm: shapes.keys.contains("layers.0.attention.norm_q.weight") || shapes.keys.contains("noise_refiner.0.attention.norm_q.weight"),
      capFeatDim: capFeatDim,
      ropeTheta: 256.0,
      tScale: 1000.0,
      axesDims: [32, 48, 48],
      axesLens: [1536, 512, 512]
    )
  }

  static func inferTextEncoderConfig(from directory: URL) throws -> ZImageTextEncoderConfig {
    let shapes = try loadTensorShapes(from: ZImageFiles.resolveWeightFiles(in: directory, componentName: "text_encoder"))
    guard let config = inferTextEncoderConfig(fromTensorShapes: shapes) else {
      throw CocoaError(.fileReadCorruptFile)
    }
    return config
  }

  static func inferTextEncoderConfig(fromTensorShapes shapes: [String: [Int]]) -> ZImageTextEncoderConfig? {
    guard let embedShape = shapes["model.embed_tokens.weight"], embedShape.count == 2 else {
      return nil
    }
    let hiddenSize = embedShape[1]
    let vocabSize = embedShape[0]
    let numHiddenLayers = (maxIndex(prefix: "model.layers", in: shapes.keys).map { $0 + 1 }) ?? 0

    guard let qShape = shapes["model.layers.0.self_attn.q_proj.weight"], qShape.count == 2,
          let kShape = shapes["model.layers.0.self_attn.k_proj.weight"], kShape.count == 2 else {
      return nil
    }

    let headDim = (shapes["model.layers.0.self_attn.q_norm.weight"]?.first)
      ?? (shapes["model.layers.0.self_attn.k_norm.weight"]?.first)
      ?? 128
    let numAttentionHeads = max(1, qShape[0] / headDim)
    let numKeyValueHeads = max(1, kShape[0] / headDim)
    let intermediateSize = (shapes["model.layers.0.mlp.gate_proj.weight"]?.first)
      ?? (shapes["model.layers.0.mlp.up_proj.weight"]?.first)
      ?? hiddenSize * 4

    return ZImageTextEncoderConfig(
      hiddenSize: hiddenSize,
      numHiddenLayers: numHiddenLayers,
      numAttentionHeads: numAttentionHeads,
      numKeyValueHeads: numKeyValueHeads,
      intermediateSize: intermediateSize,
      maxPositionEmbeddings: 40960,
      ropeTheta: 1_000_000,
      vocabSize: vocabSize,
      rmsNormEps: 1e-6,
      headDim: headDim
    )
  }

  static func inferVAEConfig(from snapshot: URL) throws -> ZImageVAEConfig {
    _ = snapshot
    // We do not have a reliable VAE config inference path yet. Keep local checkpoints
    // working by using the known Z-Image-Turbo defaults until a real shape-based
    // inference implementation is added.
    return defaultVAEConfig
  }

  private static func loadTensorShapes(from files: [URL]) throws -> [String: [Int]] {
    var shapes: [String: [Int]] = [:]
    for file in files {
      let reader = try SafeTensorsReader(fileURL: file)
      for metadata in reader.allMetadata() {
        shapes[metadata.name] = metadata.shape
      }
    }
    return shapes
  }

  private static func maxIndex(prefix: String, in keys: Dictionary<String, [Int]>.Keys) -> Int? {
    let prefixComponents = prefix.split(separator: ".").map(String.init)
    return keys.compactMap { key in
      let components = key.split(separator: ".").map(String.init)
      guard components.count > prefixComponents.count else { return nil }
      guard Array(components.prefix(prefixComponents.count)) == prefixComponents else { return nil }
      return Int(components[prefixComponents.count])
    }.max()
  }
}
