import Foundation
import Logging
import MLX
import MLXNN

/// Manages Mixture-of-Experts (MoE) expert loading and threshold switching
/// for the Wan 2.2 I2V-A14B model.
///
/// The I2V-A14B uses two transformer experts:
/// - **High-noise expert**: Used when timestep >= boundary (early denoising steps)
/// - **Low-noise expert**: Used when timestep < boundary (later denoising steps)
///
/// ## Switching Strategy
/// ```
/// boundary = 0.9 * 1000 = 900
/// model = high_noise if t >= 900 else low_noise
/// ```
///
/// ## Memory Management
/// - Lazy loading: only one expert is in GPU memory at a time
/// - When switching, the old expert is released before loading the new one
/// - High-noise expert is loaded first (~27GB BF16)
/// - One swap per generation (high->low), not per step
public final class WanMoEManager {

  // MARK: - Types

  /// Which expert is currently loaded in memory.
  public enum ActiveExpert: String {
    case highNoise = "high_noise_model"
    case lowNoise = "low_noise_model"
    case none = "none"
  }

  // MARK: - Properties

  /// The currently active transformer model.
  public private(set) var activeModel: WanTransformer3D?

  /// Which expert is currently loaded.
  public private(set) var activeExpert: ActiveExpert = .none

  /// Base directory containing both expert subdirectories.
  public let modelDir: URL

  /// Timestep boundary for switching (default 900 = 0.9 * 1000).
  public let boundary: Float

  /// Logger.
  private let logger: Logger

  /// Whether to use lazy loading (unload one before loading other).
  public let lazyLoading: Bool

  /// Transformer configuration (shared between both experts).
  public let config: WanTransformerConfig

  /// LoRA weights to apply (path, strength pairs).
  private var loraEntries: [(path: String, strength: Float)] = []

  // MARK: - Init

  /// Creates a MoE manager.
  ///
  /// - Parameters:
  ///   - modelDir: Directory containing `high_noise_model/` and `low_noise_model/` subdirs.
  ///   - boundary: Switching threshold (default 0.9, multiplied by numTrainTimesteps).
  ///   - numTrainTimesteps: Training timesteps for boundary calculation (default 1000).
  ///   - lazyLoading: Whether to unload before loading (default true).
  ///   - config: Transformer configuration.
  ///   - logger: Logger.
  public init(
    modelDir: URL,
    boundary: Float = 0.9,
    numTrainTimesteps: Int = 1000,
    lazyLoading: Bool = true,
    config: WanTransformerConfig = .i2vA14B,
    logger: Logger
  ) {
    self.modelDir = modelDir
    self.boundary = boundary * Float(numTrainTimesteps)
    self.lazyLoading = lazyLoading
    self.config = config
    self.logger = logger
  }

  // MARK: - LoRA Registration

  /// Registers a LoRA to apply to both experts during loading.
  public func registerLoRA(path: String, strength: Float = 1.0) {
    loraEntries.append((path: path, strength: strength))
    logger.info("Registered LoRA: \(path) (strength=\(strength))")
  }

  // MARK: - Expert Loading

  /// Loads the high-noise expert.
  public func loadHighNoiseExpert() throws {
    try loadExpert(.highNoise)
  }

  /// Loads the low-noise expert.
  public func loadLowNoiseExpert() throws {
    try loadExpert(.lowNoise)
  }

  /// Loads the specified expert, optionally unloading the current one first.
  private func loadExpert(_ expert: ActiveExpert) throws {
    guard expert != .none else { return }

    if lazyLoading && activeExpert != .none && activeExpert != expert {
      logger.info("Unloading \(activeExpert.rawValue) before loading \(expert.rawValue)")
      unloadCurrentExpert()
    }

    let subdir = modelDir.appendingPathComponent(expert.rawValue)
    logger.info("Loading \(expert.rawValue) from \(subdir.path)")

    let transformer = WanTransformer3D(config: config)

    try WanTransformerWeightLoader.loadTransformerWeights(
      into: transformer,
      from: subdir,
      dtype: .bfloat16,
      logger: logger
    )

    // Apply LoRAs
    for lora in loraEntries {
      try applyLoRA(to: transformer, path: lora.path, strength: lora.strength)
    }

    // Ensure weights are evaluated
    eval(transformer.parameters())

    activeModel = transformer
    activeExpert = expert
    logger.info("\(expert.rawValue) loaded successfully")

    // Diagnostic: verify critical weights after loading
    dumpWeightDiagnostics(transformer)
  }

  /// Dumps mean/std of critical weights to verify correct loading.
  ///
  /// Expected values (from Python model, high_noise_model):
  /// - patch_embedding.weight mean ≈ small, std ≈ 0.01-0.03
  /// - blocks.0.modulation mean ≈ 0, std ≈ 0.01
  /// - blocks.0.self_attn.q.weight std ≈ 0.01-0.02
  /// - head.head.weight should be near-zero (initialized to zeros in Python)
  private func dumpWeightDiagnostics(_ model: WanTransformer3D) {
    let params = model.parameters().flattened()
    var paramDict: [String: MLXArray] = [:]
    for (key, value) in params {
      paramDict[key] = value
    }

    let diagnosticKeys = [
      "patch_embedding.weight",
      "patch_embedding.bias",
      "blocks.0.modulation",
      "blocks.0.self_attn.q.weight",
      "blocks.0.self_attn.q.bias",
      "blocks.0.cross_attn.q.weight",
      "blocks.0.ffn.layers.0.weight",
      "blocks.19.modulation",
      "blocks.39.modulation",
      "head.head.weight",
      "head.head.bias",
      "head.modulation",
      "text_embedding.layers.0.weight",
      "time_embedding.layers.0.weight",
      "time_projection.layers.1.weight",
    ]

    logger.info("[WEIGHT-DIAG] === Weight Diagnostics ===")
    for key in diagnosticKeys {
      if let w = paramDict[key] {
        let wf = w.asType(.float32)
        let mean = wf.mean().item(Float.self)
        let std = MLX.sqrt(wf.variance()).item(Float.self)
        let minVal = wf.min().item(Float.self)
        let maxVal = wf.max().item(Float.self)
        logger.info("[WEIGHT-DIAG] \(key): shape=\(w.shape), dtype=\(w.dtype), mean=\(mean), std=\(std), min=\(minVal), max=\(maxVal)")
      } else {
        logger.warning("[WEIGHT-DIAG] \(key): NOT FOUND in model parameters")
      }
    }
    logger.info("[WEIGHT-DIAG] Total parameters: \(paramDict.count)")
  }

  /// Unloads the current expert to free memory.
  private func unloadCurrentExpert() {
    activeModel = nil
    activeExpert = .none
    eval(MLXArray(0))
  }

  // MARK: - Expert Selection

  /// Returns the appropriate expert for the given timestep, loading it if necessary.
  public func model(forTimestep timestep: Float) throws -> WanTransformer3D {
    let required: ActiveExpert = timestep >= boundary ? .highNoise : .lowNoise

    if required != activeExpert {
      logger.info("MoE switch: \(activeExpert.rawValue) -> \(required.rawValue) at t=\(timestep)")
      try loadExpert(required)
    }

    guard let model = activeModel else {
      throw MoEError.noActiveExpert
    }

    return model
  }

  /// Returns the guide scale for the given timestep.
  public func guideScale(forTimestep timestep: Float, scales: (Float, Float)) -> Float {
    return timestep >= boundary ? scales.1 : scales.0
  }

  // MARK: - LoRA Application

  /// Applies a LoRA to a transformer model using merge-on-load.
  ///
  /// Pattern:
  /// 1. Load LoRA safetensors
  /// 2. Find matching lora_A.weight / lora_B.weight pairs
  /// 3. Skip audio-related keys
  /// 4. Compute delta: matmul(B, A) * strength
  /// 5. Add delta to base weight
  private func applyLoRA(
    to transformer: WanTransformer3D,
    path: String,
    strength: Float
  ) throws {
    logger.info("Applying LoRA: \(path) (strength=\(strength))")

    let url = URL(fileURLWithPath: path)
    let reader = try SafeTensorsReader(fileURL: url)
    let allKeys = reader.tensorNames

    // Find lora_A/lora_B pairs
    let loraAKeys = allKeys.filter { $0.hasSuffix("lora_A.weight") }
    var applied = 0
    var skipped = 0

    // Get current model parameters as a dictionary for lookup
    let flatParams = transformer.parameters().flattened()
    var currentParams: [String: MLXArray] = [:]
    currentParams.reserveCapacity(flatParams.count)
    for (key, value) in flatParams {
      currentParams[key] = value
    }

    for aKey in loraAKeys {
      let bKey = aKey.replacingOccurrences(of: "lora_A.weight", with: "lora_B.weight")
      guard allKeys.contains(bKey) else {
        logger.warning("  Missing lora_B for \(aKey)")
        continue
      }

      // Skip audio-related keys
      let baseKey = aKey.replacingOccurrences(of: ".lora_A.weight", with: "")
      if baseKey.contains("audio_") || baseKey.contains("av_ca_") ||
         baseKey.contains("video_to_audio_attn") || baseKey.contains("audio_to_video_attn") {
        skipped += 1
        continue
      }

      // Load A and B matrices
      let loraA = try reader.tensor(named: aKey).asType(.bfloat16)
      let loraB = try reader.tensor(named: bKey).asType(.bfloat16)

      // Compute delta: matmul(B, A) * strength
      let delta = MLX.matmul(loraB, loraA) * MLXArray(strength)

      // Find the corresponding weight in the transformer
      let weightKey = baseKey + ".weight"

      if let currentWeight = currentParams[weightKey] {
        let newWeight = currentWeight + delta
        let params = ModuleParameters.unflattened([(weightKey, newWeight)])
        try? transformer.update(parameters: params, verify: [])
        applied += 1
      } else {
        logger.debug("  LoRA key not found in model: \(weightKey)")
      }
    }

    logger.info("  LoRA applied: \(applied) weights modified, \(skipped) audio keys skipped")
  }

  // MARK: - Memory Info

  /// Reports memory usage of the currently loaded expert.
  public var memoryInfo: String {
    guard activeModel != nil else {
      return "No expert loaded"
    }
    let params = activeModel!.parameters().flattened()
    let totalBytes = params.reduce(0) { $0 + $1.1.nbytes }
    let gb = Double(totalBytes) / (1024 * 1024 * 1024)
    return "\(activeExpert.rawValue): \(String(format: "%.1f", gb)) GB"
  }

  // MARK: - Errors

  public enum MoEError: Error, CustomStringConvertible {
    case noActiveExpert
    case expertLoadFailed(String)

    public var description: String {
      switch self {
      case .noActiveExpert:
        return "No transformer expert is currently loaded"
      case .expertLoadFailed(let reason):
        return "Failed to load expert: \(reason)"
      }
    }
  }
}
