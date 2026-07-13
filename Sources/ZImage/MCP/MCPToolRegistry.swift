// MCPToolRegistry.swift — Tool definitions with JSON Schema parameter schemas
//
// Registers all 11 MCP tools from the ComfyBox MCP server design doc.
// Each tool has: name, description, inputSchema (JSON Schema object).
// Tools are compiled statically — no runtime registration.

import Foundation

/// Static catalog of all MCP tool definitions for ComfyBox.
public enum MCPToolRegistry {

  /// All registered tool definitions.
  public static let tools: [MCPToolDefinition] = [
    generateImage,
    swapLoras,
    listModels,
    listStyles,
    serverHealth,
    queueStatus,
    clearQueue,
    listLoras,
    shutdownServer,
    systemStats,
    applyStyle,
    loraLibrary,
    loraScan,
    loraQuarantine,
    loadModel,
    switchModel,
    modelPool,
    unloadModel,
    generateVideo,
    videoStatus,
    upscale,
    enhancePrompt,
    listCharacters,
    listPresets,
    importLegacyPresets,
    queueList,
    interruptRender,
    cancelJob,
    nearlineList,
    nearlineScan,
    nearlineStage,
    nearlineEvict,
  ]

  // MARK: - Tool Definitions

  static let generateImage = MCPToolDefinition(
    name: "generate_image",
    description: "Generate an image from a text prompt using the loaded model. Supports text-to-image and img2img. Returns the output file path and render duration.",
    inputSchema: [
      "type": "object",
      "properties": [
        "prompt": [
          "type": "string",
          "description": "Text prompt describing the desired image.",
        ] as [String: Any],
        "negative_prompt": [
          "type": "string",
          "description": "Negative prompt — concepts to avoid. Only effective on non-distilled models (Z-Image Base, Flux 2 Klein Base, FIBO).",
        ] as [String: Any],
        "width": [
          "type": "integer",
          "description": "Image width in pixels. Must be divisible by 16. Default: model-dependent (typically 1024).",
        ] as [String: Any],
        "height": [
          "type": "integer",
          "description": "Image height in pixels. Must be divisible by 16. Default: model-dependent (typically 1024).",
        ] as [String: Any],
        "steps": [
          "type": "integer",
          "description": "Number of denoising steps. More steps = higher quality but slower. Distilled models (Turbo): 4-9. Base models: 20-50.",
        ] as [String: Any],
        "guidance": [
          "type": "number",
          "description": "CFG guidance scale. Higher values = stronger prompt adherence. 0.0 for distilled models; 3.5-7.0 for base models.",
        ] as [String: Any],
        "seed": [
          "type": "integer",
          "description": "Random seed for reproducibility. Omit for random.",
        ] as [String: Any],
        "output_path": [
          "type": "string",
          "description": "Output file path. Must be within the server's allowed output directory. Omit to write to a temp file.",
        ] as [String: Any],
        "scheduler": [
          "type": "string",
          "description": "Sampler/scheduler algorithm. Options: euler (default), heun, res_2s, ddim, dpmpp_2m.",
        ] as [String: Any],
        "sigma_schedule": [
          "type": "string",
          "description": "Sigma noise schedule. Options: flow (default), beta, beta57, linear.",
        ] as [String: Any],
        "image_path": [
          "type": "string",
          "description": "Source image path for img2img. The loaded model must support img2img.",
        ] as [String: Any],
        "image_strength": [
          "type": "number",
          "description": "Img2img denoise strength (0.0-1.0). 1.0 = full txt2img, 0.5 = preserve composition. Default: 0.7.",
        ] as [String: Any],
        "content_mode": [
          "type": "string",
          "description": "neutral | banana | avocado (gates explicit tiers). Stamped into the rendered image's embedded metadata.",
        ] as [String: Any],
      ] as [String: Any],
      "required": ["prompt"] as [String],
    ] as [String: Any]
  )

  static let swapLoras = MCPToolDefinition(
    name: "swap_loras",
    description: "Hot-swap active LoRA weights on the loaded model. Replaces all currently active LoRAs with the provided set. Pass an empty array to remove all LoRAs.",
    inputSchema: [
      "type": "object",
      "properties": [
        "loras": [
          "type": "array",
          "description": "LoRA configurations to activate.",
          "items": [
            "type": "object",
            "properties": [
              "path": [
                "type": "string",
                "description": "Path to .safetensors file, or HuggingFace model ID. Bare filenames resolve to ~/Models/loras/.",
              ] as [String: Any],
              "scale": [
                "type": "number",
                "description": "LoRA weight scale. Default: 1.0. Negative values invert the effect (useful for slider LoRAs like breast size). Typical range: -5.0 to 5.0.",
              ] as [String: Any],
            ] as [String: Any],
            "required": ["path"] as [String],
          ] as [String: Any],
        ] as [String: Any],
      ] as [String: Any],
      "required": ["loras"] as [String],
    ] as [String: Any]
  )

  static let listModels = MCPToolDefinition(
    name: "list_models",
    description: "List all supported ComfyBox model families with capabilities, recommended parameters, and HuggingFace IDs.",
    inputSchema: [
      "type": "object",
      "properties": [:] as [String: Any],
    ] as [String: Any]
  )

  static let listStyles = MCPToolDefinition(
    name: "list_styles",
    description: "List available style presets with prompt engineering templates, recommended parameters, and model pairings.",
    inputSchema: [
      "type": "object",
      "properties": [:] as [String: Any],
    ] as [String: Any]
  )

  static let serverHealth = MCPToolDefinition(
    name: "server_health",
    description: "Get ComfyBox server health: loaded model, LoRA state, memory usage, render stats, and queue depth.",
    inputSchema: [
      "type": "object",
      "properties": [:] as [String: Any],
    ] as [String: Any]
  )

  static let queueStatus = MCPToolDefinition(
    name: "queue_status",
    description: "Get the current generation queue status: pending jobs, running job, and history.",
    inputSchema: [
      "type": "object",
      "properties": [:] as [String: Any],
    ] as [String: Any]
  )

  static let clearQueue = MCPToolDefinition(
    name: "clear_queue",
    description: "Cancel all pending generation jobs in the queue. Does not affect the currently running job.",
    inputSchema: [
      "type": "object",
      "properties": [:] as [String: Any],
    ] as [String: Any]
  )

  static let listLoras = MCPToolDefinition(
    name: "list_loras",
    description: "List LoRA files available in the LoRA directory (~/Models/loras/). Returns filenames that can be passed to swap_loras.",
    inputSchema: [
      "type": "object",
      "properties": [:] as [String: Any],
    ] as [String: Any]
  )

  static let shutdownServer = MCPToolDefinition(
    name: "shutdown_server",
    description: "Gracefully shut down the ComfyBox WarmServer. In-flight renders complete before exit. Use sparingly — requires manual restart.",
    inputSchema: [
      "type": "object",
      "properties": [
        "confirm": [
          "type": "boolean",
          "description": "Must be true to confirm shutdown. Safety guard against accidental invocation.",
        ] as [String: Any],
      ] as [String: Any],
      "required": ["confirm"] as [String],
    ] as [String: Any]
  )

  static let systemStats = MCPToolDefinition(
    name: "system_stats",
    description: "Get system hardware information: Apple Silicon device name, total memory, and VRAM available for generation.",
    inputSchema: [
      "type": "object",
      "properties": [:] as [String: Any],
    ] as [String: Any]
  )

  static let applyStyle = MCPToolDefinition(
    name: "apply_style",
    description: "Apply a style preset to a prompt. Returns the enhanced prompt with prefix/suffix, negative prompt, and recommended generation parameters. Does not generate an image — use generate_image with the returned parameters.",
    inputSchema: [
      "type": "object",
      "properties": [
        "style_id": [
          "type": "string",
          "description": "Style preset ID (from list_styles).",
        ] as [String: Any],
        "prompt": [
          "type": "string",
          "description": "Base prompt to enhance with the style.",
        ] as [String: Any],
        "negative_prompt": [
          "type": "string",
          "description": "Optional negative prompt to merge with the style's negative prompt.",
        ] as [String: Any],
      ] as [String: Any],
      "required": ["style_id", "prompt"] as [String],
    ] as [String: Any]
  )


  static let loraLibrary = MCPToolDefinition(
    name: "lora_library",
    description: "List all LoRAs in the library with compatibility info. Filter by model family.",
    inputSchema: [
      "type": "object",
      "properties": [
        "model": [
          "type": "string",
          "description": "Filter by model family (e.g. \"z-image\", \"klein-9b\").",
        ] as [String: Any],
        "include_quarantined": [
          "type": "boolean",
          "description": "Include quarantined LoRAs in results. Default: false.",
        ] as [String: Any],
      ] as [String: Any],
    ] as [String: Any]
  )

  static let loraScan = MCPToolDefinition(
    name: "lora_scan",
    description: "Scan the LoRA directory for new or changed files and update the library index.",
    inputSchema: [
      "type": "object",
      "properties": [
        "force": [
          "type": "boolean",
          "description": "Force full rescan of all LoRA files. Default: false.",
        ] as [String: Any],
      ] as [String: Any],
    ] as [String: Any]
  )

  static let loraQuarantine = MCPToolDefinition(
    name: "lora_quarantine",
    description: "Quarantine or un-quarantine a LoRA. Quarantined LoRAs are hidden from model loading.",
    inputSchema: [
      "type": "object",
      "properties": [
        "id": [
          "type": "string",
          "description": "LoRA identifier from the library.",
        ] as [String: Any],
        "quarantine": [
          "type": "boolean",
          "description": "True to quarantine, false to un-quarantine.",
        ] as [String: Any],
        "reason": [
          "type": "string",
          "description": "Reason for quarantine action.",
        ] as [String: Any],
      ] as [String: Any],
      "required": ["id", "quarantine"] as [String],
    ] as [String: Any]
  )


  static let loadModel = MCPToolDefinition(
    name: "load_model",
    description: "Load a model into the pool. Optionally activate it and wait for completion. Supports all model families: Z-Image, Klein, FIBO, Chroma, SeedVR2.",
    inputSchema: [
      "type": "object",
      "properties": [
        "model": [
          "type": "string",
          "description": "Model spec to load (e.g. \"z-image-turbo-bf16\", \"klein-9b-q8\", \"briaai/FIBO\", \"chroma-8.9b\").",
        ] as [String: Any],
        "quantization": [
          "type": "string",
          "description": "Quantization level (e.g. \"4bit\", \"8bit\"). Omit for default (typically bf16).",
        ] as [String: Any],
        "activate": [
          "type": "boolean",
          "description": "Activate the model after loading. Default: true.",
        ] as [String: Any],
        "wait": [
          "type": "boolean",
          "description": "Wait for load to complete before returning. Default: true. Set false for background loading.",
        ] as [String: Any],
      ] as [String: Any],
      "required": ["model"] as [String],
    ] as [String: Any]
  )

  static let switchModel = MCPToolDefinition(
    name: "switch_model",
    description: "Switch the active model to one already loaded in the pool. Instant — no loading required.",
    inputSchema: [
      "type": "object",
      "properties": [
        "model": [
          "type": "string",
          "description": "Pool model ID to activate (from model_pool results).",
        ] as [String: Any],
      ] as [String: Any],
      "required": ["model"] as [String],
    ] as [String: Any]
  )

  static let modelPool = MCPToolDefinition(
    name: "model_pool",
    description: "List all models currently loaded in the pool with VRAM usage, active status, and last-used timestamps.",
    inputSchema: [
      "type": "object",
      "properties": [:] as [String: Any],
    ] as [String: Any]
  )

  static let unloadModel = MCPToolDefinition(
    name: "unload_model",
    description: "Unload a model from the pool to free VRAM. Cannot unload the currently active model — switch to another model first.",
    inputSchema: [
      "type": "object",
      "properties": [
        "model": [
          "type": "string",
          "description": "Pool model ID to unload (from model_pool results).",
        ] as [String: Any],
      ] as [String: Any],
      "required": ["model"] as [String],
    ] as [String: Any]
  )



  // MARK: - Video Tools

  static let generateVideo = MCPToolDefinition(
    name: "generate_video",
    description: "Generate a video clip. Supports text-to-video (T2V) and image-to-video (I2V). T2V uses LTX 2.3; I2V uses Wan 2.2. Returns a job_id for async polling via video_status. In proxy mode, generation runs on Replicate; in native mode, it runs locally on the Mac GPU.",
    inputSchema: [
      "type": "object",
      "properties": [
        "prompt": [
          "type": "string",
          "description": "For T2V: detailed cinematic scene description (longer is better). For I2V: motion description only (camera + subject + atmosphere, 80-120 words). The source image provides the scene for I2V.",
        ] as [String: Any],
        "image_path": [
          "type": "string",
          "description": "Absolute path to source image for I2V mode. Must be on the Mac filesystem. Omit for T2V mode.",
        ] as [String: Any],
        "duration": [
          "type": "number",
          "description": "Video duration in seconds (default: one ~4s chunk). Longer values render as spliced ~4s chunks, each re-anchored on the previous chunk\u{27}s last frame plus a soft identity anchor to the source \u{2014} budget ~2.5 min per chunk (e.g. 12 \u{2192} 3 chunks).",
        ] as [String: Any],
        "resolution": [
          "type": "string",
          "description": "Output resolution. I2V: '480p' or '720p' (default: '480p'). T2V: '720p' or '1080p' (default: '720p').",
        ] as [String: Any],
        "aspect_ratio": [
          "type": "string",
          "description": "Aspect ratio: '16:9' or '9:16' (default: '16:9').",
        ] as [String: Any],
        "seed": [
          "type": "integer",
          "description": "Random seed for reproducibility. Omit for random.",
        ] as [String: Any],
        "preset": [
          "type": "string",
          "description": "Optional preset id (see list_presets, mediaKind \"video\"): applies the preset\u{27}s LoRAs, prompt prefix/suffix, negative prompt, and dims budget. Explicit params override it.",
        ],
        "output_path": [
          "type": "string",
          "description": "Output file path for the .mp4. Must be within the allowed output directory. Omit to auto-generate.",
        ] as [String: Any],
      ] as [String: Any],
      "required": ["prompt"] as [String],
    ] as [String: Any]
  )

  static let videoStatus = MCPToolDefinition(
    name: "video_status",
    description: "Check the status of a video generation job. Returns status ('queued', 'processing', 'succeeded', 'failed'), and on success, the output file path and render duration.",
    inputSchema: [
      "type": "object",
      "properties": [
        "job_id": [
          "type": "string",
          "description": "Job ID returned by generate_video.",
        ] as [String: Any],
      ] as [String: Any],
      "required": ["job_id"] as [String],
    ] as [String: Any]
  )

  // MARK: - Upscale Tool

  static let upscale = MCPToolDefinition(
    name: "upscale",
    description: "Upscale an image using the SeedVR2 super-resolution pipeline. Accepts a file path and returns the upscaled output file path. Default target resolution is 1024px (safe). 2048px is experimental and may OOM.",
    inputSchema: [
      "type": "object",
      "properties": [
        "image_path": [
          "type": "string",
          "description": "Absolute path to the input image file to upscale.",
        ] as [String: Any],
        "target_resolution": [
          "type": "integer",
          "description": "Target resolution in pixels for the long edge. Default: 1024. Values above 1024 are experimental and may cause OOM errors.",
        ] as [String: Any],
        "seed": [
          "type": "integer",
          "description": "Random seed for reproducibility. Omit for random.",
        ] as [String: Any],
        "softness": [
          "type": "number",
          "description": "Softness factor (0.0-1.0). Higher values produce softer results. Default: 0.0.",
        ] as [String: Any],
        "output_path": [
          "type": "string",
          "description": "Output file path. Omit to auto-generate from input filename with '-upscaled' suffix.",
        ] as [String: Any],
        "model": [
          "type": "string",
          "description": "SeedVR2 variant: 'seedvr2-3b' (default, ~7GB) or 'seedvr2-7b' (~16GB). Auto-detected from available weights.",
          "enum": ["seedvr2-3b", "seedvr2-7b"],
        ] as [String: Any],
      ] as [String: Any],
      "required": ["image_path"] as [String],
    ] as [String: Any]
  )

  // MARK: - Creative layer & queue (added 2026-07)

  static let enhancePrompt = MCPToolDefinition(
    name: "enhance_prompt",
    description: "Optimize an image prompt using the configured prompt-optimization provider (e.g. Dan's model on LM Studio). Optionally inject a named character's description.",
    inputSchema: [
      "type": "object",
      "properties": [
        "prompt": ["type": "string", "description": "The prompt to enhance."] as [String: Any],
        "character": ["type": "string", "description": "Optional character name to inject."] as [String: Any],
        "content_mode": ["type": "string", "description": "neutral | banana | avocado (gates explicit tiers)."] as [String: Any],
      ] as [String: Any],
      "required": ["prompt"],
    ] as [String: Any]
  )

  static let listCharacters = MCPToolDefinition(
    name: "list_characters",
    description: "List creative characters and scenes (name, kind, tiered descriptions, default LoRAs, tags).",
    inputSchema: ["type": "object", "properties": [:] as [String: Any]] as [String: Any]
  )

  static let listPresets = MCPToolDefinition(
    name: "list_presets",
    description: "List saved generation presets (the canonical /v1/presets store).",
    inputSchema: ["type": "object", "properties": [:] as [String: Any]] as [String: Any]
  )

  static let importLegacyPresets = MCPToolDefinition(
    name: "import_legacy_presets",
    description: "Import presets from the old Coffee Shop image service (idempotent). Returns how many were newly imported.",
    inputSchema: ["type": "object", "properties": [:] as [String: Any]] as [String: Any]
  )

  static let queueList = MCPToolDefinition(
    name: "queue_list",
    description: "Detailed render queue: the active job (id, summary, progress) plus every pending job with its id (for cancel_job).",
    inputSchema: ["type": "object", "properties": [:] as [String: Any]] as [String: Any]
  )

  static let interruptRender = MCPToolDefinition(
    name: "interrupt_render",
    description: "Cancel the in-flight render. Pending jobs continue.",
    inputSchema: ["type": "object", "properties": [:] as [String: Any]] as [String: Any]
  )

  static let cancelJob = MCPToolDefinition(
    name: "cancel_job",
    description: "Cancel one pending render job by its id (from queue_list).",
    inputSchema: [
      "type": "object",
      "properties": ["id": ["type": "string", "description": "The pending job id."] as [String: Any]] as [String: Any],
      "required": ["id"],
    ] as [String: Any]
  )

  static let nearlineList = MCPToolDefinition(
    name: "nearline_list",
    description: "List the nearline model/LoRA catalog on attached storage: each item's name, size, kind, and whether it's staged locally.",
    inputSchema: ["type": "object", "properties": [:] as [String: Any]] as [String: Any]
  )

  static let nearlineScan = MCPToolDefinition(
    name: "nearline_scan",
    description: "Rescan the configured attached-storage roots for models/LoRAs.",
    inputSchema: ["type": "object", "properties": [:] as [String: Any]] as [String: Any]
  )

  static let nearlineStage = MCPToolDefinition(
    name: "nearline_stage",
    description: "Copy a nearline item to local storage on demand (LRU eviction keeps within the staging budget).",
    inputSchema: [
      "type": "object",
      "properties": ["name": ["type": "string", "description": "The item filename from nearline_list."] as [String: Any]] as [String: Any],
      "required": ["name"],
    ] as [String: Any]
  )

  static let nearlineEvict = MCPToolDefinition(
    name: "nearline_evict",
    description: "Remove a staged nearline copy from local storage (the attached-storage original is untouched).",
    inputSchema: [
      "type": "object",
      "properties": ["name": ["type": "string", "description": "The item filename to evict."] as [String: Any]] as [String: Any],
      "required": ["name"],
    ] as [String: Any]
  )

  // MARK: - Lookup

  /// Find a tool definition by name.
  public static func tool(named name: String) -> MCPToolDefinition? {
    tools.first { $0.name == name }
  }
}
