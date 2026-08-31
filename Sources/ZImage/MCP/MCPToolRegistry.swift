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
    repairImage,
    swapLoras,
    listModels,
    listStyles,
    serverHealth,
    queueStatus,
    clearQueue,
    pauseQueue,
    resumeQueue,
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
    composeMontage,
    renderStoryboard,
    rerenderVideo,
    extendVideo,
    importWorkflow,
    listWorkflows,
    runWorkflow,
    workflowRunStatus,
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
    civitaiSearch,
    civitaiPrompts,
    moveQueueJob,
    updateLoraTriggerwords,
    createPreset,
    deletePreset,
    setWarmPreset,
    createCharacter,
    deleteCharacter,
    getConfig,
    patchConfig,
    updateConfig,
  ]

  // MARK: - Tool Definitions

  static let repairImage = MCPToolDefinition(
    name: "repair_image",
    description: "Repair a DEFECTIVE image entirely on-device: a local vision model diagnoses render defects (mottled/damaged skin, disfigured anatomy, extra/fused fingers, mesh artifacts) and where they are, then img2img re-renders with a targeted negative prompt + optional region inpaint. Preserves composition and identity. Returns the repaired image path.",
    inputSchema: [
      "type": "object",
      "properties": [
        "image_path": [
          "type": "string",
          "description": "Path to the defective image to repair (Mac-local).",
        ] as [String: Any],
        "note": [
          "type": "string",
          "description": "Optional user description of the defect to steer the repair (e.g. 'left hand has extra fingers'). If omitted, the local vision model auto-diagnoses.",
        ] as [String: Any],
        "image_strength": [
          "type": "number",
          "description": "img2img source-preservation strength 0-1 (default 0.6; higher = closer to the source).",
        ] as [String: Any],
      ] as [String: Any],
      "required": ["image_path"],
    ],
    routes: [RouteRef(method: "POST", path: "/v1/generate")]
  )

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
        // WP-E4 (AC-17): enums generated from the engine's own kinds so the
        // schema cannot drift again (it used to advertise `linear`, which
        // does not exist). The wire key stays `scheduler` (D25).
        "scheduler": [
          "type": "string",
          "enum": SchedulerKind.allCases.map(\.rawValue),
          "description": "Sampler algorithm (wire key `scheduler`). Default: euler. ComfyUI spellings res_2s / dpmpp_2m / dpmpp_2s_ancestral are accepted aliases; an unknown name is a 400.",
        ] as [String: Any],
        "sigma_schedule": [
          "type": "string",
          "enum": SigmaScheduleKind.allCases.map(\.rawValue),
          "description": "Sigma noise schedule. Default: flow. ComfyUI names normal / simple / sgm_uniform / ddim_uniform alias to flow; an unknown name is a 400.",
        ] as [String: Any],
        "image_path": [
          "type": "string",
          "description": "Source image path for img2img. The loaded model must support img2img.",
        ] as [String: Any],
        "image_strength": [
          "type": "number",
          "description": "Img2img source-preservation strength (0.0-1.0). HIGHER = closer to the source (engine denoise = 1 - strength): 0.9 = tiny touch-up, 0.5 = preserve composition, 0.1 = near-total regeneration. Default: 0.3.",
        ] as [String: Any],
        "mask_path": [
          "type": "string",
          "description": "Optional mask PNG path for SELECTIVE inpainting (requires image_path). White pixels = regenerate/inpaint that region, black = keep the original. Lets you add or change an element in one region while locking the rest of the frame (face, composition). Mask should match the source image dimensions. Omit for standard full-frame img2img.",
        ] as [String: Any],
        "mask_region": [
          "type": "string",
          "enum": ["face", "upper", "lower"] as [String],
          "description": "Auto-generate the inpaint mask (requires image_path; mutually exclusive with mask_path). 'face' = Vision face detection on the source; 'upper'/'lower' = top/bottom half of the frame. The named region is REGENERATED. Combine with mask_invert to lock the region instead (e.g. face + mask_invert = keep the face, regenerate the rest).",
        ] as [String: Any],
        "mask_invert": [
          "type": "boolean",
          "description": "Flip the mask (white <-> black). Requires mask_path or mask_region.",
        ] as [String: Any],
        "mask_grow": [
          "type": "integer",
          "description": "Expand the inpaint mask by N pixels before rendering (default 0).",
        ] as [String: Any],
        "mask_feather": [
          "type": "integer",
          "description": "Feather the inpaint mask edges by N pixels for softer seams (default 0).",
        ] as [String: Any],
        "content_mode": [
          "type": "string",
          "description": "neutral | banana | avocado (gates explicit tiers). Stamped into the rendered image's embedded metadata.",
        ] as [String: Any],
      ] as [String: Any],
      "required": ["prompt"] as [String],
    ] as [String: Any],
    routes: [RouteRef(method: "POST", path: "/v1/generate")]
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
    ] as [String: Any],
    routes: [RouteRef(method: "POST", path: "/v1/lora/swap")]
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
    ] as [String: Any],
    routes: [RouteRef(method: "POST", path: "/v1/queue/clear")]
  )

  static let pauseQueue = MCPToolDefinition(
    name: "pause_queue",
    description: "Pause ALL creation: the current job finishes, then no queued or new jobs start until resume_queue. The pause persists across engine restarts. Gates every initiator — schedulers, chat tools, gallery actions, direct API callers.",
    inputSchema: [
      "type": "object",
      "properties": [:] as [String: Any],
    ] as [String: Any],
    routes: [RouteRef(method: "POST", path: "/v1/queue/pause")]
  )

  static let resumeQueue = MCPToolDefinition(
    name: "resume_queue",
    description: "Resume creation after pause_queue: queued jobs start processing again.",
    inputSchema: [
      "type": "object",
      "properties": [:] as [String: Any],
    ] as [String: Any],
    routes: [RouteRef(method: "POST", path: "/v1/queue/resume")]
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
    ] as [String: Any],
    routes: [RouteRef(method: "POST", path: "/v1/shutdown")]
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
    ] as [String: Any],
    routes: [RouteRef(method: "POST", path: "/v1/loras/scan")]
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
    ] as [String: Any],
    routes: [
      RouteRef(method: "POST", path: "/v1/loras/{id}/quarantine"),
      RouteRef(method: "DELETE", path: "/v1/loras/{id}/quarantine"),
    ]
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
    ] as [String: Any],
    routes: [RouteRef(method: "POST", path: "/v1/model/load")]
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
    ] as [String: Any],
    routes: [RouteRef(method: "POST", path: "/v1/model/activate")]
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
    ] as [String: Any],
    routes: [RouteRef(method: "POST", path: "/v1/model/unload")]
  )



  // MARK: - Video Tools

  static let generateVideo = MCPToolDefinition(
    name: "generate_video",
    description: "Generate a video clip. Supports LTX 2.3 text-to-video (T2V) and image-to-video (I2V) in native mode. Returns a job_id for async polling via video_status. In proxy mode, generation runs on Replicate; in native mode, it runs locally on the Mac GPU.",
    inputSchema: [
      "type": "object",
      "properties": [
        "prompt": [
          "type": "string",
          "description": "LTX 2.3: use one dominant shot in one flowing present-tense paragraph. T2V uses 4-8 descriptive sentences, roughly 45-90 words, ordered subject -> action -> camera -> mood. For I2V, the source image is ground truth: use 4-6 concise sentences describing only explicit subject, camera, and environmental motion; do not redescribe or contradict the frame. Send enhance:false when the prompt is already optimized.",
        ] as [String: Any],
        "image_path": [
          "type": "string",
          "description": "Absolute path to source image for I2V mode. Must be on the Mac filesystem. Omit for T2V mode.",
        ] as [String: Any],
        "audio": [
          "type": "boolean",
          "description": "Generate synchronized audio (native T2V single-chunk only; first audio render reloads the model with the audio branch, adding ~1 min).",
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
        "strength": [
          "type": "number",
          "description": "I2V conditioning strength 0-1 (default 1.0). Lower loosens the frame-0 pose lock; weak motion lever.",
        ] as [String: Any],
        "img_compression": [
          "type": "integer",
          "description": "Conditioning compression (libx264 CRF, 0-51). PRIMARY motion lever: low (0-2) = max fidelity but freezes motion; ~30 = strong motion for action/partnered scenes; default 35. Use low for solo/portrait, high for action.",
        ] as [String: Any],
        "guidance": [
          "type": "number",
          "description": "CFG scale (default 1.0 flat). >1 AMPLIFIES action-prompt execution; 2.0 ~doubles action motion cleanly, 3.0 over-drives into artifacts. Use ~2.0 for action/partnered, 1.0 for solo/portrait fidelity.",
        ] as [String: Any],
        "tuning": [
          "type": "object",
          "description": "Tier A per-render tuning overrides (task #9): guidanceRescale, cfgSchedule [floats], stage1Sigmas, refineSigmas, twoStage (bool), condFps, sampler, stgScale, faceAnchorStrength, icControl, colorAnchor, nagScale/nagAlpha/nagTau. Omitted fields defer to preset > config.json > env > builtin. Full precedence recorded in the job's resolved_config.",
        ] as [String: Any],
        "optimization_attempt_id": [
          "type": "string",
          "description": "Lineage reference from enhance_prompt binding this render to its optimization attempt (task #19).",
        ] as [String: Any],
        "frames": [
          "type": "integer",
          "description": "Exact frame count (1+8k: 97, 145, 289...). Forces a SINGLE render pass at this length instead of the chunked `duration` path. Use for action clips <=12s (289f) to hold motion uniform (chunking decays it). Overrides duration.",
        ] as [String: Any],
        "preset": [
          "type": "string",
          "description": "Optional preset id (see list_presets, mediaKind \"video\"): applies the preset\u{27}s LoRAs, prompt prefix/suffix, negative prompt, and dims budget. Explicit params override it.",
        ],
        "skip_character_injection": [
          "type": "boolean",
          "description": "Skip the server-side character description prepend. Send TRUE when the caller has already woven the description into the prompt. Note `enhance:false` is NOT sufficient — it only stops the optimizer's injection, and on t2v the server still defaults the character to \u{22}kira\u{22} and prepends ~110 tokens, which overruns the 128-token cap and truncates the scene and camera direction off the end.",
        ] as [String: Any],
        "loras": [
          "type": "array",
          "description": "Per-render LoRA stack override: [{path|name, scale}]. Request LoRAs REPLACE the preset/default stack for this render (precedence: request > preset > --ltx2-lora). Bare filenames resolve against the LoRA library.",
          "items": ["type": "object"] as [String: Any],
        ] as [String: Any],
        "fps": [
          "type": "integer",
          "description": "Generation frame-rate basis (default 24). Lower = slower on-screen motion per generated frame; duration maps onto the frame grid at this rate.",
        ] as [String: Any],
        "enhance": [
          "type": "boolean",
          "description": "Whether the server should optimize the prompt (default true). Send FALSE when the caller has ALREADY run its own prompt optimizer — a second rewrite drifts the prompt away from concrete staging (limb placement, figure count) and double-injects the character description.",
        ] as [String: Any],
        "output_path": [
          "type": "string",
          "description": "Output file path for the .mp4. Must be within the allowed output directory. Omit to auto-generate.",
        ] as [String: Any],
      ] as [String: Any],
      "required": ["prompt"] as [String],
    ] as [String: Any],
    routes: [RouteRef(method: "POST", path: "/v1/video/generate/async")]
  )

  static let composeMontage = MCPToolDefinition(
    name: "compose_montage",
    description: "Assemble images and video clips into a single MP4 montage with editorial motion — ken-burns pans/zooms on stills, cut/fade/dissolve transitions, and intercut real video clips (e.g. generate_video output). Memory-light (no diffusion model involved): use this to create dynamism from stills and short clips instead of rendering more video. Synchronous — returns the output path directly.",
    inputSchema: [
      "type": "object",
      "properties": [
        "segments": [
          "type": "array",
          "description": "Ordered montage segments. Each: {type: 'image'|'clip', path: <Mac absolute path>, duration_s: <seconds — required for images, optional trim for clips>, kenburns?: {zoom: [start,end], pan: [[x0,y0],[x1,y1]]}}. kenburns applies to images only: scale ramps zoom[0]→zoom[1] while the image offsets between the normalized pan points (e.g. zoom [1.0,1.2] = slow push-in).",
          "items": ["type": "object"] as [String: Any],
        ] as [String: Any],
        "transitions": [
          "type": "array",
          "description": "Transitions between segments — exactly segments-1 entries, or omit for hard cuts everywhere. Each: {type: 'cut'|'fade'|'dissolve', duration_s: <seconds, default 0.5>}. A transition overlaps its neighbors, so total duration = sum(segment durations) - sum(transition durations).",
          "items": ["type": "object"] as [String: Any],
        ] as [String: Any],
        "output": [
          "type": "object",
          "description": "Output settings: {width, height, fps, path}. Defaults: 448x768 @ 30fps, auto-generated path in the allowed output directory.",
        ] as [String: Any],
        "aspect_policy": [
          "type": "string",
          "description": "How mixed-aspect inputs map to the output frame: 'fill_crop' (scale to fill + center-crop, no bars — default) or 'fit_pad' (letterbox on black).",
        ] as [String: Any],
      ] as [String: Any],
      "required": ["segments"] as [String],
    ] as [String: Any],
    routes: [RouteRef(method: "POST", path: "/v1/montage/compose")]
  )

  static let rerenderVideo = MCPToolDefinition(
    name: "rerender_video",
    description:
      "Winner action: replay an already-rendered clip's EXACT request (same seed, same effective prompt, same init image) at a higher resolution budget — the standard flow is cheap 480p exploration, then this on clips worth keeping. Identify the clip by render_id (from video traces) or by its output path/filename. Async: returns 202 + job_id; poll video_status. A 720p re-render takes roughly 4x the original render time.",
    inputSchema: [
      "type": "object",
      "properties": [
        "render_id": [
          "type": "string",
          "description": "Trace render_id of the clip (preferred — exact).",
        ] as [String: Any],
        "path": [
          "type": "string",
          "description": "Output path or filename of the clip, matched against recent render traces.",
        ] as [String: Any],
        "resolution": [
          "type": "string",
          "description": "Named budget: '720p' (default) or '1080p'.",
        ] as [String: Any],
      ] as [String: Any],
      "required": [] as [String],
    ] as [String: Any],
    routes: [RouteRef(method: "POST", path: "/v1/video/rerender")]
  )

  static let extendVideo = MCPToolDefinition(
    name: "extend_video",
    description:
      "Winner action: chain a fresh continuation clip from an existing clip's last frame (i2v, new seed, 480p/4s standard). The source clip's stored prompt carries the scene forward unless a new motion prompt is given. Identify the clip by render_id or output path. Async: returns 202 + job_id; poll video_status. Join the pieces afterwards with compose_montage if a single file is wanted.",
    inputSchema: [
      "type": "object",
      "properties": [
        "render_id": [
          "type": "string",
          "description": "Trace render_id of the source clip.",
        ] as [String: Any],
        "path": [
          "type": "string",
          "description": "Output path or filename of the source clip.",
        ] as [String: Any],
        "seconds": [
          "type": "integer",
          "description": "Continuation length in seconds (default 4; snapped to the trained frame grid, max 12).",
        ] as [String: Any],
        "prompt": [
          "type": "string",
          "description": "Optional new motion prompt for the continuation (raw text — preset trigger words are applied automatically).",
        ] as [String: Any],
      ] as [String: Any],
      "required": [] as [String],
    ] as [String: Any],
    routes: [RouteRef(method: "POST", path: "/v1/video/extend")]
  )

  static let importWorkflow = MCPToolDefinition(
    name: "import_workflow",
    description: "Import a ComfyUI workflow (API format — the JSON from ComfyUI's 'Save (API Format)') as a stored, runnable ComfyBox workflow. Returns the workflow id plus a compatibility report: which nodes map to engine features, which are ignored glue, and which are unknown (their effect is dropped). Generic LoadImage/SaveImage nodes are supported; the UI-format nodes/links JSON is rejected with instructions.",
    inputSchema: [
      "type": "object",
      "properties": [
        "name": ["type": "string", "description": "Display name for the workflow."] as [String: Any],
        "workflow_json": [
          "type": "string",
          "description": "The ComfyUI API-format workflow JSON, as a string.",
        ] as [String: Any],
      ] as [String: Any],
      "required": ["workflow_json"] as [String],
    ] as [String: Any],
    routes: [RouteRef(method: "POST", path: "/v1/workflows/import")]
  )

  static let listWorkflows = MCPToolDefinition(
    name: "list_workflows",
    description: "List imported ComfyUI workflows: id, name, imported_at, and each workflow's compatibility report (mapped/glue/unknown nodes, whether it parses).",
    inputSchema: [
      "type": "object",
      "properties": [:] as [String: Any],
    ] as [String: Any]
  )

  static let workflowRunStatus = MCPToolDefinition(
    name: "workflow_run_status",
    description: "Check a workflow run started by run_workflow. Returns status (running | succeeded | failed) and, on success, the output image path.",
    inputSchema: [
      "type": "object",
      "properties": [
        "run_id": ["type": "string", "description": "Run id returned by run_workflow."] as [String: Any],
      ] as [String: Any],
      "required": ["run_id"] as [String],
    ] as [String: Any]
  )

  static let runWorkflow = MCPToolDefinition(
    name: "run_workflow",
    description: "Run an imported ComfyUI workflow through the native engine. Optional overrides replace the graph's prompt/negative_prompt/seed; other parameters come from the workflow itself. Waits up to ~2 minutes inline; longer renders return {status: running, run_id} — poll workflow_run_status.",
    inputSchema: [
      "type": "object",
      "properties": [
        "workflow_id": ["type": "string", "description": "Id from import_workflow / list_workflows."] as [String: Any],
        "prompt": ["type": "string", "description": "Override the positive prompt."] as [String: Any],
        "negative_prompt": ["type": "string", "description": "Override the negative prompt."] as [String: Any],
        "seed": ["type": "integer", "description": "Override the seed."] as [String: Any],
        "output_path": ["type": "string", "description": "Output file path (within the allowed output directory). Omit to auto-generate."] as [String: Any],
      ] as [String: Any],
      "required": ["workflow_id"] as [String],
    ] as [String: Any],
    routes: [RouteRef(method: "POST", path: "/v1/workflows/{id}/run")]
  )

  static let renderStoryboard = MCPToolDefinition(
    name: "render_storyboard",
    description: "Render a multi-shot video scene from a storyboard spec. Each shot is animated (i2v) from an anchor frame; by default every shot after the first chains from the PREVIOUS shot's extracted last frame, locking face/angle/character across the whole scene (no drift, no seams). A shot may instead pin an explicit anchor_image, and may apply an i2i 'insert' edit to its anchor before animating (add or change an element i2v can't invent — supports mask_path/mask_region selective inpainting). Shots are assembled with hard cuts or the requested transitions. Long-running: returns a job_id immediately — poll video_status.",
    inputSchema: [
      "type": "object",
      "properties": [
        "shots": [
          "type": "array",
          "description": "Ordered shots. Each: {prompt: <motion description>, duration_s?: <seconds, default one ~4s chunk>, anchor_image?: <Mac path — REQUIRED on the first shot, later shots default to the previous shot's last frame>, insert?: {prompt, creativity (0-1 denoise, default 0.35), negative_prompt?, mask_path?, mask_region?: face|upper|lower, mask_invert?, mask_grow?, mask_feather?, seed?}, negative_prompt?, seed?}.",
          "items": ["type": "object"] as [String: Any],
        ] as [String: Any],
        "transitions": [
          "type": "array",
          "description": "Assembly transitions between shots — exactly shots-1 entries ({type: cut|fade|dissolve, duration_s}) or omit for hard cuts (recommended: chained shots are already continuous).",
          "items": ["type": "object"] as [String: Any],
        ] as [String: Any],
        "output": [
          "type": "object",
          "description": "{width, height, fps, path}. Defaults: 640x640 @ 24fps, auto path.",
        ] as [String: Any],
        "loras": [
          "type": "array",
          "description": "LoRAs applied to every shot's i2v render: [{path, scale}].",
          "items": ["type": "object"] as [String: Any],
        ] as [String: Any],
      ] as [String: Any],
      "required": ["shots"] as [String],
    ] as [String: Any],
    routes: [RouteRef(method: "POST", path: "/v1/storyboard/render")]
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
    ] as [String: Any],
    routes: [RouteRef(method: "POST", path: "/v1/upscale")]
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
    ] as [String: Any],
    routes: [RouteRef(method: "POST", path: "/v1/enhance")]
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
    inputSchema: ["type": "object", "properties": [:] as [String: Any]] as [String: Any],
    routes: [RouteRef(method: "POST", path: "/v1/presets/import-legacy")]
  )

  static let queueList = MCPToolDefinition(
    name: "queue_list",
    description: "Detailed render queue: the active job (id, summary, progress) plus every pending job with its id (for cancel_job).",
    inputSchema: ["type": "object", "properties": [:] as [String: Any]] as [String: Any]
  )

  static let interruptRender = MCPToolDefinition(
    name: "interrupt_render",
    description: "Cancel the in-flight render. Pending jobs continue.",
    inputSchema: ["type": "object", "properties": [:] as [String: Any]] as [String: Any],
    routes: [RouteRef(method: "POST", path: "/v1/queue/interrupt")]
  )

  static let cancelJob = MCPToolDefinition(
    name: "cancel_job",
    description: "Cancel one pending render job by its id (from queue_list).",
    inputSchema: [
      "type": "object",
      "properties": ["id": ["type": "string", "description": "The pending job id."] as [String: Any]] as [String: Any],
      "required": ["id"],
    ] as [String: Any],
    routes: [RouteRef(method: "DELETE", path: "/v1/queue/{id}")]
  )

  static let nearlineList = MCPToolDefinition(
    name: "nearline_list",
    description: "List the nearline model/LoRA catalog on attached storage: each item's name, size, kind, and whether it's staged locally.",
    inputSchema: ["type": "object", "properties": [:] as [String: Any]] as [String: Any]
  )

  static let nearlineScan = MCPToolDefinition(
    name: "nearline_scan",
    description: "Rescan the configured attached-storage roots for models/LoRAs.",
    inputSchema: ["type": "object", "properties": [:] as [String: Any]] as [String: Any],
    routes: [RouteRef(method: "POST", path: "/v1/nearline/scan")]
  )

  static let nearlineStage = MCPToolDefinition(
    name: "nearline_stage",
    description: "Copy a nearline item to local storage on demand (LRU eviction keeps within the staging budget).",
    inputSchema: [
      "type": "object",
      "properties": ["name": ["type": "string", "description": "The item filename from nearline_list."] as [String: Any]] as [String: Any],
      "required": ["name"],
    ] as [String: Any],
    routes: [RouteRef(method: "POST", path: "/v1/nearline/stage")]
  )

  static let nearlineEvict = MCPToolDefinition(
    name: "nearline_evict",
    description: "Remove a staged nearline copy from local storage (the attached-storage original is untouched).",
    inputSchema: [
      "type": "object",
      "properties": ["name": ["type": "string", "description": "The item filename to evict."] as [String: Any]] as [String: Any],
      "required": ["name"],
    ] as [String: Any],
    routes: [RouteRef(method: "POST", path: "/v1/nearline/evict")]
  )

  // MARK: - CivitAI conduit + prompt repository (#234)

  static let civitaiSearch = MCPToolDefinition(
    name: "civitai_search",
    description: "Search CivitAI (or civitai.red) for models/LoRAs — name/tag search, filterable by type and base model, sorted by rating/downloads/likes/newest. Requires a CivitAI API key to be resolved server-side (--civitai-key, CIVITAI_API_KEY, or the Desktop app's saved key); returns an error if none resolves.",
    inputSchema: [
      "type": "object",
      "properties": [
        "query": [
          "type": "string",
          "description": "Text search query (model/LoRA name or keyword). Empty lists by sort order instead of searching.",
        ] as [String: Any],
        "types": [
          "type": "array",
          "items": ["type": "string"] as [String: Any],
          "description": "Filter by CivitAI model types, e.g. [\"LORA\", \"Checkpoint\"].",
        ] as [String: Any],
        "base_model": [
          "type": "string",
          "description": "Filter by base model family, e.g. \"Z-Image\", \"SDXL 1.0\". Only applied when query is empty (CivitAI's API quirk: query + baseModels together returns zero results).",
        ] as [String: Any],
        "sort": [
          "type": "string",
          "description": "Sort order: 'Highest Rated' | 'Most Downloaded' | 'Most Liked' | 'Newest' (case/spacing-insensitive, e.g. 'most_liked' also works). Default 'Most Downloaded'.",
        ] as [String: Any],
        "period": [
          "type": "string",
          "description": "Time window for the sort: 'AllTime' | 'Year' | 'Month' | 'Week' | 'Day'. Default 'AllTime'.",
        ] as [String: Any],
        "nsfw": [
          "type": "boolean",
          "description": "Include NSFW results. Default false.",
        ] as [String: Any],
        "limit": [
          "type": "integer",
          "description": "Max results to return (default 24).",
        ] as [String: Any],
        "site": [
          "type": "string",
          "description": "CivitAI host to query: \"civitai.com\" (default) or \"civitai.red\" (same API, NSFW-default mirror). Any other value is rejected with HTTP 400 — the server only ever sends its API key to these two hosts.",
        ] as [String: Any],
      ] as [String: Any],
    ]
  )

  static let civitaiPrompts = MCPToolDefinition(
    name: "civitai_prompts",
    description: "Query the local prompt repository (trained words, description excerpts, and inferred act-taxonomy category harvested from CivitAI model versions — see PromptRepositoryStore). Set harvest=true to run a fresh CivitAI harvest with the given search params FIRST, upserting results, then query. Without harvest=true this only reads what's already been harvested; it never calls CivitAI. Harvests are capped server-side at 200 models per call and a ~60s time budget (results are upserted page-by-page, so a truncated harvest keeps everything fetched so far; the summary reports truncated=true); the query step returns at most max_entries results (default 100, max 500).",
    inputSchema: [
      "type": "object",
      "properties": [
        "harvest": [
          "type": "boolean",
          "description": "If true, run a harvest against CivitAI (using query/types/base_model/sort/period/nsfw/limit/site below) before querying. Requires a resolved CivitAI API key; returns an error if none resolves. Default false.",
        ] as [String: Any],
        "query": ["type": "string", "description": "Harvest-only: text search query, same as civitai_search."] as [String: Any],
        "types": [
          "type": "array",
          "items": ["type": "string"] as [String: Any],
          "description": "Harvest-only: filter by CivitAI model types.",
        ] as [String: Any],
        "base_model": ["type": "string", "description": "Harvest-only: filter by base model family."] as [String: Any],
        "sort": ["type": "string", "description": "Harvest-only: sort order, same values as civitai_search."] as [String: Any],
        "period": ["type": "string", "description": "Harvest-only: time window, same values as civitai_search."] as [String: Any],
        "nsfw": ["type": "boolean", "description": "Harvest-only: include NSFW results."] as [String: Any],
        "limit": ["type": "integer", "description": "Harvest-only: total models to scan across pages (default 24; the server clamps this to 200 per harvest call)."] as [String: Any],
        "site": ["type": "string", "description": "Harvest-only: \"civitai.com\" (default) or \"civitai.red\". Any other value is rejected with HTTP 400."] as [String: Any],
        "filter_base_model": [
          "type": "string",
          "description": "Query filter: only entries harvested from this base model.",
        ] as [String: Any],
        "filter_act": [
          "type": "string",
          "description": "Query filter: only entries with this inferred act-taxonomy category (pose/action/clothing/body/character/style/concept).",
        ] as [String: Any],
        "filter_tag": [
          "type": "string",
          "description": "Query filter: only entries whose source model carries this tag.",
        ] as [String: Any],
        "keyword": [
          "type": "string",
          "description": "Query filter: keyword match across model name, trained words, description excerpt and tags.",
        ] as [String: Any],
        "max_entries": [
          "type": "integer",
          "description": "Query step: max repository entries to return (default 100, max 500).",
        ] as [String: Any],
      ] as [String: Any],
    ],
    routes: [RouteRef(method: "POST", path: "/v1/civitai/harvest")]
  )

  // MARK: - Headless parity Phase 1 (comfybox#300, FDD §4.2) — gap-set tools
  // for mutating routes that had no MCP tool as of c9dd27d's route inventory
  // (§2.6). Scope is intentionally the 6-item gap set only — update_config
  // and the rest of §4.2's "New tools" list are other phases'/worktrees'
  // territory (see FDD §0 row 10, §4 preamble).

  static let moveQueueJob = MCPToolDefinition(
    name: "move_queue_job",
    description: "Reorder one pending job in the render queue: move it to the top, or one slot up/down from its current position.",
    inputSchema: [
      "type": "object",
      "properties": [
        "id": ["type": "string", "description": "The pending job id (from queue_list)."] as [String: Any],
        "direction": [
          "type": "string",
          "enum": ["top", "up", "down"],
          "description": "Where to move the job: 'top' (front of queue), 'up' (one slot earlier), 'down' (one slot later). Any other value is rejected with a clean error.",
        ] as [String: Any],
      ] as [String: Any],
      "required": ["id", "direction"] as [String],
    ] as [String: Any],
    routes: [RouteRef(method: "POST", path: "/v1/queue/{id}/move")]
  )

  static let updateLoraTriggerwords = MCPToolDefinition(
    name: "update_lora_triggerwords",
    description: "Edit a LoRA's trigger-word list in the library index (does not modify the .safetensors file itself). Replaces the full list — pass every trigger word you want kept.",
    inputSchema: [
      "type": "object",
      "properties": [
        "id": ["type": "string", "description": "LoRA identifier from the library (see lora_library)."] as [String: Any],
        "triggerwords": [
          "type": "array",
          "items": ["type": "string"] as [String: Any],
          "description": "Replacement trigger-word list. Pass an empty array to clear all trigger words.",
        ] as [String: Any],
      ] as [String: Any],
      "required": ["id", "triggerwords"] as [String],
    ] as [String: Any],
    routes: [RouteRef(method: "POST", path: "/v1/loras/{id}/update")]
  )

  static let createPreset = MCPToolDefinition(
    name: "create_preset",
    description: "Create or fully replace a saved generation preset (the canonical /v1/presets store). This is a full-document write, not a patch — fields you omit are absent from the saved preset, not preserved from any existing preset with the same id. Use list_presets first to see the current shape when updating an existing preset.",
    inputSchema: [
      "type": "object",
      "properties": [
        "id": ["type": "string", "description": "Stable preset id."] as [String: Any],
        "name": ["type": "string", "description": "Display name."] as [String: Any],
        "description": ["type": "string", "description": "Preset description."] as [String: Any],
        "media_kind": [
          "type": "string",
          "enum": ["image", "video"],
          "description": "Routes the preset to the image or video pipeline.",
        ] as [String: Any],
        "provider": ["type": "string", "description": "\"local\" | \"replicate\" | \"auto\"."] as [String: Any],
        "engine": ["type": "string", "description": "\"mflux\" | \"zimage\"."] as [String: Any],
        "model": ["type": "string", "description": "Model spec (e.g. \"z-image-turbo-bf16\")."] as [String: Any],
        "prompt": ["type": "string"] as [String: Any],
        "negative_prompt": ["type": "string"] as [String: Any],
        "prompt_prefix": ["type": "string"] as [String: Any],
        "prompt_suffix": ["type": "string"] as [String: Any],
        "steps": ["type": "integer"] as [String: Any],
        "guidance": ["type": "number"] as [String: Any],
        "seed": ["type": "integer"] as [String: Any],
        "width": ["type": "integer"] as [String: Any],
        "height": ["type": "integer"] as [String: Any],
        "scheduler": ["type": "string"] as [String: Any],
        "loras": [
          "type": "array",
          "description": "LoRA stack for this preset: [{filename, scale}].",
          "items": ["type": "object"] as [String: Any],
        ] as [String: Any],
      ] as [String: Any],
      "required": ["id", "name"] as [String],
    ] as [String: Any],
    routes: [RouteRef(method: "POST", path: "/v1/presets"), RouteRef(method: "PUT", path: "/v1/presets")]
  )

  static let deletePreset = MCPToolDefinition(
    name: "delete_preset",
    description: "Delete a saved generation preset by id.",
    inputSchema: [
      "type": "object",
      "properties": [
        "id": ["type": "string", "description": "Preset id to delete (from list_presets)."] as [String: Any],
      ] as [String: Any],
      "required": ["id"] as [String],
    ] as [String: Any],
    routes: [RouteRef(method: "DELETE", path: "/v1/presets/{id}")]
  )

  static let setWarmPreset = MCPToolDefinition(
    name: "set_warm_preset",
    description: "Make a model the server's warm-start default: activate it now (loading it into the pool first if it isn't already loaded there), then persist it as the server config's modelSpec so it survives the next restart. Mirrors the Desktop app's Preset 'Set as Warm' action exactly. If activation cannot succeed (even after a load attempt), the config is left untouched — this never partially applies.",
    inputSchema: [
      "type": "object",
      "properties": [
        "model": [
          "type": "string",
          "description": "Model spec to make the warm-start default — a preset's model or customModelPath (see list_presets).",
        ] as [String: Any],
      ] as [String: Any],
      "required": ["model"] as [String],
    ] as [String: Any],
    routes: [
      RouteRef(method: "POST", path: "/v1/model/activate"),
      RouteRef(method: "POST", path: "/v1/model/load"),
      RouteRef(method: "GET", path: "/v1/config"),
      RouteRef(method: "PUT", path: "/v1/config"),
    ]
  )

  static let createCharacter = MCPToolDefinition(
    name: "create_character",
    description: "Create or update a creative character/scene (the canonical /v1/characters store). Full-document upsert: fields you omit are absent from the saved entry, not preserved from any existing entry with the same id. Use list_characters first to see the current shape when updating. 'id' defaults to a slug of 'name' when omitted.",
    inputSchema: [
      "type": "object",
      "properties": [
        "id": ["type": "string", "description": "Stable id. Defaults to a slug of 'name' when omitted."] as [String: Any],
        "name": ["type": "string", "description": "Display name."] as [String: Any],
        "kind": [
          "type": "string",
          "enum": ["character", "scene"],
          "description": "\"character\" (a subject) or \"scene\" (an environment/location). Default: character.",
        ] as [String: Any],
        "description": ["type": "string", "description": "Flat description (legacy/fallback; also used when no tiered 'base' is present)."] as [String: Any],
        "base": ["type": "string", "description": "SFW physical appearance — always included when assembling a description."] as [String: Any],
        "banana": ["type": "string", "description": "Suggestive additions — appended in banana + avocado content modes."] as [String: Any],
        "avocado": ["type": "string", "description": "Explicit additions — appended in avocado content mode only."] as [String: Any],
        "default_loras": [
          "type": "array",
          "description": "LoRAs applied by default when rendering this character: [{filename, scale}].",
          "items": ["type": "object"] as [String: Any],
        ] as [String: Any],
        "prompt_snippet": ["type": "string", "description": "A reusable prompt fragment injected when this character is selected."] as [String: Any],
        "negative_prompt": ["type": "string", "description": "Negative-prompt additions specific to this character."] as [String: Any],
      ] as [String: Any],
      "required": ["name"] as [String],
    ] as [String: Any],
    routes: [RouteRef(method: "POST", path: "/v1/characters"), RouteRef(method: "PUT", path: "/v1/characters")]
  )

  static let deleteCharacter = MCPToolDefinition(
    name: "delete_character",
    description: "Delete a creative character/scene by id.",
    inputSchema: [
      "type": "object",
      "properties": [
        "id": ["type": "string", "description": "Character id to delete (from list_characters)."] as [String: Any],
      ] as [String: Any],
      "required": ["id"] as [String],
    ] as [String: Any],
    routes: [RouteRef(method: "DELETE", path: "/v1/characters/{id}")]
  )

  // MARK: - Headless parity Phase 3 (comfybox#300, FDD §3.3/§4.4) — server-side
  // settings. `patch_config` is the primary write path going forward (RFC 7386
  // JSON Merge Patch, merged server-side against the current document);
  // `update_config` is the full-replace tool held back from Phase 1 (FDD §0 row
  // 10 — it would have proxied the clobbering whole-document PUT that this
  // phase replaces) and now lands alongside PATCH.

  static let getConfig = MCPToolDefinition(
    name: "get_config",
    description: "Read the full server configuration document (~/.comfybox/config.json): providers, replicate, krea2Models, renderDefaults (engine width/height/steps/guidance overrides, family-aware), videoDefaults (Motion tab width/height/frames), and content-mode preset mappings. Use this before update_config (a full replace) to see the current shape; patch_config does not need it.",
    inputSchema: ["type": "object", "properties": [:] as [String: Any]] as [String: Any],
    routes: [RouteRef(method: "GET", path: "/v1/config")]
  )

  static let patchConfig = MCPToolDefinition(
    name: "patch_config",
    description: "Apply an RFC 7386 JSON Merge Patch to the server configuration — the primary way to change one or a few settings (e.g. {\"renderDefaults\": {\"byFamily\": {\"fibo\": {\"steps\": 40}}}}). Fields you omit are left unchanged (unlike update_config); an explicit `null` on a field deletes it, reverting to the built-in default. Two agents patching different fields cannot conflict with each other.",
    inputSchema: [
      "type": "object",
      "properties": [
        "patch": [
          "type": "object",
          "description": "RFC 7386 JSON Merge Patch document — a partial config shape. Nested objects merge; a null value deletes that key.",
        ] as [String: Any],
      ] as [String: Any],
      "required": ["patch"] as [String],
    ] as [String: Any],
    routes: [RouteRef(method: "PATCH", path: "/v1/config")]
  )

  static let updateConfig = MCPToolDefinition(
    name: "update_config",
    description: "Replace the ENTIRE server configuration document. This is a full-document write, not a patch — fields you omit are ABSENT from the saved document, not preserved from the current one. Call get_config first, edit the returned object, and pass it back whole. Prefer patch_config for changing one or a few fields.",
    inputSchema: [
      "type": "object",
      "properties": [
        "config": [
          "type": "object",
          "description": "The full config document, as returned by get_config (with your edits applied).",
        ] as [String: Any],
      ] as [String: Any],
      "required": ["config"] as [String],
    ] as [String: Any],
    routes: [RouteRef(method: "PUT", path: "/v1/config")]
  )

  // MARK: - Lookup

  /// Find a tool definition by name.
  public static func tool(named name: String) -> MCPToolDefinition? {
    tools.first { $0.name == name }
  }
}
