# ComfyBox LoRA Library Manager + Model Storage Strategy

**Issue**: BarkadaBrew/zimage.swift#73
**Date**: 2026-05-08
**Author**: Bree
**Status**: Design spec — ready for implementation

## Problem Statement

ComfyBox has no centralized LoRA management. Today:

1. **No registry.** LoRAs are scattered across `~/Models/loras/` (Bolt SSD) and `~/bin/zimage/loras/` (WarmServer). No metadata, no searchability, no way to know what's available without `ls`.
2. **Triggerwords and scales are tribal knowledge.** The flow-dpo LoRA works best at 0.8, the chest slider at -3.0 to -5.0, fun-distill needs no triggerword — none of this is recorded anywhere machine-readable.
3. **Model compatibility is trial-and-error.** Klein 9B LoRAs silently load with 0 key matches against Z-Image Turbo (different architecture). The quarantine directory is a manual workaround.
4. **No client discovery.** Krita bridge's `zimageLoraModels()` does a flat scan of `~/bin/zimage/loras/*.safetensors` — misses subdirectories, has no metadata, and doesn't know about the larger library on Bolt.
5. **Two-copy problem.** Active LoRAs must be manually copied from `~/Models/loras/` (Bolt, ~500 MB/s) to `~/bin/zimage/loras/` (internal SSD, ~21 GB/s) for WarmServer use. No automation, no cache management.

## Current Architecture (Research Findings)

### LoRA Loading Stack

ComfyBox has **four independent LoRA loaders**, each with its own key mapper:

| Loader | Architecture | Key Mapper | Supports LoKr |
|--------|-------------|------------|---------------|
| `LoRAWeightLoader` + `LoRAApplicator` | Z-Image (Lumina2, 6B) | `LoRAKeyMapper` | Yes |
| `LoRAWeightLoader.loadForFlux2()` | Flux 2 Klein (4B/9B) | `Flux2LoRAMapping` | No |
| `Flux2LoRALoader` | Flux 2 Klein (4B/9B) | inline prefix strip | Yes |
| `ChromaLoRALoader` | Chroma | `ChromaLoRAKeyMapper` | No |

Key mapper coverage (`LoRAKeyMapper`): 238 cached target paths covering `layers.0-29`, `noise_refiner.0-1`, `context_refiner.0-1` with attention (to_q/k/v/out), feed_forward (w1/w2/w3), and adaLN_modulation.

### LoRA Naming Conventions (from safetensors)

Three conventions observed in the wild, all must be supported:

| Convention | Prefix | Used By |
|-----------|--------|---------|
| ComfyUI/diffusers | `diffusion_model.` | CivitAI community LoRAs, ai-toolkit exports |
| BFL/peft | `base_model.model.` | HuggingFace/BFL official adapters |
| Kohya/webui | `lora_unet_` (underscore-separated) | Kohya SS training exports |

### LoRA Formats

| Format | Keys | Detection | ComfyBox Support |
|--------|------|-----------|-----------------|
| Standard LoRA | `.lora_down.`/`.lora_up.` or `.lora_A.`/`.lora_B.` | Check first non-alpha key | All loaders |
| LoKr (Kronecker) | `.lokr_w1`/`.lokr_w2` + optional `.alpha` | `hasSuffix(".lokr_w1")` | Z-Image + Flux2LoRALoader only |

### Safetensors Metadata (Empirical)

Inspected metadata headers from actual files in `~/Models/loras/`:

**flow-dpo/zit_fdpo_v1.safetensors** (Z-Image LoKr, 162MB, 720 keys):
```json
{
  "ss_base_model_version": "zimage",
  "software": "{\"name\": \"ai-toolkit\", ...}",
  "name": "zit_fdpo_a90",
  "training_info": "{\"step\": 290, \"epoch\": 0}"
}
```

**fun-distill/zit-sda-v1.safetensors** (Z-Image LoKr, 162MB, 720 keys):
```json
{
  "ss_base_model_version": "zimage",
  "software": "{\"name\": \"ai-toolkit\", ...}",
  "name": "zit_sft_div_28"
}
```

**fun-distill/z-image-fun-distill-udcai-2603.safetensors** (Z-Image LoRA, 305MB, 405 keys):
```json
null  // NO metadata
```

**quarantine/klein-9b/KLEIN-Unchained-V2.safetensors** (Klein 9B LoRA, 316MB, 224 keys):
```json
{
  "ss_base_model_version": "FLUX_KLEIN_9B",
  "ss_network_module": "lora",
  "ss_network_dim": "64",
  "ss_network_alpha": "64",
  "modelspec.architecture": "flux_klein_9b",
  "extraction_method": "svd_diff"
}
```

**quarantine/chroma/flash-heun/chroma-unlocked-v47-...safetensors** (Chroma LoRA, 1.29GB, 693 keys):
```json
{
  "ss_base_model_version": "flux1",
  "ss_network_dim": "96",
  "ss_network_module": "networks.lora_flux",
  "ss_network_alpha": "96.0"
}
```

**Key insight:** `ss_base_model_version` is present in most trained LoRAs and can be used for auto-classification. But some LoRAs (like the fun-distill UDCAI) have no metadata at all — must fall back to layer name heuristics.

### Storage Layout

```
/Volumes/Bolt (4TB USB SSD, ~500 MB/s)
└── ~/Models/loras/               4.2 GB total
    ├── flow-dpo/                 162 MB  (Z-Image LoKr)
    ├── fun-distill/              468 MB  (Z-Image LoRA + LoKr)
    ├── quarantine/
    │   ├── klein-9b/             2.3 GB  (5 LoRAs + 3 sliders)
    │   └── chroma/               1.3 GB  (1 LoRA)
    ├── anime/                    empty
    ├── art-style/                empty
    ├── flux/                     empty
    ├── nsfw/                     empty (placeholder)
    ├── photography/              empty
    ├── realism-mix/              empty
    ├── uncategorized/            empty
    └── upscalers/                empty

Internal SSD (~21 GB/s, 180 GB available)
└── ~/bin/zimage/loras/           193 MB
    ├── nudeart6-e10.safetensors           32 MB
    └── RealisticSnapshot-Zimage-Turbov5.safetensors  170 MB
```

### WarmServer LoRA Handling

The WarmServer (`Sources/ZImage/Server/WarmServer.swift`) currently:

- Hardcodes `loraDirectoryPath` to `~/bin/zimage/loras`
- Resolves bare filenames against that directory; absolute paths pass through
- Supports hot-swap via `POST /v1/lora/swap` with `[{path, scale}]` payload
- Tracks `activeLoRAs: [LoRAConfiguration]` in the coordinator
- Reports LoRA state in health endpoint
- Routes Krita bridge LoRA requests through the same swap mechanism

### Krita Bridge Discovery

`ComfyBridgeObjectInfo.zimageLoraModels()` does a flat `contentsOfDirectory` scan of `~/bin/zimage/loras/` filtering for `*.safetensors`. It returns bare filenames only — no metadata, no subdirectory scanning, no compatibility info.

### CLI Interface

The CLI (`Sources/ZImageCLI/main.swift`) accepts:
- `--lora <path_or_hf_id>` (repeatable)
- `--lora-scale <float>` (per-lora override)
- `--lora-paths <comma-separated>` (batch)
- `--lora-scales <comma-separated>` (batch)
- Scale can also be embedded: `--lora path.safetensors=0.8`

Resolution order: local file check > HuggingFace download.

### Model Registry

`ComfyBoxModelRegistry` already defines `supportsLoRA` per model. Current state:

| Family | LoRA Support | Notes |
|--------|-------------|-------|
| Z-Image (all variants) | `true` | Confirmed working |
| Flux 2 Klein (all) | `false` | **Should be `true`** — Flux2LoRALoader exists and works |
| FIBO | `false` | No LoRA loader yet |
| SeedVR2 | `false` | Upscaler, LoRA not applicable |

---

## Architecture

### 1. Library Index (`library.json`)

A JSON manifest stored at `~/Models/loras/library.json` that indexes every LoRA in the collection.

#### Schema

```json
{
  "version": 1,
  "updated_at": "2026-05-08T14:30:00Z",
  "entries": [
    {
      "id": "zit-fdpo-v1",
      "filename": "zit_fdpo_v1.safetensors",
      "relative_path": "flow-dpo/zit_fdpo_v1.safetensors",
      "size_bytes": 169691320,
      "sha256": "a1b2c3...",
      "model_compatibility": ["z-image"],
      "format": "lokr",
      "rank": 8,
      "alpha": null,
      "key_count": 720,
      "layer_targets": ["attention", "feed_forward", "adaLN_modulation"],
      "triggerwords": [],
      "recommended_scale": 1.0,
      "scale_range": [0.5, 1.5],
      "tags": ["flow-dpo", "quality", "distillation"],
      "category": "flow-dpo",
      "notes": "Flow-matching DPO adapter for Z-Image Turbo. Trained with ai-toolkit.",
      "source_url": null,
      "civitai_model_id": null,
      "date_added": "2026-05-08",
      "quarantined": false,
      "quarantine_reason": null,
      "safetensors_metadata": {
        "ss_base_model_version": "zimage",
        "name": "zit_fdpo_a90"
      }
    }
  ]
}
```

#### Fields

| Field | Type | Source | Notes |
|-------|------|--------|-------|
| `id` | string | derived | Slugified from filename, must be unique |
| `filename` | string | filesystem | Bare filename |
| `relative_path` | string | filesystem | Relative to library root (`~/Models/loras/`) |
| `size_bytes` | int | filesystem | File size |
| `sha256` | string | computed | Full file hash for dedup and integrity |
| `model_compatibility` | string[] | auto + manual | Values: `z-image`, `klein-9b`, `klein-4b`, `chroma`, `flux1`, `fibo` |
| `format` | string | auto-detected | `lora`, `lokr`, `slider` |
| `rank` | int | extracted | From tensor shapes or `ss_network_dim` |
| `alpha` | float? | extracted | From `ss_network_alpha` or `adapter_config.json` |
| `key_count` | int | counted | Total tensor keys in file |
| `layer_targets` | string[] | analyzed | Which layer types are targeted |
| `triggerwords` | string[] | manual | Activation words (empty = none needed) |
| `recommended_scale` | float | manual | Best default scale |
| `scale_range` | [float, float] | manual | Useful range (for UI sliders) |
| `tags` | string[] | manual | Freeform tags for search |
| `category` | string | filesystem | Derived from parent directory name |
| `notes` | string | manual | Human-readable description |
| `source_url` | string? | manual | CivitAI or HuggingFace URL |
| `civitai_model_id` | int? | manual | For update checking |
| `date_added` | string | auto | ISO 8601 date |
| `quarantined` | bool | auto/manual | If true, excluded from active use |
| `quarantine_reason` | string? | manual | Why quarantined |
| `safetensors_metadata` | dict? | extracted | Raw metadata from safetensors header |

#### Auto-Detection Logic

Model compatibility detection in priority order:

1. **`ss_base_model_version` metadata** (most reliable):
   - `"zimage"` -> `["z-image"]`
   - `"FLUX_KLEIN_9B"` -> `["klein-9b"]`
   - `"FLUX_KLEIN_4B"` -> `["klein-4b"]`
   - `"flux1"` -> `["chroma"]` (if Chroma key patterns) or `["flux1"]`

2. **`modelspec.architecture` metadata**:
   - `"flux_klein_9b"` -> `["klein-9b"]`

3. **Layer name heuristics** (fallback when no metadata):
   - Keys contain `diffusion_model.layers.` + `context_refiner.` + `noise_refiner.` -> `["z-image"]`
   - Keys contain `double_blocks.` + `single_blocks.` with `img_attn`/`txt_attn` -> `["klein-9b"]` or `["chroma"]`
   - Distinguish Klein vs Chroma by block count: Klein 9B has 8 double + 24 single; Chroma different

4. **Format detection**:
   - Any key contains `.lokr_w1` -> `"lokr"`
   - Any key contains `.lora_down.` or `.lora_A.` -> `"lora"`
   - Key count <= 50 and name contains "slider" -> `"slider"`

### 2. Nearline/Online Storage Strategy

Two tiers of storage, with LRU caching from nearline to online:

```
┌─────────────────────────────────────────────┐
│  ONLINE (Internal SSD)                      │
│  ~/Library/Caches/ComfyBox/loras/           │
│  ~21 GB/s sequential read                   │
│  Budget: 2 GB (configurable)                │
│  Contents: hot LoRAs, LRU managed           │
│                                             │
│  Symlinks back to Bolt when space allows    │
└──────────────────────┬──────────────────────┘
                       │ stage / evict
┌──────────────────────┴──────────────────────┐
│  NEARLINE (Bolt 4TB USB SSD)                │
│  ~/Models/loras/                            │
│  ~500 MB/s sequential read                  │
│  Budget: unlimited (3.6 TB free)            │
│  Contents: full library, library.json       │
│  Source of truth for all LoRA files         │
└─────────────────────────────────────────────┘
```

#### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `COMFYBOX_MODELS` | `~/Models/loras` | Nearline library root (Bolt) |
| `COMFYBOX_CACHE` | `~/Library/Caches/ComfyBox/loras` | Online cache (internal SSD) |
| `COMFYBOX_CACHE_SIZE_GB` | `2` | Max cache size in GB |
| `COMFYBOX_PRELOAD` | (empty) | Comma-separated LoRA IDs to preload on server start |

#### Cache Behavior

- **Stage on demand:** When WarmServer needs a LoRA, check cache first. If miss, copy from Bolt to cache. The 162 MB flow-dpo LoRA copies in ~0.3s from Bolt — negligible vs. the 15-20s render time.
- **LRU eviction:** When cache exceeds budget, evict least-recently-used entries. Track access timestamps in a lightweight `cache-state.json`.
- **Preload:** `COMFYBOX_PRELOAD=zit-fdpo-v1,zit-sda-v1` stages LoRAs at server startup. WarmServer's `initialLoRAs` config triggers this automatically.
- **Direct Bolt access:** For LoRAs larger than 500 MB (e.g., SNOFS at 1 GB), skip cache and load directly from Bolt. The extra 1-2s load time is acceptable for large LoRAs that aren't hot-swapped frequently.
- **Cache-through for WarmServer:** WarmServer's `loraDirectoryPath` changes from `~/bin/zimage/loras` to the cache directory. Existing `~/bin/zimage/loras/` contents migrate to the library on first run.

#### Migration from Current Layout

1. Move `~/bin/zimage/loras/*.safetensors` to `~/Models/loras/uncategorized/`
2. Run `zimage lora scan` to build initial `library.json`
3. Update WarmServer's `loraDirectoryPath` to use cache
4. Symlink `~/bin/zimage/loras` -> cache dir for backward compatibility

### 3. Swift Implementation

#### New Files

```
Sources/ZImage/LoRA/
├── LoRALibrary.swift          // Library index manager
├── LoRALibraryEntry.swift     // Entry model (Codable)
├── LoRAScanner.swift          // Filesystem scanner + auto-detection
├── LoRACache.swift            // Nearline/online cache manager
├── LoRACompatibility.swift    // Model compatibility matrix
```

#### `LoRALibrary` (Core Manager)

```swift
public final class LoRALibrary: @unchecked Sendable {
    
    private let libraryRoot: URL        // ~/Models/loras/
    private let cacheRoot: URL          // ~/Library/Caches/ComfyBox/loras/
    private let cacheBudgetBytes: UInt64
    private var entries: [String: LoRALibraryEntry]  // keyed by id
    private let indexPath: URL          // libraryRoot/library.json
    
    // MARK: - Query
    
    /// List all entries, optionally filtered.
    public func list(
        compatibility: String? = nil,   // e.g. "z-image"
        tags: [String]? = nil,
        includeQuarantined: Bool = false
    ) -> [LoRALibraryEntry]
    
    /// Get entry by ID or filename.
    public func entry(for identifier: String) -> LoRALibraryEntry?
    
    /// Search entries by text (matches id, filename, tags, notes).
    public func search(_ query: String) -> [LoRALibraryEntry]
    
    /// Get entries compatible with a given model family.
    public func compatible(with family: ComfyBoxModelFamily) -> [LoRALibraryEntry]
    
    // MARK: - Cache
    
    /// Ensure a LoRA is in the online cache. Returns the cached file path.
    /// If already cached, returns immediately. Otherwise copies from nearline.
    public func stage(_ id: String) async throws -> URL
    
    /// Resolve a LoRA identifier to a loadable file path.
    /// Checks: cache -> nearline -> error.
    public func resolve(_ identifier: String) throws -> URL
    
    // MARK: - Mutate
    
    /// Scan filesystem and rebuild/update the index.
    public func scan() async throws -> ScanResult
    
    /// Update metadata for an entry (triggerwords, scale, tags, notes).
    public func update(_ id: String, patch: LoRAEntryPatch) throws
    
    /// Install a LoRA from a URL (download + add to library).
    public func install(from url: URL, category: String?) async throws -> LoRALibraryEntry
    
    /// Quarantine a LoRA (mark incompatible, move to quarantine/).
    public func quarantine(_ id: String, reason: String) throws
}
```

#### `LoRAScanner` (Auto-Detection)

```swift
public enum LoRAScanner {
    
    /// Scan a safetensors file and extract all discoverable metadata.
    public static func analyze(_ url: URL) throws -> LoRAScanResult
    
    /// Detect model compatibility from safetensors metadata + key patterns.
    public static func detectCompatibility(
        metadata: [String: String]?,
        sampleKeys: [String]
    ) -> [String]
    
    /// Detect LoRA format from key patterns.
    public static func detectFormat(keys: [String]) -> LoRAFormat
    
    /// Infer rank from tensor shapes.
    public static func inferRank(keys: [String], reader: SafeTensorsReader) -> Int
}

public struct LoRAScanResult {
    let compatibility: [String]
    let format: LoRAFormat          // .lora, .lokr, .slider
    let rank: Int
    let alpha: Float?
    let keyCount: Int
    let layerTargets: [String]
    let safetensorsMetadata: [String: String]?
}

public enum LoRAFormat: String, Codable {
    case lora
    case lokr
    case slider
}
```

#### `LoRACompatibility` Matrix

```swift
public enum LoRACompatibility {
    
    /// Check if a LoRA is compatible with the currently loaded model.
    /// Uses the key mapper for the model family to do a trial mapping.
    public static func check(
        entry: LoRALibraryEntry,
        modelFamily: ComfyBoxModelFamily
    ) -> CompatibilityResult
    
    /// Map model_compatibility strings to ComfyBoxModelFamily.
    public static func familyMapping(_ compat: String) -> ComfyBoxModelFamily? {
        switch compat {
        case "z-image": return .zImage
        case "klein-9b", "klein-4b": return .flux2Klein
        // chroma not yet in ComfyBoxModelFamily
        default: return nil
        }
    }
}

public struct CompatibilityResult {
    let isCompatible: Bool
    let matchedKeys: Int
    let totalKeys: Int
    let matchRatio: Float       // matchedKeys / totalKeys
    let missingLayers: [String] // keys that didn't map
    let warnings: [String]      // e.g. "LoKr format not supported for this model"
}
```

### 4. CLI Commands

New subcommand group: `zimage lora`.

```
USAGE: zimage lora <subcommand>

SUBCOMMANDS:
  list          List available LoRAs
  info          Show detailed info for a LoRA
  scan          Scan filesystem and rebuild library index
  install       Download and install a LoRA from URL
  check         Check LoRA compatibility with a model
  stage         Stage a LoRA into the online cache
  quarantine    Quarantine an incompatible LoRA

EXAMPLES:
  zimage lora list                              # all LoRAs
  zimage lora list --model z-image              # compatible with Z-Image
  zimage lora list --tag nsfw                   # by tag
  zimage lora info zit-fdpo-v1                  # detailed metadata
  zimage lora scan                              # rebuild index from disk
  zimage lora install https://civitai.com/...   # download + register
  zimage lora check KLEIN-Unchained-V2 --model z-image  # compatibility test
  zimage lora stage zit-fdpo-v1                 # copy to online cache
  zimage lora quarantine KLEIN-Unchained-V2 --reason "Klein 9B only"
```

#### `zimage lora list` Output

```
ID                          Model       Format  Size     Scale  Tags
─────────────────────────── ─────────── ─────── ──────── ────── ────────────────
zit-fdpo-v1                 z-image     lokr    162 MB   1.0    flow-dpo, quality
zit-sda-v1                  z-image     lokr    162 MB   1.0    fun-distill
z-image-fun-distill-udcai   z-image     lora    305 MB   1.0    fun-distill
KLEIN-Unchained-V2          klein-9b    lora    316 MB   0.8    nsfw [Q]
sexgod-nudity-helper        klein-9b    lora    632 MB   0.8    nsfw [Q]
chroma-unlocked-v47         chroma      lora    1.29 GB  1.0    distillation [Q]

[Q] = quarantined (incompatible with current base model)
6 LoRAs in library (3 active, 3 quarantined)
```

### 5. WarmServer API Extensions

#### `GET /v1/loras` — List Available LoRAs

Returns all non-quarantined LoRAs compatible with the currently loaded model.

```json
{
  "loras": [
    {
      "id": "zit-fdpo-v1",
      "filename": "zit_fdpo_v1.safetensors",
      "format": "lokr",
      "size_bytes": 169691320,
      "model_compatibility": ["z-image"],
      "recommended_scale": 1.0,
      "scale_range": [0.5, 1.5],
      "triggerwords": [],
      "tags": ["flow-dpo", "quality"],
      "cached": true,
      "active": false
    }
  ],
  "active_loras": [
    {"id": "zit-fdpo-v1", "scale": 1.0}
  ],
  "model_family": "z-image",
  "count": 3
}
```

#### `GET /v1/lora/info/<id>` — LoRA Detail

Returns full metadata for a single LoRA including safetensors_metadata, compatibility check results, and cache state.

```json
{
  "id": "zit-fdpo-v1",
  "filename": "zit_fdpo_v1.safetensors",
  "relative_path": "flow-dpo/zit_fdpo_v1.safetensors",
  "size_bytes": 169691320,
  "sha256": "a1b2c3...",
  "model_compatibility": ["z-image"],
  "format": "lokr",
  "rank": 8,
  "alpha": null,
  "key_count": 720,
  "layer_targets": ["attention", "feed_forward", "adaLN_modulation"],
  "triggerwords": [],
  "recommended_scale": 1.0,
  "scale_range": [0.5, 1.5],
  "tags": ["flow-dpo", "quality"],
  "notes": "Flow-matching DPO adapter for Z-Image Turbo.",
  "source_url": null,
  "date_added": "2026-05-08",
  "quarantined": false,
  "cached": true,
  "cache_path": "/Users/toddwalderman/Library/Caches/ComfyBox/loras/zit_fdpo_v1.safetensors",
  "nearline_path": "/Volumes/Bolt/Users/toddwalderman/Models/loras/flow-dpo/zit_fdpo_v1.safetensors",
  "compatibility_check": {
    "current_model": "z-image-turbo-q8",
    "is_compatible": true,
    "matched_keys": 720,
    "total_keys": 720,
    "match_ratio": 1.0
  },
  "safetensors_metadata": {
    "ss_base_model_version": "zimage",
    "name": "zit_fdpo_a90",
    "software": "{\"name\": \"ai-toolkit\", ...}"
  }
}
```

#### `POST /v1/lora/swap` — Enhanced (backward compatible)

Extend the existing swap endpoint to accept LoRA IDs in addition to paths:

```json
{
  "loras": [
    {"id": "zit-fdpo-v1", "scale": 0.8},
    {"path": "/absolute/path/to/custom.safetensors", "scale": 1.0}
  ]
}
```

Resolution: `id` -> library lookup -> stage to cache -> resolve path. Falls back to `path` for backward compatibility.

#### `POST /v1/lora/scan` — Trigger Library Rescan

```json
{"force": true}
```

Returns scan results (new/updated/removed counts).

### 6. Krita Bridge Integration

#### `ComfyBridgeObjectInfo.zimageLoraModels()` Upgrade

Replace the flat directory scan with library-backed discovery:

```swift
private static func zimageLoraModels() -> [String] {
    guard let library = LoRALibrary.shared else {
        return fallbackDirectoryScan()
    }
    
    // Return LoRAs compatible with the active model family
    let activeFamily = WarmServerState.shared?.modelFamily ?? .zImage
    return library
        .compatible(with: activeFamily)
        .map { $0.filename }
        .sorted()
}
```

#### `/api/etn/model_info/loras` Enhancement

Krita queries `GET /api/etn/model_info/loras` for LoRA metadata. Extend the response:

```json
{
  "lora_name.safetensors": {
    "trigger_words": ["triggerword1"],
    "recommended_scale": 0.8,
    "tags": ["nsfw", "realism"]
  }
}
```

This enables Krita's LoRA picker to show metadata without extra API calls.

### 7. Compatibility Matrix

#### How Compatibility Is Determined

```
┌─────────────────────────────┐
│ 1. Check ss_base_model_ver  │──→ Known string? Map directly.
│    from safetensors metadata│
└──────────┬──────────────────┘
           │ unknown/missing
┌──────────▼──────────────────┐
│ 2. Sample 20 tensor keys    │──→ Pattern match on key prefixes:
│    from safetensors          │    diffusion_model.layers.* → Z-Image
│                              │    double_blocks.*/single_* → Klein/Chroma
│                              │    transformer_blocks.*     → Flux 1
└──────────┬──────────────────┘
           │ ambiguous
┌──────────▼──────────────────┐
│ 3. Trial key mapping        │──→ Run through each family's key mapper,
│                              │    count successful matches.
│                              │    Highest match ratio wins.
└─────────────────────────────┘
```

#### Compatibility Table (Current Known LoRAs)

| LoRA | Z-Image | Klein 9B | Klein 4B | Chroma | Flux 1 |
|------|---------|----------|----------|--------|--------|
| zit_fdpo_v1 (LoKr) | **720/720** | 0 | 0 | 0 | 0 |
| zit-sda-v1 (LoKr) | **720/720** | 0 | 0 | 0 | 0 |
| z-image-fun-distill-udcai (LoRA) | **405/405** | 0 | 0 | 0 | 0 |
| KLEIN-Unchained-V2 | 0 | **224/224** | partial | 0 | 0 |
| sexgod-nudity-helper | 0 | **224/224** | partial | 0 | 0 |
| chroma-unlocked-v47 (LoRA) | 0 | 0 | 0 | **693/693** | partial |

## Implementation Plan

### Phase 1: Library Index + Scanner (Priority: P0)

**Files:** `LoRALibrary.swift`, `LoRALibraryEntry.swift`, `LoRAScanner.swift`

1. Implement `LoRAScanner.analyze()` — read safetensors header, extract metadata, detect format/compatibility
2. Implement `LoRALibrary` — load/save `library.json`, query methods, scan
3. CLI: `zimage lora list`, `zimage lora info`, `zimage lora scan`
4. Build initial `library.json` from existing `~/Models/loras/`

**Estimate:** 2-3 days. No changes to WarmServer or bridge.

### Phase 2: Cache Manager + WarmServer Integration (P0)

**Files:** `LoRACache.swift`, WarmServer changes

1. Implement `LoRACache` — stage, evict, LRU tracking
2. Update WarmServer to use library for LoRA resolution
3. Add `GET /v1/loras` and `GET /v1/lora/info/<id>` endpoints
4. Update `POST /v1/lora/swap` to accept IDs
5. Migrate `~/bin/zimage/loras/` contents to library

**Estimate:** 2 days. WarmServer restart required.

### Phase 3: Krita Bridge + Compatibility (P1)

**Files:** `LoRACompatibility.swift`, `ComfyBridgeObjectInfo.swift` changes

1. Implement trial key mapping for compatibility checking
2. Update `zimageLoraModels()` to use library
3. Extend `/api/etn/model_info/loras` with metadata
4. CLI: `zimage lora check`

**Estimate:** 1-2 days.

### Phase 4: Install + CivitAI (P2)

1. `zimage lora install <url>` — download, scan, add to library
2. CivitAI API integration for metadata enrichment (triggerwords, model version)
3. Update checking for installed LoRAs with `civitai_model_id`

**Estimate:** 2-3 days. Nice-to-have, not blocking core workflow.

## Dependencies

- **ComfyBoxModelRegistry:** Add `chroma` to `ComfyBoxModelFamily` enum when Chroma support lands.
- **Flux 2 Klein LoRA:** Set `supportsLoRA: true` in registry for all Klein models — the `Flux2LoRALoader` already works.
- **SafeTensorsReader:** Already exists in ComfyBox, used by all LoRA loaders. Need to add a `metadata()` method if not already present.

## Open Questions

1. **Should quarantined LoRAs be physically moved or just flagged?** Current manual approach moves to `quarantine/` subdir. The library index could just flag them and leave files in place. Recommendation: flag only — moving files breaks absolute paths in scripts.

2. **Cache on internal SSD vs. ramdisk?** The 162 MB LoRAs load in <0.3s from Bolt, so the cache benefit is marginal for initial load. The real win is for hot-swap during live painting sessions where sub-second swap matters. Internal SSD is the right choice — ramdisk adds complexity for minimal gain.

3. **Multi-LoRA stacking limits?** Z-Image supports stacking via `LoRAApplicator.applyDynamically()` with additive adapters. Should the library track known-good stacking combinations? Defer to Phase 4 — let users discover combos, then record them.

4. **HuggingFace LoRAs:** `LoRAConfiguration` already supports `.huggingFace(modelId, filename)`. Should the library index HF-downloaded LoRAs? Yes — they're cached in `~/.cache/huggingface/` and should appear in the library with `source_url` set.
