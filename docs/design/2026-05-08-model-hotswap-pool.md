# ComfyBox Model Hot-Swap Design

**Issue**: (to be created)
**Date**: 2026-05-08
**Author**: Bree
**Status**: Design draft

## Problem

WarmServer loads one model at startup and cannot switch models without a full server restart. Krita users switching between Z-Image Turbo (fast), Klein 4B (creative), and FIBO (maquette) must restart the server each time — a 15-30 second interruption that breaks creative flow.

## Current Architecture

The `WarmCoordinator` actor holds:
- `pipeline: ZImagePipeline` — always created (Z-Image)
- `flux2Pipeline: Flux2Pipeline?` — lazy, created when Flux 2 detected
- `fiboPipeline: FiboPipeline?` — lazy, created when FIBO detected
- `chromaPipeline: ChromaPipeline?` — lazy, created when Chroma detected
- `currentModelFamily: WarmModelFamily` — routing flag

Model is loaded in `prepare()` which runs once at startup. The model spec comes from `configuration.modelSpec` which is immutable.

### Memory Profile (M3 Max 128GB)

| Model | VRAM (approx) | Load Time |
|-------|---------------|-----------|
| Z-Image Turbo BF16 | ~12 GB | ~5s |
| Z-Image Turbo Q8 | ~7 GB | ~3s |
| Klein 4B Q8 | ~5 GB | ~3s |
| Klein 9B Q8 | ~10 GB | ~5s |
| FIBO 8B 4-bit | ~8 GB | ~7s |
| Chroma 8.9B | ~17 GB | ~8s |
| SeedVR2 3B | ~7 GB | ~3s |

Total available: ~90 GB (128 GB - OS/apps). Can hold 2-3 models simultaneously.

## Design

### Approach: Multi-Model Pool (not swap)

Instead of unload-then-load (slow, wastes GPU memory during swap), keep multiple models resident and switch routing:

```
┌──────────────────────────────────────────────┐
│ Model Pool                                    │
│                                               │
│  [Z-Image Turbo] ← active                    │
│  [FIBO 4-bit]    ← loaded, idle              │
│  [SeedVR2 3B]    ← loaded, idle (upscaler)   │
│                                               │
│  Budget: 40 GB (configurable)                 │
│  LRU eviction when budget exceeded            │
└──────────────────────────────────────────────┘
```

### API

#### `POST /v1/model/load`

Load a model into the pool (does not activate it).

```json
{
  "model": "briaai/FIBO",          // model spec (HF ID or local path)
  "activate": true,                 // also make it the active model
  "quantization": "4bit",           // optional override
  "wait": true                      // block until loaded (default: true)
}
```

Response:
```json
{
  "status": "loaded",
  "model": "briaai/FIBO",
  "family": "fibo",
  "loadTimeMs": 7200,
  "vramEstimateMB": 8192,
  "poolSize": 3,
  "poolBudgetMB": 40960
}
```

#### `POST /v1/model/activate`

Switch the active model (must already be loaded).

```json
{
  "model": "briaai/FIBO"
}
```

Response: instant (just flips `currentModelFamily` + sets active pipeline).

#### `GET /v1/model/pool`

List all loaded models.

```json
{
  "active": "briaai/FIBO",
  "pool": [
    {"model": "z-image-turbo-bf16", "family": "flux1", "vramMB": 12288, "active": false, "lastUsed": "..."},
    {"model": "briaai/FIBO", "family": "fibo", "vramMB": 8192, "active": true, "lastUsed": "..."},
    {"model": "seedvr2-3b", "family": "seedvr2", "vramMB": 7168, "active": false, "lastUsed": "..."}
  ],
  "totalVramMB": 27648,
  "budgetMB": 40960
}
```

#### `POST /v1/model/unload`

Explicitly unload a model from the pool.

```json
{
  "model": "z-image-turbo-bf16"
}
```

Cannot unload the active model (switch first).

### Krita Bridge Integration

When Krita selects a different model from the model dropdown:
1. The bridge's `ComfyBridgeWorkflowParser` detects the model from `CheckpointLoaderSimple.ckpt_name`
2. If the model isn't in the pool → `POST /v1/model/load` with `activate: true`
3. If already in pool but not active → `POST /v1/model/activate`
4. If already active → no-op

From the user's perspective: select model in dropdown → next render uses it. First render after model switch has a load delay; subsequent renders are instant.

### Memory Management

**Budget enforcement:** When loading a new model would exceed the pool budget:
1. Find the least-recently-used non-active model
2. Unload it (release MLX arrays, nil the pipeline)
3. Force MLX garbage collection: `MLX.GPU.synchronize(); MLX.GPU.resetPeakMemory()`
4. Retry the load

**Pre-emptive loading:** The `COMFYBOX_PRELOAD` env var (from LoRA Library Manager design) also applies to models. Start the server with commonly used models already in the pool.

### Implementation

#### Phase 1: Core Pool + API

1. **New `ModelPool` actor** (`Sources/ZImage/Server/ModelPool.swift`):
   - Manages pipeline lifecycle for all model families
   - LRU tracking per model
   - Budget enforcement with eviction
   - Thread-safe via Swift actor

2. **Refactor `WarmCoordinator`**:
   - Replace individual pipeline properties with `ModelPool` reference
   - `prepare()` loads initial model into pool and activates
   - Generation methods query pool for active pipeline

3. **New endpoints** in WarmServer route handler:
   - `POST /v1/model/load`
   - `POST /v1/model/activate`
   - `GET /v1/model/pool`
   - `POST /v1/model/unload`

#### Phase 2: Krita Integration

4. **Bridge model detection** in `ComfyBridgeWorkflowParser`:
   - Extract model from `CheckpointLoaderSimple.ckpt_name`
   - Map to ComfyBox model spec
   - Auto-load/activate before generation

5. **ObjectInfo model list** from pool:
   - `CheckpointLoaderSimple.ckpt_name` options = pool-aware model list
   - Include both loaded and known-available models

#### Phase 3: MCP Integration

6. **MCP tools**:
   - `comfybox_load_model` — load model into pool
   - `comfybox_switch_model` — activate a pooled model
   - `comfybox_model_pool` — list pool status

### Safety

- Drain pending queue before unloading any model
- Never unload the active model — force explicit switch first
- Budget enforcement is advisory, not hard — a single model can exceed budget if needed (e.g., Chroma 17GB > 40GB/3)
- Log all load/unload/activate events for debugging

### Open Questions

1. Should activating a new model cancel pending renders from the old model?
2. Should the pool persist across server restarts (save pool state to JSON)?
3. Should Krita see ALL models or only pool-compatible ones in the dropdown?
