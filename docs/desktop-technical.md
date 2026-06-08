# ComfyBox Desktop Technical Documentation

Developer reference for the ComfyBox Desktop SwiftUI application. Covers architecture, module responsibilities, data flow, and extension points.

---

## 1. Architecture Overview

ComfyBox Desktop is one of two executable targets in the `comfybox` Swift package. Both share the `ZImage` library for core ML inference types and the `WarmServerClient` HTTP client.

```
Package.swift
  |
  +-- ZImage (library) -- ML models, inference, WarmServerClient
  |
  +-- ComfyBox (executable) -- CLI server (WarmServer) + generate tool
  |
  +-- ComfyBoxDesktop (executable) -- SwiftUI GUI client
```

The desktop app is a **pure client**. It does not run any ML inference directly. All generation, model loading, LoRA swapping, and prompt enhancement happens on the WarmServer via HTTP API calls. The app connects to a running ComfyBox server instance.

**Platform requirement:** macOS 14+ (Sonoma). Defined in `Package.swift` via `.macOS(.v14)`.

**Dependencies from ZImage:** The desktop target depends on `ZImage` primarily for the `WarmServerClient` type (HTTP client). It does not use MLX, Transformers, or any ML-specific types at runtime.

---

## 2. Module Map

### App Entry

| File | Purpose | Key Types |
|------|---------|-----------|
| `ComfyBoxDesktopApp.swift` | SwiftUI `@main` app, scene definition, tab routing, toolbar, initialization | `ComfyBoxDesktopApp`, `AppTab` |

### Core Services

| File | Purpose | Key Types |
|------|---------|-----------|
| `EngineService.swift` | Observable wrapper around WarmServerClient; bridges HTTP API to SwiftUI state | `EngineService`, `GenerationRequest`, `LoRASelection`, `ServerConnectionState`, `ModelInfo`, `PoolModelInfo`, `LoRAInfo`, `QueueInfo` |
| `PresetManager.swift` | Preset CRUD with JSON persistence | `PresetManager`, `GenerationPreset`, `PresetLoRA` |

### DAM (Digital Asset Management)

| File | Purpose | Key Types |
|------|---------|-----------|
| `DAM/DAMStore.swift` | Actor-isolated SQLite database for asset tracking | `DAMStore`, `DAMStoreError` |
| `DAM/DAMAsset.swift` | Asset data model (value type) | `DAMAsset` |
| `DAM/AssetIngestor.swift` | File watcher + thumbnail generator | `AssetIngestor` |

### Views

| File | Purpose | Key Types |
|------|---------|-----------|
| `Views/GenerationView.swift` | Main generation interface (prompt, params, preview) | `GenerationView`, `ResolutionPreset`, `SavePresetSheet` |
| `Views/GalleryView.swift` | Asset grid with search, filter, sort | `GalleryView`, `GalleryCellView`, `DraggableAsset`, `GallerySortOrder` |
| `Views/AssetDetailView.swift` | Full image view with metadata editing | `AssetDetailView` |
| `Views/ModelSelector.swift` | Model pool management UI | `ModelSelector` |
| `Views/LoRAPicker.swift` | LoRA selection with scale sliders | `LoRAPicker` |
| `Views/QueuePanel.swift` | Render queue status display | `QueuePanel` |
| `Views/SettingsView.swift` | App settings with persistence | `SettingsView`, `DesktopSettings` |
| `Views/PresetView.swift` | Preset list with CRUD | `PresetView`, `PresetEditorSheet` |
| `Views/CharacterLibraryView.swift` | Character browser with insert action | `CharacterLibraryView`, `CharacterEntry`, `CharacterCard`, `FlowLayout` |
| `Views/ComparisonGridView.swift` | Side-by-side image comparison with metadata diff | `ComparisonGridView`, `AsyncThumbnail` |

---

## 3. EngineService

`EngineService` is the central bridge between the WarmServer HTTP API and SwiftUI views. It is marked `@Observable` and all its published properties drive reactive UI updates.

### Lifecycle

1. **Construction:** `EngineService()` creates an idle instance with no server connection.
2. **Connection:** `connect()` creates a `WarmServerClient`, starts a health polling `Task` that fires every 3 seconds.
3. **First poll success:** State transitions to `.connected`. Triggers `refreshModels()` and `refreshLoras()`.
4. **Ongoing:** Health polls update `currentModel`, `queueCount`, `queueInfo`, and `connectionState`.
5. **Disconnection:** `disconnect()` cancels the poll task, nils the client, resets all state.

### Key API Methods

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `generate(_:)` | `POST /v1/generate` | Submit generation, return output path |
| `refreshModels()` | `GET /v1/models` | Fetch model registry |
| `refreshPool()` | `GET /v1/model/pool` | Fetch loaded model pool |
| `loadModel(id:quantization:activate:)` | `POST /v1/model/load` | Load model into pool |
| `activateModel(id:)` | `POST /v1/model/activate` | Switch active model |
| `unloadModel(id:)` | `POST /v1/model/unload` | Remove model from pool |
| `refreshLoras()` | `GET /v1/loras` | Fetch LoRA library |
| `swapLoras(_:)` | `POST /v1/lora/swap` | Hot-swap active LoRAs |
| `enhancePrompt(_:)` | `POST /v1/enhance` | LLM prompt rewriting |
| `fetchCharacters()` | `GET /v1/characters` | Get character registry |
| `pollHealth()` | `GET /health` | Server status (private, called by poll loop) |

### Threading

`EngineService` is `@Observable` (not an actor). All property mutations happen on the calling context. The health poll task uses `[weak self]` to avoid retain cycles. Generation and model operations are `async` methods called from SwiftUI `Task` blocks.

---

## 4. DAM System

### DAMStore (SQLite)

`DAMStore` is an **actor** providing serialized access to the SQLite database. This eliminates data races when the UI thread reads assets while the ingestor writes new ones.

**Database path:** `~/.comfybox/dam.sqlite3`

**Initialization:** `DAMStore.open()` is an async factory. It opens the database with `SQLITE_OPEN_FULLMUTEX`, enables WAL journal mode, and creates tables if needed.

**Schema:**

```sql
CREATE TABLE assets (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL DEFAULT 'image',
    filename TEXT NOT NULL,
    absolute_path TEXT NOT NULL UNIQUE,
    file_size INTEGER NOT NULL DEFAULT 0,
    sha256 TEXT,
    width INTEGER,
    height INTEGER,
    created_at REAL NOT NULL,
    modified_at REAL NOT NULL,
    ingested_at REAL NOT NULL,
    orphaned INTEGER NOT NULL DEFAULT 0,
    prompt TEXT,
    negative_prompt TEXT,
    seed INTEGER,
    steps INTEGER,
    guidance REAL,
    model_family TEXT,
    rating INTEGER NOT NULL DEFAULT 0,
    favorite INTEGER NOT NULL DEFAULT 0,
    content_mode TEXT,
    character_name TEXT
);

CREATE INDEX idx_assets_created ON assets(created_at DESC);

CREATE VIRTUAL TABLE assets_fts USING fts5(
    id UNINDEXED,
    prompt,
    negative_prompt
);
```

**FTS5:** Full-text search is maintained as a shadow table. When an asset with a prompt is inserted, the FTS index is updated. Search queries use `MATCH` with relevance ranking.

**WAL mode:** Enables concurrent reads during writes, important when the UI fetches assets while the ingestor is inserting.

### DAMAsset

A plain `Sendable` struct with all asset fields. Immutable by convention (let properties). Used for both database reads and inserts via `INSERT OR REPLACE`.

### AssetIngestor

`@Observable` class that watches the output directory for new image files.

**Polling strategy:** Scans the watch directory every 5 seconds for new `.png`, `.jpg`, `.jpeg` files. Maintains a `knownPaths` set to avoid re-ingesting.

**Stability check:** A file must have a modification time older than 1 second to be considered fully written.

**Ingestion pipeline per file:**
1. Read file attributes (size, dates).
2. Read image dimensions via `CGImageSourceCopyPropertiesAtIndex` (no full decode).
3. Read JSON sidecar metadata (`{filename}.json`) if present.
4. Build `DAMAsset` and insert into `DAMStore`.
5. Generate 256px JPEG thumbnail via `CGImageSourceCreateThumbnailAtIndex`.

**Sidecar format:** JSON file with optional fields: `prompt`, `negative_prompt`/`negativePrompt`, `seed`, `steps`, `guidance`, `model`/`modelFamily`/`model_family`, `contentMode`/`content_mode`, `characterName`/`character_name`/`character`. Both camelCase and snake_case variants are accepted.

**Thumbnail storage:** `~/.comfybox/thumbnails/{assetId}.jpg`, 256px max dimension, 80% JPEG quality.

---

## 5. View Hierarchy

```
ComfyBoxDesktopApp (Scene)
  |
  +-- WindowGroup
  |     +-- NavigationSplitView
  |           +-- Sidebar: Tab list (Generate, Gallery, Compare, Presets)
  |           +-- Detail: Router (switch on selectedTab)
  |                 +-- GenerationView
  |                 |     +-- ModelSelector (collapsible)
  |                 |     +-- LoRAPicker (collapsible)
  |                 |     +-- CharacterLibraryView (collapsible)
  |                 |     +-- QueuePanel (collapsible)
  |                 |     +-- SavePresetSheet (modal)
  |                 |
  |                 +-- GalleryView
  |                 |     +-- GalleryCellView (per cell)
  |                 |     +-- AssetDetailView (modal sheet)
  |                 |
  |                 +-- ComparisonGridView
  |                 |     +-- PickerCell (selection mode)
  |                 |     +-- AsyncThumbnail (comparison mode)
  |                 |
  |                 +-- PresetView
  |                       +-- PresetRow (per row)
  |                       +-- PresetEditorSheet (modal)
  |
  +-- Settings (Scene)
        +-- SettingsView (TabView: Server, Generation, Gallery)
```

### Dependency Injection

All services are created at the app level and passed down:

- `EngineService` -- created as `@State` in app, passed as `@Bindable` to views.
- `DAMStore` -- created async on launch, passed by value.
- `AssetIngestor` -- created after DAMStore, passed by value.
- `PresetManager` -- created as `@State` in app, passed to GenerationView and PresetView.
- `characters: [CharacterEntry]` -- fetched on launch, passed as array.

---

## 6. Data Flow

### Generation Request Flow

```
User fills prompt/params in GenerationView
    |
    v
GenerationView.submitGeneration()
    |
    +-- If LoRAs selected: engine.swapLoras(selections)
    |       POST /v1/lora/swap
    |
    +-- engine.generate(request)
    |       POST /v1/generate {prompt, width, height, steps, guidance, seed, outputPath}
    |       <- {success, outputPath, durationMs}
    |
    +-- Load NSImage from outputPath for preview display
    |
    +-- onGenerated(outputPath, request)  -- callback to app
            |
            v
        ComfyBoxDesktopApp.handleGenerated()
            |
            v
        ingestor.ingestFile(at: outputPath)
            |
            +-- Build DAMAsset from file + sidecar
            +-- store.insertAsset(asset)
            +-- generateThumbnail()
            |
            v
        GalleryView auto-refreshes (observes ingestor.ingestedCount)
```

### Health Polling Flow

```
EngineService.connect()
    |
    v
healthPollTask (every 3s)
    |
    +-- GET /health
    |       <- {status, model, model_family, is_rendering, pending_count, ...}
    |
    +-- Update @Observable properties:
            connectionState, currentModel, queueCount, queueInfo
    |
    +-- SwiftUI views reactively update
```

### Preset Flow

```
Save: GenerationView -> SavePresetSheet -> presetManager.create()
                                              -> append to presets array
                                              -> JSON encode to ~/.comfybox/presets.json

Load: PresetView tap -> onApply(preset) -> app switches to .generate tab
                                        -> GenerationView.applyPreset()
                                              -> populate prompt, steps, guidance, etc.
```

---

## 7. Preset System

### Storage Format

File: `~/.comfybox/presets.json`

```json
[
  {
    "id": "UUID",
    "name": "My Preset",
    "promptTemplate": "a detailed prompt...",
    "modelId": "huggingface/model-id",
    "loras": [
      {"id": "lora-id", "filename": "lora.safetensors", "scale": 0.8}
    ],
    "steps": 9,
    "guidance": 3.5,
    "width": 1024,
    "height": 1024,
    "sampler": null,
    "createdAt": "2026-06-08T10:00:00Z",
    "modifiedAt": "2026-06-08T10:00:00Z"
  }
]
```

Dates use ISO 8601 encoding. JSON is pretty-printed with sorted keys.

### CRUD Operations

- **Create:** `PresetManager.create()` builds a `GenerationPreset` from parameters, appends to array, saves.
- **Update:** `PresetManager.update()` finds preset by ID, updates `modifiedAt`, saves.
- **Delete:** `PresetManager.delete(id:)` removes by ID, saves.
- **Duplicate:** `PresetManager.duplicate()` copies all fields, assigns new UUID and name suffix, saves.

All mutations call `save()` which re-encodes the entire array to disk. There is no incremental persistence.

---

## 8. Configuration

### DesktopSettings

Defined in `SettingsView.swift`. Codable struct persisted at `~/.comfybox/desktop-config.json`.

```json
{
  "serverHost": "127.0.0.1",
  "serverPort": 7862,
  "autoConnect": true,
  "outputDirectory": "/Users/todd/Pictures/ComfyBox",
  "defaultSteps": 9,
  "defaultGuidance": 3.5,
  "defaultWidth": 1024,
  "defaultHeight": 1024,
  "thumbnailSize": 180,
  "gallerySortDefault": "date"
}
```

**Load precedence:** File on disk > hardcoded defaults. No environment variable overrides.

**Application on launch:** `ComfyBoxDesktopApp.applySettings()` reads `DesktopSettings.load()` and pushes values to `EngineService` properties (`serverHost`, `serverPort`, `outputDirectory`). If `autoConnect` is true, calls `engine.connect()`.

**Save:** `DesktopSettings.save()` creates the parent directory if needed, encodes to JSON with pretty printing and sorted keys, writes atomically via `Data.write(to:)`.

---

## 9. Build and Distribution

### SPM Build

```bash
swift build --product ComfyBoxDesktop              # debug
swift build -c release --product ComfyBoxDesktop    # release
```

The desktop target compiles as an executable (not an app bundle). For distribution as a `.app`, an Xcode project wrapping the SPM package is recommended.

### Xcode

Open `Package.swift` in Xcode. Select the `ComfyBoxDesktop` scheme. Build and run targets macOS 14+.

### Code Signing

Currently unsigned. For distribution outside the App Store:
1. Add a signing identity to the Xcode project.
2. Enable Hardened Runtime.
3. Notarize with `xcrun notarytool`.

### Required Entitlements

- **Network Client** -- outbound HTTP to WarmServer.
- **File System Access** -- read/write to `~/.comfybox/` and the output directory.

---

## 10. API Surface

All endpoints are on the WarmServer HTTP API. The desktop app is a client.

### Health

```
GET /health
Response: {
  status: string,
  model: string?,
  model_family: string?,
  loaded: bool?,
  is_rendering: bool?,
  pending_count: int?,
  render_count: int?,
  uptime_seconds: int?,
  last_render_duration_ms: int?,
  last_error: string?,
  loras: [{source: string, scale: float}]?,
  memory_usage_mb: uint64?
}
```

### Generation

```
POST /v1/generate
Body: {
  prompt: string,
  width: int,
  height: int,
  steps: int,
  guidance: float,
  outputPath: string,
  seed: uint64?    (omit or 0 for random)
}
Response: {
  success: bool,
  outputPath: string,
  durationMs: int
}
```

### Models

```
GET /v1/models
Response: {
  models: [{
    id: string,
    family: string,
    variant: string,
    quantization: string,
    display_name: string,
    description: string,
    parameters_b: float?,
    default_steps: int?,
    default_guidance: float?,
    supports_guidance: bool?,
    supports_lora: bool?,
    default_resolution: string?,
    estimated_vram_gb: float?,
    huggingface_id: string?
  }]
}

GET /v1/model/pool
Response: {
  pool: [{
    model: string,
    family: string,
    vram_mb: int?,
    active: bool?,
    last_used: string?
  }]
}

POST /v1/model/load
Body: { model: string, quantization: string?, activate: bool, wait: bool }

POST /v1/model/activate
Body: { model: string }

POST /v1/model/unload
Body: { model: string }
```

### LoRAs

```
GET /v1/loras
Response: {
  loras: [{
    id: string,
    filename: string,
    model_compatibility: string?,
    format: string?,
    rank: int?,
    size_bytes: int?,
    quarantined: bool?,
    tags: [string]?,
    category: string?,
    triggerwords: [string]?,
    recommended_scale: float?
  }],
  active_loras: [string]
}

POST /v1/lora/swap
Body: { loras: [{path: string, scale: float}] }
```

### Enhancement

```
POST /v1/enhance
Body: { prompt: string }
Response: { prompt: string }
```

### Characters

```
GET /v1/characters
Response: {
  characters: [{
    id: string,
    name: string,
    description: string?,
    default_loras: [string]?,
    prompt_snippet: string?,
    tags: [string]?
  }]
}
```

---

## 11. Concurrency Model

### Actor Isolation

- **DAMStore** is the only actor in the codebase. All database reads and writes are serialized through it. Callers `await` its methods.
- **EngineService** is `@Observable`, not an actor. It relies on being called from the main actor context (SwiftUI views).
- **PresetManager** is `@Observable`, not an actor. Array mutations and file I/O happen synchronously.

### @MainActor Usage

Views are implicitly `@MainActor` via SwiftUI. `EngineService` property updates happen on whatever thread the caller is on -- in practice, always the main actor since all calls originate from SwiftUI `Task {}` blocks or `.task` modifiers.

### Task Patterns

- **Health polling:** Long-running `Task` started in `connect()`, cancelled in `disconnect()`. Uses `[weak self]` to allow deallocation.
- **Thumbnail loading:** `Task.detached` is used in `GalleryCellView` and `AsyncThumbnail` to load `NSImage` off the main thread, then `MainActor.run` to assign the result.
- **Generation:** Async `Task` blocks in `submitGeneration()`. LoRA swap and generate are sequential awaits.
- **File scanning:** Background `Task` in `AssetIngestor` polls every 5 seconds with `Task.sleep`.

### Sendable Conformance

- `DAMAsset` is `Sendable` (all stored properties are value types or optionals thereof).
- `GenerationRequest` is `Sendable`.
- `LoRASelection` is `Sendable`.
- `CharacterEntry` is `Sendable`.
- `GenerationPreset` is `Sendable` and `Codable`.

---

## 12. Extension Points

### Adding a New Generation Parameter

1. Add the field to `GenerationRequest` in `EngineService.swift`.
2. Add a UI control in `GenerationView.swift` (in the `parameterSection`).
3. Include the field in the JSON payload dict in `EngineService.generate()`.
4. Add the field to `GenerationPreset` and `PresetManager` if it should be saveable.
5. Update the preset editor sheet to expose the field.

### Adding a New View/Tab

1. Add a case to `AppTab` in `ComfyBoxDesktopApp.swift` with icon and shortcut key.
2. Add a `case` to the `detailView` switch in the app.
3. Create the view file in `Sources/ComfyBoxDesktop/Views/`.
4. Add the keyboard shortcut in the `commands` block.

### Adding DAM Metadata Fields

1. Add the column to `DAMAsset`.
2. Add the column to the `CREATE TABLE` statement in `DAMStore.createTables()`.
3. Add bind/read calls in `insertAsset()` and `assetFromRow()`.
4. Read from sidecar JSON in `AssetIngestor.readSidecar()`.
5. Display in `AssetDetailView.generationSection`.
6. Add to `ComparisonGridView.metadataDiffView` if it should appear in comparisons.

Note: Adding columns to an existing database requires a migration. The current schema uses `CREATE TABLE IF NOT EXISTS`, so new columns need an `ALTER TABLE ADD COLUMN` migration path, or the user must delete the database.

### Adding a New WarmServer API Integration

1. Add the response type (private `Decodable` struct) in `EngineService.swift`.
2. Add the public method with `async throws` signature.
3. Add any `@Observable` state properties for the UI.
4. Wire the method to a view action.
5. Document the endpoint in the API Surface section above.

### Adding Gallery Filters

1. Add the filter state variable in `GalleryView` (e.g., `@State private var filterMyField: String?`).
2. Extract distinct values in `extractFilterValues()`.
3. Add filter logic in the `filteredAssets` computed property.
4. Add a `Picker` or `Toggle` in the `toolbarView`.
