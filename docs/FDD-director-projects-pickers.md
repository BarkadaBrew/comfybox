# FDD: Director projects and pickers

Status: draft v0.1 (2026-09-17). Nothing here is built.
Requested by Todd: "we should design the ui to include projects and pickers to make assembly for the user to be intuitive and easy, refer to the github for design language."
Design reference: [WhatDreamsCost-ComfyUI](https://github.com/WhatDreamsCost/WhatDreamsCost-ComfyUI) `js/ltx_director.js` (LTX Director 2.0). Copy its behaviour and look, not its code (GPL-3.0, same rule as `docs/FDD-ltx-director-tab.md` §4.5).
Builds on: `docs/FDD-ltx-director-tab.md` v1.2 plus its §10 Phase 1 deltas. Engine Phase 1 is merged in `~/Projects/zimage-temporal`.

## 1. Problem, goals, non-goals

**Problem.** The Director tab (`Sources/ComfyBoxDesktop/Views/Director/*`) edits one loose `.cbdirector` file at a time. Its only memory is one global autosave (`~/.comfybox/director-autosave.cbdirector`, `Director/DirectorAutosaveStore.swift`). Every asset comes from a Finder drop (`DirectorKeyframeTrack.onDrop`, `DirectorAudioTrack.onDrop`). Renders land as `director-<epoch>.mp4` in the output directory (`DirectorView.generate`), and nothing links a render back to the timeline and seed that made it. To assemble a clip today you have to find stills in Finder, remember which render came from which edit, and hand-copy a last frame to continue a shot.

**Goals**
1. **A project is the unit of work.** It holds timeline versions, its assets, and every render with the exact timeline snapshot and resolved seed that produced it.
2. **Pickers bring in assets without Finder.** Stills come from the gallery, clips and last frames from renders, audio with trim, and a character or preset in one or two clicks. The same entry points work everywhere: toolbar, gap "+", right-click "Replace with…", and drag-drop.
3. **Assembly flows forward.** The empty state leads to a first keyframe. Drops show where they will land. Validation issues sit on the clip they concern. Progress shows per chunk, and the result lands in the project, ready for "continue from last frame".
4. **The look follows LTX Director.** The timeline and pickers use its visual language, adapted to macOS conventions.

**Non-goals**
- No engine changes to rendering, the recipe, or the timeline wire schema. `DirectorTimeline` v1 stays byte-identical.
- No server-side project store. The deferred `GET/PUT /v1/director/timelines/{id}` stays deferred.
- No Phase 2 tracks (Retake, Reference, Regions). The pickers reserve their entry points but leave them disabled.
- No new content filtering. Pickers inherit the existing `AppContentGate` behaviour exactly as the Gallery tab does. That is parity, not a new rule.
- No multi-user or cloud sync.

## 2. Project model

### 2.1 On-disk layout

Projects live under the desktop output directory (`DesktopSettings.outputDirectory`). The engine contains output paths to its allowed output directory (`WarmServer.swift` submit-time output resolution, `DirectorPayload.outputPath` doc comment). A project inside that directory can therefore receive its own renders without relaxing containment.

```
<outputDirectory>/Director/
  <project-slug>-<shortid>/
    project.json                 # manifest (§2.2)
    timelines/<timeline-id>.cbdirector   # DirectorDocument bytes, unchanged format
    assets/<asset-id>.<ext>      # imported/pasted/extracted media (copied in)
    renders/<render-id>/
      timeline.cbdirector        # snapshot submitted
      plan.json                  # DirectorPlan from 202 / final status
      output.mp4                 # engine output_path
      lastframe.png              # extracted on success (LastFrameExtractor)
      thumb.jpg                  # poster frame for pickers
~/.comfybox/director/recents.json          # [{project_path, name, opened_at}] (desktop-owned)
```

- **Gallery picks are referenced, not copied.** They go by absolute path plus catalog id. Finder imports, pastes and extracted frames are copied into `assets/` so the project stays self-contained.
- **Moving a project folder breaks references.** Gallery references hold absolute paths, so they break when the folder moves; project-internal paths resolve relative to `project.json` at load time and survive the move. Wire timelines always carry absolute paths, because the engine reads them.
- **Legacy autosave migrates once.** On first launch the old autosave file becomes an "Untitled (recovered)" project, then the file is left in place.

### 2.2 `project.json` schema (v1, snake_case, sorted keys)

```jsonc
{
  "schema": "comfybox.director.project", "version": 1,
  "id": "p_7f3a", "name": "Rooftop dusk", "created_at": "...", "updated_at": "...",
  "archived": false,
  "current_timeline_id": "t_03",
  "timelines": [
    { "id": "t_01", "name": "v1", "file": "timelines/t_01.cbdirector",
      "parent_id": null, "created_at": "...", "note": "" }
  ],
  "assets": [
    { "id": "a_12", "kind": "image|audio|video", "path": "assets/a_12.png",
      "source": "import|paste|gallery|render_lastframe|render_frame",
      "catalog_id": null, "render_id": null, "sha256": "...", "label": "", "added_at": "..." }
  ],
  "renders": [
    { "id": "r_05", "timeline_id": "t_03", "job_id": "…",
      "snapshot": "renders/r_05/timeline.cbdirector", "plan": "renders/r_05/plan.json",
      "output": "renders/r_05/output.mp4",
      "preset": "<preset id>", "seed": 771144, "fps": 24, "width": 576, "height": 896,
      "length_frames": 289, "chunks": 1,
      "status": "queued|processing|done|failed|interrupted",
      "error": null, "submitted_at": "...", "finished_at": "...", "elapsed_s": 912.0,
      "rating": null, "note": "" }
  ]
}
```

- **Timeline files are plain `DirectorDocument` output** (`Sources/ZImage/Video/Director/DirectorDocument.swift`). They open with the CLI (`comfybox director-render`) and with the existing Open panel.
- **The recorded seed is the effective one.** It comes from `plan.chunks[0].seed`, so a preset-resolved seed is captured even when `settings.seed` is nil.
- **Preset and LoRAs are recorded by id and path only.** Prompt text lives in the snapshot, never duplicated into the manifest.

### 2.3 Lifecycle

| Action | Behaviour |
|---|---|
| New | Named sheet: name, resolution, length. Creates the folder and `t_01`. ⌘N inside the tab. |
| Open / Recent | Project switcher popover in the toolbar, listing recents with a poster thumbnail (the latest render's `thumb.jpg`). File ▸ Open Recent. |
| Autosave | The current `DirectorAutosaveStore` debounce (1 s), retargeted to `timelines/<current>.cbdirector`. The manifest `updated_at` is written in the same flush. |
| Save Version | ⌘S snapshots the current timeline into a new `t_NN` whose `parent_id` is the previous one. Autosave keeps editing the head. Versions are cheap; no "unsaved changes" dialog inside a project. |
| Duplicate | Copies the manifest, timelines and assets. Renders are left out, with a checkbox to include them. |
| Rename | Changes `name` only. The folder slug stays. |
| Archive | `archived: true` hides the project from recents and the switcher unless "Show archived" is on. No deletion from the UI; deleting happens in Finder. |
| Export `.cbdirector` | Existing Save As with "Embed assets" (`DirectorView.saveDocument`). |
| Import `.cbdirector` | Creates a project around it. Referenced files stay referenced. |

### 2.4 Render history and compare

A **Renders** strip under the player shows cards newest first, each with a poster, status dot, v-name, seed, chunks, elapsed time and rating.

**Card actions**
- Play.
- Compare.
- Restore timeline: opens the snapshot as a new version.
- Continue from last frame: a new version whose keyframe at frame 0 is `lastframe.png`.
- Use frame at playhead as keyframe.
- Reveal in Finder.
- Copy seed.

**Compare** opens two renders side by side, each in its own `AVPlayer` with one shared scrubber. A diff table sits underneath: settings, global prompt, segments, keyframe frames and strengths, seed. It follows the `diffRow` pattern in `Views/ComparisonGridView.swift`.

**Durability.** Local video jobs are non-durable (FDD §4.3). When a project opens, any `queued` or `processing` record gets one `GET /v1/video/status/{job_id}`. A 404 or a failure marks it `interrupted` and offers "Resubmit snapshot".

## 3. Pickers

### 3.1 One picker shell

All pickers share one shell, `DirectorPickerPanel`. It opens as a popover from a point on the timeline, or as a docked **Library** drawer on the left of the timeline, like an NLE media bin.

**Anatomy**
- **Source tabs:** Project, Renders, Gallery, Finder, Clipboard.
- **Search field:** filters as you type.
- **Filter chips:** character, rating ≥, kind, date.
- **Grid:** thumbnail cells, 96–160 pt, resizable with ⌘+/⌘−.
- **Recents row:** the last 12 items used in this project, then across projects.

**Keyboard**
- Arrows move the selection.
- Return inserts at the playhead, or at the invoking gap or clip.
- Space opens Quick Look.
- ⌘F focuses search.
- Esc closes.

**Drag-out.** Every cell drags out as an `NSItemProvider` carrying a file URL plus a `com.barkadabrew.director.asset` payload (asset id and kind). The timeline gets ghost placement for picker drags and Finder drags alike.

**Data sources**
- **Gallery:** the ComfyBoxGallery catalog on `:7871`, `GET /v1/catalog/search?kind=image|video&q=&character=&min_rating=&order=newest&limit=&offset=` (`Sources/ComfyBoxCatalog/GalleryServer.swift:80`, params `:246-259`). The desktop already wraps this in `DAM/CatalogBrowser.swift`, which also owns `hiddenAssetIDs`.
- **Local path:** the row's `path`, or `locations` host `mac` from `GET /v1/catalog/asset/{id}`. A row without a local copy is shown but disabled, marked "not on this Mac".
- **Thumbnails:** `AsyncThumbnail` (`Views/ComparisonGridView.swift:366`).
- **Clip lineage:** the catalog `i2v_source` edge (`CatalogModels.swift:16`) lets a clip cell show "animated from" and offer that still.

### 3.2 Per picker

| Picker | Sources | Result on the timeline | Specifics |
|---|---|---|---|
| **Keyframe image** | Project assets, Renders (last frame or any frame), Gallery `kind=image`, Finder (`NSOpenPanel`, `ImageDropSupport.handleImageDrops`), Clipboard (`NSPasteboard` image, written to `assets/`) | `Keyframe{image_path, frame snapped to 8, strength 1.0}` through `DirectorDocumentModel.addKeyframe` | The cell badge shows the source aspect ratio and warns when it differs from the timeline (engine center-crops). "Set as end frame" is an insert option. |
| **Video / last frame** | Project renders, Gallery `kind=video` | A still extracted to `assets/` (source `render_lastframe` or `render_frame`), inserted as a keyframe | Inline scrubber with a "Last frame" default button. Extraction uses ZImage `LastFrameExtractor` (the exact last frame, #461) or `AVAssetImageGenerator` at a chosen time with zero tolerance. "Use as reference clip" is shown disabled (Phase 2). |
| **Audio (Load Audio UI style)** | Project assets, Finder (WAV/AIFF/MP3/M4A/AAC), Recents | `AudioClip{audio_path, start_frame, length_frames, trim_start_frames, gain}` via `addAudioClip`/`trimAudioClip`; switches `audio.mode` to `imported` after confirming | Waveform (`DirectorWaveformView` peaks), draggable in and out handles in frames at timeline fps, play selection, gain slider, and a readout of trimmed length vs the timeline. |
| **Character** | `GET /v1/characters` (`EngineService.fetchCharacters`, `CharacterEntry`) | `settings.character` (nil = no injection) | `CharacterEntry` has no image, so the cell thumbnail is the newest top-rated catalog still with `character=<name>`. A secondary action inserts that still as a keyframe. Shows name, kind and tag chips only. |
| **Preset / recipe** | `GET /v1/presets` filtered to `mediaKind == "video"` (already used in `DirectorSidebar.swift:172`) | `settings.preset` | A popover list replaces the sidebar `Picker`. Each row shows id, name, steps, fps, LoRA count and whether it forces temporal upscale, which is a known submit-time 400 (`temporal_upscale_unsupported`). Shows id and name only, never preset prompt text. LoRAs reuse `Views/LoRAPicker.swift`. |
| **Render version** | `project.json` renders | Player, compare, restore, continue | The same cards as §2.4, as a picker for "Compare with…" and "Continue from…". |

### 3.3 Entry points (uniform across pickers)

| Entry | Opens |
|---|---|
| Toolbar **Add Image / Add Audio / Add Video** (left group, as upstream "Add Image/Text/Audio/Video") | The picker for that kind. Inserts at the playhead. |
| Gap **"+"** button on a track (shown on hover in empty spans ≥ 16 frames) | The kind for that track, anchored to the gap. Inserts at the gap start. |
| Right-click clip ▸ **Replace with…** ▸ Project asset / Render frame / Gallery / Finder / Copied image | The picker pre-filtered. Replaces `image_path` or `audio_path` and keeps frame, strength, trim and gain. Upstream: "Replace Segment", "Replace with Copied Image". |
| Right-click gap ▸ Add Image here / Add Audio here / Add Prompt here / Paste | Same as "+". |
| Drag from Library drawer, Finder or Gallery tab | Ghost placement (§4.2). |
| Sidebar Character / Preset rows | Their pickers. |
| Keyboard ⇧⌘I (image), ⇧⌘A (audio), ⇧⌘K (character) | The picker at the playhead. |

## 4. Assembly flow

### 4.1 Empty and first run

**No project open.** The tab shows a centered card with "New Project", "Open Recent" (up to 6 posters) and "Import .cbdirector".

**Empty timeline in a new project.** The Keyframes track shows two ghost targets, **"Drop first frame"** at frame 0 and **"Drop last frame"** at the end. Each is also a click target that opens the Keyframe picker. The Prompt row reads "Describe the whole shot" and focuses the GLOBAL PROMPT field. The validator already reports `missing_global_prompt` and `no_keyframes`. In the empty state these show as inline hints, not red errors, until the first edit.

### 4.2 Placing

**Ghost preview.** While dragging, a translucent clip is drawn at the snapped frame: the 8-frame latent grid for keyframes, always, and edge or clip magnet snapping within 15 px, as upstream at `ltx_director.js:1624`. A "Drop to Place" label and a vertical guide line appear. An invalid target (past the end, a Phase 2 track) shows the ghost in red with the reason.

**Push physics.**
- **Keyframe drops.** Today a collision is resolved silently by `addKeyframe` searching for the nearest free bucket, and `moveKeyframe` refuses an occupied one. The UI changes both:
  - On drop, the neighbour is pushed forward one grid step (and so on down the chain) if there is room. Otherwise the ghost shows the nearest free frame it will actually use.
  - While dragging, neighbours visibly slide instead of the move being refused.
- **Prompt segments and audio clips.** Dropping or dragging into an occupied span pushes later items right up to the timeline end. At the end it trims the pushed item, with a toast and undo.
- **Joints.** Where two prompt segments abut, the shared edge is a joint handle. Dragging it ripple-trims both sides, as upstream.

All of this is model code (`DirectorDocumentModel`), unit-tested, with the views only hit-testing. That matches the WP3 rule that the canvas stays a thin renderer.

### 4.3 Editing prompts

- **Where prompts are shown.** Selecting a segment shows its text in the **SEGMENT PROMPT** textarea below the timeline, beside **GLOBAL PROMPT**. Both have floating uppercase labels, as upstream. A subtitle strip on the prompt track shows the first line of each segment.
- **Keyboard.** Tab and ⇧Tab walk segments in time order. ⌘Return leaves the text field.
- **Existing features stay.** Enhance per segment (existing optimizer path) and the SpeechLength readout (`Director/SpeechLength.swift`) are kept.

### 4.4 Validate as you go

- **Two validation passes.**
  - Local validation already runs after every edit (`model.autoValidate`, `validateLocally`).
  - When the engine is connected, a debounced (750 ms) `POST /v1/video/director/validate` also runs. It is pure: no GPU, no queue (`WarmServer.swift:1558`, `directorValidateBody :6191`). It returns `snapped_length_frames`, `plan` and `issues`.
- **Rendering issues.** Each `DirectorIssue` has `ids`.
  - Errors draw a red outline and a corner badge on those clips; warnings draw amber.
  - Hover shows the message. Clicking the badge selects the clips (`model.select(issue:)`).
  - Issues with no ids (`timeline_too_short`, `generated_audio_seams`, `length_snapped`) appear as a slim banner above the ruler with a count, expandable to a list.
- **Plan overlay.** The plan draws chunk boundaries (`boundary_frames`) as dashed ruler ticks with chunk numbers, and keyframe ticks for the compiled placement, which shows an end frame pinned to `length-1`.
- **Generate button.** Disabled only by errors (`canGenerate`), with the first error as its tooltip.

### 4.5 Submit, progress, result

1. **Snapshot and record.** Generate writes `renders/<id>/timeline.cbdirector` and appends a `queued` record.
2. **Submit.** It calls `POST /v1/video/director` (`EngineService+Director.submitDirectorJob`) with `output_path = renders/<id>/output.mp4` and `source: "desktop-director"`. It stores `job_id` and the 202 `plan`.
3. **Chunk progress.** The poll (`pollVideoStatus`) reads `stage_index`, `stage_count` and `progress_percent`. The timeline ruler shows a segmented progress bar, one segment per plan chunk: finished chunks solid green, the active one filling, pending ones hollow. The status line keeps `directorStatusLine`. The record updates on every poll.
4. **Success.**
   - The record is marked `done` with elapsed time, and the effective seed is taken from the final plan.
   - `lastframe.png` and `thumb.jpg` are extracted, and the player loads the output.
   - The card slides into the Renders strip, and a "Continue from last frame" chip is offered.
5. **Failure.** The record is marked `failed` with the `chunk k/n (stage): message` text. The card offers "Open snapshot" and "Resubmit". No automatic retry, the same as today.

## 5. Design language

### 5.1 Patterns adopted from LTX Director

| Element | Upstream (`js/ltx_director.js`) | ComfyBox mapping |
|---|---|---|
| Surface greys | ComfyUI neutrals `#111`–`#2a2a2a` | `DirectorTheme` tokens: `canvasBG #111`, `trackBG #1a1a1a`, `clipBG #242424`, `rulerBG #161616`, `border #2a2a2a` |
| Selection | White stroke, pill-shaped trim handles | 1.5 pt white stroke, 6×18 pt capsule handles on the leading and trailing edges |
| Playhead | Red `#ff4444` line with a shield handle | Same colour. A `Path` shield at the ruler top, draggable. |
| Active / audio | Green `#4fff8f` | Waveforms, ON pills, finished-chunk progress |
| Toggles | Muted navy `#1c222d` "toggle-on" | Toggle backgrounds (snap, track eye/speaker) |
| Marquee | Blue | Marquee rect only (system accent is not used elsewhere on the canvas) |
| Geometry | Ruler 24 px, image block 160, audio 80, motion 80 (`:5-9`) | Ruler 24, Keyframes 120 (tiled thumbnails need height; today 76), Prompts 56 plus subtitle strip, Audio 80, Reference 28 disabled |
| Track label sidebar | 120 px, MAIN / AUDIO / IC-LoRA, eye/speaker toggles, tiny ON/OFF pills | 120 pt (today `labelWidth` 86). Labels KEYFRAMES / PROMPTS / AUDIO / REFERENCE in 10 pt semibold tracking. Eye (hide) and speaker (mute preview) toggles, plus an ON/OFF pill for audio mode imported vs generated. |
| Clips | Thumbnails tiled across image clips; IMAGE / VIDEO badges | Keyframe thumbnail tiled across its hold span up to the next keyframe (visual only). Badges IMAGE / LAST FRAME / END. |
| Gaps | "+" buttons, ghost "Drop to Place" | §3.3, §4.2 |
| Snapping | Magnet icon, 15 px threshold, S toggles | Same threshold. Magnet button with `toggle-on` navy. |
| Toolbar | Left: Add Image/Text/Audio/Video + Delete. Right: snap / in / out / mark / help / settings icons, 28 px buttons | Left: Add Image, Add Prompt, Add Audio, Add Video (disabled), Delete. Right: Snap, Zoom −/+, Help (shortcut sheet), Timeline Settings popover. 28 pt square borderless buttons. |
| Transport | Monospace timecode row | `mm:ss:ff · frame` in `.monospacedDigit()`, play/pause, to-start/to-end |
| Prompt fields | SEGMENT PROMPT, GLOBAL PROMPT textareas with floating uppercase labels | Two `TextEditor`s under the timeline with overlaid 9 pt uppercase labels |
| Context menus | Copy Segment, Paste, Replace Segment, Replace with Copied Image, Split; gap menu | Native `NSMenu` via `.contextMenu`: Copy, Paste, Replace with ▸, Split at Playhead, Set as End Frame, Strength ▸, Delete |
| Save/load | "Save Timeline", "Save Timeline As", "Load Timeline" JSON with paths | Project versions (§2) plus `.cbdirector` export/import |

### 5.2 SwiftUI and AppKit mapping

- **Timeline in one `Canvas`.** Render the ruler, tracks, clips, thumbnails, waveforms, issue badges, ghosts, playhead and chunk-progress bar in a single SwiftUI `Canvas` inside the horizontal `ScrollView`. Today they are per-clip `View`s in `DirectorTrackViews.swift`. Tiled thumbnails and push animations at many clips need immediate-mode drawing. Put a transparent hit-test overlay on top, with a `DragGesture`, `onDrop` and `contextMenu` per hit region. `DirectorLayout` stays the only px↔frame mapping.
- **Resolved images.** Thumbnails come from a small `NSCache` keyed by path and size, handed to `Canvas` via `context.resolve(Image(nsImage:))`.
- **Popovers.** The picker is an `NSPopover`-backed SwiftUI `.popover`. The Library drawer is an `HSplitView` pane.
- **Commands.** Project commands move to `.commands` (`CommandGroup(replacing: .newItem)` scoped by focused tab). The hidden-button pattern in `DirectorView.shortcutButtons` goes, and ⌘N is freed.

### 5.3 Deliberate macOS deviations

- **Appearance.** The timeline canvas is always dark, as in Final Cut and Resolve, because colour judgement of thumbnails needs a neutral dark surround. The sidebar, pickers, sheets and menus follow system light or dark. Upstream is dark-only because ComfyUI is.
- **Menus.** Native context menus and a menu bar instead of HTML floating menus. They inherit key equivalents and accessibility.
- **Shortcuts (upstream first, then macOS):**
  - Space: play.
  - ⌫ or Delete: remove.
  - ⌘B: split at playhead (upstream Ctrl+B).
  - ⌘C/⌘V: copy and paste clips; paste of an image from the pasteboard creates a keyframe.
  - S: toggle snap (upstream). This changes the current tab, where S splits; the split moves to ⌘B.
  - ⌘Z/⇧⌘Z: undo and redo.
  - ⌘=/⌘−: zoom.
  - Not adopted: upstream I/O (in and out points were dropped in Phase 1, §10 of the Director FDD) and X "mark".
  - Shortcuts fire only when the timeline has focus, which is the macOS counterpart of upstream's hover check.
- **Units.** Points not pixels. Selection strokes are hairline-aware on Retina.
- **Accessibility.** Each hit region exposes an accessibility element: label, frame, and issues.

## 6. Engine and API mapping

| Need | Route / code | Status |
|---|---|---|
| Validate as you go | `POST /v1/video/director/validate` | Exists |
| Submit and plan | `POST /v1/video/director` (202, `plan`) | Exists |
| Chunk progress | `GET /v1/video/status/{id}` + `plan`, `stage_index`, `stage_count` | Exists |
| Gallery stills and clips | `:7871 /v1/catalog/search`, `/v1/catalog/asset/{id}`, `/v1/catalog/facets` | Exists (desktop `CatalogBrowser`) |
| Presets | `GET /v1/presets` | Exists |
| Characters | `GET /v1/characters` | Exists |
| LoRAs | Existing LoRA catalog used by `LoRAPicker` | Exists |
| Last frame | ZImage `LastFrameExtractor` (in-process in the desktop) | Exists |
| Project store, assets, renders | Desktop-local files (§2) | **New, desktop only** |
| Render thumbnails | Desktop extracts `thumb.jpg` | **New, desktop only** |
| Director renders in the catalog | Backfill indexing of `Director/**/renders/*/output.mp4` | **Verify.** Whether the catalog ingests `director-*` or project-folder outputs is unconfirmed. Optional additive: record an `i2v_source` edge from a render to its keyframe-0 catalog asset. |
| Catalog thumbnail route | None found in `GalleryServer.swift` | **Optional engine addition** `GET /v1/catalog/thumb/{id}?size=`. Only needed if `AsyncThumbnail` from local paths is too slow on large grids. Measure first. |

No required engine change. The one integration check is the render output path: `renders/<id>/output.mp4` under the output directory must pass submit-time containment. Confirm with a `/director` submit in WP-P2's live check, when the queue allows.

## 7. Work packages

Order: model first, then pickers, then canvas, then polish. Every WP is desktop-only unless marked, and tests run on `ComfyBoxDesktopTests` (no GPU).

**WP-P1: Project store (S–M)**
- Scope: a pure `DirectorProjectStore` plus `DirectorProject` Codable (§2.2), with create, open, recents, duplicate, rename, archive, save version, and legacy autosave migration. `DirectorAutosaveStore` is retargeted per project.
- Accepts:
  - Round-trip `project.json` byte-stable (sorted keys).
  - Timelines are readable by `DirectorDocument.read`.
  - A moved project folder still resolves project-internal paths.
  - Migration runs exactly once.
- Tests: temp-dir unit tests for each lifecycle action; duplicate excludes renders by default.

**WP-P2: Render records (S)**
- Scope: the submit path writes the snapshot and record, the poll updates it, success extracts `lastframe.png` and `thumb.jpg`, and open-project re-polls in-flight records to `interrupted`.
- Accepts:
  - The record's seed equals `plan.chunks[0].seed`.
  - A failed poll marks the record `failed` or `interrupted`, never retried.
  - Live check: one short render lands in `renders/<id>/` (Todd-run or queue-permitting, never during a declared soak).
- Tests: `EngineService` stubbed with a canned 202, status and failure; extraction on a fixture mp4.

**WP-P3: Picker shell and Keyframe / Last-frame pickers (M)**
- Scope: `DirectorPickerPanel`, the Project, Renders, Gallery, Finder and Clipboard sources, recents, keyboard, drag-out providers, frame scrubber extraction.
- Accepts:
  - Return inserts at the playhead.
  - Clipboard image becomes an `assets/` PNG keyframe.
  - A gallery row without a Mac location is disabled.
  - `AppContentGate` parity with the Gallery tab.
  - "Continue from last frame" creates a version with a keyframe at 0.
- Tests: a source adapter against a stub catalog JSON; recents ordering; extraction frame-exactness against `LastFrameExtractor`.

**WP-P4: Audio, Character, Preset pickers (M)**
- Scope: the trim UI writing `trim_start_frames` and `length_frames`; the catalog-thumbnail character cells; the preset popover with the upscale warning; the LoRA picker reuse.
- Accepts:
  - Trim handles snap to frames at timeline fps.
  - Inserting audio prompts for the `imported` mode switch.
  - Character nil means "None (no injection)".
  - No preset prompt text is rendered.
- Tests: trim math; preset row flags from fixture presets (ids only).

**WP-P5: Timeline canvas rewrite, design language and placement (L)**
- Scope: `DirectorTheme`, a single `Canvas` renderer plus hit-test layer, 120 pt label sidebar, toolbar, transport, gap "+", ghost drops, push physics and joints in the model, context menus with Replace with…, shortcut remap (S snap, ⌘B split).
- Accepts:
  - Every existing WP3 interaction still works.
  - Push and ripple are undoable as one step.
  - A keyframe push never creates `keyframes_collide`.
  - Smooth at 16 keyframes, 32 segments and 4 audio clips.
- Tests: model tests for push, ripple and joint trim, collisions, and end-frame pinning; `DirectorLayout` hit-test tests; snapshot render of the theme in light and dark system appearance.

**WP-P6: Validate inline and chunk progress (S–M)**
- Scope: debounced server validate with local fallback, per-clip badges, id-less banner, segmented chunk progress on the ruler.
- Accepts:
  - Each validator code with `ids` badges the right clip.
  - Id-less codes land in the banner.
  - The progress bar matches `stage_index/stage_count`.
- Tests: fixture issue lists mapped to badge regions; debounce coalescing.

**WP-P7: Empty state, render strip, compare (M)**
- Scope: the no-project card, first and last frame ghost targets, Renders strip actions, compare view with a synced scrubber and timeline diff.
- Accepts:
  - A new user reaches a valid FFLF timeline with two picks and a prompt, and no Finder.
  - Restore creates a version and never overwrites the head.
- Tests: timeline diff function (settings, prompts, keyframes); restore and continue produce the expected versions.

## 8. Open questions (Todd's call)

1. **Project root.** Should projects live under the output directory (proposed, keeps renders inside engine containment) or somewhere else, such as `~/Documents/Director`? Somewhere else means renders stay in the output directory and the project only references them.
2. **Gallery picks.** Reference them (proposed, no duplication) or copy them into `assets/` so a project folder is fully portable?
3. **Shortcut remap.** Adopt upstream's S = snap and ⌘B = split, which changes today's S = split?
