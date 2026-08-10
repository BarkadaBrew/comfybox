# LoRA Import — Models Tab (2026-08-10)

Approved by Todd 2026-08-10 ("lora import is a go") after design Q&A:
category picked at import time; picker accepts files AND folders.

## Purpose

Let the user import LoRAs from anywhere on disk into the collection from the
desktop Models tab, so they immediately appear in the LoRA library and are
usable from presets and the LoRA picker.

## What already exists (reuse, don't rebuild)

- `POST /v1/loras/import {path, category}` — copies the file into
  `~/Models/loras/<category>/` (default `vault`), rescans, returns the indexed
  entry (`id`, `filename`, `model_compatibility`, `triggerwords`).
  Backed by `LoRALibrary.importFile(from:category:)`.
- Presets + LoRAPicker read the library index — imported entries are
  available to them automatically after the rescan.
- CivitAI browser proves the copy → rescan → refresh flow.

## New pieces

1. **`LoRAImportPlanner` (ComfyBoxDesktop, pure + unit-tested)**
   - `expand(urls:)` — resolve a mixed file/folder selection into a deduped,
     sorted list of `.safetensors` file URLs (folders enumerated recursively,
     hidden files skipped, non-safetensors counted as skipped).
   - `categories(from: [LoRAInfo])` — distinct existing categories for the
     dropdown, sorted, always including `vault` first.
2. **`EngineService.importLora(path:category:) -> ImportedLoRA`** — thin
   client for the route, following the `scanLoras()` pattern.
3. **Models tab UI (`ModelsView`)**
   - "Import LoRA…" button in the LoRA section header.
   - `NSOpenPanel`: multi-select, `.safetensors` files and directories.
   - Import sheet: resolved file list (name + size), category dropdown
     (existing categories + "New category…" free text, default `vault`),
     Import button.
   - Sequential per-file import; outcomes shown inline (imported /
     already-in-library / failed + reason); one failure never stops the batch.
   - On completion: refresh the LoRA list (`refreshAll()`).
   - Button disabled with tooltip when the engine base URL is not local —
     the API takes a server-local path.

## Error handling

- Folder with no `.safetensors` → sheet shows "nothing to import".
- Duplicate filename already in library → route returns the existing entry;
  shown as "already in library", not an error.
- Per-file API failures surface the server message inline.

## Testing

- Unit (ComfyBoxDesktopTests): planner expansion/dedupe/skip counting,
  category derivation.
- Manual: import a file + a folder from ~/Downloads, confirm entries appear
  in Models tab groups, the LoRA picker, and are attachable to a preset.
