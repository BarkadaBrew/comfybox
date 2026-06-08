# ComfyBox Desktop User Guide

ComfyBox Desktop is a native macOS app for AI image generation. It connects to a running WarmServer instance (the ComfyBox CLI backend) and provides a graphical interface for prompt-based image generation, model management, LoRA selection, asset browsing, and image comparison.

---

## 1. Getting Started

### Building the App

ComfyBox Desktop is built with Swift Package Manager alongside the rest of the ComfyBox project.

```bash
cd ~/Projects/zimage.swift
swift build -c release --product ComfyBoxDesktop
```

The compiled binary lands in `.build/release/ComfyBoxDesktop`. You can also open the project in Xcode by opening `Package.swift` directly.

### First Launch

On first launch, the app will:

1. Create the configuration directory at `~/.comfybox/` if it does not exist.
2. Initialize the DAM (Digital Asset Management) database at `~/.comfybox/dam.sqlite3`.
3. Create the thumbnail cache directory at `~/.comfybox/thumbnails/`.
4. Attempt to auto-connect to WarmServer at `127.0.0.1:7862` (the default).

### Connecting to WarmServer

WarmServer must be running before the app can generate images. Start it from the CLI:

```bash
ComfyBox serve --port 7862
```

The connection indicator in the toolbar shows the current state:

- **Green dot** -- Connected and ready
- **Yellow dot** -- Connecting
- **Gray dot** -- Disconnected
- **Red dot** -- Connection error

Click the connection button in the toolbar to connect or disconnect manually. To change the server address, open Settings (Cmd+,) and edit the Host and Port fields.

### App Layout

The app uses a sidebar navigation with four tabs:

| Tab | Shortcut | Purpose |
|-----|----------|---------|
| Generate | Cmd+1 | Write prompts and submit renders |
| Gallery | Cmd+2 | Browse and manage generated images |
| Compare | Cmd+3 | Side-by-side image comparison |
| Presets | Cmd+4 | Manage saved generation configurations |

---

## 2. Generation

The Generate tab is the main workspace. The left panel contains all controls; the right panel shows the image preview.

### Writing Prompts

Type your prompt in the text editor. Describe what you want to see. The prompt field expands vertically to accommodate longer text.

To clear the prompt and start fresh, click "New" or press Cmd+N.

### Selecting a Model

The Model section at the top of the control panel shows the currently active model. Expand the section to see:

- **Active Model** -- The model currently loaded and ready for generation.
- **Model Pool** -- All models loaded in memory. You can have multiple models loaded simultaneously and switch between them instantly.
- **Available Models** -- The full registry of models the server knows about, grouped by family (Flux, FIBO, Chroma, etc.).

To use a different model, either:
- Click "Activate" next to a model already in the pool (instant switch).
- Click "Load" next to a model in the Available Models list (downloads/loads to pool, then activates).

### Adjusting Parameters

- **Resolution** -- Pick from preset sizes: 512x512, 768x1024, 1024x1024, or 1024x768. Displayed as a segmented control.
- **Steps** -- Drag the slider from 1 to 50. More steps means more detail but slower renders. Default: 9.
- **Guidance** -- Controls how closely the image follows your prompt. Range: 0 to 20, step 0.5. Default: 3.5. Higher values produce more literal interpretations.
- **Seed** -- Leave empty for a random seed each time. Enter a specific number to reproduce an exact result.

### Submitting a Render

Click "Generate" or press Cmd+Return. While rendering:

- A progress spinner appears in the generate button and the preview panel.
- The queue badge in the sidebar updates.
- The button is disabled to prevent duplicate submissions.

When the render completes, the image appears in the preview panel with the render duration displayed below it. The image is automatically saved to the output directory and ingested into the gallery.

---

## 3. Presets

Presets save your generation settings so you can recall them later.

### Saving a Preset

1. Set up your prompt, model, LoRAs, and parameters as desired.
2. Click "Save Preset" or press Cmd+S.
3. Enter a name in the dialog that appears.
4. Click "Save Preset" to confirm.

A preset captures: prompt template, model ID, selected LoRAs with scales, steps, guidance, and resolution.

### Loading a Preset

Switch to the Presets tab (Cmd+4) and click on a preset. This switches back to the Generate tab and fills in all the saved settings.

### Managing Presets

Right-click any preset to:

- **Apply** -- Load its settings into the Generate tab.
- **Edit** -- Open the editor to modify name, prompt, parameters, or model.
- **Duplicate** -- Create a copy with "(Copy)" appended to the name.
- **Delete** -- Remove the preset permanently.

### Storage

Presets are stored as JSON at `~/.comfybox/presets.json`. The file is human-readable and can be edited directly if needed.

---

## 4. Characters

The Character Library is a browsable registry of characters available through the engine API. Each character has a name, description, tags, default LoRAs, and a prompt snippet.

### Browsing Characters

Expand the Characters section in the Generate tab control panel. Characters appear as cards. Click a card to expand it and see full details including tags, default LoRAs, and the prompt snippet.

Use the search field at the top to filter characters by name, description, or tags.

### Inserting a Character

Click the "Insert" button on a character card. The character prompt snippet is appended to your current prompt. If the prompt is empty, the snippet becomes the entire prompt; otherwise it is appended after a comma.

---

## 5. Prompt Enhancement

The Enhance button (sparkle icon next to the Prompt header) sends your current prompt to the server LLM enhancement endpoint. The server rewrites your prompt to be more detailed and effective for image generation, then returns the enhanced version which replaces your original text.

Requirements:
- The server must be connected.
- The prompt field must not be empty.
- The enhancement endpoint must be available on the server. If the server returns a 404, the button is disabled for the session.

---

## 6. Gallery

The Gallery tab (Cmd+2) displays all images tracked by the DAM database.

### Browsing

Images appear as a responsive grid of thumbnails. Each cell shows:
- A 256px thumbnail (generated on ingestion).
- The first two lines of the prompt.
- Favorite heart indicator and star rating if set.
- Content mode badge if present.

Click any thumbnail to open the detail view.

### Searching

Type in the search field at the top of the gallery. Searches match against prompts, negative prompts, filenames, and character names. Results update in real time.

For larger datasets, the gallery uses FTS5 (SQLite full-text search) for fast prompt matching.

The search field can be focused with Cmd+F.

### Filtering

Use the toolbar controls to narrow results:

- **Favorites** -- Toggle the heart icon to show only favorited images.
- **Content Mode** -- Filter by content mode if your images have mode metadata.
- **Character** -- Filter by character name.

### Sorting

Choose a sort order from the dropdown:

| Sort | Behavior |
|------|----------|
| Date | Newest first (default) |
| Rating | Highest rating first |
| Favorites First | Favorites sorted to top, then by date |

### Quick Look

Select an image and press Space to open it with the system file viewer.

### Drag-and-Drop

Drag any thumbnail from the gallery to Finder, Desktop, or any app that accepts image drops. Images are transferred as PNG files.

### Context Menu

Right-click a thumbnail for additional options:
- **Reveal in Finder** -- Opens a Finder window with the image file selected.
- **Favorite / Unfavorite** -- Toggle the favorite flag.
- **Add to Comparison** -- Add the image to the comparison selection (up to 4).

### Asset Detail View

Click any thumbnail to open the detail panel as a sheet. The left side shows the full-size image; the right side shows:

**File Info:** Filename, dimensions, file size, creation date.

**Annotations:** Star rating (click stars to rate 1-5, click the current rating to clear). Favorite toggle.

**Generation Metadata:** Model, steps, guidance, seed, content mode, character name.

**Prompt:** Full prompt text (selectable for copy). Negative prompt if present.

Click "Save Changes" to persist rating and favorite changes.

---

## 7. Image Comparison

The Compare tab (Cmd+3) provides side-by-side image comparison for A/B testing prompt variations, model differences, or parameter tuning.

### Selecting Images for Comparison

**From the Gallery:**
1. Toggle comparison mode using the grid icon button in the gallery toolbar.
2. Click thumbnails to select them (up to 4). Selected images show a blue checkmark.
3. Click "Compare N" to send them to the Compare tab.

**From the Compare tab directly:**
If you have fewer than 2 images selected, the Compare tab shows a picker grid. Click images to add or remove them from the selection.

### Reading the Comparison

When 2-4 images are selected, they display side-by-side at equal sizes. Below the images, a metadata diff table highlights differences:

Fields compared: Prompt, Seed, Steps, Guidance, Model, Content Mode, Character, and Resolution. Values that differ between images are shown in bold with a subtle yellow background, making it easy to spot what changed.

### Controls

- **Show Diff** toggle -- Show or hide the metadata comparison table.
- **Clear Selection** -- Remove all selected images and return to the picker.

---

## 8. Model Management

### Understanding the Model Pool

WarmServer maintains a model pool -- multiple models can be loaded into memory simultaneously. Only one model is "active" at a time (used for the next generation), but switching between loaded models is instantaneous with no reload delay.

### Loading a Model

1. Expand the Model section in the Generate tab.
2. In the "Available Models" list, find the model you want. Models are grouped by family.
3. Click "Load". The model downloads (if needed) and loads into the pool. This can take seconds to minutes depending on model size.

While loading, the "Load" button is disabled across all models to prevent concurrent loads.

### Switching the Active Model

In the Model Pool section, click "Activate" next to any loaded model. The switch is immediate.

### Unloading a Model

Click the red X button next to a loaded (but inactive) model to unload it and free its memory. You cannot unload the currently active model.

### Model Information

Hover over a model in the Available Models list to see its description tooltip. Each model row shows:
- Display name
- Parameter count (e.g., "12.1B")
- Quantization level (e.g., "4bit", "bf16")
- Estimated VRAM usage

---

## 9. LoRA Management

LoRAs (Low-Rank Adaptations) modify the active model behavior, adding styles, concepts, or character knowledge.

### Browsing LoRAs

Expand the "LoRA Adapters" section in the Generate tab. Available LoRAs from the server library are listed. Each row shows:
- LoRA name
- File size
- Category and trigger words
- "Active" badge if currently loaded on the server

When more than 5 LoRAs are available, a search field appears for filtering by name, filename, tags, category, or trigger words.

### Enabling a LoRA

Click the checkbox next to a LoRA to add it to your generation. A scale slider appears (range 0.0 to 2.0, step 0.05). The default scale uses the LoRA recommended value.

### Adjusting Scale

Higher scale values increase the LoRA effect. Typical values are 0.5 to 1.5. Going above 1.5 can produce artifacts.

### Applying LoRAs

Click "Apply LoRAs" to send the current selection to the server. LoRAs are also automatically applied before each generation. The server performs a hot-swap, loading/unloading LoRAs without reloading the base model.

### Ordering

Selected LoRAs appear at the top of the list, followed by LoRAs active on the server, then the rest alphabetically.

---

## 10. Queue and Status

The Queue section in the Generate tab shows real-time server status, updated every 3 seconds.

### Status Indicators

- **Rendering** (orange dot + spinner) -- The server is actively generating an image.
- **Idle** (green dot) -- The server is ready for the next job.

### Information Displayed

| Field | Description |
|-------|-------------|
| Status | Current render state |
| Pending | Number of jobs waiting in queue |
| Completed renders | Total renders since server start |
| Last render | Duration of the most recent render |
| Server uptime | Time since the server started |
| Memory usage | Current memory consumption |
| Models loaded | Number of models in the pool |
| Last Error | Most recent error message, if any |

---

## 11. Settings

Open Settings with Cmd+, or from the application menu.

### Server Tab

| Setting | Default | Description |
|---------|---------|-------------|
| Host | 127.0.0.1 | WarmServer IP address or hostname |
| Port | 7862 | WarmServer port number |
| Auto-connect on launch | On | Automatically connect when the app starts |

The current connection status and active model are displayed as read-only fields.

### Generation Tab

| Setting | Default | Description |
|---------|---------|-------------|
| Steps | 9 | Default step count for new generations |
| Guidance | 3.5 | Default guidance value |
| Width | 1024 | Default output width |
| Height | 1024 | Default output height |
| Output Directory | ~/Pictures/ComfyBox | Where generated images are saved |

### Gallery Tab

| Setting | Default | Description |
|---------|---------|-------------|
| Thumbnail Size | Medium (180) | Grid cell size: Small (140), Medium (180), Large (240) |
| Default Sort | Date | Default sort order: Date, Rating, Favorites First |

### Saving

Click "Apply and Save" to persist changes. Settings are stored at `~/.comfybox/desktop-config.json`. Click "Reset to Defaults" to revert all settings.

---

## 12. Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| Cmd+1 | Switch to Generate tab |
| Cmd+2 | Switch to Gallery tab |
| Cmd+3 | Switch to Compare tab |
| Cmd+4 | Switch to Presets tab |
| Cmd+Return | Submit generation |
| Cmd+S | Save current settings as preset |
| Cmd+N | Clear prompt and start new |
| Cmd+F | Focus gallery search field |
| Cmd+, | Open Settings |
| Space | Quick Look selected gallery image |
| Escape | Cancel or dismiss dialogs |

---

## 13. Working with Krita

ComfyBox Desktop and Krita can both connect to the same WarmServer instance simultaneously. This enables a workflow where:

1. You generate base images in ComfyBox Desktop.
2. You open them in Krita for painting and editing.
3. You send edited images back through the pipeline for img2img refinement.

Both clients share the same model pool, LoRA state, and render queue. Jobs are queued in order regardless of which client submitted them.

To use both:
1. Start WarmServer on a known port (e.g., `ComfyBox serve --port 7862`).
2. Connect ComfyBox Desktop to that port.
3. Configure the Krita ComfyBox plugin to connect to the same host and port.

Note: Model and LoRA changes from either client affect the other. If you load a model from Krita, ComfyBox Desktop will reflect the change on its next health poll (within 3 seconds).

---

## 14. Troubleshooting

### Cannot Connect to Server

- Verify WarmServer is running: `ps aux | grep ComfyBox`
- Check the host and port in Settings match the server configuration.
- If connecting to a remote machine, ensure the port is not blocked by a firewall.

### Model Will Not Load

- Check that the model files exist in the expected location on the server.
- Check server logs for download or memory errors.
- Large models may require more memory than available. Check the estimated VRAM in the model list.

### Generation Takes Too Long or Hangs

- Reduce step count. 9 steps is usually sufficient for turbo-quantized models.
- Check the Queue panel for error messages.
- If the queue shows "Rendering" indefinitely, the server process may need to be restarted.

### Images Not Appearing in Gallery

- The gallery watches the output directory configured in Settings. Verify it matches where the server writes output files.
- The file watcher polls every 5 seconds. New images may take a few seconds to appear.
- Click the refresh button in the gallery toolbar to force a reload.
- Check the ingestor status in the toolbar -- it shows the count of ingested files.

### Thumbnails Not Loading

- Thumbnails are stored in `~/.comfybox/thumbnails/`. Check that the directory exists and is writable.
- If thumbnails are missing for existing images, delete the `thumbnails/` directory and restart the app to regenerate them.

### Database Errors

- The SQLite database is at `~/.comfybox/dam.sqlite3`. If it becomes corrupted, delete it and restart the app. The database will be recreated and existing images will be re-ingested from the output directory on the next file scan.

### Settings Not Persisting

- Settings are saved to `~/.comfybox/desktop-config.json`. Check that the `~/.comfybox/` directory is writable.
- Remember to click "Apply and Save" -- settings are not saved automatically when you close the Settings window.
