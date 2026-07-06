# ComfyBox Desktop — Session Backlog

Durable record of work so a crash/OOM reboot can't lose state. Updated as work
lands. "Shipped" = committed on `main` + app redeployed to
`/Applications/ComfyBox Desktop.app`.

Deploy recipe: `swift build -c release --product ComfyBoxDesktop`, then
`/bin/cp -f .build/release/ComfyBoxDesktop "/Applications/ComfyBox Desktop.app/Contents/MacOS/ComfyBoxDesktop"`,
then `codesign --force --deep --sign - "/Applications/ComfyBox Desktop.app"`, relaunch.
Server deploy: `swift build -c release --product ComfyBox`, then restart the
`com.barkadabrew.comfybox` launchd agent (only when its render queue is idle).

## Shipped this session (committed on main)

- Characters: legacy image-service migration + `kind` (character/scene) + tier
  round-trip fix + editor rework (edde109, d392a5b)
- Gallery: masonry aspect-ratio layout, folders, lightbox, multi-select,
  delete-to-Trash, Photo-Mechanic Finder color tags, Copy/Reveal, folder import,
  orphan self-healing, secure-with-Touch-ID, detail-card nav/full-screen
  (56a87f9, 0cd6dca, b26ffb6, d79e31b, f622bb5, 32221c7, 96da7b4)
- Parameters: 11 resolution presets + typed slider entry (924fad9)
- Presets: server-backed CRUD + import-from-image-service (aba6c2d, 94281f3)
- Prompt Library tab (abd45b4)
- Health board: activity heatmap, uptime tracking, littleroundbox Netdata card,
  This Mac card (17828d0, 15f209f, ea8ed32)
- CivitAI browser: search, prompt scraping, LoRA download (13c96b3)
- UI scale setting (d221655)
- Nearline model/LoRA storage — stage from attached disk on demand (a30f6ca)
- Queue management — list/cancel/interrupt/clear (a9e5ccf)
- Canvas board v1 — infinite board, projects, right-click (re-render/replace/
  duplicate/flip/rotate/z-order) (e67201d)
- Camera-placement tool v1 + AI Assistant (Dan's v1.3) (f7539a7)

## New server endpoints added this session (API surface)

- `POST /v1/enhance` — prompt optimization via configured provider
- `GET /v1/queue`, `POST /v1/queue/interrupt`, `POST /v1/queue/clear`,
  `DELETE /v1/queue/{id}`
- `GET /v1/nearline`, `POST /v1/nearline/{scan,stage,evict}`
- `POST /v1/presets/import-legacy`
- CharacterEntry gained `kind`; character + preset legacy import run at startup

## Also shipped

- Assistant embedded in Generate — AgentAction + parseAction (json action
  block), auto-applies prompt/steps/guidance/size/seed and can trigger a
  render; standalone Assistant tab retained (f57ad27).
- MCP + API sync — 11 new MCP tools (enhance/characters/presets/queue/nearline)
  + docs/api-reference.md (e71f487).

## Pending / TODO (all explicit requests done; these are proposed/deferred)

- Camera tool **V2** — img2img/Klein reference for identity-preserving
  multi-angle + captioning. **Explicitly deferred by owner.**
- Draw Things derivations (backends exist server-side; need desktop UI):
  1. Creative Upscale right-click action — DONE (d76ef0b).
  2. img2img in Generate — deferred (shares the i2i plumbing owner deferred).
  3. inpaint/outpaint on Canvas — i2i-adjacent; hold until i2i is un-deferred.
  4. ControlNet UI — i2i-adjacent; hold.
  5. Moodboard — reference-image conditioning; hold.
- Canvas: export board — DONE (9b45131). Later phases still open: node
  connections, on-canvas generation, prompt/text nodes.

## Server routes ACTIVATED + verified live (2026-07-05)

Photoshoot finished (153 renders, queue idle). Restarted the daemon onto the
current binary; verified live: /v1/nearline 200 (scan cataloged 61 Seagate
items), /v1/queue 200, /v1/presets/import-legacy {imported:0} (idempotent),
/v1/enhance success (528-char reply from Dan's model). All session server
routes now operational.

## Next: un-deferring img2img (foundational i2i tier)

With everything else done and the owner signalling continued momentum, building
img2img in Generate next — it unlocks the whole Draw Things tier (camera V2,
inpaint/outpaint, ControlNet, moodboard). Backend exists: ImageToImagePipeline
(imageStrength→denoise). If the owner still wants i2i held, redirect.

## GUI verification owed on owner's return (laptop locked, validated by tests)

- Assistant-in-Generate live round-trip (Dan's model → applies controls).
- Masonry gallery visual, canvas board interactions, camera panel, folder
  import — all built + unit-tested but not screenshot-verified while locked.

## Operational notes

- Serve daemon (`com.barkadabrew.comfybox`) has been running a photoshoot batch;
  new SERVER routes (enhance/queue/nearline/preset-import) go live only after
  its next restart. Desktop UIs for them are already deployed.
- Bree handoff: retracted the Kira-health-endpoint ask (Kira dropped from
  ComfyBox); Netdata LAN ufw rule on littleroundbox stays.
- littleroundbox root disk ~85% full (watch on Health board).
