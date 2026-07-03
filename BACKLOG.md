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

## In progress

- **#30 assistant-in-Generate** (STARTED): AgentService has `AgentAction`
  (structured param changes) + updated systemPrompt teaching a ```json action
  block. TODO: `AgentService.parseAction(from:)` (pure+tested), have `send()`
  return/store the action, build a Generate-embedded assistant panel that
  applies prompt/negative/steps/guidance/size/seed/loras and can trigger a
  render. Keep the standalone Assistant tab.

## Pending / TODO

- **#31 API + MCP sync**: document the new endpoints; add matching
  `mcp_comfybox_*` MCP tools (queue control, nearline stage/evict, enhance,
  preset import). Verify against running server.
- Camera tool **V2**: img2img/Klein reference for identity-preserving
  multi-angle; captioning (deferred by user).
- Draw Things derivations (backends already exist server-side; need desktop UI),
  recommended order: img2img in Generate → inpaint/outpaint on Canvas →
  ControlNet UI → Creative Upscale right-click → Moodboard.
- Canvas later phases: node connections, on-canvas generation, prompt/text
  nodes, export board.

## Operational notes

- Serve daemon (`com.barkadabrew.comfybox`) has been running a photoshoot batch;
  new SERVER routes (enhance/queue/nearline/preset-import) go live only after
  its next restart. Desktop UIs for them are already deployed.
- Bree handoff: retracted the Kira-health-endpoint ask (Kira dropped from
  ComfyBox); Netdata LAN ufw rule on littleroundbox stays.
- littleroundbox root disk ~85% full (watch on Health board).
