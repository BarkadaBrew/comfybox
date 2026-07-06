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

## Hub features — Desktop as the Coffeeshop suite hub (2026-07-06)

- Service control plane: launchd (launchctl) / SSH+shell start/stop/restart on
  Health cards + config sheet; daemon seeded (82399ee).
- API keys → macOS Keychain (AppSecrets), one-time migration from JSON (1ab4dd0).
- Bree panel (⌘B): vault handoff two-pane + composer; append-only convention
  (313e776). Follow-on: live ask_bree/tasks need a Bree HTTP API (:3779 WS-only).
- ⌘K command palette: fuzzy nav + actions; Canvas moved to ⌘Y (a38c890).
- Menu-bar item + sidebar grouping (Create/Library/Operate/Suite) + ActivityLog
  feed (75314a3). Follow-on: wire more activity emitters (cloud/mflux/video gen,
  service actions, downloads); first-run wizard; panels for video-conductor/
  capture-node/BBS/SnapAI; global content-mode toggle.

## CivitAI fixes (2026-07-06)

- Download was failing with opaque NSURLErrorDomain -1011. Root cause: gated/
  NSFW model + NO CivitAI API key configured (none in Keychain or JSON). Fixed:
  header+token auth with a redirect delegate that strips the auth header at the
  CDN handoff (B2), and descriptive errors (401/403 → "needs API key/gated",
  404, else HTTP+body). ACTION FOR TODD: add a CivitAI key in Settings → CivitAI
  to download gated/NSFW LoRAs (4f566b2).
- "View on CivitAI" link in the model sheet (honors .com/.red).
- On import, trigger words + up to 8 sample prompts saved as <name>.civitai.json
  sidecar AND pushed into the Prompt Library.

## mflux frontend — desktop-only backend (2026-07-06)

Refined the no-Python stance (memory comfybox-no-python): the SERVER stays
pure Swift/MLX and clients never touch mflux, but the DESKTOP app may shell out
to the local mflux venv for UI use.
- MfluxService: runs ~/Projects/mflux/.venv/bin scripts with live streamed
  output; version (pip show), update (pip install -U). Pure arg-builders for
  generate/train/save unit-tested (8 tests) (a90a4a0).
- mflux tab (⌘X): Generate (all base-model variants, quantize, LoRA, img2img),
  Train (mflux-train config/resume + dry-run), Tools (mflux-save quantize/bake),
  Update — with a shared live console.
- NOT verified with a live mflux generation/training run (heavy: model
  download + GPU); wrapper + args verified by tests and the real --help surface.
- Follow-on ideas: mflux as a choice in the Generate Backend picker too;
  training-config authoring form (currently config-file/resume only);
  mflux-upscale-seedvr2 / controlnet variants in Tools.

## CivitAI + cloud providers (2026-07-06)

- CivitAI browser: civitai.com/civitai.red source toggle (.red verified live,
  same /api/v1, NSFW-on default), richer filters (Type/Base/Sort/Period +
  NSFW toggle) with display→API mapping, search clear button (000e12a).
- Replicate + Fal cloud generation backends (desktop-side): CloudImageProvider
  with pure, unit-tested request/parse logic; Generate gains a Backend picker
  (Local/Replicate/Fal) + model id; keys in Settings → AI Providers; result
  downloaded + ingested like a local render (c478a3b). Live cloud render needs
  a configured key to verify. v1 targets the Flux family (flux-schnell/flux);
  custom models may need matching input params.

## LTX-2 integrated into ComfyBox server (2026-07-06)

Local video generation wired into the warm server:
- LTX2VideoGenerator (reusable service lifted from the ltx2-i2v CLI): lazy
  model load, T2V + I2V, chunked extend, MP4 out; pure helpers tested (b34b0ce).
- /v1/video/generate local backend behind --ltx2-weights + --ltx2-gemma; runs
  through the coordinator queue (.localVideo op) so it never shares the GPU
  with an image render; falls through to Replicate when unset (1f2ebce).
- API reference + serve help updated. 574 server tests pass.
- Weights present at ~/Models/ltx2-distilled (38GB transformer + VAE +
  upsamplers). Gemma text-encoder path NOT at the default HF location — the
  operator must pass --ltx2-gemma <snapshot dir>.
- NOT YET verified with a live 38GB render (would evict the image model + take
  minutes; needs a dedicated run, ideally a separate instance, with the owner).
  Do NOT enable --ltx2-weights on the shared image-serving daemon casually.
- Follow-on DONE: desktop Motion tab (⌘M) — T2V + I2V form, inline AVKit
  preview, gallery "Animate" action; Settings → Motion defaults (0ea34ae,
  cfe5f4c). Still awaiting a live 38GB render to confirm end-to-end.
- Settings surface now complete: prompt-optimizer/vision/captioning models
  (AI Providers), server/generation/gallery/motion defaults, CivitAI key, UI
  scale; watched health services edited on the Health board.

## img2img SHIPPED + verified live (2026-07-05)

Un-deferred and delivered:
- GenerationRequest gains initImagePath + imageStrength → /v1/generate payload
  (imagePath + imageStrength). "Reference (img2img)" section in Generate:
  choose image, thumbnail, Strength slider, Remove (a186098).
- "Use as Reference (img2img)" in Gallery + Canvas menus → Generate (684b093).
- VERIFIED LIVE: img2img render of the apple as oil-painting kept the
  reference composition; server accepted imagePath+imageStrength. NOTE: server
  sandboxes outputPath to ~/Pictures/ComfyBox.

This unlocks the rest of the Draw Things tier. Next candidates (all bigger,
owner-steer welcome):
- Camera V2 — reference + camera directive already works manually; a "Camera
  Angles" batch action (N angles from one reference) would automate it.
- Inpaint/outpaint on Canvas — server supports inpaintImageData+maskData;
  needs a mask-painting UI (substantial, own effort).
- ControlNet UI — pose/depth/canny reference (enqueueControlGenerate exists).
- Moodboard — multi-reference conditioning.

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
