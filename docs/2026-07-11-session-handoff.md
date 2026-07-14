# Session handoff — 2026-07-11 (overnight session)

Written by Claude Code so a fresh session can pick up exactly where this one left off. Read this fully before doing anything — several items below are "verified working" and several are "written but not yet verified," and mixing those up would waste time re-deriving context.

## Repo/deploy state right now

- `main` is at `eb77916`, pushed. Working tree clean (only pre-existing untracked stray files: `.playwright-mcp/`, `CODEX_PROMPT.txt`, `CODEX_TASK.md`, `mindcraft-snapshot.md`, `scripts/convert_to_mlx.py` — not mine, leave alone).
- `com.barkadabrew.comfybox` (server) is running the `eb77916` build, deployed and health-checked.
- CoffeeShop Desktop.app is running an earlier build from tonight (commit `e8be6aa`) — Smart Tabs + live preview + CivitAI/thumbnail fixes are in it. It does **not** need a redeploy for the model/LoRA fix below (that's server-only code Desktop doesn't touch).
- Server's active model is currently **krea2** (empty LoRAs) — left over from my last smoke test, not necessarily what you want loaded. Reset via `/v1/model/activate` or restart if it matters.
- `~/Library/LaunchAgents/com.barkadabrew.comfybox.plist` was edited tonight: added `--ltx2-weights /Volumes/Bolt/hf-cache/models--dgrauet--ltx-2.3-mlx-q8/snapshots/03da129baa459c9a70fc5858dee52fa417b3a93d`. **`--ltx2-gemma` is NOT set** — see "Open decision" below, local video is still not actually usable yet.

## Verified working tonight (tested live, not just compiled)

1. **Smart Tabs** (Gallery) — save/apply/rename/delete named filter combos. `Sources/ComfyBoxDesktop/DAM/SmartTab.swift` + `GalleryView.swift`.
2. **CivitAI search fix** — was silently returning zero results whenever a text query + base-model filter were both set (confirmed live against the real CivitAI API: `query=girl&baseModels=Krea 2` → 0 results, `query=girl` alone → 2). Fixed by dropping the server-side `baseModels` param for text queries and filtering client-side instead.
3. **Thumbnail fix** — `CGImageDestinationFinalize` failures left 0-byte files masquerading as "already generated," permanently blocking regeneration. Fixed + added a gallery-load backfill pass for already-broken thumbnails.
4. **Live denoising preview** (GH #216, closed) — `GET /v1/generate/preview` serves the latest JPEG frame via the same fast latent-approximation the Krita bridge uses. Desktop's `GenerationView` shows it during a render. **Scope note: only wired for the Z-Image/Flux1 family** — Krea2/Flux2/Fibo/Chroma don't have per-step progress callbacks in the dispatcher at all (pre-existing gap).
5. **Async queue-submit API** — `POST /v1/generate/async` (returns `job_id` immediately, 202) + `GET /v1/generate/status/{id}` (poll `queued`→`processing`→`succeeded`/`failed`). Mirrors the existing `/v1/video/generate` + `/v1/video/status/{id}` pattern exactly. Verified end-to-end multiple times, including through a full server redeploy.
6. **GH #153 fix** — the MCP-bridge-blamed port conflict was actually `com.barkadabrew.comfybox`'s `KeepAlive=true` launchd config silently re-occupying the port within ~5s of a manual kill. Fixed: `NWListener` failure on `EADDRINUSE` now prints the actual cause + exact `launchctl bootout`/`bootstrap` commands instead of a bare NWError. Closed on GitHub.
7. **GH #12, #149 closed** — both stale/already-fixed, see issue comments for why.
8. **The critical one — per-job model+LoRA fix.** `GeneratePayload` gained optional `model`/`loras` fields; `runGenerate` now applies a job's own model+LoRAs at dequeue time instead of trusting the pool's global "currently active" state. **Why this mattered**: the other session (coffeeshop-server, building `generate_image` against the new async endpoint) proved 3× that a job submitted with `model=krea2` while the global model was `cyber` rendered on `cyber` — because the old design assumed "caller activates model synchronously right before calling generate," which async queue-submit breaks (a job can dequeue long after a different request changed global state). **Verified live tonight**: submitted an async job with `model: "krea2"` while the server's active model was cyberrealistic; output PNG's embedded metadata confirms `"model":"krea-2-turbo"`. This was flagged by the other session as **the one blocker — "nothing else about async matters until this lands."** It has landed, is committed (`eb77916`), pushed, and deployed. **The other session needs to know this is done** — I have not yet written that to the handoff file (see Pending below).

## Written but NOT yet verified

- **LTX-2 local video auto-download wiring**: `--ltx2-weights`/`--ltx2-gemma` now accept either a local path or an `org/repo[:rev]` HuggingFace spec, resolved via `ModelResolution.resolve()` (downloads + caches on first use if not already local) instead of requiring a hand-populated directory. Compiles clean, but **never actually exercised** — doing so requires a real local video request, which will trigger a ~24GB Gemma-3 download that hasn't happened (see Open decision). Don't assume this works until it's been through one real request.

## NOT started yet

- **"Make sure LTX LoRAs can be managed as are other models"** (your explicit instruction, not yet acted on). This is also Spec B from the other session ("LTX LoRAs for local NSFW video" — `generate_video` gaining a `loras: [{path, scale}]` field, applied per-job to the LTX pipeline, config-gated). Confirmed tonight: `LocalVideoRequest` already has a *single* `loraPath`/`loraStrength` (see `WarmServer.swift` around `localVideoResponseIfConfigured`, ~line 1421) — NOT a list, and NOT integrated with the LoRA library/tagging/filtering system the image side got earlier this session (family auto-detect, keyword extraction, CivitAI import, etc.). Scope for a real fix: (1) extend `LocalVideoRequest`/`LTX2VideoRequest` to accept multiple LoRAs, (2) wire `LTX2VideoGenerator` to load+apply a list instead of one, (3) tag LTX2 LoRAs in `LoRAScanner.detectCompatibility` so the existing family-filter UI in Desktop's LoRA picker recognizes them, matching how krea2/flux/z-image LoRAs already work. Not started.

- **AI_Art (#204–209) / chroma-generate (#210–215) triage and build.** Full triage is done (see below); execution had just started (#209, mflux Kontext in Desktop) when the Replicate-cost investigation interrupted it. Recommended build order, cheapest-first:
  1. **#209** — mflux Kontext mode in Desktop's mflux panel. `mflux-generate-kontext` confirmed installed at `~/.local/bin`. Desktop-only mflux subprocess use is explicitly allowed (see memory `comfybox-no-python`). Follow the exact pattern `MfluxView.swift`/`MfluxService.swift` already use for generate/img2img/train/save.
  2. **#215** — queue/status CLI. Confirmed zero queue CLI commands exist in `main.swift` today; WarmServer already exposes pause/resume/reorder/status over HTTP. Thin wrapper.
  3. **#214** — DAM lineage (source/derived asset tracking). Confirmed zero lineage fields exist on `DAMAsset` today. Foundational — #211/#212/#208 all assume it exists, so building it first avoids retrofitting three features later.
  4. **#205** — local still-image animation (Ken Burns/pan/zoom, title overlays, GIF/MP4 export via FFmpeg). Self-contained, reuses `MediaToolsService`.
  5. **#211** — vision QC scorecards. `VisionService.describe()` already has the caption/tag LLM-vision-call pattern to extend; sequence after #214 so results attach via lineage.
  - Blocked (need #197/#195/#198/#200/#202 first, correctly deferred): #208, #204, #212.
  - Needs a benchmarking session with you looking at real output, not solo code work: #210 (preset tiers), #213 (naturalness/CFG-decay research).
  - Big standalone builds needing their own design pass: #206 (native SVG editor — no SVG engine exists at all today), #207 (posable mannequin — genuinely valuable given the existing Healthcare Studio Pack, but real rigging/geometry work).

## Open decisions needing your input

1. **Gemma-3 download (~24GB)**: LTX-2 transformer weights already exist locally (`/Volumes/Bolt/hf-cache/models--dgrauet--ltx-2.3-mlx-q8`), but the required Gemma-3 text-encoder does not exist anywhere on this Mac — confirmed via full-disk search. LTX-2's architecture has a dedicated trained connector (`LTX2Connector1D`, `connector.safetensors`) bridging Gemma-3's specific output space into the DiT — **it cannot use a different/smaller encoder without retraining that connector, which isn't realistic.** Gemma-3 is a hard requirement, not a choice. Until it's downloaded (either explicitly now, or transparently on the first real local-video request via tonight's auto-download wiring), every video request still silently falls through to paid Replicate — the actual root cause of the Replicate usage you flagged. Decide: download now, or let it happen lazily on first real use (bigger latency on that first request).
2. **Bree's campaign Replicate hardcode** (`orchestrator.ts:337` in coffeeshop-server, unrelated repo/code I can't touch from here): the other session asked whether to fix it themselves or wait for you. I recommended: let them fix it, it's a clear violation with zero ComfyBox dependency. You haven't confirmed either way yet.
3. Confirm you're fine with the `krea2` LoRA-filter/download-flux1.1-pro root-cause conclusions reported earlier (Bree's separate Replicate usage for image/voice, not ComfyBox) — nothing further needed from me here unless something doesn't add up.

## Pending — needs to happen next turn regardless of anything else

**Write to `Coffee Shop/Handoff/desktop-to-bree.md`** telling the other session the per-job model+LoRA fix has landed (commit `eb77916`, deployed, verified live with the exact repro they described). This is the single most time-sensitive item in this whole handoff — they said "nothing else about async matters until this lands," and it has landed, but they don't know that yet. Do this first in the next session unless it's already been done.

Also worth a shorter follow-up note acknowledging receipt of the 4-spec priority list (per-job fix → LTX LoRAs → Kokoro TTS → reference-centric generation FDD) so they know the ordering was received, even though only #1 is actually done.

## Key facts learned tonight (avoid re-discovering these)

- `com.barkadabrew.comfybox`'s launchd plist has `KeepAlive=true, ThrottleInterval=5` — a manual `kill` on the server process gets silently replaced within 5s. Always use `scripts/deploy-server.sh` (does `launchctl kickstart -k`) or `launchctl bootout`/`bootstrap` for a real stop, never a bare `kill` if you want it to actually stay down.
- Two concurrent `xcodebuild` invocations against the same `-derivedDataPath` can deadlock the build system silently (near-0% CPU, no progress, no error) — this happened once tonight and cost real time before being caught. If a build/test run seems stuck, check `ps aux` for a second xcodebuild process before assuming it's a real hang.
- Render-state-check protocol (established this session, still correct): always `curl /health` and check `is_rendering` before killing/restarting the server. A busy coordinator and a hung one look identical to a naive health-timeout check — verify via the actual field, not connection-timeout-as-proxy.
- `GeneratePayload`, `WarmServerCoordinator`, and most server internals live in one file, `Sources/ZImage/Server/WarmServer.swift` (~4900 lines). Cross-type static-function calls need the explicit type name (`WarmServer.parseModelSpec(...)`, not `Self.` from inside `WarmServerCoordinator` — they're different types in the same file, `Self` resolves per-type not per-file).
