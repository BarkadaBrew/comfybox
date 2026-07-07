# CoffeeShop Desktop — Session Handoff (2026-07-06)

Durable record ahead of a model change. Learnings are also in Claude memory
(`krita-comfybox-bridge`, `comfybox-queue-management`, `comfybox-desktop-surfaces`).

## Shipped today (all committed + deployed to `/Applications/CoffeeShop Desktop.app`)

| Commit | What |
|--------|------|
| `202b7cd` | **Queue management** — server: per-job **source** (desktop/comfyui/bree/api), **pause/resume** (`/v1/queue/pause\|resume`), **reorder** (`/v1/queue/{id}/move` up/down/top/bottom); desktop **Queue tab** (live poll, source badges, cancel, reorder, pause, clear). |
| `fff7ef9` | **Applications tab** (surfaces: ComfyUI bridge, Krita, Krita-MCP, MCP/Bree) + **SeedVR2 upscale** option in Generate. |
| `01cd818` | Server tab shows the copy-ready Krita/ComfyUI endpoint (This Mac + LAN IP). |
| `a50b2be` | **Inpaint/outpaint** — server decodes `inpaint_image_base64`/`mask_base64`; desktop Inpaint tab with mask painting. Verified live (apple→lemon). |
| `2f44986` | Video first-class in gallery (AVAssetImageGenerator thumbs + AVKit playback). |
| `9a05ebd` | Live render progress bar (server `progress_percent`, fast poll during render). |
| `af2b967` | Batch / multi-seed generation. |
| `1aa56b7` | Presets capture the seed (round-trip). |
| `4cfe2ad` | Face swap backend (insightface + inswapper) installed & wired. |

Earlier in the session: local vision captioning/tagging, stable signing identity,
NSFW gallery gate, CoffeeShop rebrand, Krea 2 support, Zeta-Chroma, Découpage tab.

## Krita ↔ ComfyBox
- Krita AI Diffusion (v1.51.0) uses ComfyBox as its ComfyUI backend at
  `127.0.0.1:7870`. Bridge is complete (58 nodes, inpaint, img2img, latent preview).
- The "missing docker" was a **hidden panel**, not a broken plugin
  (Settings → Dockers → AI Image Generation). Plugin `batch_size` was 6 → set to 1.
- Helper: `scripts/fix-krita-comfybox.sh`. Full notes in memory + the fix script.

## Open items (need the daemon / live testing — deferred)
1. **Verify Krita inpaint / outpaint / Live Paint end-to-end.** Code paths exist in
   the bridge; not yet observed with a live Krita request. Plan: tail
   `/tmp/comfybox-serve.log` while testing each in Krita; fix any parse/exec error.
2. **`/v1/queue/interrupt` does not reliably stop an in-flight render** — only a
   daemon restart guarantees cancellation. Worth fixing (observe cancellation in the
   pipeline denoise loop) so Queue-tab Interrupt actually works.
3. **Bridge upscale workflow execution** (from the gap-analysis) — Krita's Upscale
   workspace sends `UpscaleModelLoader`/`ImageUpscaleWithModel` workflows the bridge
   doesn't yet execute; SeedVR2/ESRGAN pipelines exist to back it. Biggest remaining
   Krita gap for daily use.
4. Krita renders are slow because styles default to **20 steps** on full checkpoints
   (`cyberrealisticZImage_v50`). Lower to ~8 for speed; keep canvas ≤1024 (DyPE).

## Handoff hygiene
- Before stepping away while Kira (or anything) uses the daemon: **verify the queue
  is not paused** (`/v1/queue` → `is_paused: false`) or their renders won't run.
- Do not restart the daemon while another project is using it (cold start ~40–56s).
