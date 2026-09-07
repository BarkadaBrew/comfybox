# ComfyBox ↔ ComfyUI capability parity map

**Repo:** `BarkadaBrew/comfybox` (`~/Projects/zimage.swift`, Swift/MLX)
**Maintained:** living reference — update as pipelines are ported/verified.
**Last updated:** 2026-09-06

> **Framing.** ComfyBox is *not* "ComfyUI + MLX." ComfyUI is a general node-graph engine plus an unbounded custom-node ecosystem on PyTorch/CUDA. ComfyBox is a **curated set of pipelines hand-ported to mlx-swift** — a capability exists only if it was ported *and* verified against the reference. It is a vertically-integrated, Apple-Silicon-native **subset** that is faster and better-integrated for the models it covers, not a superset. Parity is per-feature and never automatic (see the LTX-2 fps-RoPE bug, `FDD-ltx2-temporal-motion.md`).

**Legend:** ✅ Working (production or integration-tested) · 🟡 Wired but unverified / experimental · 🔴 Stub or broken · ⬜ Not ported

## Image generation — base models
| Capability | ComfyBox | Status |
|---|---|---|
| Z-Image-Turbo | native MLX | ✅ Working (foundation) |
| Krea-2-Turbo (q8) | native MLX | ✅ Working (prod warm default) |
| CyberRealistic / "Kira" (Z-Image Base variant) | native MLX | ✅ Working (prod) |
| Flux.2 / Klein-9b | ported, wired | 🟡 Loadable, no integration test |
| Chroma (Flux-based) | ported, wired | 🟡 Loadable, unverified |
| FIBO | ported, wired | 🟡 Loadable, unverified |
| Flux.1 dev/schnell | fallback family | 🟡 Supported path, unproven |
| Zeta-Chroma (Z-Image art model) | — | ⬜ Planned, not ported (its schedule shift IS ported — #154) |
| SD 1.5 / SDXL / SD3 | — | ⬜ Not ported (different lineage) |

## Conditioning & control
| Capability | Status |
|---|---|
| ControlNet — canny / depth / hed / pose | ✅ Working (integration-tested) |
| LoRA (baked + dynamic runtime) | ✅ Working (tested) |
| LoKr (Kronecker) + quantized-base LoRA | ✅ Working |
| Inpaint / outpaint (mask region/invert/grow/feather) | ✅ Working (#239) |
| img2img (strength semantics) | ✅ Working |
| Prompt enhancement (Qwen LLM) | ✅ Working |
| IPAdapter / InstantID / T2I-Adapter / regional prompt | ⬜ Not ported |

## Video
| Capability | Status |
|---|---|
| LTX-2 text-to-video | ✅ Working — base motion fixed (fps-RoPE) |
| LTX-2 image-to-video | ✅ Working (validated dynamic) |
| LTX-2 two-stage refine | 🟡 **WIP** — damps motion 11.9→~5 (FDD §7) |
| LTX-2 long-video identity anchor | ✅ Working (#231) |
| Storyboard chaining (last-frame) | ✅ Working (#237) |
| Montage / compositor (ken-burns, transitions) | ✅ Working (#232) |
| LTX-2 audio (A2V / V2A cross-modal) | 🔴 **Stub** — Phase 5 deferred |
| Wan 2.2 i2v | 🔴 **Broken** — purple/magenta cast, unmerged branch |
| Replicate cloud video proxy | ✅ Working (fallback) |
| AnimateDiff / SVD / Hunyuan / CogVideo | ⬜ Not ported |

## Upscale / restoration
| Capability | Status |
|---|---|
| ESRGAN / RRDBNet | ✅ Working (`upscale`) |
| SeedVR2 super-resolution | 🟡 Working @1024px; 2048px experimental (OOM risk) |
| Style packs — engine-applied post-process looks (`phone`, `trix-bw`, `hp5-soft`) | ✅ Working (#399) — pure CPU pass after decode, request/preset selected; ComfyUI has no equivalent single node (it is an image-op subgraph there) |

## Samplers / schedulers
| Capability | Status |
|---|---|
| Euler (FlowMatch), DDIM, DEIS, DPM++ 2M, DPM++ 2S-ancestral, Heun, RES2s | ✅ Working set |
| `ModelSamplingAuraFlow` / `ModelSamplingSD3` schedule shift (`shift`, `time_snr_shift`) | ✅ Working — request/preset/bridge, Z-Image family (#154); unit-pinned, live-unverified |
| `ModelSamplingFlux` log-shift (`mu`) | ✅ Working (Krea 2, same `shift` field) |
| CFG++ / `euler_ancestral_cfg_pp` (video) | ✅ Working |
| STG (spatiotemporal guidance) | 🟡 Optional; +50% motion but artifact-prone |

## Style system
| Capability | Status |
|---|---|
| Prompt-style presets (`GET /v1/styles`, `apply_style`/`list_styles` MCP) | ✅ Working |
| Studio Packs (bundled preset + LoRA + style, desktop) | ✅ Working |
| Server presets (LoRA picker/sliders, negative prompt, Save-as) | ✅ Working (PR #250) |
| LoRA registry + prompt-based suggestion engine (JIT catalog) | ⬜ Designed (FDD-krea2-standard-jit-registry), not built |

## Infrastructure & ops — *ComfyBox's edge (ComfyUI doesn't do these)*
| Capability | Status |
|---|---|
| Warm server + model pool + LRU eviction + VRAM budget | ✅ Working |
| Native group-wise quantization (4/8-bit Affine/MXFP4) | ✅ Working (65→40 GB LTX gate) |
| MCP tool surface (39 tools) | ✅ Working |
| Queue management (pause/resume/reorder/attribution) | ✅ Working |
| Desktop app + Kira Suite + NSFW gate | ✅ Working |
| Krita AI-Diffusion bridge | ✅ Working |

## ComfyUI interop
| Capability | ComfyBox | Status |
|---|---|---|
| Import & run ComfyUI **API-format** workflows | ~25-node allowlist + compat report | ✅ Phase 1 (unsupported nodes reported, not run) |
| Arbitrary **custom-node** execution | — | ⬜ Not supported (architectural) |
| Node-graph editor UI | — | ⬜ Not a graph editor |

---

**Planning takeaways**
- The proven core (Z-Image family + LoRA + ControlNet + inpaint + LTX-2 video + upscale + ops) is production-grade and beats ComfyUI-on-Mac on speed, memory, and integration.
- 🟡 rows are wired but need a verification pass (smoke render + reference diff) before parity can be claimed.
- 🔴 rows are the real open engineering: LTX-2 refine damping, LTX audio cross-modal, Wan i2v color cast.
- ⬜ rows are native-port costs (days each), not config flags. Treat every "can ComfyBox do X?" as "has X been ported *and verified*?"
