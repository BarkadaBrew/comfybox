# FDD: Glimmer GPU slot — every Glimmer call holds a top-priority GPU slot

**Status:** v1 · 2026-09-17 · Author: Opus 5 (Claude Code) · Repos: comfybox (engine) + coffeeshop-server (daemons)

## 1. Requirement (Todd, 2026-09-17, verbatim)

- "any glimmer action requires a scheduled GPU lease slot with top priority."
- Fallback decision after the Ollama eval ("build it"): **wait for the slot** for chat turns, tool calls, vision, long prompts and t2v authoring; **Ollama CPU fallback allowed only** for image-prompt authoring and i2v motion text.

Glimmer = `Muse-Glimmer-30B-heretic-MLX-Q6` on the Mac's mlx-serve `:11234`, sharing the GPU with ComfyBox renders (1.6 tok/s under a render vs 15–35 idle).

## 2. What exists and why it is not enough

- `coffeeshop-server/src/image/gpu-lease.ts` `withGpuPriority` / `openGpuPriority`: refcounted **admission pause** only (`POST /v1/queue/pause`). A render in flight keeps the GPU; the pause bit is global across processes; holders have no rank. Used only by prompt-optimizer, prompt-pregen, avocado-text.
- Unleased Glimmer callers: `llm-router` local chat turns, `kira/turn-hooks` planner, `kira/vision/vision-client`, `kira/vision/local-text`, `src/vision/vision-client` (Bree), comfybox MCP `repair_image` diagnosis (`VisionChat`), desktop captioning/assistant.
- Engine #1479 can checkpoint an **in-flight LTX-2 video** at a step boundary, but only for a `preempt: true` image job. Krea-2 image renders cannot be preempted.
- `llm-route` arbiter routes several kinds to the Ollama failover while the engine is busy — including t2v authoring and vision enrichment, which the eval showed Ollama cannot do (0/6 BEATS, no vision, no tools).

## 3. Design

### 3.1 Engine: inference slots (comfybox)

A lock-based `InferenceSlotTable` owned by `WarmServer`, served on the **sync control plane** (no actor hop, answers during a render):

| Route | Body | Returns |
|---|---|---|
| `POST /v1/queue/inference-slot` | `{"holder": String, "ttl_sec": Int?}` (ttl default 180, max 900) | `SlotStatus` |
| `GET /v1/queue/inference-slot/{id}` | — | `SlotStatus` or 404 |
| `POST /v1/queue/inference-slot/{id}/renew` | `{"ttl_sec": Int?}` | `SlotStatus` or 404 |
| `DELETE /v1/queue/inference-slot/{id}` | — | `{"released": Bool, "active_slots": Int}` |
| `GET /v1/queue/inference-slot` | — | `{"active_slots": Int, "slots": [SlotStatus]}` |

`SlotStatus` = `{"slot_id", "holder", "state": "granted"|"preempting"|"waiting", "eta_sec": Double?, "expires_at": ISO8601, "active_slots": Int}`.

Semantics:
1. **Admission:** while ≥1 unexpired slot is held, the job loop starts no render. Independent of the operator's manual pause: releasing slots never un-pauses a manual pause.
2. **Grant:** `granted` when no render is on the GPU, or the in-flight video is parked in an inference hold.
3. **Top priority over video:** acquiring while an LTX-2 video renders raises the #1479 preemption signal with an **inference hold** instead of an image preemptor. The video checkpoints at its next step boundary **without evicting weights** (Glimmer needs compute, not the video's memory), parks until the slot table is empty, then resumes. State is `preempting` until it parks. No refusal guard (no evict/reload cost). If another preemption is already in flight, or the video does not yield within the existing fallback window, state is `waiting` until the render ends.
4. **Image render in flight:** `waiting`, with `eta_sec` from progress (`elapsed / pct - elapsed`). Admission is already closed, so the slot is granted the moment that render ends.
5. **Leak safety:** TTL expiry releases a slot (logged at warning). `/health` and `/v1/queue` report `inference_slots`.

### 3.2 Daemon: `glimmer-slot.ts` (coffeeshop-server, platform tier)

- `isGlimmerEndpoint(url)`: host:port in `BREE_GLIMMER_ENDPOINTS` (default `10.0.100.134:11234,127.0.0.1:11234,localhost:11234`). Explicit list, no probing.
- `acquireGlimmerSlot(consumer, opts)`: POST, poll GET every 1 s until granted or `maxWaitMs`, renew every ttl/2 while held, idempotent `release()`. Outcomes: `granted`, `timed-out`, `engine-unreachable` (fail open, like the lease), `unsupported` (404 on an older engine → fall back to the existing admission-pause lease), `disabled` (`BREE_GPU_ADMISSION_DISABLED=1`).
- `withGlimmerSlot(consumer, baseUrl, fn, opts)`: non-Glimmer URL → `fn()` unchanged; Glimmer → acquire, run, release in `finally`.
- Wire every daemon caller in §2. Existing `withGpuPriority` stays for non-Glimmer local backends.
- Arbiter policy: only `prompt-optimization` (image) and `motion-optimization` (i2v) may route to the failover; new kind `t2v-authoring` (t2v + beats-only) and `motion-enrichment` never route; `persona-override` keeps its **cloud** failover untouched (cognition may use cloud) but a local Glimmer turn now waits for the slot rather than crawling.

### 3.3 Engine-local callers (comfybox)

`repair_image` diagnosis (MCP) and desktop `VisionService`/`AgentService` acquire the same slot through `WarmServerClient` / `EngineService` when their provider URL is a Glimmer endpoint.

## 4. Out of scope (named)

- Obsidian plugins (coffee-shop, sheet-plus) calling `:11234` directly — separate repos; follow-up.
- Preempting in-flight Krea-2 image renders (no resumable entry point).
- Cross-process fairness between Bree and Kira slots (FIFO by acquire time is enough at this volume).

## 5. Rollout

1. Daemon PR deploys first: against today's engine the slot call 404s → `unsupported` → the existing lease, so behaviour is unchanged until the engine ships.
2. Engine PR deploys when no other session is rendering (a serve restart drops the in-flight job). Then slots engage everywhere with no daemon redeploy.
3. Verify: `GET /v1/queue/inference-slot` shows holders during Kira turns; a video render logs `inference hold` checkpoints; Glimmer decode during a render returns to idle-class tok/s.
