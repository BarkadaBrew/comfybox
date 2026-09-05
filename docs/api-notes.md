# ComfyBox Server — API Notes (hand-maintained)

Operational notes and body schemas that complement the generated
[`api-reference.md`](api-reference.md). That file is regenerated wholesale by
`comfybox docs generate` (and byte-checked in CI); THIS file is hand-maintained
and never touched by the generator — put prose, schemas and examples here.
Content below carried verbatim from the pre-Phase-4 hand-maintained reference
(`docs/api-reference.md` @ 3654f8b4).

## Video generation (LTX-2 / Replicate)

`POST /v1/video/generate`: Video generation. **Local LTX-2** (T2V + I2V) when
the server is started with `--ltx2-weights` + `--ltx2-gemma` (runs through the
render queue, returns 200 with the MP4 path); otherwise the Replicate cloud
proxy (202, job-based). `GET /v1/video/status/{id}` reports video job status
(Replicate proxy).

Local LTX-2 body (snake_case): `{prompt, negative_prompt?, image_path?, width?, height?, frames? (1+8k), steps?, seed?, strength?, extend_to_seconds?, fps?, output_path?}`
→ `{success, output_path, frame_count, duration_seconds, elapsed_seconds, backend: "ltx2-local"}`.
`image_path` present = image-to-video; absent = text-to-video. Output is
contained to the server's allowed output directory.

## Prompt enhancement

`POST /v1/enhance` body: `{prompt, character?, character_description?, content_mode?}`
→ `{success, prompt, enhanced, note?}`.

## LoRA roles (`POST /v1/lora/swap`)

`POST /v1/lora/swap` accepts an optional semantic role on every entry:

```json
{
  "loras": [
    {
      "path": "krea2_turbo_distill_r256.safetensors",
      "scale": 0.6,
      "role": "accel"
    }
  ]
}
```

Valid roles are `kroma`, `accel`, `bypass`, and `control`; omit `role` for an
ordinary style/character LoRA. Roles are declarations, not filename guesses.
In particular, Krea-2 distillation files such as
`krea2_turbo_distill_r256.safetensors` must declare `"role":"accel"` when
they fill the accelerator slot. Auto-staging may change `path`, but preserves
`role`.

## Per-request LoRA stacks and the warm default (#282)

**Every render carries its own stack.** A job's adapters are resolved once, at
submit, and applied at dequeue. There are three sources, in strict precedence:

| # | Source | When it wins |
|---|---|---|
| 1 | the request's own `loras` | whenever the key is present — including `"loras": []`, which means *no adapters* |
| 2 | the named `preset`'s expanded stack | when the request sent no `loras` and the preset was expandable (#286) |
| 3 | the **warm default** | only when the request named neither `preset` nor `loras` |

The generate response says which, in the additive `lora_stack_origin` field:
`"request"`, `"preset"` or `"warm_default"` (absent on the ControlNet arm,
which has always rendered its own request's stack). `GET /v1/generate/status/{id}`
carries the same field once the job has dequeued.

**The warm default is only valid for the base it was published under.** A swap
records the family and model spec it applied to. A bare request that dequeues
onto a *different* checkpoint does **not** take that default: it renders with
**no adapters** and the response carries `warm_default_skipped`:

| code | meaning |
|---|---|
| `family_mismatch` | the default was published under another model family |
| `model_mismatch` | same family, different checkpoint |

This is never a 4xx or 5xx — forcing another base's adapters is what could
throw, and a request that always rendered must not start failing. An **empty**
default (clear the adapters) and the engine's launch-time `--lora` stack are
admitted everywhere; an unknown spec on either side is not a mismatch.

**`lora_reload`** is set on the response when a job actually cleared the
resident adapters and bound a different stack, rather than taking the
same-stack shortcut. Alternating bare and preset renders (Krita + Kira at the
same time) legitimately pay a full clear+reload per job; this field and a
matching warning log are how that cost is measured rather than merely assumed.

**What `POST /v1/lora/swap` now means.** The route, its payload and its
response JSON are unchanged. What changed is its scope: a swap publishes the
**warm default** — the stack a request carrying neither `preset` nor `loras`
renders with — instead of mutating a shared resident stack that later jobs
silently inherit. A swap still applies its stack to the resident pipeline
(a swap-first client expects that, and `SwapResidencyRestore` exists so it can
happen against an evicted pipeline), but no later job picks it up except
through the default.

Read the current warm default from `GET /v1/model/pool`, field
`warm_default_stack` (additive, same per-entry shape as `/health.loras`):

```json
{
  "active": "krea2-raw",
  "warm_default_stack": [
    {"name": "kroma.safetensors", "path": "/…/kroma.safetensors", "scale": 0.6, "role": "kroma"}
  ]
}
```

`/health.loras` answers a different question — what is **resident**, i.e. the
last job's stack. Since #282 that is a consequence of the last render, not a
prediction of the next one.

**Consequences for daemon owners.**

- A bare `/v1/generate` no longer inherits the previous job's adapters. On an
  engine launched without `--lora` and with no swap yet, a bare request renders
  with **no adapters** — deterministically, instead of on whatever happened to
  be resident.
- To keep a stack across bare requests, swap it (it becomes the warm default)
  or name it on every request.
- FIBO and Chroma have **no LoRA application path at all**: `ChromaPipeline
  .generate` takes no adapters, the model pool never forwards `initialLoRAs` to
  either family, and `/v1/lora/swap` refuses them. They render with no adapters,
  always. Nothing is applied for them now — a warm default *or* a stack the
  request named — where the request-named case previously loaded into the
  Flux-1 pipeline they do not render through. Nothing 4xx's that did not
  before; the render is the same bare render it always was, and the Chroma PNG
  now records the empty stack it actually used instead of the coordinator's
  unrelated resident list.
- The video path (`/v1/video/generate`) has always resolved its own stack per
  request (`loras` → the video preset's `loras` → `--ltx2-lora`) and is
  unchanged.

## Preset LoRA references (Krea-2)

Preset LoRA references use `filename` rather than `path` and preserve the
same optional `role`:

```json
{
  "loras": [
    {
      "filename": "krea2_turbo_distill_r256.safetensors",
      "scale": 0.6,
      "role": "accel"
    }
  ]
}
```

### Kroma is a regular LoRA (structured `kroma` is DEPRECATED, 2026-09-04)

Todd reversed the #276/#350-era design: kroma has no special engine
semantics anywhere. `PresetLoRAStack.decide` (the `POST /v1/generate
{"preset": id}` expansion) applies a preset's `loras[]` exactly as declared,
in order — no prepend, no strip, no reordering for a `role: "kroma"` entry.
Declare kroma the same way as any other adapter:

```json
{
  "loras": [
    {
      "filename": "kroma-v0.3-base-lora-rank-384-fro-0985.safetensors",
      "scale": 0.6,
      "role": "kroma"
    }
  ]
}
```

**One-release compatibility shim.** The structured `kroma` object
(`{"kroma": {"strength": <number>, "file": <optional>}}`) still decodes on
`PUT /v1/presets`, but `PresetStore` migrates it on every load and save
(`ImagePreset.migratingKromaDeprecation`): a declared `kroma` with an
explicit, non-empty `file` folds into `loras[]` as a `role: "kroma"` entry
(idempotently — a matching entry already present is never duplicated), and
the structured field itself becomes a DERIVED, read-only echo of that
`loras[]` entry, never an independent value a client can set. A `kroma`
with no `file` (the old "engine-default file" case) has nothing concrete to
become a LoRA of — it migrates to nothing, and the echo is `nil`. Every
response that carries a non-nil `kroma` also carries an additive
`"kroma_deprecated": true` marker. The krea2-family "a preset must declare
kroma" validation rule (O4a) is retired along with this — its absence is as
legal as any other adapter's.

Existing consumers that read `.kroma` (the daemon, the desktop app) keep
working unmodified during the compatibility window; new code should read
`loras[]` and stop relying on the structured field. See
[Krea-2 Raw + r256 preset stack](methods/krea2-r256-preset-stack.md) (some
of that document's structured-kroma framing predates this reversal).

## Gallery output filenames

Default render filenames (no `output_path` in the request) are built by
`ComfyBoxOutputNaming.defaultFilename` (`Sources/ZImage/Server/ComfyBoxOutputNaming.swift`):

```
comfybox-<model>[-<preset>]-<tier>[-<source>]-<yyyyMMdd-HHmmss>-<4-hex-salt>.<ext>
```

e.g. `comfybox-krea2-avocado-20260904-143022-a3f2.png`. `<model>` is the
short name of the active model spec (`krea2`, `kroma-v0.2`, `fibo`, …),
`<tier>` is the request's content mode (`manual` when absent). This
replaced the legacy `zimage-<uuid>.png` / `zimage-krea2-<uuid>.png` scheme
in 2026-08 (commits `3ed1996`, `1cb123e`) — the model segment already
carries the family, so **no code should reintroduce a hardcoded `zimage-`
prefix on a persisted gallery file** (issue #251).

Nothing in ComfyBox or the daemon (`coffeeshop-server`) parses this prefix
to classify a render — model family, mode, preset etc. all come from the
JSON metadata embedded in the PNG itself (`ComfyBoxCatalog/MetadataReader`)
or from the request/response body, never from filename text. Existing
`zimage-krea2-*` files on disk from before the 2026-08 change keep working
unmodified — nothing needs to read or migrate them. `WarmServer.swift`'s
`"zimage-…"` temp-file names (control image, mask, inpaint init, ESRGAN
scratch files) are unrelated: they're process-local scratch paths deleted
before the response returns, never a persisted gallery filename, and are
unreachable for Krea-2 (`ControlNet is not supported for Flux 2 or
Krea-2 models` — the route throws before any temp file is written).

## Queue lifecycle ledger (comfybox#283 / comfybox#217)

`GET /v1/queue/lifecycle?job_id=&limit=` — read-only, append-only record of
what actually happened to queue jobs: `enqueued`, `admitted`, `started`,
`progress` (bounded rate — see below), `checkpointed`, `resumed`,
`interrupted`, `completed`, `failed`, `replayed_after_restart`, `dropped`.
Built as the TELEMETRY that #283 (a restart re-enqueues the active job and
re-renders it from step 1, and nothing reported that accurately) and #217
(the Desktop queue/progress UI goes stale during a render) need before either
issue's proposed behavior changes can be evaluated safely — it changes
nothing about queue behavior itself.

Query params: `job_id` (optional — filter to one job) and `limit` (optional,
default 200, clamped to 1–2000). Response:

```json
{"boot_id": "…", "count": 3, "events": [
  {"sequence": 41, "boot_id": "…", "wall_time": "2026-09-04T…Z", "job_id": "…",
   "kind": "admitted", "job_kind": "generate", "source": "api"}
]}
```

`boot_id` is a fresh UUID generated once per process start — two events with
different `boot_id`s straddle a restart, which is the direct answer to
#283's "nothing distinguishes a recovered job from a new one." `sequence` is
a process-wide (not per-job) monotonic counter, reseeded from the JSONL tail
on restart so it never resets to 0.

Additive fields on existing routes, `null`/absent-safe for older clients:

- `GET /v1/queue`: each `pending[]` entry gains `last_event`; the response
  gains `active_last_event` when a job is active. Both are one
  `QueueLifecycleEvent` object (see above), or absent if the ledger has
  never seen that job (e.g. a snapshot recovered from before this instrument
  shipped).
- `GET /v1/generate/status/{id}`: gains `lifecycle_tail`, the last 5 events
  recorded for that job id, or absent if none.

**Diagnosing #283 from the ledger**: after a bounce, `GET
/v1/queue/lifecycle?job_id=<the id from queue-state.json>` shows the pre-crash
history ending mid-render (no `completed`/`failed`), followed by a
`replayed_after_restart` event under a NEW `boot_id` — `from_step1: true`
(image generate/LoRA swap never checkpoint today, so this is currently
always the answer) confirms the render actually restarted from step 1 rather
than resuming, and `original_job_id` names the job. This is the
operator-visible signal #283 finding 1 says is missing; it does not by
itself change whether a restart drops or resumes the job — that is #283's
open decision, not this instrument's.

Storage: an in-memory ring (default last 4000 events, actor-hop-free reads)
plus `~/.comfybox/queue-lifecycle.jsonl` (append-only, survives a restart;
honors `COMFYBOX_STATE_DIR` like every other engine state path). `progress`
events are throttled to at most one per job per second — a fast-ticking
render cannot flood either store.

See `Sources/ZImage/Server/QueueLifecycleLedger.swift` for the full event
schema and `ReplayClassifier`'s pure from-step-1-vs-resumed logic.

## Startup imports

- Character + preset legacy imports also run **once at server startup**
  (idempotent), merging from `~/.coffeeshop/image-service/`.
