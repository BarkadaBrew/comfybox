# SPEC: LTX-2 Parameter Externalization + Desktop Tuning

**Status:** design · **Author:** 2026-08-02 session · **Owner:** desktop/UI queue

## 1. Problem

36 `LTX2_*` environment variables shape LTX-2 rendering. They live in a launchd
plist, are invisible at runtime, and take a service restart to change. This has
produced repeated *silent wrong-value* failures — the value looked set, or looked
absent, and no surface said otherwise:

| Incident | Cost |
|---|---|
| `LTX2_GUIDANCE_RESCALE` dropped from the plist (2026-07-30) | 3 days of over-saturated/shadow-crushed high-CFG renders |
| `LTX2_TWO_STAGE=0` left after the refine bug | weeks of soft/unrefined output |
| Trailing whitespace in env values (`'1  '`) | `== "1"` false → two-stage silently off, paths "not found" |
| `guidance` never forwarded to `generateT2V` | t2v ran CFG-off for 3 days despite the daemon sending 3.5 |

**The lesson is not "add knobs." It is: the system cannot currently answer
"what settings did this render actually use, and where did each come from?"**
Every incident above would have been caught on sight by that one answer.

Secondary problem: the tuning loop costs ~5 minutes and a service outage per
change (edit plist → bootout → resign → bootstrap → 2 min model reload), plus a
TCC-permission hazard on every restart. Parameters that are already per-request
(`guidance`, `img_compression`, `frames`, `strength`) need none of that, and were
the only pleasant part of the 2026-08-02 tuning session.

## 2. Goals / Non-goals

**Goals**
1. Make the *effective* configuration of any render observable, with provenance.
2. Move render-shaping parameters to per-request + preset, so tuning needs no restart.
3. Give the desktop app a tuning surface, including matched-seed A/B.

**Non-goals**
- Exposing all 36 env vars. Only the ~10 that are actually tuned (§4).
- Replacing `~/.comfybox/config.json` or the plist for machine-level settings.
- Removing env support: env stays as the lowest-precedence override for ops/debug.

## 3. Parameter tiers

Three homes, by what the parameter is *about*:

### Tier A — Render-shaping (→ per-request + preset field)
Varies per shot type (action vs portrait vs t2v). Must be tunable without restart.

`guidance` · `guidance_rescale` · `cfg_schedule` · `stage1_sigmas` ·
`refine_sigmas` · `two_stage` · `cond_fps` · `img_compression` · `strength` ·
`sampler` · `stg_scale` · `face_anchor_strength` · `ic_control`

(`guidance`, `img_compression`, `strength`, `frames` are ALREADY per-request —
this tier is finishing a job that was started.)

### Tier B — Machine-shaped (→ `~/.comfybox/config.json`)
About this box's memory/disk, not about the shot. Wrong values here cause OOM or
corruption, not a different look. Should NOT be in a creative UI.

`plain_decode_max_vol` · `refine_max_vol` · `decode_mode` / `decode_tile` ·
`upsampler_path` · `video_bits_per_px`

### Tier C — Instrumentation (→ env stays, unchanged)
Debug tooling, off in normal operation. Env is the right ergonomics.

`LTX2_TRAJ_DUMP` · `LTX2_REFINE_ROWSTATS` · `LTX2_REFINE_DUMP_DIR` ·
`LTX2_DUMP_LATENT` · `LTX2_REFINE_DECODE_ONLY`

### Precedence (single rule, applied everywhere)
```
request field  >  preset field  >  config.json  >  env  >  built-in default
```
Every resolution records which level supplied the value.

## 4. Phase 1 — Effective-config readout (DO THIS FIRST)

Smallest piece, highest value, useful even if nothing else ships.

**Engine:** a `ResolvedVideoConfig` struct built once per render, holding every
Tier A + B value AND its provenance. Replaces scattered inline
`ProcessInfo.processInfo.environment[...]` reads at point of use.

```swift
struct ResolvedParam<T> { let value: T; let source: Source }
enum Source { case request, preset(String), configFile, env, builtin }
```

**API:**
- `GET /v1/video/config/effective` → current resolution for a hypothetical render
- `GET /v1/video/status/{id}` gains `resolved_config` → what THAT render used
- Log one line per render: the full resolved set (so history is greppable)

**Desktop:** a read-only "Effective Config" card in the Kira/Video tab —
parameter, value, source badge. Highlight anything sourced from `env` or
`builtin` that the user might expect to be set (this is the missing-rescale
detector).

**Validation, at resolution time:** trim whitespace on every env read; reject
non-finite/out-of-range numbers with a loud log line rather than a silent
fallback; verify path-valued params exist on disk.

**Acceptance:** with `LTX2_GUIDANCE_RESCALE` deleted from the plist, the panel
shows `guidance_rescale 0.0 (builtin)` flagged — i.e. the 2026-07-30 incident is
visible in one glance.

## 5. Phase 2 — Tier A to per-request + preset

- Extend `LocalVideoRequest` + the `generate_video` MCP schema with the Tier A
  fields (`enhance` landed 2026-08-02 as the first of these).
- Extend `ImagePreset` (mediaKind "video") with the same fields.
- Thread through `LTX2VideoRequest` → pipeline. **Audit every call site**: the
  t2v `guidance` drop happened because one call omitted a parameter the others
  passed. A single resolved-config object passed down prevents recurrence.
- Move Tier B to `config.json` with env fallback.

**Payoff for the daemon:** kira-daemon stops hardcoding recipe numbers in
`video-tools.ts` and just names a preset ("action", "portrait", "t2v-solo").

## 6. Phase 3 — Desktop tuning + A/B

- **Preset editor** for video presets: Tier A fields with ranges, defaults, and
  inline notes on what each does (the hard-won findings: cfg over-drives above
  ~3.5, rescale counters it, compression is the motion lever…).
- **Matched-seed A/B**: pick a parameter, give 2-4 values, fire N renders at one
  seed, show frames side by side with saturation/sharpness/flicker readouts.
  This is exactly the loop run by hand on 2026-08-02 (strength, compression,
  rescale) — automating it turns a multi-hour restart-driven sweep into one submit.
- **Provenance-aware editing**: show the inherited value and its source next to
  every field, so an override is visibly an override.

## 7. Risks

- **More knobs = more wrong values.** Mitigated by validation (§4) and by making
  the resolved value visible everywhere. Observability lands first for this reason.
- **Preset sprawl.** Cap the UI to the tuned subset; leave the rest env-only.
- **Behavior drift for existing callers.** Absent request keys must resolve
  exactly as today — Phase 2 ships with byte-identical defaults.

## 8. Sequencing

1. Phase 1 (engine resolution + readout + validation) — independent, ship alone
2. Phase 2 (per-request/preset plumbing) — depends on 1's resolution object
3. Phase 3 (desktop editor + A/B) — depends on 2; rides with the desktop rebuild

Estimated: Phase 1 ~250 LOC engine + small desktop card; Phase 2 mechanical but
broad (audit every pipeline call site); Phase 3 is the largest, mostly UI.

---

## Rev 2 — Codex findings resolution (2026-08-03, findings #14–20 in reviews/codex-specs-rereview-2026-08-03.md)

14. **Split-brain readout (#14 — critical, ACCEPTED):** shipped Phase 1
    reports a resolution the renderer does not consume; point-of-use env
    reads remain authoritative. Interim mitigation shipped same day: the
    resolver's bool kinds now mirror renderer semantics EXACTLY
    (`two_stage` = raw `== "1"`, `ic_control` = raw `!= "0"`), with
    non-canonical values flagged — the readout can no longer claim a value
    the renderer won't act on. The full fix IS Phase 2: one typed
    `ResolvedVideoConfig` built per render and passed through every call
    site, replacing every inline env read. Phase 2's definition of done is
    "grep for `environment[\"LTX2_` in render paths returns only Tier C".
15. **Per-job resolved_config (#15):** `VideoJobStatus` gains
    `resolved_config`, snapshotted onto the job BEFORE execution; persisted
    into the render trace (jobs prune after 1h — the trace is the durable
    record).
16. **Effective endpoint contract (#16):** the endpoint accepts optional
    request-shaped params + `preset_id` and returns `requested_config` AND
    the derived `render_plan` (aspect matching, dim snapping, two-stage
    halving + stage-1 floor, duration folding) with per-step notes. Existing
    per-request fields (guidance, strength, frames, dims, fps) join the
    registry in Phase 2.
17. **Validation consistency (#17):** bool fix shipped (see #14). Remaining
    for Phase 2: enum validation for sampler/decode strings from one
    canonical descriptor table; on invalid high-precedence values, resolution
    FALLS THROUGH to the next lower source (flagged) instead of jumping to
    builtin.
18. **Lifecycle column (#18):** the parameter manifest gains
    `lifecycle: load-time | per-render`. `two_stage`'s upsampler dependency
    is load-time today; Phase 2 lazy-loads the upsampler on first enabled
    request so the param can become per-render truthfully.
19. **Field matrix (#19):** Phase 2 specifies per parameter: wire name,
    Swift type, range/enum, units, applicability (t2v/i2v/refine), default,
    eligible precedence sources, lifecycle, preset serialization, MCP schema
    fragment, UI control, cross-field constraints. Tier B precedence starts
    at configFile (no request/preset overrides).
20. **A/B integrity (#20):** batches resolve ONCE at submission and freeze
    prompt/template/exemplars, source-image hash, model+LoRA set, preset
    revision, and full effective config; only the declared factor varies.
    Arm order randomized; batch/arm IDs persisted before enqueue; metrics
    named per axis (brightness/saturation drift, cyan-blown fraction of ALL
    pixels, contrast-normalized sharpness).
