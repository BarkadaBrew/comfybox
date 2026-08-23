# Changelog

Behaviour changes that a caller could observe, announced rather than slipped
in. Newest first. Entries name the work package of the design document that
decided them.

## Unreleased

### Added

- **A LoRA pair whose target is a bare parameter now binds, as a delta.**
  `LoRAApplicator.applyDynamically` binds by walking `namedModules()` and
  wrapping Linears, so it can only reach a target that IS a `Linear` or
  `QuantizedLinear`. The Krea 2 turbo distills
  (`krea2_turbo_distill_r256`, `…_r128`) offer
  `diffusion_model.last.modulation.lin.lora_A/lora_B` — a rank-2 SVD of the
  (2, 6144) modulation delta — and `Krea2SimpleModulation.lin` is a bare
  `MLXArray` added to the timestep vector, not a Linear. The strict Krea 2
  apply therefore refused all 531 keys over the one it could not reach
  ("did not bind completely — 1 key(s) matched nothing:
  `last.modulation.lin.weight`"), and no r256/r128 render was possible.
  `LoRABareParameterPairs.split` now rewrites such a pair into a `.diff` on
  the real parameter path — the same route Kroma's
  `last.modulation.lin.diff` [2, 6144] has always taken — so it applies
  through `LoRAPatchSession` with an exact first-write-wins snapshot and is
  restored on rollback. It shows up in provenance as
  `loras[].deltas_applied`, never as a silently dropped layer.
  **Fail-closed is unchanged:** a pair diverts only when no bindable Linear
  answers to its key AND a real parameter path does AND `up @ down` fits that
  shape exactly, so a key naming something the architecture does not have
  still throws `partialApplication`, and one naming a parameter it cannot fit
  throws `incompatibleWeights`. The module walk wins unconditionally, so
  every adapter that bound before binds identically
  (`krea2_turbo_lora_rank_64_bf16`: 264/264 pairs, 7 deltas, before and
  after).
- **Every Krea 2 render now carries a provenance record, `applied`, on four
  sinks.** The `/v1/generate` response, the async job status
  (`GET /v1/generate/async/{id}`), `/health.last_recipe`, and the PNG's EXIF
  `UserComment` JSON all carry the same block: what LOADED (`base_model`,
  `base_variant`, `base_model_file`, `quantization`, `vae`, `vae_layout`,
  `vae_source`, `text_encoder`, `loras[]` with `pairs_bound` /
  `shape_rejected` / `deltas_applied` / `role`, `control_lora`) and what RAN
  (`stages[]` with the resolved `sampler` / `sigma_schedule`, the shift, the
  sigma head and tail, `steps_requested` / `steps_effective` / `steps_run` /
  `model_evals`, and `negative_prompt` **only when guidance > 1**, because
  below that it did not apply). Every value is read back from the pipeline and
  the loop — none is echoed from the request. Krea 2 only for now; other
  families emit no `applied` block rather than a half-filled one.
  FDD-krea2-raw-recipe WP-E10 §3.10, D8/D12/D15/D22, AC-60..64.
- **`/health` gains `build_sha`, `model_alias`, `model_variant` for Krea 2,
  and `last_recipe`.** `build_sha` is the git short sha stamped into the binary
  by `scripts/gen-build-info.sh` (`"unknown"` for an unstamped build), so a
  clobbered or wrong-branch binary is detectable from outside. `model_alias`
  restores the declared alias (`krea2-raw`) beside the resolved directory that
  `model` carries. `last_recipe` is dropped the moment the checkpoint that
  produced it stops being resident — a different base activated, a different
  family prepared, or the whole image stack evicted for an LTX-2 render
  (#218). A record whose checkpoint is no longer in memory is not
  provenance. WP-E10, AC-34b.
- **PNG generation metadata is byte-stable.** `parametersJSON` is now written
  with sorted keys; it previously followed Swift's per-process dictionary hash
  order, so the same render produced different EXIF bytes after every restart
  and the whole-file SHA of a PNG could not be compared across runs.
- **`/health.progress_percent` advances during a Krea 2 render.** The Krea 2
  arm's per-step callback only wrote a log line, so the field sat at 0 for the
  whole render; both families now publish through one mapping.
- **`loras[].role`** on `/v1/generate` and `/v1/lora/swap` — `kroma` | `accel`
  | `bypass` | `control`. The sender declares which configuration slot an
  adapter fills; the engine stores it on the applied configuration and reads
  it back into `applied.loras[].role`, so a client reports `kroma_strength`
  as applied instead of matching filenames. An unknown label is a 400.

### Changed

- **`vae` on a non-Krea-2 family is now a 400, not silently ignored.** A
  caller that named a decoder for flux1/flux2/fibo/chroma previously got the
  family's default with no error and no log. WP-E10 ("E9b"), D18.
- **A Krea 2 VAE file carrying only a SUBSET of the decoder's parameters is
  refused (`vaeIncomplete`) instead of loading a mixed decoder.** The subset
  used to overwrite its targets and leave every other parameter at whatever
  was resident — half one VAE, half another, named as the new file. Nothing is
  written when the check fails. WP-E10 ("E9b").
- **The `krea2 handoff:` log line fires only on an actual base swap.** A no-op
  re-activation of the resident base, and a cold start with nothing to hand
  off from, emit nothing. D17, AC-59a.
- **An async job's id is the id the queue persists and replays under.** A job
  that failed replay after a restart was recorded under the coordinator's own
  private id — a second UUID the `/v1/generate/async` caller never saw — so it
  could not be polled. AC-18.

- **Krea 2 `shift` is mu, and Krea 2's `beta`/`beta57`/`karras`/`exponential`
  are built on ComfyUI's `ModelSamplingFlux` table (deliberate behaviour
  change; supersedes the two WP-E12 entries below).** ComfyUI registers Krea 2
  as `ModelSamplingFlux(shift=1.15, timesteps=10000)`, whose
  `flux_time_shift(mu=shift, t)` is the same function as Krea 2's native warp —
  so a request `shift` now **is** `mu` (effective linear shift `e^shift`;
  `shift: 1.15` reproduces the published workflow's grid) instead of
  `mu = log(shift)`. Under the Krea 2 family the table-backed schedules index
  the 10 000-entry Flux table built from that `mu` — `beta(6)` at `shift 1.15`
  is now `[1.0, 0.969095, 0.892582, 0.759584, 0.545649, 0.241540, 0.0]`, not the
  1000-entry DiscreteFlow grid (`σ₁ 0.919919`) — and `karras`/`exponential`
  take their bounds from the same table (`σ_min 3.1575e-4` at 1.15). With
  `shift` omitted the same table is built at the resolution-derived `mu`, so
  Krea 2 `beta`/`beta57`/`karras`/`exponential` renders **without** `shift`
  also move (they previously indexed an unshifted 1000-entry table).
  `SchedulerFactory` refuses those schedules for Krea 2 without `mu`
  (`missingMu`), never defaults. Z-Image and every family that decodes a
  `scheduler_config.json` stay on the DiscreteFlow table, `mu`-free and
  bit-unchanged (`ZImageSchedulerConfig.modelSampling` defaults to
  `.discreteFlow`). `Krea2Sampling.schedulerConfig(shift:)` lost its parameter.
  `deis_3m` warm-up is documented as 4 steps (`order + 1`), not 3.
  FDD-krea2-raw-recipe Addendum A.1, WP-E12b, AC-21 (re-pinned) / AC-24.

- **`beta` / `beta57` sigma schedules replaced, in place, with ComfyUI's
  algorithm (deliberate behaviour change).** The previous `SigmaSchedule.beta`
  integrated the beta CDF and interpolated in log-sigma space; it was not the
  schedule its name claims. It is now ComfyUI's `beta_scheduler` exactly: beta
  **PPF** at `1 − i/steps`, `rint`-ed to an index into the model's discrete
  sigma table (`σ[i] = shift·t/(1+(shift−1)·t)`, `t=(i+1)/T`, from the
  scheduler config's `shift`/`num_train_timesteps`), consecutive duplicates
  dropped, trailing 0. At 6 steps / shift 1.15 the old σ₁ was 0.1596; ComfyUI's
  (and now ours) is 0.9199 — both pinned in
  `BetaScheduleComfyParityTests.testBeforeAfterFixture`, and 6/9/30-step grids
  are pinned against the E18 oracle dump. **Every caller of `beta`/`beta57`
  renders differently from earlier builds** — the Z-Image CLI (`--sigma-schedule
  beta`), `/v1/generate {"sigma_schedule":"beta"|"beta57"}`, the Krita bridge,
  and the daemon's tile pipeline (`seamless-processor.ts`). Because ComfyUI
  de-duplicates, a request can run **fewer** steps than asked: the scheduler's
  `numInferenceSteps` is the produced count, the pipelines loop over it and log
  `steps_effective` when it differs (on the 1000-entry table `beta` first
  shrinks at 139 steps → 138, `beta57` at 97 → 96 — exactly where ComfyUI
  does; every production budget ≤ 52 is unaffected). Rollback = revert
  WP-E12; `bong_tangent` (WP-E11) is independent.
  FDD-krea2-raw-recipe D5 / §3.11, WP-E12, AC-21/22.

### Added

- **`shift` request field (Krea 2).** `POST /v1/generate {"shift": 1.15}` and
  `Krea2Pipeline.Request.shift` / `Img2ImgRequest.shift` override the
  resolution-dependent log-shift `mu` with `log(shift)` for schedule
  construction; absent = unchanged behaviour. Non-positive values and `shift`
  on a non-Krea-2 model family are **400**s naming the field, never ignored.
  `Krea2Sampling.resolveShift` is the one seam that yields `mu`, the effective
  shift, its source (`dynamic` | `explicit`) and the scheduler config together.
  FDD-krea2-raw-recipe D3, WP-E12.

- **Z-Image `res_2s` output changes (deliberate).** `RES2sScheduler` is an
  exponential integrator in the data prediction `x₀ = x − σ·v`; `ZImagePipeline`
  and `ZImageControlPipeline` were feeding it the flow velocity, which is
  dimensionally wrong. Every `ZImageScheduler` now declares a
  `modelOutputConvention` (`.velocity` for all existing schedulers,
  `.dataPrediction` for `res_2s`) and the pipelines convert once per model
  evaluation through `modelInput(velocity:sample:sigma:)`. A Z-Image render with
  `scheduler: res_2s` therefore differs from earlier builds — measured in
  `ZImageRES2sCorrectionTests` (pre-fix feed misses the exact-field x₀ by >1.0
  relative; post-fix reconstructs it to ≤1e-5). The default euler/flow path and
  every other sampler are byte-identical. FDD-krea2-raw-recipe D2 / §3.2, WP-E2,
  AC-10/11/74.
