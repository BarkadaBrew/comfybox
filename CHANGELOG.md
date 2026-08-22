# Changelog

Behaviour changes that a caller could observe, announced rather than slipped
in. Newest first. Entries name the work package of the design document that
decided them.

## Unreleased

### Changed

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
