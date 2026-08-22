# Upstream oracle sources — provenance

These files are verbatim copies of the upstream sources the scheduler parity
fixtures were generated from (FDD-krea2-raw-recipe §5.2, WP-E18). They are
**validation oracles only**: ComfyBox never imports or executes them, and no
engine code path depends on anything in `scripts/oracles/`. They are pinned here
so a reviewer can read the exact algorithm a fixture encodes without a network,
and so a future re-run of the generator can be diffed against the same commit.

Fetched 2026-08-22 with `curl` from `raw.githubusercontent.com` at the commits
below. Whole repositories are **not** vendored; the generator clones both repos
at these same commits into a scratch cache because RES4LYF's sampler imports the
entire package and ComfyUI at runtime.

## RES4LYF

| field | value |
|---|---|
| repo | https://github.com/ClownsharkBatwing/RES4LYF |
| commit | `26036f647ca15d3048a193daf99a40cecfc3820d` |
| commit date | 2026-08-07T00:13:47Z (`main` as of 2026-08-22) |
| files | `beta/rk_sampler_beta.py`, `beta/rk_method_beta.py`, `beta/rk_coefficients_beta.py`, `beta/rk_noise_sampler_beta.py`, `beta/rk_guide_func_beta.py`, `beta/phi_functions.py`, `beta/constants.py`, `beta/deis_coefficients.py`, `beta/noise_classes.py`, `sigmas.py`, `helper.py` |

What each is load-bearing for:

- `beta/rk_sampler_beta.py` — `sample_rk_beta`, the loop the step traces are exported from (the `ClownsharKSampler_Beta` node's sampler function).
- `beta/rk_method_beta.py` — `RK_Method_Exponential` / `RK_Method_Linear` (the model call and epsilon anchoring, `noise_anchor = 1.0`), `set_coeff`, `update_substep`, `bong_iter` (T3).
- `beta/rk_coefficients_beta.py` — every tableau (`ralston_2s/3s/4s`, `res_2s`, `res_3s`, …), `get_rk_methods_beta` with the DEIS order ramp and the `multistep_extra_initial_steps = 1` warm-up rule, the `c.append(1)` final node.
- `beta/rk_noise_sampler_beta.py` — `get_sde_step` / `get_sde_coeff` (the hard-mode VP split), `set_sde_substep`, `swap_noise_step` / `swap_noise_substep` (where the re-noise is applied, after z-scoring the noise), `prepare_sigmas` (the σ_min insertion before the trailing 0).
- `beta/phi_functions.py` — `Phi` (mpmath, 80 digits, `analytic_solution=True` by default) used for the exponential tableaus.
- `beta/deis_coefficients.py` — `get_deis_coeff_list(…, deis_mode="rhoab")`, the closed-form branch with the corrected `min(i+1, max_order)` ramp.
- `sigmas.py` — `bong_tangent_scheduler` / `get_bong_tangent_sigmas`, and `get_sigmas` (the node's `denoise` handling: `total_steps = int(steps/denoise)`, then the last `steps+1` sigmas).
- `beta/noise_classes.py`, `beta/constants.py`, `helper.py`, `beta/rk_guide_func_beta.py` — imported by the sampler; copied so the import chain is reviewable.

## ComfyUI

| field | value |
|---|---|
| repo | https://github.com/comfyanonymous/ComfyUI |
| commit | `783545f689a0af730065994b46b382ae24844c99` |
| commit date | `master` as of 2026-08-22 |
| files | `comfy/samplers.py`, `comfy/model_sampling.py`, `comfy/k_diffusion/sampling.py`, `comfy/k_diffusion/deis.py` |

What each is load-bearing for:

- `comfy/samplers.py` — `beta_scheduler` (PPF → rounded index into the model's sigma table, de-duplicate, append 0), `calculate_sigmas`, `SCHEDULER_HANDLERS` (RES4LYF registers `bong_tangent` and `beta57` here), `KSAMPLER.sample` (initial `noise_scaling`).
- `comfy/model_sampling.py` — `CONST`, `ModelSamplingFlux` (**what ComfyUI builds for Krea 2**: `supported_models.py` `class Krea2` → `model_base.Krea2(model_type=ModelType.FLUX)` → `model_sampling()` composes `ModelSamplingFlux` + `CONST`; 10 000-entry table, `sigma(t) = e^shift / (e^shift + 1/t − 1)`, shift 1.15), and `ModelSamplingDiscreteFlow` (the class FDD D3/D5/AC-21 assumed; 1000-entry table, `σ = shift·t / (1 + (shift−1)·t)`).
- `comfy/k_diffusion/deis.py` — ComfyUI's uncorrected `rhoab` (`min(i, max_order)` ramp); kept for the diff against RES4LYF's, which is what the workflow runs.
- `comfy/k_diffusion/sampling.py` — reference for the k-diffusion samplers the existing Swift schedulers were ported from.

## Re-fetching

```
R=26036f647ca15d3048a193daf99a40cecfc3820d
curl -sfL https://raw.githubusercontent.com/ClownsharkBatwing/RES4LYF/$R/beta/rk_sampler_beta.py -o scripts/oracles/upstream/res4lyf/beta/rk_sampler_beta.py
C=783545f689a0af730065994b46b382ae24844c99
curl -sfL https://raw.githubusercontent.com/comfyanonymous/ComfyUI/$C/comfy/samplers.py -o scripts/oracles/upstream/comfyui/comfy/samplers.py
```

If either upstream moves and a regenerated fixture no longer matches the values
pinned in `Tests/ZImageTests/Scheduler/OracleFixtureTests.swift`, the port
source moved and the FDD is stale — that is the alarm, not a test to loosen.
