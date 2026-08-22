#!/usr/bin/env python3
"""
gen_scheduler_fixtures.py — one-off oracle fixture generator for the scheduler
parity tests (FDD-krea2-raw-recipe WP-E18, §5.2).

VALIDATION TOOL ONLY. ComfyUI and RES4LYF are oracles here, never backends;
this script is not a runtime dependency of ComfyBox and nothing in the engine
imports it. It exists so the Swift tests can pin *what upstream computes* and
fail loud when upstream moves.

What it writes (into --out, default Tests/ZImageTests/Fixtures/Scheduler):

  comfy_sigmas.json
      `beta` / `beta57` at 6/9/30 steps under BOTH model-sampling classes —
      `ModelSamplingFlux(shift=1.15)`, which is what ComfyUI builds for Krea 2
      (supported_models.py `class Krea2` → model_base.Krea2 with
      ModelType.FLUX → model_sampling(): CONST + ModelSamplingFlux), and
      `ModelSamplingDiscreteFlow(shift=1.15)`, which is what the FDD's D3/D5/
      AC-21 assumed — plus `bong_tangent` at 2/6/8/9/10/12/20 (shift-free, and
      asserted identical under both classes), and the reference stage-2 grid
      `get_sigmas(model, "bong_tangent", steps=2, denoise=0.2)`.

  res4lyf_deis_coeffs.json
      RES4LYF `get_deis_coeff_list(sigmas, max_order, deis_mode="rhoab")` for
      orders 2/3/4 on fixed sigma arrays (RES4LYF's copy carries the corrected
      `min(i+1, max_order)` ramp; ComfyUI's `comfy/k_diffusion/deis.py` is the
      uncorrected `min(i, max_order)` — the workflow runs RES4LYF's).

  res4lyf_trace_<recipe>_<tier>.json + .safetensors   (6 traces)
      RES4LYF `sample_rk_beta` step traces against the scripted denoiser
      `0.5*tanh(x) + 0.25*sigma - 0.1*x` on a 1x16x8x8 latent, for
      {res_2s + beta, 6 steps} and {deis_3m + bong_tangent, 2 steps @ denoise
      0.2} at tiers T1 (eta 0, bongmath off), T2 (eta 0.5), T3 (eta 0.5 +
      bongmath). Per step: sigma, sigma_next, the tableau RES4LYF actually ran,
      the eta split (sigma_up / sigma_down / alpha_ratio), every model call's
      input and output, every substep re-noise with its injected (z-scored)
      noise tensor, every bongmath fixed-point result, x_next and x_out.
      Tensors are float32 (RES4LYF's work_dtype) in the .safetensors; scalars
      are the sampler's float64 in the .json.

Interpreter
  A torch-capable Python is required. This fixture set was produced with a
  dedicated venv (no project venv was touched):
      python3 -m venv <scratch>/oracle-venv
      <scratch>/oracle-venv/bin/pip install torch numpy scipy mpmath einops \
          safetensors tqdm psutil pyyaml aiohttp torchsde Pillow torchvision \
          torchaudio transformers packaging comfy-kitchen==0.2.31 comfy-aimdo \
          comfyui-frontend-package comfyui-workflow-templates \
          comfyui-embedded-docs pydantic pydantic-settings alembic SQLAlchemy \
          filelock requests simpleeval blake3 yarl comfy-angle av \
          PyWavelets kornia opencv-python-headless matplotlib
  (ComfyUI's requirements.txt, plus what RES4LYF's package import pulls in:
  mpmath for its analytic phi functions, PyWavelets/kornia/opencv/matplotlib
  for modules the sampler never calls but the package imports.) The exact
  versions used are written into every fixture's `provenance` block. RES4LYF's
  res4lyf.py registers HTTP routes on ComfyUI's PromptServer at import time;
  `import_res4lyf` installs a no-op stand-in for that one attribute.

Upstream sources
  Pinned by commit in scripts/oracles/upstream/PROVENANCE.md. The script
  clones both repos at those commits into --cache (default <scratch>) when
  --comfyui / --res4lyf are not given. The load-bearing files are also copied
  verbatim under scripts/oracles/upstream/ for review; the clones are needed
  because RES4LYF's sampler imports the whole package and ComfyUI.

Usage
  oracle-venv/bin/python scripts/oracles/gen_scheduler_fixtures.py \
      --comfyui <ComfyUI checkout> --res4lyf <RES4LYF checkout> \
      --out Tests/ZImageTests/Fixtures/Scheduler
"""

from __future__ import annotations

import argparse
import datetime as _dt
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import types
from pathlib import Path

COMFYUI_URL = "https://github.com/comfyanonymous/ComfyUI.git"
COMFYUI_SHA = "783545f689a0af730065994b46b382ae24844c99"
RES4LYF_URL = "https://github.com/ClownsharkBatwing/RES4LYF.git"
RES4LYF_SHA = "26036f647ca15d3048a193daf99a40cecfc3820d"

KREA2_SHIFT = 1.15            # comfy/supported_models.py Krea2.sampling_settings["shift"]
LATENT_SHAPE = (1, 16, 8, 8)  # FDD §5.2
SEED = 4242                   # initial latent / noise; SDE noise seed = SEED + 1 (the node adds 1)

RECIPES = {
    # The published workflow's two ClownsharKSampler_Beta nodes (krea2_simple.json ids 265 / 274).
    "res2s_beta6":  dict(sampler="res_2s",  scheduler="beta",         steps=6, denoise=1.0),
    "deis3m_bong2": dict(sampler="deis_3m", scheduler="bong_tangent", steps=2, denoise=0.2),
}
TIERS = {
    "T1": dict(eta=0.0, bongmath=False),
    "T2": dict(eta=0.5, bongmath=False),
    "T3": dict(eta=0.5, bongmath=True),
}


# --------------------------------------------------------------------------- setup

def ensure_repo(url: str, sha: str, dest: Path) -> Path:
    if not (dest / ".git").exists():
        subprocess.check_call(["git", "clone", "-q", "--filter=blob:none", url, str(dest)])
    head = subprocess.check_output(["git", "-C", str(dest), "rev-parse", "HEAD"]).decode().strip()
    if head != sha:
        subprocess.check_call(["git", "-C", str(dest), "fetch", "-q", "origin", sha])
        subprocess.check_call(["git", "-C", str(dest), "checkout", "-q", sha])
    return dest


def repo_sha(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"]).decode().strip()


def import_res4lyf(path: Path):
    """Import the RES4LYF checkout as a package named RES4LYF regardless of its dir name.

    RES4LYF's res4lyf.py registers aiohttp routes on `server.PromptServer.instance` at
    import time; outside a running ComfyUI that attribute does not exist. A minimal
    stand-in is installed: no-op route decorators, `supports` advertising
    custom_nodes_from_web (so init() skips its JS install), no client. Nothing in the
    sampler path touches it.
    """
    import server as comfy_server

    if not hasattr(comfy_server.PromptServer, "instance"):
        def _noop_route(*_a, **_k):
            return lambda fn: fn
        comfy_server.PromptServer.instance = types.SimpleNamespace(
            routes=types.SimpleNamespace(post=_noop_route, get=_noop_route),
            supports=["custom_nodes_from_web"],
            client_id=None,
            send_sync=lambda *_a, **_k: None,
        )

    spec = importlib.util.spec_from_file_location(
        "RES4LYF", path / "__init__.py", submodule_search_locations=[str(path)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["RES4LYF"] = mod
    spec.loader.exec_module(mod)
    return mod


def scripted_denoised(x, sigma):
    """The denoiser both stacks evaluate (FDD §5.2). sigma: [B] tensor or scalar."""
    import torch
    s = sigma.view(x.shape[0], *([1] * (x.ndim - 1))) if torch.is_tensor(sigma) and sigma.ndim == 1 else sigma
    return 0.5 * torch.tanh(x) + 0.25 * s - 0.1 * x


def build_fake_model(model_sampling):
    """The minimal `model.inner_model.inner_model.model_sampling` chain sample_rk_beta reads.

    RES4LYF's RK_Method_Beta.__init__ checks `hasattr(model, "model")` first, so the
    wrapper must not carry a `.model` attribute.
    """
    import torch

    # `diffusion_model` is probed with hasattr() for test-only overrides (eps_out,
    # y0_standard_guide); an empty namespace answers every probe with "absent".
    inner = types.SimpleNamespace(
        model_sampling=model_sampling, device=torch.device("cpu"),
        diffusion_model=types.SimpleNamespace())
    guider = types.SimpleNamespace(inner_model=inner)

    class FakeDenoiser:
        def __init__(self):
            self.inner_model = guider

        def __call__(self, x, sigma, **kwargs):
            return scripted_denoised(x, sigma)

    return FakeDenoiser()


# --------------------------------------------------------------------------- tracer

class Tracer:
    """Monkeypatches RES4LYF's classes to export one record per step."""

    def __init__(self, rk_method_mod, noise_mod, rk_coeff_mod):
        self.rkm = rk_method_mod
        self.nsm = noise_mod
        self.rkc = rk_coeff_mod
        self.tensors: dict = {}
        self.steps: list = []
        self.cur: dict | None = None
        self.sigmas_run = None
        self._last_noise = None
        self._last_swap = None
        self._orig: dict = {}

    def put(self, key, t):
        import torch
        assert key not in self.tensors, key
        self.tensors[key] = t.detach().clone().to(torch.float32).cpu().contiguous()
        return key

    def __enter__(self):
        rkm, nsm, tr = self.rkm, self.nsm, self
        o = self._orig

        o["prepare_sigmas"] = nsm.RK_NoiseSampler.prepare_sigmas
        def prepare_sigmas(self_, *a, **k):
            sigmas, unsample = o["prepare_sigmas"](self_, *a, **k)
            tr.sigmas_run = [float(v) for v in sigmas]
            return sigmas, unsample
        nsm.RK_NoiseSampler.prepare_sigmas = prepare_sigmas

        o["set_sde_step"] = nsm.RK_NoiseSampler.set_sde_step
        def set_sde_step(self_, sigma, sigma_next, eta, overshoot, s_noise):
            o["set_sde_step"](self_, sigma, sigma_next, eta, overshoot, s_noise)
            tr.cur = dict(
                index=len(tr.steps),
                sigma=float(sigma), sigma_next=float(sigma_next),
                sigma_up_eta=float(self_.sigma_up_eta), sigma_down_eta=float(self_.sigma_down_eta),
                alpha_ratio_eta=float(self_.alpha_ratio_eta),
                h=float(self_.h),
                model_calls=[], substeps=[], bongmath=[],
            )
            tr.steps.append(tr.cur)
        nsm.RK_NoiseSampler.set_sde_step = set_sde_step

        # get_rk_methods_beta swaps the sampler during a multistep warm-up / fallback and
        # logs "step N: <requested> -> <effective>"; RK.rk_type keeps the requested name.
        # Capture the log line so the effective sampler is observed, not re-derived.
        rkc = self.rkc
        o["rkc.RESplain"] = rkc.RESplain
        def resplain(*a, **k):
            msg = " ".join(str(v) for v in a)
            if " -> " in msg and msg.startswith("step "):
                tr._last_swap = msg.split(" -> ", 1)[1].split(" ", 1)[0].split("(", 1)[0].strip()
            return o["rkc.RESplain"](*a, **k)
        rkc.RESplain = resplain

        o["set_coeff"] = rkm.RK_Method_Beta.set_coeff
        def set_coeff(self_, *a, **k):
            tr._last_swap = None
            o["set_coeff"](self_, *a, **k)
            c = tr.cur
            c["rk_type_requested"] = self_.rk_type
            if tr._last_swap is not None:
                c["rk_type"] = tr._last_swap
            elif int(self_.multistep_stages) > 0 and self_.rk_type.startswith("deis"):
                c["rk_type"] = "deis"
            else:
                c["rk_type"] = self_.rk_type
            c["exponential"] = bool(self_.EXPONENTIAL)
            c["rows"] = int(self_.rows)
            c["row_offset"] = int(self_.row_offset)
            c["multistep_stages"] = int(self_.multistep_stages)
            c["a_matrix"] = [[float(v) for v in row] for row in self_.A]
            c["b_weights"] = [[float(v) for v in row] for row in self_.B]
            c["c_nodes"] = [float(v) for v in self_.C]
        rkm.RK_Method_Beta.set_coeff = set_coeff

        o["set_substep_list"] = nsm.RK_NoiseSampler.set_substep_list
        def set_substep_list(self_, RK):
            o["set_substep_list"](self_, RK)
            tr.cur["substep_sigmas"] = [float(v) for v in self_.s_]
        nsm.RK_NoiseSampler.set_substep_list = set_substep_list

        o["set_sde_substep"] = nsm.RK_NoiseSampler.set_sde_substep
        def set_sde_substep(self_, row, multistep_stages, *a, **k):
            o["set_sde_substep"](self_, row, multistep_stages, *a, **k)
            tr.cur["substeps"].append(dict(
                row=int(row),
                sub_sigma=float(self_.sub_sigma), sub_sigma_next=float(self_.sub_sigma_next),
                sub_sigma_up_eta=float(self_.sub_sigma_up_eta), sub_sigma_down_eta=float(self_.sub_sigma_down_eta),
                sub_alpha_ratio_eta=float(self_.sub_alpha_ratio_eta),
                h_new=float(self_.h_new),
                x0=None, x_pre=None, noise=None, x_post=None,
            ))
        nsm.RK_NoiseSampler.set_sde_substep = set_sde_substep

        for cls_name in ("RK_Method_Exponential", "RK_Method_Linear"):
            cls = getattr(rkm, cls_name)
            o[f"{cls_name}.__call__"] = cls.__call__
            def make_call(orig):
                def __call__(self_, x, sub_sigma, x_0=None, sigma=None, transformer_options=None):
                    eps, denoised = orig(self_, x, sub_sigma, x_0, sigma, transformer_options)
                    c = tr.cur
                    i = len(c["model_calls"])
                    p = f"step{c['index']:02d}/call{i}"
                    c["model_calls"].append(dict(
                        row=i,
                        s_tmp=float(sub_sigma), sigma=float(sigma if sigma is not None else sub_sigma),
                        x_in=tr.put(f"{p}/x_in", x),
                        x0=tr.put(f"{p}/x_0", x_0 if x_0 is not None else x),
                        denoised=tr.put(f"{p}/denoised", denoised),
                        eps=tr.put(f"{p}/eps", eps),
                    ))
                    return eps, denoised
                return __call__
            cls.__call__ = make_call(o[f"{cls_name}.__call__"])

        o["normalize_zscore"] = nsm.normalize_zscore
        def normalize_zscore(*a, **k):
            out = o["normalize_zscore"](*a, **k)
            tr._last_noise = out.detach().clone()
            return out
        nsm.normalize_zscore = normalize_zscore

        o["swap_noise_substep"] = nsm.RK_NoiseSampler.swap_noise_substep
        def swap_noise_substep(self_, x_0, x_next, *a, **k):
            tr._last_noise = None
            out = o["swap_noise_substep"](self_, x_0, x_next, *a, **k)
            if tr._last_noise is not None:
                c = tr.cur
                sub = c["substeps"][-1]
                p = f"step{c['index']:02d}/sub{sub['row']}"
                # x_0 as swap_noise_substep saw it — in T3 bong_iter re-derives x_0 after
                # this point, so the step-level x_0 is not the one this re-noise used.
                sub["x0"] = tr.put(f"{p}/x_0", x_0)
                sub["x_pre"] = tr.put(f"{p}/x_pre", x_next)
                sub["noise"] = tr.put(f"{p}/noise", tr._last_noise)
                sub["x_post"] = tr.put(f"{p}/x_post", out)
            return out
        nsm.RK_NoiseSampler.swap_noise_substep = swap_noise_substep

        o["rebound_overshoot_step"] = nsm.RK_NoiseSampler.rebound_overshoot_step
        def rebound_overshoot_step(self_, x_0, x):
            out = o["rebound_overshoot_step"](self_, x_0, x)
            c = tr.cur
            p = f"step{c['index']:02d}"
            c["x0"] = tr.put(f"{p}/x_0", x_0)
            c["x_next"] = tr.put(f"{p}/x_next", out)
            c["noise_step"] = None
            c["x_out"] = c["x_next"]
            return out
        nsm.RK_NoiseSampler.rebound_overshoot_step = rebound_overshoot_step

        o["swap_noise_step"] = nsm.RK_NoiseSampler.swap_noise_step
        def swap_noise_step(self_, x_0, x_next, *a, **k):
            tr._last_noise = None
            out = o["swap_noise_step"](self_, x_0, x_next, *a, **k)
            if tr._last_noise is not None:
                c = tr.cur
                p = f"step{c['index']:02d}"
                c["noise_step"] = tr.put(f"{p}/noise_step", tr._last_noise)
                c["x_out"] = tr.put(f"{p}/x_out", out)
            return out
        nsm.RK_NoiseSampler.swap_noise_step = swap_noise_step

        o["bong_iter"] = rkm.RK_Method_Beta.bong_iter
        def bong_iter(self_, x_0, x_, eps_, eps_prev_, data_, sigma, s_, row, row_offset, h, *a, **k):
            x_0o, x_o, eps_o = o["bong_iter"](self_, x_0, x_, eps_, eps_prev_, data_, sigma, s_, row, row_offset, h, *a, **k)
            c = tr.cur
            i = len(c["bongmath"])
            p = f"step{c['index']:02d}/bong{i}"
            n = int(self_.rows) + 1
            c["bongmath"].append(dict(
                row=int(row),
                x0=tr.put(f"{p}/x_0", x_0o),
                x_rows=[tr.put(f"{p}/x_row{r}", x_o[r]) for r in range(n)],
                eps_rows=[tr.put(f"{p}/eps_row{r}", eps_o[r]) for r in range(n)],
            ))
            return x_0o, x_o, eps_o
        rkm.RK_Method_Beta.bong_iter = bong_iter
        return self

    def __exit__(self, *exc):
        rkm, nsm, o = self.rkm, self.nsm, self._orig
        self.rkc.RESplain = o["rkc.RESplain"]
        nsm.RK_NoiseSampler.prepare_sigmas = o["prepare_sigmas"]
        nsm.RK_NoiseSampler.set_sde_step = o["set_sde_step"]
        rkm.RK_Method_Beta.set_coeff = o["set_coeff"]
        nsm.RK_NoiseSampler.set_substep_list = o["set_substep_list"]
        nsm.RK_NoiseSampler.set_sde_substep = o["set_sde_substep"]
        rkm.RK_Method_Exponential.__call__ = o["RK_Method_Exponential.__call__"]
        rkm.RK_Method_Linear.__call__ = o["RK_Method_Linear.__call__"]
        nsm.normalize_zscore = o["normalize_zscore"]
        nsm.RK_NoiseSampler.swap_noise_substep = o["swap_noise_substep"]
        nsm.RK_NoiseSampler.rebound_overshoot_step = o["rebound_overshoot_step"]
        nsm.RK_NoiseSampler.swap_noise_step = o["swap_noise_step"]
        rkm.RK_Method_Beta.bong_iter = o["bong_iter"]
        return False


# --------------------------------------------------------------------------- fixtures

def make_provenance(comfy_dir: Path, r4_dir: Path) -> dict:
    import numpy, scipy, torch, mpmath
    return dict(
        generator="scripts/oracles/gen_scheduler_fixtures.py",
        generated_at=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        comfyui_url=COMFYUI_URL, comfyui_sha=repo_sha(comfy_dir),
        res4lyf_url=RES4LYF_URL, res4lyf_sha=repo_sha(r4_dir),
        python=platform.python_version(), torch=torch.__version__, numpy=numpy.__version__,
        scipy=scipy.__version__, mpmath=mpmath.__version__,
        note="ComfyUI/RES4LYF are validation oracles only (FDD-krea2-raw-recipe §5.2); never a ComfyBox backend.",
    )


def gen_sigma_fixture(prov, ms_flux, ms_df, r4_sigmas, comfy_samplers, fake_model_flux) -> dict:
    import torch

    def dump(fn):
        return [float(v) for v in fn]

    betas = {}
    for alpha, beta, key in ((0.6, 0.6, "beta"), (0.5, 0.7, "beta57")):
        betas[key] = {}
        for ms_key, ms in (("flux", ms_flux), ("discrete_flow", ms_df)):
            betas[key][ms_key] = {
                str(n): dump(comfy_samplers.beta_scheduler(ms, n, alpha=alpha, beta=beta))
                for n in (6, 9, 30)
            }

    bong = {}
    shift_free = True
    for n in (2, 6, 8, 9, 10, 12, 20):
        a = dump(r4_sigmas.bong_tangent_scheduler(ms_flux, n))
        b = dump(r4_sigmas.bong_tangent_scheduler(ms_df, n))
        shift_free = shift_free and (a == b)
        bong[str(n)] = a

    # The node path for stage 2: get_sigmas(model, "bong_tangent", steps=2, denoise=0.2)
    # → total_steps = int(2/0.2) = 10, then the last steps+1 entries.
    stage2 = dump(r4_sigmas.get_sigmas(fake_model_flux, "bong_tangent", 2, 0.2))

    def ms_block(ms, cls):
        return dict(klass=cls, shift=KREA2_SHIFT, table_size=int(ms.sigmas.numel()),
                    sigma_min=float(ms.sigma_min), sigma_max=float(ms.sigma_max))

    return dict(
        provenance=prov,
        comfy_krea2_model_sampling="ModelSamplingFlux",
        comfy_krea2_model_sampling_note=(
            "comfy/supported_models.py Krea2 (sampling_settings shift 1.15, multiplier 1.0) → "
            "model_base.Krea2(model_type=ModelType.FLUX) → model_sampling(): CONST + ModelSamplingFlux "
            "(10000-entry table, sigma(t) = e^shift / (e^shift + 1/t − 1)). The FDD's D3/D5/AC-21 "
            "assumed ModelSamplingDiscreteFlow(shift=1.15); both grids are recorded here, keyed."),
        model_samplings=dict(
            flux=ms_block(ms_flux, "ModelSamplingFlux+CONST"),
            discrete_flow=ms_block(ms_df, "ModelSamplingDiscreteFlow+CONST"),
        ),
        beta=betas["beta"],
        beta57=betas["beta57"],
        beta_params=dict(beta=[0.6, 0.6], beta57=[0.5, 0.7]),
        bong_tangent=bong,
        bong_tangent_shift_free=bool(shift_free),
        bong_tangent_source="RES4LYF sigmas.py bong_tangent_scheduler (model_sampling unused)",
        stage2_bong_tangent_denoise=dict(steps=2, denoise=0.2, total_steps=10, sigmas=stage2,
                                         note="RES4LYF sigmas.get_sigmas(): total_steps=int(steps/denoise), then sigmas[-(steps+1):]"),
    )


def prepare_like_res4lyf(sigmas: list[float], sigma_min: float) -> list[float]:
    """RK_NoiseSampler.prepare_sigmas for sampler_mode='standard': dedupe consecutive, then
    insert sigma_min before the trailing 0 (or raise sigmas[-2] to it)."""
    out = [sigmas[0]] + [s for p, s in zip(sigmas, sigmas[1:]) if s != p]
    if out[-1] == 0:
        if out[-2] < sigma_min:
            out[-2] = sigma_min
        elif abs(out[-2] - sigma_min) > 1e-4:
            out = out[:-1] + [sigma_min, 0.0]
    return out


def gen_deis_fixture(prov, deis_mod, sigma_cases: dict) -> dict:
    import torch
    cases = []
    for name, sigmas in sigma_cases.items():
        t = torch.tensor(sigmas, dtype=torch.float64)
        for order in (2, 3, 4):
            coeffs = deis_mod.get_deis_coeff_list(t, order, deis_mode="rhoab")
            cases.append(dict(
                name=f"{name}/order{order}", max_order=order, sigmas=[float(v) for v in t],
                coeffs=[[float(v) for v in row] for row in coeffs],
            ))
    return dict(
        provenance=prov,
        deis_mode="rhoab",
        source="RES4LYF beta/deis_coefficients.py",
        ramp="order = min(i+1, max_order); order 1 → []  (RES4LYF's corrected ramp; ComfyUI's comfy/k_diffusion/deis.py uses min(i, max_order))",
        usage=("rk_coefficients_beta.py get_rk_methods_beta('deis'): coeff_list[step] / h, with h = NS.h = "
               "sigma_down − sigma (linear frame); b *= (sigma_down − sigma)/(sigma_next − sigma) (=1 at overshoot 0). "
               "Steps with step < order + 1 run ralston_{order}s instead (multistep_extra_initial_steps = 1)."),
        cases=cases,
    )


def run_trace(prov, r4, comfy_samplers, ms, recipe_name, recipe, tier_name, tier) -> tuple[dict, dict]:
    import torch
    from safetensors.torch import save_file  # noqa: F401  (presence check)

    rks = r4.beta.rk_sampler_beta
    rkm = r4.beta.rk_method_beta
    nsm = r4.beta.rk_noise_sampler_beta
    rkc = r4.beta.rk_coefficients_beta
    r4_sigmas = r4.sigmas

    model = build_fake_model(ms)

    # Sigmas exactly as SharkSampler.main builds them (default_dtype float64).
    sigmas = r4_sigmas.get_sigmas(model, recipe["scheduler"], recipe["steps"], abs(recipe["denoise"])).to(torch.float64)
    sigmas_schedule = [float(v) for v in sigmas]

    # Initial latent: stage 1 starts from an empty latent; stage 2 from a seeded
    # stand-in for a stage-1 output. Noise is seeded and exported; x is comfy's
    # CONST.noise_scaling(sigmas[0], noise, latent).
    g = torch.Generator().manual_seed(SEED)
    noise = torch.randn(LATENT_SHAPE, generator=g, dtype=torch.float32)
    if recipe["denoise"] >= 1.0:
        latent = torch.zeros(LATENT_SHAPE, dtype=torch.float32)
    else:
        latent = 0.5 * torch.randn(LATENT_SHAPE, generator=g, dtype=torch.float32)
    x = ms.noise_scaling(sigmas[0], noise, latent)
    assert x.dtype == torch.float32

    eta = tier["eta"]
    with Tracer(rkm, nsm, rkc) as tr:
        x_final = rks.sample_rk_beta(
            model, x.clone(), sigmas.clone(),
            extra_args={"model_options": {"transformer_options": {}}},
            callback=None, disable=True,
            sampler_mode="standard",
            rk_type=recipe["sampler"], implicit_sampler_name="use_explicit",
            c1=0.0, c2=0.5, c3=1.0,
            noise_sampler_type="gaussian", noise_sampler_type_substep="gaussian",
            noise_mode_sde="hard", noise_mode_sde_substep="hard",
            eta=eta, eta_substep=eta,                      # ClownsharKSampler: eta_substep = eta
            s_noise=1.0, s_noise_substep=1.0,
            noise_anchor=1.0, noise_boost_normalize=True,
            overshoot_mode="hard", overshoot_mode_substep="hard", overshoot=0.0, overshoot_substep=0.0,
            implicit_type="bongmath", implicit_type_substeps="bongmath",
            implicit_steps_diag=0, implicit_steps_full=0,
            LGW_MASK_RESCALE_MIN=True, guides=None,
            noise_seed=SEED + 1,                           # SharkSampler: clown noise seed = seed + 1
            cfgpp=0.0, cfg_cw=1.0,
            BONGMATH=tier["bongmath"],
            rk_swaps=[], steps_to_run=-1,
            extra_options="",
        )

    steps = tr.steps
    assert tr.sigmas_run is not None
    assert len(steps) == len(tr.sigmas_run) - 2, (len(steps), tr.sigmas_run)

    # Final tail: x_final = x_out − σ_min · eps_last (model_sampling.calculate_denoised).
    last = steps[-1]
    x0 = tr.tensors[last["x0"]]
    xn = tr.tensors[last["x_next"]]
    eps_last = (x0 - xn) / (last["sigma"] - last["sigma_next"])
    x_out = tr.tensors[last["x_out"]]
    tail = x_out - float(ms.sigma_min) * eps_last
    tail_err = float((tail - x_final.to(torch.float32)).abs().max())
    assert tail_err < 1e-5, tail_err

    tensors = dict(tr.tensors)
    tensors["latent_image"] = latent.clone()
    tensors["noise_init"] = noise.clone()
    tensors["x_init"] = x.clone()
    tensors["final/x"] = x_final.to(torch.float32).clone()
    tensors["final/eps_last"] = eps_last.to(torch.float32).clone()

    manifest = dict(
        provenance=prov,
        recipe=dict(
            tier=tier_name, sampler=recipe["sampler"], scheduler=recipe["scheduler"],
            steps=recipe["steps"], denoise=recipe["denoise"],
            eta=eta, eta_substep=eta, bongmath=tier["bongmath"],
            noise_mode_sde="hard", s_noise=1.0, noise_anchor=1.0,
            shift=KREA2_SHIFT, model_sampling="ModelSamplingFlux",
            seed=SEED, noise_seed_sde=SEED + 1,
            node="ClownsharKSampler_Beta (RES4LYF beta/samplers.py), defaults as the node passes them",
        ),
        denoiser="0.5*tanh(x) + 0.25*sigma - 0.1*x",
        latent_shape=list(LATENT_SHAPE),
        sigmas_schedule=sigmas_schedule,
        sigmas_run=tr.sigmas_run,
        sigma_min=float(ms.sigma_min), sigma_max=float(ms.sigma_max),
        model_calls_total=sum(len(s["model_calls"]) for s in steps),
        latent_image="latent_image", noise_init="noise_init", x_init="x_init",
        steps=steps,
        final=dict(x="final/x", eps_last="final/eps_last", linear_tail_from_sigma_min=True),
    )
    return manifest, tensors


# --------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).resolve()
    repo = here.parents[2]
    ap.add_argument("--out", type=Path, default=repo / "Tests/ZImageTests/Fixtures/Scheduler")
    ap.add_argument("--cache", type=Path, default=Path(os.environ.get("ORACLE_CACHE", str(repo / ".build/oracles"))))
    ap.add_argument("--comfyui", type=Path, default=None)
    ap.add_argument("--res4lyf", type=Path, default=None)
    ap.add_argument("--only", choices=["sigmas", "deis", "traces"], default=None)
    args = ap.parse_args()

    comfy_dir = args.comfyui or ensure_repo(COMFYUI_URL, COMFYUI_SHA, args.cache / "ComfyUI")
    r4_dir = args.res4lyf or ensure_repo(RES4LYF_URL, RES4LYF_SHA, args.cache / "RES4LYF")
    for d, sha in ((comfy_dir, COMFYUI_SHA), (r4_dir, RES4LYF_SHA)):
        got = repo_sha(d)
        if got != sha:
            print(f"refusing: {d} is at {got}, pinned {sha}", file=sys.stderr)
            return 2

    sys.path.insert(0, str(comfy_dir))
    import torch
    torch.set_grad_enabled(False)
    import comfy.samplers
    import comfy.model_sampling as cms
    r4 = import_res4lyf(r4_dir)
    import RES4LYF.beta.rk_sampler_beta  # noqa: F401  — registers nothing, but forces the import chain
    import RES4LYF.beta.deis_coefficients as deis_mod
    assert "bong_tangent" in comfy.samplers.SCHEDULER_HANDLERS, "RES4LYF __init__ did not register bong_tangent"

    class FluxMS(cms.ModelSamplingFlux, cms.CONST):
        pass

    class DiscreteFlowMS(cms.ModelSamplingDiscreteFlow, cms.CONST):
        pass

    ms_flux = FluxMS()
    ms_flux.set_parameters(shift=KREA2_SHIFT)
    ms_df = DiscreteFlowMS()
    ms_df.set_parameters(shift=KREA2_SHIFT, multiplier=1.0)

    prov = make_provenance(comfy_dir, r4_dir)
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    def write_json(name, obj):
        with open(out / name, "w") as f:
            json.dump(obj, f, indent=1)
            f.write("\n")
        print(f"wrote {out / name}")

    if args.only in (None, "sigmas"):
        fx = gen_sigma_fixture(prov, ms_flux, ms_df, r4.sigmas, comfy.samplers, build_fake_model(ms_flux))
        write_json("comfy_sigmas.json", fx)

    traces = {}
    if args.only in (None, "traces", "deis"):
        from safetensors.torch import save_file
        for recipe_name, recipe in RECIPES.items():
            for tier_name, tier in TIERS.items():
                manifest, tensors = run_trace(prov, r4, comfy.samplers, ms_flux, recipe_name, recipe, tier_name, tier)
                traces[(recipe_name, tier_name)] = manifest
                if args.only in (None, "traces"):
                    stem = f"res4lyf_trace_{recipe_name}_{tier_name}"
                    write_json(f"{stem}.json", manifest)
                    save_file(tensors, str(out / f"{stem}.safetensors"))
                    print(f"wrote {out / stem}.safetensors  ({len(tensors)} tensors, "
                          f"{manifest['model_calls_total']} model calls, {len(manifest['steps'])} steps)")

    if args.only in (None, "deis"):
        sigma_cases = {
            "res2s_beta6_run": traces[("res2s_beta6", "T1")]["sigmas_run"],
            "deis3m_bong2_run": traces[("deis3m_bong2", "T1")]["sigmas_run"],
            "bong_tangent8_prepared": prepare_like_res4lyf(
                [float(v) for v in r4.sigmas.bong_tangent_scheduler(ms_flux, 8)], float(ms_flux.sigma_min)),
            "beta9_flux_prepared": prepare_like_res4lyf(
                [float(v) for v in comfy.samplers.beta_scheduler(ms_flux, 9)], float(ms_flux.sigma_min)),
        }
        write_json("res4lyf_deis_coeffs.json", gen_deis_fixture(prov, deis_mod, sigma_cases))

    return 0


if __name__ == "__main__":
    sys.exit(main())
