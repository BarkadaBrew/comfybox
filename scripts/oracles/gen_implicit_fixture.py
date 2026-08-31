#!/usr/bin/env python3
"""
gen_implicit_fixture.py — oracle fixture for the implicit-RK refinement port.

VALIDATION TOOL ONLY (sibling of gen_scheduler_fixtures.py; same oracle
discipline — ComfyUI/RES4LYF are oracles, never a ComfyBox backend). It emits a
single trace fixture exercising RES4LYF's `full_iter` re-iteration loop:

  res4lyf_trace_implicit_heun2s.json + .safetensors

    RES4LYF `sample_rk_beta` with `rk_type="heun_2s"`, `implicit_steps_full=1`,
    `implicit_steps_diag=0`, eta 0 (tier T1, no SDE noise), bongmath off, guides
    off, on a `beta` grid of 4 steps against the scripted denoiser
    `0.5*tanh(x) + 0.25*sigma - 0.1*x` on a 1x16x8x8 latent.

    Captured PER STEP and PER full_iter pass: the model calls (x_in, the sigma
    the model was evaluated at, denoised, the x_0/sigma-anchored eps) and the
    pass's resulting x_next — so the Swift port can be checked pass-by-pass.

Run with a torch-capable interpreter (ComfyUI venv), same as the harness:

  <venv>/bin/python scripts/oracles/gen_implicit_fixture.py \
      --comfyui <ComfyUI checkout> --res4lyf <RES4LYF checkout> \
      --out Tests/ZImageTests/Fixtures/Scheduler
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

# Reuse the harness helpers verbatim (import the sibling script as a module).
_HERE = Path(__file__).resolve()
_HARNESS = _HERE.parent / "gen_scheduler_fixtures.py"
_spec = importlib.util.spec_from_file_location("gen_scheduler_fixtures", _HARNESS)
H = importlib.util.module_from_spec(_spec)
sys.modules["gen_scheduler_fixtures"] = H
_spec.loader.exec_module(H)


RECIPE = dict(sampler="heun_2s", scheduler="beta", steps=4, denoise=1.0)
# T1: eta 0 (no SDE noise), bongmath off, guides off.
TIER = dict(eta=0.0, bongmath=False)
IMPLICIT_STEPS_DIAG = 0

# Two fixtures: the plain explicit reference (full=0, one pass/step — the
# byte-identical-when-off baseline) and the implicit refinement (full=1).
VARIANTS = {
    "explicit_heun2s": 0,
    "implicit_heun2s": 1,
}


class ImplicitTracer:
    """Minimal tracer for the guides-off / eta-0 / bongmath-off implicit path.

    Records one record per solver step, and inside it one record per full_iter
    pass: the model calls made during that pass and the x_next it produced.
    """

    def __init__(self, rk_method_mod, noise_mod):
        self.rkm = rk_method_mod
        self.nsm = noise_mod
        self.tensors: dict = {}
        self.steps: list = []
        self.cur: dict | None = None
        self.sigmas_run = None
        self._orig: dict = {}

    def put(self, key, t):
        import torch
        assert key not in self.tensors, key
        self.tensors[key] = t.detach().clone().to(torch.float32).cpu().contiguous()
        return key

    def _cur_pass(self):
        # The pass currently executing is the one not yet closed by a
        # rebound_overshoot_step: index == number of already-closed passes.
        return len(self.cur["full_iters"])

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
                h=float(self_.h),
                full_iters=[],          # one entry per completed pass (rebound_overshoot_step)
                _pass_calls={},         # pass_index -> list of model-call records
            )
            tr.steps.append(tr.cur)
        nsm.RK_NoiseSampler.set_sde_step = set_sde_step

        # Record every model call (RK_Method_Linear.__call__ for heun_2s).
        cls = rkm.RK_Method_Linear
        o["Linear.__call__"] = cls.__call__
        def make_call(orig):
            def __call__(self_, x, sub_sigma, x_0=None, sigma=None, transformer_options=None):
                eps, denoised = orig(self_, x, sub_sigma, x_0, sigma, transformer_options)
                c = tr.cur
                p = tr._cur_pass()
                calls = c["_pass_calls"].setdefault(p, [])
                i = len(calls)
                tag = f"step{c['index']:02d}/pass{p}/call{i}"
                calls.append(dict(
                    row=i,
                    s_eval=float(sub_sigma),
                    sigma_anchor=float(sigma if sigma is not None else sub_sigma),
                    x_in=tr.put(f"{tag}/x_in", x),
                    x0=tr.put(f"{tag}/x_0", x_0 if x_0 is not None else x),
                    denoised=tr.put(f"{tag}/denoised", denoised),
                    eps=tr.put(f"{tag}/eps", eps),
                ))
                return eps, denoised
            return __call__
        cls.__call__ = make_call(o["Linear.__call__"])

        # Close each full_iter pass and capture its x_next.
        o["rebound_overshoot_step"] = nsm.RK_NoiseSampler.rebound_overshoot_step
        def rebound_overshoot_step(self_, x_0, x):
            out = o["rebound_overshoot_step"](self_, x_0, x)
            c = tr.cur
            p = len(c["full_iters"])
            tag = f"step{c['index']:02d}/pass{p}"
            c["full_iters"].append(dict(
                pass_index=p,
                model_calls=c["_pass_calls"].get(p, []),
                x0=tr.put(f"{tag}/x_0", x_0),
                x_next=tr.put(f"{tag}/x_next", out),
            ))
            return out
        nsm.RK_NoiseSampler.rebound_overshoot_step = rebound_overshoot_step
        return self

    def __exit__(self, *exc):
        nsm, o = self.nsm, self._orig
        nsm.RK_NoiseSampler.prepare_sigmas = o["prepare_sigmas"]
        nsm.RK_NoiseSampler.set_sde_step = o["set_sde_step"]
        self.rkm.RK_Method_Linear.__call__ = o["Linear.__call__"]
        nsm.RK_NoiseSampler.rebound_overshoot_step = o["rebound_overshoot_step"]
        for s in self.steps:
            s.pop("_pass_calls", None)
        return False


def run_implicit_trace(prov, r4, ms, implicit_steps_full):
    import torch

    rks = r4.beta.rk_sampler_beta
    rkm = r4.beta.rk_method_beta
    nsm = r4.beta.rk_noise_sampler_beta
    r4_sigmas = r4.sigmas

    model = H.build_fake_model(ms)

    sigmas = r4_sigmas.get_sigmas(model, RECIPE["scheduler"], RECIPE["steps"], abs(RECIPE["denoise"])).to(torch.float64)
    sigmas_schedule = [float(v) for v in sigmas]

    g = torch.Generator().manual_seed(H.SEED)
    noise = torch.randn(H.LATENT_SHAPE, generator=g, dtype=torch.float32)
    latent = torch.zeros(H.LATENT_SHAPE, dtype=torch.float32)
    x = ms.noise_scaling(sigmas[0], noise, latent)
    assert x.dtype == torch.float32

    eta = TIER["eta"]
    with ImplicitTracer(rkm, nsm) as tr:
        x_final = rks.sample_rk_beta(
            model, x.clone(), sigmas.clone(),
            extra_args={"model_options": {"transformer_options": {}}},
            callback=None, disable=True,
            sampler_mode="standard",
            rk_type=RECIPE["sampler"], implicit_sampler_name="use_explicit",
            c1=0.0, c2=0.5, c3=1.0,
            noise_sampler_type="gaussian", noise_sampler_type_substep="gaussian",
            noise_mode_sde="hard", noise_mode_sde_substep="hard",
            eta=eta, eta_substep=eta,
            s_noise=1.0, s_noise_substep=1.0,
            noise_anchor=1.0, noise_boost_normalize=True,
            overshoot_mode="hard", overshoot_mode_substep="hard", overshoot=0.0, overshoot_substep=0.0,
            implicit_type="bongmath", implicit_type_substeps="bongmath",
            implicit_steps_diag=IMPLICIT_STEPS_DIAG, implicit_steps_full=implicit_steps_full,
            LGW_MASK_RESCALE_MIN=True, guides=None,
            noise_seed=H.SEED + 1,
            cfgpp=0.0, cfg_cw=1.0,
            BONGMATH=TIER["bongmath"],
            rk_swaps=[], steps_to_run=-1,
            extra_options="",
        )

    steps = tr.steps
    assert tr.sigmas_run is not None
    assert len(steps) == len(tr.sigmas_run) - 2, (len(steps), tr.sigmas_run)
    for s in steps:
        assert len(s["full_iters"]) == implicit_steps_full + 1, (s["index"], len(s["full_iters"]))

    # Model-free tail: x_final = x_out - sigma_min * eps_last (last pass eps).
    last = steps[-1]["full_iters"][-1]
    x0 = tr.tensors[last["x0"]]
    xn = tr.tensors[last["x_next"]]
    eps_last = (x0 - xn) / (steps[-1]["sigma"] - steps[-1]["sigma_next"])
    tail = xn - float(ms.sigma_min) * eps_last
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
            tier="T1", sampler=RECIPE["sampler"], scheduler=RECIPE["scheduler"],
            steps=RECIPE["steps"], denoise=RECIPE["denoise"],
            eta=eta, eta_substep=eta, bongmath=TIER["bongmath"],
            implicit_steps_full=implicit_steps_full, implicit_steps_diag=IMPLICIT_STEPS_DIAG,
            shift=H.KREA2_SHIFT, model_sampling="ModelSamplingFlux",
            seed=H.SEED, noise_seed_sde=H.SEED + 1,
            note=("RES4LYF sample_rk_beta full_iter loop; explicit heun_2s re-iterated "
                  "implicit_steps_full+1 times as a fixed point. row_offset=1, so pass>0 "
                  "re-anchors row 0 on the previous pass's x_next at sigma_next."),
        ),
        denoiser="0.5*tanh(x) + 0.25*sigma - 0.1*x",
        latent_shape=list(H.LATENT_SHAPE),
        sigmas_schedule=sigmas_schedule,
        sigmas_run=tr.sigmas_run,
        sigma_min=float(ms.sigma_min), sigma_max=float(ms.sigma_max),
        latent_image="latent_image", noise_init="noise_init", x_init="x_init",
        steps=steps,
        final=dict(x="final/x", eps_last="final/eps_last", linear_tail_from_sigma_min=True),
    )
    return manifest, tensors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    repo = _HERE.parents[2]
    ap.add_argument("--out", type=Path, default=repo / "Tests/ZImageTests/Fixtures/Scheduler")
    ap.add_argument("--cache", type=Path, default=Path(os.environ.get("ORACLE_CACHE", str(repo / ".build/oracles"))))
    ap.add_argument("--comfyui", type=Path, default=None)
    ap.add_argument("--res4lyf", type=Path, default=None)
    args = ap.parse_args()

    comfy_dir = args.comfyui or H.ensure_repo(H.COMFYUI_URL, H.COMFYUI_SHA, args.cache / "ComfyUI")
    r4_dir = args.res4lyf or H.ensure_repo(H.RES4LYF_URL, H.RES4LYF_SHA, args.cache / "RES4LYF")
    for d, sha in ((comfy_dir, H.COMFYUI_SHA), (r4_dir, H.RES4LYF_SHA)):
        got = H.repo_sha(d)
        if got != sha:
            print(f"refusing: {d} is at {got}, pinned {sha}", file=sys.stderr)
            return 2

    sys.path.insert(0, str(comfy_dir))
    import torch
    torch.set_grad_enabled(False)
    import comfy.samplers  # noqa: F401
    import comfy.model_sampling as cms
    r4 = H.import_res4lyf(r4_dir)
    import RES4LYF.beta.rk_sampler_beta  # noqa: F401

    class FluxMS(cms.ModelSamplingFlux, cms.CONST):
        pass

    ms_flux = FluxMS()
    ms_flux.set_parameters(shift=H.KREA2_SHIFT)

    prov = H.make_provenance(comfy_dir, r4_dir)
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    from safetensors.torch import save_file
    for label, full in VARIANTS.items():
        manifest, tensors = run_implicit_trace(prov, r4, ms_flux, full)
        stem = f"res4lyf_trace_{label}"
        with open(out / f"{stem}.json", "w") as f:
            json.dump(manifest, f, indent=1)
            f.write("\n")
        save_file(tensors, str(out / f"{stem}.safetensors"))
        total_calls = sum(len(fi["model_calls"]) for s in manifest["steps"] for fi in s["full_iters"])
        print(f"wrote {out / stem}.json")
        print(f"wrote {out / stem}.safetensors  ({len(tensors)} tensors, "
              f"{total_calls} model calls, {len(manifest['steps'])} steps, "
              f"{full + 1} passes/step)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
