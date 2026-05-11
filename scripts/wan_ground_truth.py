#!/usr/bin/env python3
"""Run Wan 2.2 I2V model and dump ground truth values for comparison with Swift.

This script loads the actual model weights and runs a single forward pass,
printing intermediate values at each pipeline stage.

Usage: python3 scripts/wan_ground_truth.py --model-dir /path/to/wan22-i2v --image /path/to/image.png

Requirements: torch, torchvision, PIL, diffusers, tqdm, easydict
"""
import argparse
import math
import sys
import os

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image


def setup_wan_path(model_dir):
    """Add the Wan2.2 source to sys.path."""
    wan_dir = os.path.expanduser("~/Projects/Wan2.2")
    if os.path.exists(wan_dir):
        sys.path.insert(0, wan_dir)
    else:
        print(f"WARNING: Wan2.2 source not found at {wan_dir}", file=sys.stderr)


def sinusoidal_embedding_1d(dim, position):
    """Reference sinusoidal embedding (float64, matches Wan model.py)."""
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


def dump_stats(name, tensor):
    """Print tensor statistics."""
    t = tensor.float()
    print(f"[GROUND] {name}: shape={list(t.shape)}, dtype={tensor.dtype}, "
          f"mean={t.mean().item():.8f}, std={t.std().item():.8f}, "
          f"min={t.min().item():.8f}, max={t.max().item():.8f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, help="Path to Wan 2.2 I2V model dir")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--steps", type=int, default=5, help="Number of denoising steps")
    parser.add_argument("--frames", type=int, default=5, help="Number of frames")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--shift", type=float, default=5.0, help="Noise shift")
    parser.add_argument("--guidance", type=float, default=3.5, help="CFG scale")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("mps")  # Mac GPU

    setup_wan_path(args.model_dir)

    # Import Wan modules
    from wan.modules.model import WanModel
    from wan.modules.t5 import T5EncoderModel
    from wan.modules.vae2_1 import Wan2_1_VAE
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

    # Config
    vae_stride = (4, 8, 8)
    patch_size = (1, 2, 2)
    num_train_timesteps = 1000
    boundary = 0.9 * num_train_timesteps
    max_area = 720 * 1280
    text_len = 512
    freq_dim = 256
    model_dim = 5120

    F = args.frames
    prompt = "a woman smiling"
    neg_prompt = ""  # Empty for simplicity

    # Load image
    img = Image.open(args.image).convert("RGB")
    img_tensor = TF.to_tensor(img).sub_(0.5).div_(0.5).to(device)

    # Compute resolution
    h, w = img_tensor.shape[1:]
    aspect_ratio = h / w
    lat_h = round(
        np.sqrt(max_area * aspect_ratio) // vae_stride[1] //
        patch_size[1] * patch_size[1])
    lat_w = round(
        np.sqrt(max_area / aspect_ratio) // vae_stride[2] //
        patch_size[2] * patch_size[2])
    h_out = lat_h * vae_stride[1]
    w_out = lat_w * vae_stride[2]

    max_seq_len = ((F - 1) // vae_stride[0] + 1) * lat_h * lat_w // (
        patch_size[1] * patch_size[2])

    print(f"Resolution: {h_out}x{w_out} (latent: {lat_h}x{lat_w})")
    print(f"Frames: {F}, max_seq_len: {max_seq_len}")
    print(f"Steps: {args.steps}, shift: {args.shift}, guidance: {args.guidance}")
    print()

    # Load T5
    print("Loading T5...")
    t5 = T5EncoderModel(
        text_len=text_len,
        dtype=torch.bfloat16,
        device=torch.device('cpu'),
        checkpoint_path=os.path.join(args.model_dir, "models_t5_umt5-xxl-enc-bf16.safetensors"),
        tokenizer_path=os.path.join(args.model_dir, "google/umt5-xxl"),
    )

    # Encode text
    t5.model.to(device)
    context = t5([prompt], device)
    context_null = t5([neg_prompt], device)
    t5.model.cpu()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    dump_stats("context[0]", context[0])
    dump_stats("context_null[0]", context_null[0])

    # Load VAE
    print("Loading VAE...")
    vae = Wan2_1_VAE(
        vae_pth=os.path.join(args.model_dir, "Wan2.1_VAE.safetensors"),
        device=device)

    # VAE encode init image
    y = vae.encode([
        torch.concat([
            torch.nn.functional.interpolate(
                img_tensor[None].cpu(), size=(h_out, w_out), mode='bicubic').transpose(0, 1),
            torch.zeros(3, F - 1, h_out, w_out)
        ], dim=1).to(device)
    ])[0]
    dump_stats("vae_encoded", y)

    # Build mask
    msk = torch.ones(1, F, lat_h, lat_w, device=device)
    msk[:, 1:] = 0
    msk = torch.concat([
        torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
    ], dim=1)
    msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
    msk = msk.transpose(1, 2)[0]

    # Conditioning
    y = torch.concat([msk, y])
    dump_stats("conditioning (msk+vae)", y)

    # Generate noise
    seed_g = torch.Generator(device=device)
    seed_g.manual_seed(args.seed)
    noise = torch.randn(
        16, (F - 1) // vae_stride[0] + 1, lat_h, lat_w,
        dtype=torch.float32, generator=seed_g, device=device)
    dump_stats("noise", noise)
    print(f"[GROUND] noise[0,0,0,0:5] = {noise[0,0,0,:5].tolist()}")

    # Scheduler
    sample_scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=num_train_timesteps,
        shift=1,
        use_dynamic_shifting=False)
    sample_scheduler.set_timesteps(args.steps, device=device, shift=args.shift)
    timesteps = sample_scheduler.timesteps

    print(f"\n[GROUND] sigmas = {sample_scheduler.sigmas.tolist()}")
    print(f"[GROUND] timesteps = {timesteps.tolist()}")

    # Load model (high noise expert for initial steps)
    print("\nLoading high_noise_model...")
    model = WanModel.from_pretrained(
        args.model_dir, subfolder="high_noise_model")
    model.eval().requires_grad_(False)
    model.to(torch.bfloat16).to(device)

    # Dump weight diagnostics
    print("\n[GROUND] === Weight Diagnostics ===")
    for name, param in model.named_parameters():
        if any(k in name for k in [
            "patch_embedding.weight", "patch_embedding.bias",
            "blocks.0.modulation", "blocks.0.self_attn.q.weight",
            "blocks.0.self_attn.q.bias", "blocks.19.modulation",
            "blocks.39.modulation", "head.head.weight", "head.head.bias",
            "head.modulation", "text_embedding.0.weight",
            "time_embedding.0.weight", "time_projection.1.weight",
        ]):
            pf = param.float()
            print(f"[GROUND] {name}: shape={list(param.shape)}, dtype={param.dtype}, "
                  f"mean={pf.mean().item():.8f}, std={pf.std().item():.8f}")

    # Sinusoidal embedding trace
    print("\n[GROUND] === Sinusoidal Embedding Trace ===")
    ts_val = timesteps[0].item()
    sin_emb = sinusoidal_embedding_1d(freq_dim, torch.tensor([float(int(ts_val))])).float()
    print(f"[GROUND] sinEmb for t={int(ts_val)}, first 8: {sin_emb[0, :8].tolist()}")

    # Denoising loop
    print("\n[GROUND] === Denoising Loop ===")
    latent = noise

    with (torch.amp.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda'
          else torch.no_grad()):
        for step_idx, t in enumerate(timesteps):
            latent_model_input = [latent.to(device)]
            timestep = torch.stack([t]).to(device)

            guide_scale = args.guidance

            # Forward pass
            noise_pred_cond = model(
                latent_model_input, t=timestep,
                context=[context[0]], seq_len=max_seq_len, y=[y])[0]
            noise_pred_uncond = model(
                latent_model_input, t=timestep,
                context=context_null, seq_len=max_seq_len, y=[y])[0]

            noise_pred = noise_pred_uncond + guide_scale * (
                noise_pred_cond - noise_pred_uncond)

            dump_stats(f"step {step_idx} noisePredCond", noise_pred_cond)
            dump_stats(f"step {step_idx} noisePredUncond", noise_pred_uncond)
            dump_stats(f"step {step_idx} noisePred (CFG)", noise_pred)

            # Scheduler step
            temp_x0 = sample_scheduler.step(
                noise_pred.unsqueeze(0), t,
                latent.unsqueeze(0),
                return_dict=False, generator=seed_g)[0]
            latent = temp_x0.squeeze(0)

            dump_stats(f"step {step_idx} latent after", latent)
            print()

    print("[GROUND] === Done ===")


if __name__ == "__main__":
    main()
