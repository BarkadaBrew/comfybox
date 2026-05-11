#!/usr/bin/env python3
"""Compare sinusoidal embedding between float32 and float64.

Run on the Mac to get ground truth values for comparison with Swift.

Usage: python3 scripts/compare_sinusoidal.py
"""
import torch
import numpy as np


def sinusoidal_embedding_1d_f64(dim, position):
    """Python reference: float64 computation."""
    half = dim // 2
    position = position.type(torch.float64)
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


def sinusoidal_embedding_1d_f32(dim, position):
    """Swift equivalent: float32 computation."""
    half = dim // 2
    position = position.type(torch.float32)
    exponents = torch.arange(half, dtype=torch.float32) / half
    freqs = torch.pow(torch.tensor(10000.0, dtype=torch.float32), -exponents)
    sinusoid = torch.outer(position, freqs)
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


def main():
    dim = 256
    half = dim // 2

    # Test timesteps (5 steps with shift=5.0)
    timesteps = [999, 952, 882, 768, 555]

    print("=== Sinusoidal Embedding: float64 vs float32 ===")
    print(f"dim={dim}, half={half}")
    print()

    for ts in timesteps:
        pos = torch.tensor([float(ts)])

        emb_f64 = sinusoidal_embedding_1d_f64(dim, pos)
        emb_f32 = sinusoidal_embedding_1d_f32(dim, pos)

        # Convert both to float32 for comparison (as the pipeline does)
        emb_f64_f32 = emb_f64.float()

        diff = (emb_f64_f32 - emb_f32).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        print(f"Timestep {ts}:")
        print(f"  f64 -> f32: first 8 values = {emb_f64_f32[0, :8].tolist()}")
        print(f"  f32 direct: first 8 values = {emb_f32[0, :8].tolist()}")
        print(f"  max abs diff = {max_diff:.10e}")
        print(f"  mean abs diff = {mean_diff:.10e}")
        print(f"  f64 mean={emb_f64_f32.mean().item():.8f}, std={emb_f64_f32.std().item():.8f}")
        print(f"  f32 mean={emb_f32.mean().item():.8f}, std={emb_f32.std().item():.8f}")
        print()

    # Also test the full pipeline: sinusoidal -> time_embedding MLP -> time_projection
    print("=== Impact through random MLP ===")
    torch.manual_seed(42)
    time_emb = torch.nn.Sequential(
        torch.nn.Linear(dim, 5120),
        torch.nn.SiLU(),
        torch.nn.Linear(5120, 5120)
    )
    time_proj = torch.nn.Sequential(
        torch.nn.SiLU(),
        torch.nn.Linear(5120, 5120 * 6)
    )

    for ts in timesteps:
        pos = torch.tensor([float(ts)])
        emb_f64 = sinusoidal_embedding_1d_f64(dim, pos).float()
        emb_f32 = sinusoidal_embedding_1d_f32(dim, pos)

        with torch.no_grad():
            te_f64 = time_emb(emb_f64)
            te_f32 = time_emb(emb_f32)
            tp_f64 = time_proj(te_f64)
            tp_f32 = time_proj(te_f32)

        diff_te = (te_f64 - te_f32).abs()
        diff_tp = (tp_f64 - tp_f32).abs()
        print(f"Timestep {ts}:")
        print(f"  After time_emb: max_diff={diff_te.max().item():.6e}, mean_diff={diff_te.mean().item():.6e}")
        print(f"  After time_proj: max_diff={diff_tp.max().item():.6e}, mean_diff={diff_tp.mean().item():.6e}")
        print(f"  time_proj f64 std={tp_f64.std().item():.6f}, f32 std={tp_f32.std().item():.6f}")
        print()


if __name__ == "__main__":
    main()
