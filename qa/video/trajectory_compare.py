#!/usr/bin/env python3
"""Compare two LTX-2 denoise-trajectory dumps (.jsonl) step by step.

Each input file is produced by running the ComfyBox LTX-2 pipeline with
LTX2_TRAJ_DUMP=<path-prefix> set (one JSON object per denoising step),
or by an equivalent dump from a reference implementation (e.g. ComfyUI)
using the same field names.

Usage:
    python3 qa/video/trajectory_compare.py ours.jsonl reference.jsonl
    python3 qa/video/trajectory_compare.py a.jsonl b.jsonl --threshold 0.15

Prints an aligned per-step table of the key stats and their relative
deltas, flags the first step where delta_norm or latent_std diverges by
more than the threshold (default 15%), and renders ASCII sparklines for
each metric. No third-party dependencies (no matplotlib).
"""

import argparse
import json
import math
import sys

SPARK_CHARS = " .:-=+*#%@"

# Metrics shown in the per-step table: (field, short label)
TABLE_METRICS = [
    ("latent_std", "lat_std"),
    ("delta_norm", "delta"),
    ("velocity_mean_abs", "vel|.|"),
    ("x0_std", "x0_std"),
]

# Metrics that get sparklines (cond/gen variants added if present).
SPARK_METRICS = [
    "latent_mean",
    "latent_std",
    "x0_mean",
    "x0_std",
    "velocity_mean_abs",
    "delta_norm",
    "noise_injected",
]

DIVERGENCE_FIELDS = ("delta_norm", "latent_std")


def load_jsonl(path):
    steps = []
    with open(path) as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                steps.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"warning: {path}:{lineno}: bad JSON line ({exc})", file=sys.stderr)
    steps.sort(key=lambda rec: rec.get("step", 0))
    return steps


def rel_delta(a, b):
    """Relative difference of b vs a, using a as the reference scale."""
    if a is None or b is None:
        return None
    denom = max(abs(a), abs(b), 1e-12)
    return (b - a) / denom


def fmt(value, width=11):
    if value is None:
        return "-".rjust(width)
    if value == 0:
        return "0".rjust(width)
    if abs(value) >= 1e4 or abs(value) < 1e-3:
        return f"{value:.3e}".rjust(width)
    return f"{value:.5f}".rjust(width)


def fmt_pct(value, width=8):
    if value is None:
        return "-".rjust(width)
    return f"{value * 100:+.1f}%".rjust(width)


def sparkline(values):
    vals = [v for v in values if v is not None and math.isfinite(v)]
    if not vals:
        return "(no data)"
    lo, hi = min(vals), max(vals)
    span = hi - lo
    out = []
    for v in values:
        if v is None or not math.isfinite(v):
            out.append("?")
            continue
        t = 0.5 if span < 1e-30 else (v - lo) / span
        idx = min(len(SPARK_CHARS) - 1, int(t * (len(SPARK_CHARS) - 1) + 0.5))
        out.append(SPARK_CHARS[idx])
    return "".join(out)


def series(steps, field):
    return [rec.get(field) for rec in steps]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("file_a", help="first trajectory dump (ours)")
    parser.add_argument("file_b", help="second trajectory dump (reference or other config)")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.15,
        help="relative divergence threshold for flagging (default 0.15 = 15%%)",
    )
    args = parser.parse_args()

    steps_a = load_jsonl(args.file_a)
    steps_b = load_jsonl(args.file_b)
    if not steps_a or not steps_b:
        print("error: one or both files contain no steps", file=sys.stderr)
        return 2

    n = min(len(steps_a), len(steps_b))
    if len(steps_a) != len(steps_b):
        print(
            f"note: step count differs (A={len(steps_a)}, B={len(steps_b)}); "
            f"comparing first {n} steps\n"
        )

    label_a = args.file_a
    label_b = args.file_b
    print(f"A = {label_a}")
    print(f"B = {label_b}")
    print()

    # ---- Aligned per-step table ----
    header = f"{'step':>4} {'sigma':>9}"
    for _, label in TABLE_METRICS:
        header += f" | {label + '(A)':>11} {label + '(B)':>11} {'d%':>8}"
    print(header)
    print("-" * len(header))

    first_divergence = None  # (step, field, rel)
    for i in range(n):
        a, b = steps_a[i], steps_b[i]
        row = f"{a.get('step', i):>4} {fmt(a.get('sigma'), 9)}"
        for field, _ in TABLE_METRICS:
            va, vb = a.get(field), b.get(field)
            rd = rel_delta(va, vb)
            row += f" | {fmt(va)} {fmt(vb)} {fmt_pct(rd)}"
            if (
                first_divergence is None
                and field in DIVERGENCE_FIELDS
                and rd is not None
                and abs(rd) > args.threshold
            ):
                first_divergence = (a.get("step", i), field, rd)
        flag = ""
        if first_divergence is not None and first_divergence[0] == a.get("step", i):
            flag = "  <-- FIRST DIVERGENCE"
        print(row + flag)

    print()
    if first_divergence is not None:
        step, field, rd = first_divergence
        print(
            f"*** FIRST DIVERGENCE at step {step}: {field} differs by "
            f"{rd * 100:+.1f}% (threshold {args.threshold * 100:.0f}%) ***"
        )
    else:
        print(
            f"no divergence in {'/'.join(DIVERGENCE_FIELDS)} beyond "
            f"{args.threshold * 100:.0f}% over {n} compared steps"
        )

    # ---- Sparklines ----
    spark_fields = list(SPARK_METRICS)
    present = set()
    for rec in steps_a[:1] + steps_b[:1]:
        present.update(rec.keys())
    for base in SPARK_METRICS:
        for suffix in ("_cond", "_gen"):
            if base + suffix in present:
                spark_fields.append(base + suffix)

    print()
    print("sparklines (each char = one step, scaled per-row):")
    width = max(len(f) for f in spark_fields)
    for field in spark_fields:
        sa = series(steps_a, field)[:n]
        sb = series(steps_b, field)[:n]
        if all(v is None for v in sa) and all(v is None for v in sb):
            continue
        print(f"  {field:<{width}}  A |{sparkline(sa)}|")
        print(f"  {'':<{width}}  B |{sparkline(sb)}|")

    # ---- Summary aggregates ----
    print()
    print("aggregates (mean over compared steps):")
    for field in ("delta_norm", "velocity_mean_abs", "latent_std", "x0_std"):
        va = [v for v in series(steps_a, field)[:n] if v is not None]
        vb = [v for v in series(steps_b, field)[:n] if v is not None]
        if not va or not vb:
            continue
        ma, mb = sum(va) / len(va), sum(vb) / len(vb)
        ratio = mb / ma if abs(ma) > 1e-30 else float("inf")
        print(f"  {field:<20} A={ma:.5e}  B={mb:.5e}  B/A={ratio:.3f}x")

    return 1 if first_divergence is not None else 0


if __name__ == "__main__":
    sys.exit(main())
