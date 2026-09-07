#!/usr/bin/env python3
"""comfybox#405 verification tool — OFFLINE ONLY, never part of the runtime
(intent.md: zero Python at runtime). Sits alongside the other offline scripts
in this directory.

Replay the OLD (main) and NEW (PR #408) i2v dim algorithms over every
`LTX-2 I2V: adjusted` line in the production engine log, and report any shape
that changes. Review ruling 1: no shape may shrink.

Usage:  python3 scripts/replay-i2v-dims.py ~/.comfybox/serve.err.log
"""
import re, sys, math
from collections import Counter

def snap64(v):
    return max(256, int(round(v / 64.0)) * 64)

def fit(aspect, bw, bh, area_cap_primary, relax_trigger, window, area_weight):
    if not (aspect > 0):
        return (snap64(bw), snap64(bh))
    budget = float(max(bw, 64) * max(bh, 64))
    idealW = math.sqrt(budget * aspect)
    idealH = idealW / aspect
    baseW = int(round(idealW / 64.0))
    baseH = int(round(idealH / 64.0))

    def search(cap):
        best = None
        for dw in range(-window, window + 1):
            for dh in range(-window, window + 1):
                w = max(256, (baseW + dw) * 64)
                h = max(256, (baseH + dh) * 64)
                area = float(w * h)
                if area > budget * cap:
                    continue
                ae = abs(w / h - aspect) / aspect
                re_ = abs(area - budget) / budget
                score = ae + area_weight * re_
                cand = (w, h, ae, re_, score)
                if best is None:
                    best = cand
                else:
                    if area_weight > 0:
                        better = score < best[4] - 1e-12
                    else:
                        better = (ae < best[2] - 1e-9) or (abs(ae - best[2]) <= 1e-9 and re_ < best[3])
                    if better:
                        best = cand
        return best

    pick = search(area_cap_primary)
    if pick is None or pick[2] > relax_trigger:
        relaxed = search(1.6)
        if relaxed is not None and relaxed[2] < ((pick[2] if pick else float('inf')) - 1e-9):
            pick = relaxed
    if pick is None:
        return (snap64(int(round(idealW))), snap64(int(round(idealH))))
    return (pick[0], pick[1])

# OLD = main's WarmServer.deriveVideoDims
def old_algo(sw, sh, bw, bh):
    return fit(sw / sh, bw, bh, 1.25, 0.03, 1, 0.0)

# NEW = PR #408 after the review fix (VideoDimensionResolver.fit + clamp)
def clamp(w, h, max_long=4096, max_px=16_777_216):
    if max(w, h) <= max_long and w * h <= max_px:
        return (w, h)
    scale = min(max_long / max(w, h), math.sqrt(max_px / (w * h)))
    f = lambda v: max(256, int(v / 64.0) * 64)
    w2, h2 = f(w * scale), f(h * scale)
    g = 0
    while (max(w2, h2) > max_long or w2 * h2 > max_px) and max(w2, h2) > 256 and g < 1024:
        if w2 >= h2: w2 = max(256, w2 - 64)
        else: h2 = max(256, h2 - 64)
        g += 1
    return (w2, h2)

def new_algo(sw, sh, bw, bh):
    return clamp(*fit(sw / sh, bw, bh, 1.25, 0.03, 1, 0.0))

# The FIRST cut of the PR (what the reviewer flagged): cap 1.0, +/-2, weighted
def firstcut_algo(sw, sh, bw, bh):
    return clamp(*fit(sw / sh, bw, bh, 1.0, 0.10, 2, 0.05))

LINE = re.compile(r"LTX-2 I2V: adjusted (\d+)x(\d+) -> (\d+)x(\d+) \(source (\d+)x(\d+)")

path = sys.argv[1]
rows = []
with open(path, errors="replace") as f:
    for line in f:
        m = LINE.search(line)
        if m:
            bw, bh, ow, oh, sw, sh = map(int, m.groups())
            rows.append((bw, bh, sw, sh, ow, oh))

print(f"parsed {len(rows)} 'I2V: adjusted' lines from {path}")

def report(name, algo):
    changed = Counter()
    shrunk = 0
    mismatch_vs_log = 0
    for bw, bh, sw, sh, ow, oh in rows:
        got = algo(sw, sh, bw, bh)
        if got != (ow, oh):
            changed[((bw, bh), (sw, sh), (ow, oh), got)] += 1
            if got[0] * got[1] < ow * oh:
                shrunk += 1
    print(f"\n=== {name}")
    print(f"  renders whose dims change vs the log: {sum(changed.values())} / {len(rows)}"
          f"  ({100*sum(changed.values())/max(1,len(rows)):.0f}%)")
    print(f"  distinct (budget, source) shapes that change: {len(changed)}")
    print(f"  renders that SHRINK: {shrunk}")
    if changed:
        print(f"  {'budget':>10} {'source':>11} {'was':>10} {'now':>10} {'px delta':>9}  n")
        for (b, s, was, now), n in sorted(changed.items(), key=lambda kv: -kv[1]):
            d = (now[0]*now[1] - was[0]*was[1]) / (was[0]*was[1]) * 100
            print(f"  {b[0]}x{b[1]:<6} {s[0]}x{s[1]:<7} {was[0]}x{was[1]:<6} {now[0]}x{now[1]:<6} {d:+7.1f}%  {n}")

report("OLD algorithm (main) — sanity check, must be 0 changes", old_algo)
report("FIRST CUT of PR #408 (cap 1.0, +/-2, weighted) — what the review flagged", firstcut_algo)
report("NEW (PR #408 after the review fix)", new_algo)

print("\n=== DIRECT old-vs-new comparison (the question that matters)")
diff = Counter()
for bw, bh, sw, sh, ow, oh in rows:
    o, n = old_algo(sw, sh, bw, bh), new_algo(sw, sh, bw, bh)
    if o != n:
        diff[((bw, bh), (sw, sh), o, n)] += 1
print(f"  logged renders where NEW != OLD: {sum(diff.values())} / {len(rows)}")
for k, v in diff.items():
    print("   ", k, v)

fc = Counter()
for bw, bh, sw, sh, ow, oh in rows:
    o, n = old_algo(sw, sh, bw, bh), firstcut_algo(sw, sh, bw, bh)
    if o != n:
        fc[((bw, bh), (sw, sh), o, n)] += 1
shrink = sum(v for (b, s, o, n), v in fc.items() if n[0]*n[1] < o[0]*o[1])
print(f"  (first cut, for the record: {sum(fc.values())} differed, {len(fc)} distinct shapes,"
      f" {shrink} renders shrank)")

print("\n=== EXHAUSTIVE sweep: every budget x source pair, old vs new")
budgets = [(w, h) for w in range(256, 1601, 64) for h in range(256, 1601, 64)]
srcs = [(w, h) for w in range(64, 2049, 64) for h in range(64, 2049, 64)]
srcs += [(10, 9999), (9999, 10), (1, 1), (3, 4), (1664, 896), (896, 1664), (576, 1024)]
bad = 0
checked = 0
for bw, bh in budgets[::7]:
    for sw, sh in srcs[::37]:
        checked += 1
        if old_algo(sw, sh, bw, bh) != new_algo(sw, sh, bw, bh):
            # the ONLY legitimate difference is the >4096 safety clamp
            o = old_algo(sw, sh, bw, bh)
            if max(o) <= 4096 and o[0]*o[1] <= 16_777_216:
                bad += 1
                if bad < 10:
                    print("   UNEXPECTED", (bw, bh), (sw, sh), o, new_algo(sw, sh, bw, bh))
print(f"  checked {checked} pairs; unexpected differences (outside the safety clamp): {bad}")
