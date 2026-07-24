"""Objective video metrics for the ComfyBox video QA harness.

All metrics are computed on downscaled frames (320px wide) for speed; they are
comparative fingerprints, not absolute units. Calibration anchors (2026-07-24):
a good deliberate i2v ≈ motion 0.31 / jerk 0.14; a near-static t2v ≈ motion
0.08-0.14 / jerk 0.015-0.03. Ground-truth ComfyUI renders scored with this same
module define per-case target bands in suite.json.
"""
import cv2
import numpy as np


def load_frames(path, maxw=320):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if f.shape[1] > maxw:
            h = int(f.shape[0] * maxw / f.shape[1])
            f = cv2.resize(f, (maxw, h))
        frames.append(f)
    cap.release()
    return frames


def load_frames_full(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames


def validity(frames):
    """Blank/corrupt frame detection. std < 10 = uniform (blank/gray/white)."""
    bad = [i for i, f in enumerate(frames) if float(f.std()) < 10]
    return {"valid": len(bad) == 0, "corrupt_frames": bad, "total_frames": len(frames)}


def motion_profile(frames):
    """Flow magnitude series -> motion, jerk (spikiness), direction reversals."""
    if len(frames) < 3:
        return {"motion": 0.0, "jerk": 0.0, "jerk_p95": 0.0, "reversal_rate": 0.0}
    pg = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    mags, vecs = [], []
    for f in frames[1:]:
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        fl = cv2.calcOpticalFlowFarneback(pg, g, None, 0.5, 2, 12, 2, 5, 1.1, 0)
        mags.append(float(np.sqrt(fl[..., 0] ** 2 + fl[..., 1] ** 2).mean()))
        vecs.append((float(fl[..., 0].mean()), float(fl[..., 1].mean())))
        pg = g
    m = np.array(mags)
    accel = np.abs(np.diff(m))
    dots = []
    for i in range(1, len(vecs)):
        a, b = vecs[i - 1], vecs[i]
        na, nb = np.hypot(*a), np.hypot(*b)
        if na > 0.01 and nb > 0.01:
            dots.append((a[0] * b[0] + a[1] * b[1]) / (na * nb))
    return {
        "motion": round(float(m.mean()), 3),
        "jerk": round(float(accel.mean()), 4),
        "jerk_p95": round(float(np.percentile(accel, 95)), 4) if len(accel) else 0.0,
        "reversal_rate": round(float(np.mean(np.array(dots) < 0)), 3) if dots else 0.0,
        "jerk_ratio": round(float(accel.mean() / (m.mean() + 1e-6)), 3),
    }


def sharpness_stability(frames, step=4):
    lap = [cv2.Laplacian(cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
           for i in range(0, len(frames), step)]
    return {"sharp_mean": round(float(np.mean(lap)), 1),
            "sharp_std": round(float(np.std(lap)), 1),
            "flicker_index": round(float(np.std(lap) / (np.mean(lap) + 1e-6)), 3)}


def source_similarity(frames, source_path):
    """i2v identity fingerprint: HSV histogram correlation of each sampled frame
    vs the source still + first-frame structural agreement. 1.0 = identical."""
    src = cv2.imread(source_path)
    if src is None or not frames:
        return {"src_hist_corr": None, "first_frame_corr": None}
    f0 = frames[0]
    src_r = cv2.resize(src, (f0.shape[1], f0.shape[0]))

    def hist(im):
        h = cv2.calcHist([cv2.cvtColor(im, cv2.COLOR_BGR2HSV)], [0, 1], None,
                         [32, 32], [0, 180, 0, 256])
        return cv2.normalize(h, h).flatten()

    hsrc = hist(src_r)
    idx = np.linspace(0, len(frames) - 1, min(9, len(frames))).astype(int)
    corrs = [float(cv2.compareHist(hsrc, hist(frames[i]), cv2.HISTCMP_CORREL)) for i in idx]
    g0 = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gs = cv2.cvtColor(src_r, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ff = float(np.corrcoef(g0.flatten(), gs.flatten())[0, 1])
    return {"src_hist_corr": round(float(np.mean(corrs)), 3),
            "first_frame_corr": round(ff, 3)}


def contact_sheet(path, out_png, n=8, height=300):
    frames = load_frames_full(path)
    if not frames:
        return None
    idx = np.linspace(0, len(frames) - 1, min(n, len(frames))).astype(int)
    sel = []
    for i in idx:
        f = frames[i]
        sel.append(cv2.resize(f, (int(f.shape[1] * height / f.shape[0]), height)))
    w = min(s.shape[1] for s in sel)
    cv2.imwrite(out_png, np.hstack([s[:, :w] for s in sel]))
    return out_png


def score(path, source_path=None):
    frames = load_frames(path)
    if not frames:
        return {"error": "unreadable", "validity": {"valid": False}}
    r = {"validity": validity(frames)}
    r.update(motion_profile(frames))
    r.update(sharpness_stability(frames))
    if source_path:
        r.update(source_similarity(frames, source_path))
    return r


if __name__ == "__main__":
    import sys, json
    src = sys.argv[2] if len(sys.argv) > 2 else None
    print(json.dumps(score(sys.argv[1], src), indent=1))
