"""Perceptual judge for the ComfyBox video QA harness.

Sends a temporal contact sheet (two rows: spread + consecutive burst) to the
local LM Studio VLM (the same provider ComfyBox uses for vision) and asks for
structured 0-10 scores. The model is a reasoner; we demand strict JSON and
parse defensively. Scores are RELATIVE judgments for regression tracking and
training-loop reward, not absolute truth.
"""
import base64
import json
import os
import re
import urllib.request

import cv2
import numpy as np

VISION_URL = os.environ.get("COMFYBOX_VISION_URL", "http://localhost:1234/v1")
VISION_MODEL = os.environ.get("COMFYBOX_VISION_MODEL", "qwen3.5-9b-distilled-opus-heretic-mlx-vlm")

PROMPT = """You are a strict video-quality QA judge for AI-generated adult video. The image shows frames from ONE generated clip: top row = 8 frames spread across the whole clip (temporal order left to right); bottom row = 8 CONSECUTIVE frames from the middle.

The clip was generated from this prompt: "{prompt}"

Score 0-10 (10 = flawless) on EXACTLY these axes:
- motion_quality: is implied movement smooth, continuous, physically plausible (weight, momentum)? Penalize jumpy/spastic pose changes between the consecutive frames, frozen stillness when motion was prompted, sliding/floating.
- choreography: does the visible action sequence match what the prompt asked for, including later beats?
- anatomy: correct human anatomy throughout — hands, breasts, genitals, limbs; penalize extra/missing/fused parts, melting, garbling.
- identity_consistency: same person, same clothing/scene across all frames (ignore intended motion).
- artifacts: 10 = clean; penalize mottling, mosaic, glitter/noise, hallucinated text/captions, ghost figures, blank or corrupted regions.
- solo: 10 if exactly the prompted number of people (default one), 0 if clearly wrong.

Respond with ONLY a JSON object, no prose, no markdown fences:
{{"motion_quality": n, "choreography": n, "anatomy": n, "identity_consistency": n, "artifacts": n, "solo": n, "worst_defect": "one short sentence"}}"""


def build_judge_sheet(video_path, out_png, height=280):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if len(frames) < 10:
        return None

    def rz(im):
        return cv2.resize(im, (int(im.shape[1] * height / im.shape[0]), height))

    spread_idx = np.linspace(0, len(frames) - 1, 8).astype(int)
    mid = len(frames) // 2
    consec_idx = list(range(mid - 4, mid + 4))
    r1 = np.hstack([rz(frames[i]) for i in spread_idx])
    r2 = np.hstack([rz(frames[i]) for i in consec_idx])
    w = min(r1.shape[1], r2.shape[1])
    cv2.imwrite(out_png, np.vstack([r1[:, :w], r2[:, :w]]))
    return out_png


def judge(video_path, prompt, workdir):
    sheet = os.path.join(workdir, "judge_sheet.png")
    if build_judge_sheet(video_path, sheet) is None:
        return {"error": "too few frames"}
    b64 = base64.b64encode(open(sheet, "rb").read()).decode()
    payload = {
        "model": VISION_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Do NOT think out loud. No reasoning, no <coding_plan>, no preamble. FIRST character of your reply must be {. " + PROMPT.format(prompt=prompt[:400])},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ],
        }],
        "max_tokens": 1600,
        "temperature": 0.1,
    }
    req = urllib.request.Request(
        VISION_URL.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            content = json.load(r)["choices"][0]["message"]["content"]
    except Exception as e:  # judge unreachable -> objective metrics still stand
        return {"error": f"vlm: {e}"}
    content = re.sub(r"<think>.*?</think>|<coding_plan>.*?</coding_plan>", "", content, flags=re.S)
    ms = re.findall(r"\{[^{}]*\"motion_quality\"[^{}]*\}", content, re.S)
    m = ms[-1] if ms else None
    class _M:  # tiny adapter to keep .group(0) call
        def __init__(s2, t): s2.t = t
        def group(s2, _): return s2.t
    m = _M(m) if m else None
    if not m:
        return {"error": "unparseable", "raw": content[-400:]}
    try:
        scores = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"error": "bad json", "raw": m.group(0)[:400]}
    scores["judge_sheet"] = sheet
    return scores


if __name__ == "__main__":
    import sys
    print(json.dumps(judge(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "a video",
                           os.path.dirname(sys.argv[1]) or "."), indent=1))
