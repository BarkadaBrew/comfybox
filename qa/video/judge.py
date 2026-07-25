"""Perceptual judge for the ComfyBox video QA harness.

Sends a temporal contact sheet to the local LM Studio VLM (the same provider
ComfyBox uses for vision) and asks for structured 0-10 scores. The model is a
reasoner; we demand strict JSON and parse defensively. Scores are RELATIVE
judgments for regression tracking and training-loop reward, not absolute truth.

Hardening (2026-07-25, from the 20260724-170839 full run where ~50% of calls
came back unparseable/empty):
- up to 3 attempts per clip, 15s apart; temperature drops to 0 on retries
- if chat `content` is empty, the LM Studio reasoning-field variants
  (message.reasoning / message.reasoning_content / choices[0].reasoning /
  message.thinking) are scanned for the JSON blob too
- motion-aware judge sheet (see build_judge_sheet): spread row + every-3rd-frame
  midwindow row + frame-difference heatmap row so periodic/cyclic motion is
  visible to the VLM
- if ALL attempts fail the result is {"error": ..., "skipped": true} and the
  harness must NOT count judge-band failures for that case (objective metrics
  still apply).
"""
import base64
import json
import os
import re
import time
import urllib.request

import cv2
import numpy as np

VISION_URL = os.environ.get("COMFYBOX_VISION_URL", "http://localhost:1234/v1")
VISION_MODEL = os.environ.get("COMFYBOX_VISION_MODEL", "qwen3.5-9b-distilled-opus-heretic-mlx-vlm")

JUDGE_RETRIES = 3
JUDGE_RETRY_WAIT = 15  # seconds between attempts

PROMPT = """You are a strict video-quality QA judge for AI-generated adult video. The image shows frames from ONE generated clip, in THREE rows:
- Row 1: 8 frames spread evenly across the WHOLE clip (temporal order left to right).
- Row 2: 8 frames sampled every 3rd frame across a 24-frame window from the MIDDLE of the clip — close enough in time that real motion shows as clear pose progression between neighbors.
- Row 3: 4 frame-difference heatmaps (absolute difference of two consecutive frames, colormapped: dark/black = no change, bright/hot = pixels that moved). Bright silhouettes of limbs/hips/hair mean real motion is happening at that moment; near-black tiles mean the clip is frozen there. Rhythmic/cyclic motion (grinding, thrusting, dancing) shows as bright repeating shapes — use this row to detect motion that rows 1-2 may hide.

The clip was generated from this prompt: "{prompt}"

Score 0-10 (10 = flawless) on EXACTLY these axes:
- motion_quality: is implied movement smooth, continuous, physically plausible (weight, momentum)? Use row 2 for pose progression and row 3 for motion energy. Penalize jumpy/spastic pose changes, frozen stillness when motion was prompted (near-black row 3), sliding/floating.
- choreography: does the visible action sequence match what the prompt asked for, including later beats?
- anatomy: correct human anatomy throughout — hands, breasts, genitals, limbs; penalize extra/missing/fused parts, melting, garbling.
- identity_consistency: same person, same clothing/scene across all frames (ignore intended motion; ignore row 3, it is a heatmap, not a person).
- artifacts: 10 = clean; penalize mottling, mosaic, glitter/noise, hallucinated text/captions, ghost figures, blank or corrupted regions (judge rows 1-2 only).
- solo: 10 if exactly the prompted number of people (default one), 0 if clearly wrong.

Respond with ONLY a JSON object, no prose, no markdown fences:
{{"motion_quality": n, "choreography": n, "anatomy": n, "identity_consistency": n, "artifacts": n, "solo": n, "worst_defect": "one short sentence"}}"""


def build_judge_sheet(video_path, out_png, height=280):
    """Three-row motion-aware sheet:
    row 1: 8 frames spread across the whole clip
    row 2: every 3rd frame over a 24-frame window centered mid-clip (8 frames)
    row 3: 4 colormapped absdiff heatmaps of consecutive-frame pairs from the
           same midwindow — makes periodic/cyclic motion visible to the VLM.
    """
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
    n = len(frames)

    def rz(im):
        return cv2.resize(im, (int(im.shape[1] * height / im.shape[0]), height))

    def clamp(i):
        return max(0, min(n - 1, i))

    spread_idx = np.linspace(0, n - 1, 8).astype(int)
    mid = n // 2
    # 24-frame midwindow sampled every 3rd frame -> 8 frames
    win_idx = [clamp(i) for i in range(mid - 12, mid + 12, 3)]
    r1 = np.hstack([rz(frames[i]) for i in spread_idx])
    r2 = np.hstack([rz(frames[i]) for i in win_idx])

    # row 3: absdiff heatmaps of 4 consecutive pairs spread within the midwindow
    heat_at = [clamp(i) for i in (mid - 9, mid - 3, mid + 3, mid + 9)]
    tiles = []
    for i in heat_at:
        j = clamp(i + 1)
        d = cv2.absdiff(frames[i], frames[j])
        g = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
        # fixed gain (not per-tile normalize) so a frozen clip stays dark
        g = np.clip(g.astype(np.float32) * 4.0, 0, 255).astype(np.uint8)
        tiles.append(rz(cv2.applyColorMap(g, cv2.COLORMAP_INFERNO)))
    r3 = np.hstack(tiles)

    w = min(r1.shape[1], r2.shape[1])
    if r3.shape[1] < w:  # pad heatmap row to full width
        r3 = np.hstack([r3, np.zeros((height, w - r3.shape[1], 3), np.uint8)])
    cv2.imwrite(out_png, np.vstack([r1[:, :w], r2[:, :w], r3[:, :w]]))
    return out_png


def _texts_from_response(resp):
    """Collect every text field the VLM may have answered in: the chat
    `content` plus LM Studio reasoning-model variants (the qwen3.5-9b reasoner
    frequently returns empty content with the answer buried in reasoning)."""
    texts = []
    try:
        choice = resp["choices"][0]
    except (KeyError, IndexError, TypeError):
        return texts
    msg = choice.get("message") or {}
    content = msg.get("content")
    if isinstance(content, list):  # some servers return content parts
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    if content:
        texts.append(content)
    for key in ("reasoning", "reasoning_content", "thinking"):
        v = msg.get(key)
        if isinstance(v, str) and v:
            texts.append(v)
    v = choice.get("reasoning")
    if isinstance(v, str) and v:
        texts.append(v)
    return texts


def _extract_scores(texts):
    """Find the last flat JSON object containing motion_quality in any text."""
    for text in texts:
        text = re.sub(r"<think>.*?</think>|<coding_plan>.*?</coding_plan>", "", text, flags=re.S)
        ms = re.findall(r"\{[^{}]*\"motion_quality\"[^{}]*\}", text, re.S)
        if not ms:
            continue
        try:
            return json.loads(ms[-1]), None
        except json.JSONDecodeError:
            return None, {"error": "bad json", "raw": ms[-1][:400]}
    joined = "\n".join(texts)
    if not joined.strip():
        return None, {"error": "empty content (and no reasoning field)"}
    return None, {"error": "unparseable", "raw": joined[-400:]}


def _call_vlm(b64, prompt, temperature, max_tokens=3000, no_think=False, timeout=180):
    text = ("Do NOT think out loud. No reasoning, no <coding_plan>, no preamble. "
            "FIRST character of your reply must be {. " + PROMPT.format(prompt=prompt[:400]))
    if no_think:
        # Qwen3-family soft switch — the observed failure mode is the reasoner
        # burning the whole token budget thinking (truncated mid-sentence, no
        # JSON ever emitted). Harmless if the model ignores it.
        text = "/no_think " + text
    payload = {
        "model": VISION_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ],
        }],
        # reasoning models silently burn the budget thinking and return empty
        # content when max_tokens is tight — give real headroom (see /k3 lesson)
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    req = urllib.request.Request(
        VISION_URL.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            resp = json.load(r)
    except Exception as e:  # judge unreachable -> objective metrics still stand
        return None, {"error": f"vlm: {e}"}
    scores, err = _extract_scores(_texts_from_response(resp))
    return scores, err


def judge(video_path, prompt, workdir, retries=JUDGE_RETRIES, retry_wait=JUDGE_RETRY_WAIT):
    stem = os.path.splitext(os.path.basename(video_path))[0]
    sheet = os.path.join(workdir, f"{stem}_judge_sheet.png")
    if build_judge_sheet(video_path, sheet) is None:
        return {"error": "too few frames", "skipped": True}
    b64 = base64.b64encode(open(sheet, "rb").read()).decode()
    last_err = {"error": "no attempts"}
    for attempt in range(retries):
        if attempt:
            time.sleep(retry_wait)
        # retries: temperature 0 (deterministic), /no_think soft switch, and a
        # bigger budget — the dominant failure is reasoning-burned max_tokens
        temperature = 0.1 if attempt == 0 else 0.0
        scores, err = _call_vlm(b64, prompt, temperature,
                                max_tokens=3000 if attempt == 0 else 6000,
                                no_think=attempt > 0,
                                timeout=180 if attempt == 0 else 420)
        if scores is not None:
            scores["judge_sheet"] = sheet
            scores["attempts"] = attempt + 1
            return scores
        last_err = err
        print(f"[judge] attempt {attempt + 1}/{retries} failed: {err.get('error')}")
    out = dict(last_err)
    out["skipped"] = True  # harness: do NOT count judge bands for this case
    out["attempts"] = retries
    out["judge_sheet"] = sheet
    return out


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("usage: judge.py <video.mp4> [prompt] [workdir]\n"
              "env: COMFYBOX_VISION_URL, COMFYBOX_VISION_MODEL")
        sys.exit(0)
    print(json.dumps(judge(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "a video",
                           sys.argv[3] if len(sys.argv) > 3 else (os.path.dirname(sys.argv[1]) or ".")),
                     indent=1))
