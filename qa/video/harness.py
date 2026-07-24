#!/usr/bin/env python3
"""ComfyBox video QA harness.

Runs the declarative test suite (suite.json) against the warm server, scores
every render objectively (metrics.py) and perceptually (judge.py), and appends
machine-readable records to runs/<ts>/results.jsonl — (prompt, config) ->
(scores) pairs suitable as reward data for prompt/config optimization loops.

Ops rules baked in (learned the hard way, 2026-07-24):
- pause the Kira content scheduler for the run, resume after (always)
- restart the warm server between renders (memory drain; admission fails otherwise)
- validate frames before judging anything (blank/corrupt detection)
- purge rendered mp4s from ~/Pictures/ComfyBox after archiving (the orphan
  reconciler sweeps leftovers into Kira's gallery)

Usage:
  python3 harness.py                 # smoke suite
  python3 harness.py --full          # all cases
  python3 harness.py --cases id1,id2 # explicit
  python3 harness.py --no-judge      # objective metrics only
  python3 harness.py --no-restart    # skip server restarts (fast, riskier)
"""
import argparse
import datetime
import json
import os
import plistlib
import shutil
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import metrics  # noqa: E402
import judge as judge_mod  # noqa: E402

SERVER = "http://127.0.0.1:7870"
KIRA = "http://localhost:3787"
PLIST = os.path.expanduser("~/Library/LaunchAgents/com.barkadabrew.comfybox.plist")
PICTURES = os.path.expanduser("~/Pictures/ComfyBox")


def http(method, url, body=None, timeout=30):
    req = urllib.request.Request(url, method=method,
                                 data=json.dumps(body).encode() if body else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def kira_pause(paused):
    try:
        http("POST", f"{KIRA}/v1/kira/content-scheduler/{'pause' if paused else 'resume'}")
        print(f"[harness] kira scheduler {'paused' if paused else 'resumed'}")
    except Exception as e:
        print(f"[harness] WARN kira scheduler unreachable: {e}")


def server_restart():
    uid = os.getuid()
    subprocess.run(["launchctl", "bootout", f"gui/{uid}/com.barkadabrew.comfybox"],
                   capture_output=True)
    time.sleep(4)
    subprocess.run(["launchctl", "bootstrap", f"gui/{uid}", PLIST], capture_output=True)
    for _ in range(60):
        try:
            http("GET", f"{SERVER}/health", timeout=3)
            print("[harness] server ready")
            return True
        except Exception:
            time.sleep(3)
    raise RuntimeError("server did not come back after restart")


def config_snapshot():
    snap = {}
    try:
        with open(PLIST, "rb") as f:
            d = plistlib.load(f)
        args = d.get("ProgramArguments", [])
        if "--ltx2-weights" in args:
            snap["ltx2_weights"] = os.path.basename(args[args.index("--ltx2-weights") + 1])
        if "--ltx2-gemma" in args:
            snap["ltx2_gemma"] = os.path.basename(args[args.index("--ltx2-gemma") + 1].rstrip("/"))
        snap["env"] = {k: v for k, v in d.get("EnvironmentVariables", {}).items()
                       if k.startswith("LTX2_")}
    except Exception as e:
        snap["plist_error"] = str(e)
    try:
        snap["git"] = subprocess.run(["git", "-C", os.path.join(HERE, "..", ".."),
                                      "rev-parse", "--short", "HEAD"],
                                     capture_output=True, text=True).stdout.strip()
    except Exception:
        pass
    return snap


def render(case, run_dir):
    body = {
        "prompt": case["prompt"],
        "negative_prompt": case.get("negative"),
        "seed": case.get("seed", 42),
        "frames": case.get("frames", 97),
        "backend": "local",
        "enhance": case.get("enhance", False),
        "content_mode": case.get("content_mode", "neutral"),
        "character": case.get("character", ""),
    }
    for k_json, k_case in [("image_path", "init_image"), ("width", "width"),
                           ("height", "height"), ("strength", "strength"),
                           ("aspect_ratio", "aspect_ratio"), ("resolution", "resolution")]:
        v = case.get(k_case)
        if v is not None:
            if k_case == "init_image":
                v = os.path.join(HERE, v)
            body[k_json] = v
    body = {k: v for k, v in body.items() if v is not None}

    sub = http("POST", f"{SERVER}/v1/video/generate/async", body)
    job = sub.get("job_id")
    if not job:
        return {"error": f"submit failed: {sub}"}
    print(f"[harness] {case['id']}: job {job}")
    t0 = time.time()
    while time.time() - t0 < 3600:
        time.sleep(20)
        try:
            st = http("GET", f"{SERVER}/v1/video/status/{job}", timeout=10)
        except Exception:
            continue
        s = st.get("status")
        if s == "succeeded":
            out = st.get("output_path")
            dst = os.path.join(run_dir, f"{case['id']}.mp4")
            shutil.copy2(out, dst)
            try:
                os.remove(out)  # anti-pollution: keep the reconciler away
            except OSError:
                pass
            return {"output": dst, "render_seconds": round(time.time() - t0)}
        if s == "failed":
            return {"error": st.get("error", "failed"), "render_seconds": round(time.time() - t0)}
    return {"error": "timeout"}


def evaluate(case, rendered, run_dir, do_judge=True):
    rec = {"id": case["id"], "mode": case["mode"], "category": case.get("category"),
           "prompt": case["prompt"], "negative": case.get("negative"),
           "seed": case.get("seed"), "frames": case.get("frames"),
           "ts": datetime.datetime.now().isoformat(timespec="seconds"),
           "config": config_snapshot()}
    rec.update(rendered)
    if "output" not in rendered:
        rec["verdict"] = "RENDER_FAILED"
        return rec
    src = os.path.join(HERE, case["init_image"]) if case.get("init_image") else None
    m = metrics.score(rendered["output"], src)
    rec["metrics"] = m
    sheet = metrics.contact_sheet(rendered["output"],
                                  os.path.join(run_dir, f"{case['id']}_sheet.png"))
    rec["contact_sheet"] = sheet
    if do_judge:
        rec["judge"] = judge_mod.judge(rendered["output"], case["prompt"], run_dir)

    exp = case.get("expected", {})
    fails = []
    if exp.get("valid") and not m["validity"]["valid"]:
        fails.append(f"corrupt frames {m['validity']['corrupt_frames'][:5]}")
    if "motion_min" in exp and m.get("motion", 0) < exp["motion_min"]:
        fails.append(f"motion {m.get('motion')} < {exp['motion_min']}")
    if "motion_max" in exp and m.get("motion", 0) > exp["motion_max"]:
        fails.append(f"motion {m.get('motion')} > {exp['motion_max']}")
    if "jerk_max" in exp and m.get("jerk", 0) > exp["jerk_max"]:
        fails.append(f"jerk {m.get('jerk')} > {exp['jerk_max']}")
    j = rec.get("judge") or {}
    if not j.get("error"):
        if "identity_min" in exp and j.get("identity_consistency", 10) < exp["identity_min"]:
            fails.append(f"identity {j.get('identity_consistency')} < {exp['identity_min']}")
        if "anatomy_min" in exp and j.get("anatomy", 10) < exp["anatomy_min"]:
            fails.append(f"anatomy {j.get('anatomy')} < {exp['anatomy_min']}")
        if exp.get("solo") and j.get("solo", 10) < 5:
            fails.append("not solo")
    rec["failures"] = fails
    rec["verdict"] = "PASS" if not fails else "FAIL"
    return rec


def _apply_judge_bands(rec, case):
    exp = case.get("expected", {})
    j = rec.get("judge") or {}
    fails = rec.get("failures", [])
    if not j.get("error"):
        if "identity_min" in exp and j.get("identity_consistency", 10) < exp["identity_min"]:
            fails.append(f"identity {j.get('identity_consistency')} < {exp['identity_min']}")
        if "anatomy_min" in exp and j.get("anatomy", 10) < exp["anatomy_min"]:
            fails.append(f"anatomy {j.get('anatomy')} < {exp['anatomy_min']}")
        if exp.get("solo") and j.get("solo", 10) < 5:
            fails.append("not solo")
        if "motion_quality_min" in exp and j.get("motion_quality", 10) < exp["motion_quality_min"]:
            fails.append(f"motion_quality {j.get('motion_quality')} < {exp['motion_quality_min']}")
        if "choreography_min" in exp and j.get("choreography", 10) < exp["choreography_min"]:
            fails.append(f"choreography {j.get('choreography')} < {exp['choreography_min']}")
    rec["failures"] = fails
    rec["verdict"] = "PASS" if not fails else "FAIL"


def write_report(records, run_dir):
    lines = [f"# Video QA run {os.path.basename(run_dir)}", ""]
    npass = sum(1 for r in records if r["verdict"] == "PASS")
    lines.append(f"**{npass}/{len(records)} PASS**\n")
    lines.append("| case | verdict | motion | jerk | identity | anatomy | artifacts | worst defect |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in records:
        m = r.get("metrics", {})
        j = r.get("judge") or {}
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
            r["id"], r["verdict"], m.get("motion", "-"), m.get("jerk", "-"),
            j.get("identity_consistency", "-"), j.get("anatomy", "-"),
            j.get("artifacts", "-"), (j.get("worst_defect") or "-")[:60]))
        if r.get("failures"):
            lines.append(f"|  | failures: {'; '.join(r['failures'])} | | | | | | |")
    open(os.path.join(run_dir, "report.md"), "w").write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--cases", default=None)
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--no-restart", action="store_true")
    args = ap.parse_args()

    suite = json.load(open(os.path.join(HERE, "suite.json")))
    cases = {c["id"]: c for c in suite["cases"]}
    if args.cases:
        selected = [cases[c] for c in args.cases.split(",")]
    elif args.full:
        selected = list(cases.values())
    else:
        selected = [cases[c] for c in suite["smoke"]]

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(HERE, "runs", ts)
    os.makedirs(run_dir, exist_ok=True)
    print(f"[harness] run {ts}: {len(selected)} case(s) -> {run_dir}")

    kira_pause(True)
    records = []
    try:
        for i, case in enumerate(selected):
            if not args.no_restart:
                server_restart()
            print(f"[harness] ({i+1}/{len(selected)}) {case['id']}")
            rendered = render(case, run_dir)
            rec = evaluate(case, rendered, run_dir, do_judge=False)
            records.append(rec)
            print(f"[harness] {case['id']}: rendered ({rec.get('render_seconds','-')}s)")
    finally:
        kira_pause(False)
    # JUDGE PHASE: after all renders — the VLM shares the GPU and starves
    # if called while the video model is rendering.
    if not args.no_judge:
        print("[harness] judge phase")
        for rec, case in zip(records, selected):
            if rec.get("output"):
                for attempt in range(3):
                    j = judge_mod.judge(rec["output"], case["prompt"], run_dir)
                    if not j.get("error"):
                        break
                    time.sleep(10)
                rec["judge"] = j
                _apply_judge_bands(rec, case)
    for rec in records:
        with open(os.path.join(run_dir, "results.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[harness] {rec['id']}: {rec['verdict']}"
              + (f" — {rec['failures']}" if rec.get("failures") else ""))
    write_report(records, run_dir)
    print(f"[harness] done: {sum(1 for r in records if r['verdict']=='PASS')}/{len(records)} PASS")
    print(f"[harness] report: {run_dir}/report.md")


if __name__ == "__main__":
    main()
