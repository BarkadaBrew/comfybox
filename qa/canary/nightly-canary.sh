#!/bin/bash
# Nightly video-quality canary: one anchor render + metrics vs bands.
# Installed as com.barkadabrew.video-canary (launchd, 04:15 daily).
# Writes ~/.comfybox/canary/last.json and alerts (macOS notification) on
# regression or failure. Pauses/resumes Kira around the render.
set -u
REPO=/Users/toddwalderman/Projects/zimage.swift
OUT=~/.comfybox/canary
mkdir -p "$OUT"
TS=$(date +%Y%m%d-%H%M)
LOG="$OUT/run-$TS.log"
exec >> "$LOG" 2>&1
echo "canary start $TS"

fail() {
  echo "CANARY FAIL: $1"
  /usr/bin/osascript -e "display notification \"$1\" with title \"ComfyBox video canary FAILED\"" 2>/dev/null
  printf '{"ok": false, "reason": "%s", "ts": "%s"}\n' "$1" "$TS" > "$OUT/last.json"
  curl -s -m 10 -X POST http://localhost:3787/v1/kira/content-scheduler/resume >/dev/null 2>&1
  exit 1
}

curl -s -m 5 http://127.0.0.1:7870/health >/dev/null 2>&1 || fail "server down"
R=$(curl -s -m 5 http://127.0.0.1:7870/v1/queue | python3 -c "import sys,json;print(json.load(sys.stdin)['is_rendering'])" 2>/dev/null)
[ "$R" = "True" ] && { echo "busy — skipping tonight"; exit 0; }
curl -s -m 10 -X POST http://localhost:3787/v1/kira/content-scheduler/pause >/dev/null 2>&1

PROMPT="standing sex, he thrusts into her petite body in a steady fluid rhythm, her hips rocking with each thrust, breasts moving, penetration clearly visible, she reacts, natural continuous coherent motion, sharp photorealistic skin, fine detail"
NEG="subtitle, caption, text, text on screen, watermark, logo, timestamp"
JOB=$(curl -s -m 30 -X POST http://127.0.0.1:7870/v1/video/generate/async -H "Content-Type: application/json" \
  -d "{\"prompt\":\"$PROMPT\",\"negative_prompt\":\"$NEG\",\"image_path\":\"$REPO/qa/video/assets/seed-girl.png\",\"width\":384,\"height\":640,\"frames\":49,\"seed\":43,\"backend\":\"local\",\"content_mode\":\"avocado\",\"strength\":1.0,\"enhance\":false,\"character\":\"\"}" \
  | python3 -c "import sys,json;print(json.load(sys.stdin).get('job_id',''))")
[ -z "$JOB" ] && fail "submit failed"

for i in $(seq 1 60); do
  sleep 20
  S=$(curl -s -m 10 "http://127.0.0.1:7870/v1/video/status/$JOB" | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('status',''),d.get('output_path',''))" 2>/dev/null)
  case "$S" in succeeded*|completed*) OUTPATH="${S#* }"; break;; failed*) fail "render failed";; esac
done
[ -z "${OUTPATH:-}" ] && fail "render timeout"

M=$(python3 "$REPO/qa/video/metrics.py" "$OUTPATH" "$REPO/qa/video/assets/seed-girl.png")
echo "$M" > "$OUT/metrics-$TS.json"
rm -f "$OUTPATH"
VERDICT=$(echo "$M" | python3 -c "
import sys, json
d = json.load(sys.stdin)
bad = []
if not d['validity']['valid']: bad.append('corrupt frames')
if d['sharp_mean'] < 28: bad.append('sharp %.1f < 28' % d['sharp_mean'])
if d['flicker_index'] > 0.40: bad.append('flicker %.2f > 0.40' % d['flicker_index'])
if d['motion'] < 1.0: bad.append('motion %.2f < 1.0' % d['motion'])
print(';'.join(bad) if bad else 'OK')")
curl -s -m 10 -X POST http://localhost:3787/v1/kira/content-scheduler/resume >/dev/null 2>&1
if [ "$VERDICT" != "OK" ]; then fail "band regression: $VERDICT"; fi
printf '{"ok": true, "ts": "%s"}\n' "$TS" > "$OUT/last.json"
echo "canary OK"
