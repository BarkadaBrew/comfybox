#!/usr/bin/env bash
# deploy-mlx-serve.sh — install/upgrade the Mac's mlx-serve launchd agent.
#
#   scripts/deploy-mlx-serve.sh <version>       # e.g. 26.8.11: expects ~/mlx-serve/<version>/mlx-serve
#   scripts/deploy-mlx-serve.sh --rollback      # restore the previous plist backup + previous `current`
#
# One binary on the machine (FDD D5): ~/mlx-serve/current -> <version>. The plist in
# ops/launchd/ is the source of truth. Glimmer is unavailable to Bree/Kira for ~15-40 s
# during the restart; the first 1-2 calls afterwards crawl while the DFlash drafter
# calibrates its cost table (expected). Verifies /health, the model list and /metrics.json.
set -euo pipefail
LABEL=com.barkadabrew.mlx-serve
AGENT=~/Library/LaunchAgents/$LABEL.plist
SRC="$(cd "$(dirname "$0")/.." && pwd)/ops/launchd/$LABEL.plist"
ROOT=~/mlx-serve
uid=$(id -u)
if [ "${1:-}" = "--rollback" ]; then
  bak=$(ls -t "$AGENT".bak-* 2>/dev/null | head -1); [ -n "$bak" ] || { echo "no plist backup"; exit 1; }
  prev=$(cat "$ROOT/current.prev" 2>/dev/null || true)
  launchctl bootout "gui/$uid/$LABEL" 2>/dev/null || true
  cp "$bak" "$AGENT"; [ -n "$prev" ] && ln -sfn "$prev" "$ROOT/current"
  launchctl bootstrap "gui/$uid" "$AGENT"; echo "rolled back to $(basename "$bak") current->$(readlink "$ROOT/current")"; exit 0
fi
ver=${1:?version, e.g. 26.8.11}
bin="$ROOT/$ver/mlx-serve"; [ -x "$bin" ] || { echo "missing $bin"; exit 1; }
got=$("$bin" --version 2>/dev/null | grep -oE '^mlx-serve [0-9.]+' | awk '{print $2}')
[ "$got" = "$ver" ] || { echo "binary reports $got, expected $ver"; exit 1; }
extra=$(ls -d "$ROOT"/[0-9]*/mlx-serve 2>/dev/null | grep -v "/$ver/" || true); [ -n "$extra" ] && echo "note: other versions present (rollback candidates): $extra"
[ -d "/Applications/MLX Core.app" ] && echo "WARNING: MLX Core.app is installed — it launches a second server on :11234 (FDD §0.6a). Uninstall it."
mkdir -p ~/.mlx-serve/models
[ -L "$ROOT/current" ] && readlink "$ROOT/current" > "$ROOT/current.prev" || true
ln -sfn "$ver" "$ROOT/current"
ts=$(date +%Y%m%d-%H%M%S); [ -f "$AGENT" ] && cp "$AGENT" "$AGENT.bak-$ts"
launchctl bootout "gui/$uid/$LABEL" 2>/dev/null || true
# bootout returns before the job is gone; bootstrap races it (error 5). Wait for the label to disappear.
for i in $(seq 1 30); do launchctl print "gui/$uid/$LABEL" >/dev/null 2>&1 || break; sleep 1; done
cp "$SRC" "$AGENT"; launchctl bootstrap "gui/$uid" "$AGENT"
for i in $(seq 1 90); do curl -s -m 2 http://127.0.0.1:11234/health 2>/dev/null | grep -q ok && break; sleep 1; done
curl -s -m 2 http://127.0.0.1:11234/health | grep -q ok || { echo "health never came up; rolling back"; "$0" --rollback; exit 1; }
for i in $(seq 1 180); do curl -s -m 3 http://127.0.0.1:11234/v1/models 2>/dev/null | grep -q '"Muse-Glimmer-30B-heretic-MLX-Q6"[^}]*"loaded":true' && break; sleep 1; done
echo "== $(curl -s http://127.0.0.1:11234/health) after ~${i}s; version $("$bin" --version | head -1)"
curl -s -m 5 http://127.0.0.1:11234/v1/models | python3 -c 'import sys,json; d=json.load(sys.stdin); [print(" ", m["id"], "|", m.get("state"), "| drafter", (m.get("meta") or {}).get("drafter_loaded")) for m in d["data"]]'
curl -s -m 5 http://127.0.0.1:11234/metrics.json | head -c 160; echo
grep -E '^\[args\]' ~/.mlx-serve/logs/mlx-serve-11234.log | tail -5
