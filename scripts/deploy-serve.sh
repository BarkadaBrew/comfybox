#!/bin/zsh
# deploy-serve.sh — FDD-krea2-raw-recipe §7.3: versioned engine deploy.
#
# Builds the ComfyBox serve binary from THIS worktree, installs it as
# ~/.comfybox/bin/ComfyBox-<sha> with mlx.metallib beside it, repoints the
# `current` symlink, restarts the launchd agent, and runs the ordered smoke.
# Never writes into any worktree's .build/. Rollback is a named path:
#   ln -sfn ComfyBox-<previous-sha> ~/.comfybox/bin/current && launchctl kickstart -k gui/$UID/com.barkadabrew.comfybox
#
# Usage: scripts/deploy-serve.sh [--no-pause] [--smoke-only]
# Requires the launchd plist ProgramArguments[0] to be ~/.comfybox/bin/current
# (one-time change, Todd's call — FDD §9 Q9). If it is not, the script stops
# after installing the binary and prints the plist edit instead of restarting.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
BIN_DIR=$HOME/.comfybox/bin
PLIST=$HOME/Library/LaunchAgents/com.barkadabrew.comfybox.plist
LABEL=com.barkadabrew.comfybox
PORT=7870
PAUSE=1; SMOKE_ONLY=0
for a in "$@"; do case $a in --no-pause) PAUSE=0;; --smoke-only) SMOKE_ONLY=1;; esac; done

sha=$(git -C "$ROOT" rev-parse --short HEAD)
say() { print -P "%F{cyan}[deploy $sha]%f $*"; }
fail() { print -P "%F{red}[deploy $sha] FAIL:%f $*" >&2; exit 1; }

health() { curl -sf --max-time 10 "http://127.0.0.1:$PORT/health"; }
queue_pause()  { curl -sf -X POST --max-time 10 "http://127.0.0.1:$PORT/v1/queue/pause"  >/dev/null; }
queue_resume() { curl -sf -X POST --max-time 10 "http://127.0.0.1:$PORT/v1/queue/resume" >/dev/null; }

smoke() {
  say "smoke a) LTX2 face-anchor fix present in build: git log"
  git -C "$ROOT" log --oneline -50 | grep -q c39136a || fail "c39136a (face-anchor mask pad) missing from this build's history"
  say "smoke c) /health reachable"
  local h; h=$(health) || fail "/health not reachable"
  print -- "$h" | python3 -c 'import sys,json; d=json.load(sys.stdin); print("  model_family", d.get("model_family"), "variant", d.get("model_variant"), "build_sha", d.get("build_sha"))'
  say "smoke d) unknown sampler must 400"
  local code; code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 -X POST "http://127.0.0.1:$PORT/v1/generate" -H 'content-type: application/json' -d '{"prompt":"x","scheduler":"uni_pc","steps":1,"width":64,"height":64}')
  [[ "$code" == "400" ]] || fail "POST /v1/generate scheduler=uni_pc returned $code, expected 400 (WP-E4 not live?)"
  say "smoke e) build_sha matches deployed sha"
  print -- "$h" | python3 -c "import sys,json; d=json.load(sys.stdin); s=d.get('build_sha') or ''; sys.exit(0 if s.startswith('$sha') else 1)" || fail "/health build_sha != $sha (clobbered binary?)"
  say "smoke b/f) seed-44821 turbo SHA-256 fixture and a Krita default-style render are manual until WP-E3/E19 land — see FDD §7.3"
}

if (( SMOKE_ONLY )); then smoke; exit 0; fi

say "1) Todd was told before this ran (pause is visible; codesign can prompt)."
if (( PAUSE )); then say "2) pausing queue"; queue_pause || fail "could not pause queue"; fi

say "3) building release in $ROOT (never rm -rf .build)"
( cd "$ROOT" && swift build -c release --product ComfyBox 2>&1 | tail -3 )
src="$ROOT/.build/release/ComfyBox"; [[ -x "$src" ]] || fail "no binary at $src"

say "4) face-anchor fix check"; git -C "$ROOT" log --oneline -50 | grep -q c39136a || fail "c39136a missing"

say "5) installing $BIN_DIR/ComfyBox-$sha"
mkdir -p "$BIN_DIR"
/bin/cp -f "$src" "$BIN_DIR/ComfyBox-$sha"
metallib="$ROOT/.build/release/mlx.metallib"
[[ -f "$metallib" ]] || metallib="$HOME/Projects/zimage.swift/.build/release/mlx.metallib"
[[ -f "$metallib" ]] || fail "mlx.metallib not found (a clean .build never regenerates it — copy from the desktop app bundle)"
/bin/cp -f "$metallib" "$BIN_DIR/mlx.metallib"
codesign --force --sign - "$BIN_DIR/ComfyBox-$sha" 2>&1 | tail -1 || true
prev=$(readlink "$BIN_DIR/current" 2>/dev/null || echo none)
ln -sfn "ComfyBox-$sha" "$BIN_DIR/current"
say "   current -> ComfyBox-$sha (previous: $prev)"

if ! grep -q "$BIN_DIR/current" "$PLIST"; then
  say "PLIST NOT REPOINTED (Q9). Binary installed; to activate, set ProgramArguments[0] in $PLIST to $BIN_DIR/current, then: launchctl bootout gui/$UID/$LABEL; launchctl bootstrap gui/$UID $PLIST"
  (( PAUSE )) && { say "resuming queue (nothing restarted)"; queue_resume || true; }
  exit 2
fi

say "6) restarting $LABEL"
launchctl bootout "gui/$UID/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$UID" "$PLIST"
for i in {1..60}; do health >/dev/null 2>&1 && break; sleep 2; done
health >/dev/null || fail "engine did not come back within 120s — rollback: ln -sfn $prev $BIN_DIR/current && launchctl kickstart -k gui/$UID/$LABEL"

say "7) smoke"; smoke
if (( PAUSE )); then say "8) resuming queue"; queue_resume || fail "RESUME FAILED — resume manually, pause persists across restarts"; fi
health | python3 -c 'import sys,json; d=json.load(sys.stdin); print("  is_paused", d.get("is_paused"), "status", d.get("status"))'
say "done"
