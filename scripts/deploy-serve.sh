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
  git -C "$ROOT" merge-base --is-ancestor c39136a HEAD || fail "c39136a (face-anchor mask pad) missing from this build's history"
  say "smoke c) /health reachable"
  local h; h=$(health) || fail "/health not reachable"
  print -- "$h" | python3 -c 'import sys,json; d=json.load(sys.stdin); print("  model_family", d.get("model_family"), "variant", d.get("model_variant"), "build_sha", d.get("build_sha"))'
  say "smoke d) unknown sampler must 400"
  local code; code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 -X POST "http://127.0.0.1:$PORT/v1/generate" -H 'content-type: application/json' -d '{"prompt":"x","scheduler":"uni_pc","steps":1,"width":64,"height":64}')
  [[ "$code" == "400" ]] || fail "POST /v1/generate scheduler=uni_pc returned $code, expected 400 (WP-E4 not live?)"
  say "smoke e) build_sha matches deployed sha (WP-E10: stamped by scripts/gen-build-info.sh before the build)"
  local bs; bs=$(print -- "$h" | python3 -c "import sys,json; print(json.load(sys.stdin).get('build_sha') or '')")
  if [[ -z "$bs" || "$bs" == "unknown" ]]; then fail "/health build_sha is '${bs:-absent}' — the running binary was not stamped (pre-E10, or built without scripts/gen-build-info.sh); binary identity unverifiable"
  elif [[ "$bs" != "$sha"* ]]; then fail "/health build_sha=$bs != $sha (clobbered binary?)"; fi
  say "smoke c2) /health carries last_recipe + model_alias keys (WP-E10 sinks; null until the first krea2 render)"
  print -- "$h" | python3 -c 'import sys,json; d=json.load(sys.stdin); assert "last_recipe" in d and "model_alias" in d, "last_recipe/model_alias missing from /health"; print("  model_alias", d.get("model_alias"), "last_recipe", "present" if d.get("last_recipe") else "null")' || fail "/health lacks the WP-E10 keys"
  say "smoke b/f) seed-44821 turbo SHA-256 fixture and a Krita default-style render are manual until WP-E3/E19 land — see FDD §7.3"
}

if (( SMOKE_ONLY )); then smoke; exit 0; fi

say "1) Todd was told before this ran (pause is visible; codesign can prompt)."
PAUSED=0
STAMPED=0
on_exit() {
  # WP-E10: never leave a real sha in the committed BuildInfo.swift.
  if (( STAMPED )); then "$ROOT/scripts/gen-build-info.sh" --reset >/dev/null 2>&1 || print -P "%F{red}gen-build-info --reset FAILED — run it by hand before committing%f" >&2; fi
  if (( PAUSED )); then say "exit: resuming queue (pause persists across restarts — never leave it paused)"; queue_resume && PAUSED=0 || print -P "%F{red}RESUME FAILED — run: curl -X POST http://127.0.0.1:$PORT/v1/queue/resume%f" >&2; fi
}
trap on_exit EXIT
if (( PAUSE )); then say "2) pausing queue"; queue_pause || fail "could not pause queue"; PAUSED=1; fi

say "3) stamping build sha into Sources/ZImage/Support/BuildInfo.swift (reset on exit), then building release in $ROOT (never rm -rf .build)"
"$ROOT/scripts/gen-build-info.sh" || fail "gen-build-info failed"; STAMPED=1
( cd "$ROOT" && swift build -c release --product ComfyBox 2>&1 | tail -3 )
src="$ROOT/.build/release/ComfyBox"; [[ -x "$src" ]] || fail "no binary at $src"

say "4) face-anchor fix check"; git -C "$ROOT" merge-base --is-ancestor c39136a HEAD || fail "c39136a missing"

say "5) installing $BIN_DIR/ComfyBox-$sha"
mkdir -p "$BIN_DIR"
/bin/cp -f "$src" "$BIN_DIR/ComfyBox-$sha"
metallib="$ROOT/.build/release/mlx.metallib"
[[ -f "$metallib" ]] || metallib="$HOME/Projects/zimage.swift/.build/release/mlx.metallib"
[[ -f "$metallib" ]] || fail "mlx.metallib not found (a clean .build never regenerates it — copy from the desktop app bundle)"
/bin/cp -f "$metallib" "$BIN_DIR/mlx.metallib"
# Sign with the STABLE Developer ID identity and a STABLE identifier so macOS
# permission grants (Removable Volumes, Local Network) persist across deploys.
# An ad-hoc signature with the sha in the identifier is a NEW code identity every
# deploy: on 2026-09-05 that silently dropped the Removable-Volumes grant and the
# engine blocked forever in open() on /Volumes/Bolt with nobody there to approve
# the prompt (comfybox#395). Same policy as scripts/deploy-server.sh.
IDENT="Developer ID Application: Todd Walderman (STHPB624H2)"
if security find-identity -v -p codesigning 2>/dev/null | grep -q "$IDENT"; then
  codesign --force --sign "$IDENT" --identifier "$LABEL" "$BIN_DIR/ComfyBox-$sha" 2>&1 | tail -1 || true
  say "   signed with the stable identity ($LABEL)"
else
  codesign --force --sign - --identifier "$LABEL" "$BIN_DIR/ComfyBox-$sha" 2>&1 | tail -1 || true
  print -P "%F{red}[deploy $sha] WARN: stable identity missing — ad-hoc signed; Removable-Volumes/Local-Network grants will re-prompt%f" >&2
fi
prev=$(readlink "$BIN_DIR/current" 2>/dev/null || echo none)
ln -sfn "ComfyBox-$sha" "$BIN_DIR/current"
say "   current -> ComfyBox-$sha (previous: $prev)"

if ! grep -q "$BIN_DIR/current" "$PLIST"; then
  say "PLIST NOT REPOINTED (Q9). Binary installed; to activate, set ProgramArguments[0] in $PLIST to $BIN_DIR/current, then: launchctl bootout gui/$UID/$LABEL; launchctl bootstrap gui/$UID $PLIST"
  exit 2  # trap resumes the queue
fi

say "6) restarting $LABEL"
launchctl bootout "gui/$UID/$LABEL" 2>/dev/null || true
# Confirm the label is fully unregistered before bootstrap — otherwise
# bootstrap races a half-registered label and fails "5: Input/output error",
# leaving the engine DOWN (observed 2026-08-29). See reference: bootout first,
# confirm gone, then bootstrap.
for i in {1..15}; do launchctl print "gui/$UID/$LABEL" >/dev/null 2>&1 || break; sleep 1; done
launchctl print "gui/$UID/$LABEL" >/dev/null 2>&1 && fail "label still registered after bootout — engine left stopped; run: launchctl bootout gui/$UID/$LABEL; launchctl bootstrap gui/$UID $PLIST"
launchctl bootstrap "gui/$UID" "$PLIST"
for i in {1..180}; do health >/dev/null 2>&1 && break; sleep 2; done
health >/dev/null || fail "engine did not come back within 360s (cold krea2 load takes ~4min) — rollback: ln -sfn $prev $BIN_DIR/current && launchctl kickstart -k gui/$UID/$LABEL"

say "7) smoke"; smoke
if (( PAUSED )); then say "8) resuming queue"; queue_resume && PAUSED=0 || fail "RESUME FAILED — resume manually, pause persists across restarts"; fi
health | python3 -c 'import sys,json; d=json.load(sys.stdin); print("  is_paused", d.get("is_paused"), "status", d.get("status"))'
# Provenance: tag the deployed commit (annotated, best-effort push). Pairs with
# /health build_sha so a deploy can be found in git history without the ledger.
tag="deploy-$(date +%Y-%m-%d)-$sha"
if git -C "$ROOT" tag -a "$tag" -m "engine deployed $(date -Iseconds) as ComfyBox-$sha" 2>/dev/null; then
  git -C "$ROOT" push -q origin "$tag" 2>/dev/null && say "   tagged $tag (pushed)" || say "   tagged $tag (push failed — push it later)"
fi
say "done"
