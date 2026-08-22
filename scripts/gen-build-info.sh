#!/bin/zsh
# gen-build-info.sh — stamp the git short sha into Sources/ZImage/Support/BuildInfo.swift
# (WP-E10, FDD §3.10 sink 3 / §7.3 smoke step e: /health.build_sha).
#
# The Swift file is COMMITTED with the placeholder "unknown". This script
# rewrites exactly one line — the `gitSHA` constant — with the short sha of
# HEAD in THIS worktree (suffix `-dirty` when there are uncommitted changes),
# so a release build carries its own identity and a clobbered / wrong-branch
# binary is detectable from outside. `--reset` restores the placeholder;
# `--print` only prints what would be stamped.
#
# Usage:
#   scripts/gen-build-info.sh            # stamp HEAD's sha (run right before `swift build -c release`)
#   scripts/gen-build-info.sh --reset    # restore the committed placeholder
#   scripts/gen-build-info.sh --print    # print the sha that would be stamped
#
# deploy-serve.sh runs the stamp before the build and the reset from its EXIT
# trap, so the tree is never left dirty and the stamped line is never
# committed by accident. Never commit BuildInfo.swift with a real sha in it.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
FILE="$ROOT/Sources/ZImage/Support/BuildInfo.swift"
MARK='// gen-build-info:'
[[ -f "$FILE" ]] || { print -u2 "gen-build-info: $FILE not found"; exit 1; }

sha() {
  local s; s=$(git -C "$ROOT" rev-parse --short HEAD)
  if [[ -n "$(git -C "$ROOT" status --porcelain --untracked-files=no -- . ':!Sources/ZImage/Support/BuildInfo.swift' 2>/dev/null)" ]]; then
    s="${s}-dirty"
  fi
  print -- "$s"
}

stamp() {
  local value=$1
  # Replace the string literal on the marked line only.
  local tmp; tmp=$(mktemp)
  awk -v v="$value" -v mark="$MARK" '
    index($0, mark) && /public static let gitSHA = "/ {
      sub(/public static let gitSHA = "[^"]*"/, "public static let gitSHA = \"" v "\"")
    }
    { print }
  ' "$FILE" > "$tmp"
  grep -q "public static let gitSHA = \"$value\"" "$tmp" || { rm -f "$tmp"; print -u2 "gen-build-info: marked gitSHA line not found in $FILE"; exit 1; }
  mv "$tmp" "$FILE"
}

case "${1:-}" in
  --print) sha ;;
  --reset) stamp unknown; print "gen-build-info: reset to placeholder" ;;
  "")      s=$(sha); stamp "$s"; print "gen-build-info: stamped $s" ;;
  *)       print -u2 "usage: $0 [--reset|--print]"; exit 2 ;;
esac
