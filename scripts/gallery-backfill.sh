#!/usr/bin/env bash
# Rebuild the catalog from what is on disk.
#
# Reads three trees: the ComfyBox gallery home on this Mac, and the two studio
# trees on the server, reached over SMB at /Volumes/todd. NEVER reads any path
# containing "Vaults" — Bree's vault is out of scope, and it is genuinely
# reachable from /Volumes/todd, so the exclusion is load-bearing rather than
# theoretical.
#
# The studio archives are NOT local. Sidecars, the render journal and both
# history files record paths in the SERVER spelling (/home/todd/...), which
# exists nowhere on this Mac. The *-remote-prefix flags are what let the sweep
# translate them; without them every i2v edge and every journal row misses
# silently and the sweep reports success having linked nothing.
set -euo pipefail

BIN="${BIN:-.build/release/ComfyBoxGallery}"
HOME_TREE="${HOME_TREE:-$HOME/Pictures/ComfyBox}"
MOUNT="${MOUNT:-/Volumes/todd}"
KIRA_STUDIO="${KIRA_STUDIO:-$MOUNT/.kira/studio}"
BREE_STUDIO="${BREE_STUDIO:-$MOUNT/.bree/studio}"
RENDER_JOURNAL="${RENDER_JOURNAL:-$MOUNT/.kira/render-journal.jsonl}"
KIRA_HISTORY="${KIRA_HISTORY:-$KIRA_STUDIO/history.json}"
BREE_HISTORY="${BREE_HISTORY:-$BREE_STUDIO/history.json}"
KIRA_REMOTE="${KIRA_REMOTE:-/home/todd/.kira/studio}"
BREE_REMOTE="${BREE_REMOTE:-/home/todd/.bree/studio}"

if [[ ! -x "$BIN" ]]; then
  echo "build first: swift build -c release --product ComfyBoxGallery" >&2
  exit 1
fi

args=(backfill --home "$HOME_TREE")

# An unmounted share is the failure mode that looks most like success: the sweep
# finds nothing, reports zero, and exits 0. Skip a missing tree loudly instead.
if [[ -d "$KIRA_STUDIO" ]]; then
  args+=(--kira-studio "$KIRA_STUDIO" --kira-remote-prefix "$KIRA_REMOTE")
  [[ -f "$RENDER_JOURNAL" ]] && args+=(--render-journal "$RENDER_JOURNAL")
  [[ -f "$KIRA_HISTORY" ]] && args+=(--kira-history "$KIRA_HISTORY")
else
  echo "WARNING: $KIRA_STUDIO not mounted — skipping Kira's archive" >&2
fi

if [[ -d "$BREE_STUDIO" ]]; then
  args+=(--bree-studio "$BREE_STUDIO" --bree-remote-prefix "$BREE_REMOTE")
  [[ -f "$BREE_HISTORY" ]] && args+=(--bree-history "$BREE_HISTORY")
else
  echo "WARNING: $BREE_STUDIO not mounted — skipping Bree's archive" >&2
fi

exec "$BIN" "${args[@]}" "$@"
