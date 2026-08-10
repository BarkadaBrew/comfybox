#!/bin/bash
# GUI-session launchd agent: re-signs ComfyBox with Todd's Developer ID when a
# flag file appears. Runs in the Aqua session (has Security Server / keychain
# access that SSH sessions lack), so codesign with the Developer ID key succeeds
# headlessly. Triggered via WatchPaths on ~/.comfybox/resign-request.
FLAG="$HOME/.comfybox/resign-request"
LOG="$HOME/.comfybox/resign-agent.log"
[ -e "$FLAG" ] || exit 0
{
  echo "=== resign $(date) ==="
  cd "$HOME/Projects/zimage.swift" || { echo "no repo"; exit 1; }
  BIN=".build/release/ComfyBox"
  xattr -cr "$BIN" 2>/dev/null
  if codesign --force --sign "Developer ID Application: Todd Walderman (STHPB624H2)" --identifier com.barkadabrew.comfybox "$BIN"; then
    echo "SIGNED OK"
  else
    echo "SIGN FAILED rc=$?"
  fi
  codesign -dv "$BIN" 2>&1 | grep -iE "Authority=Developer|TeamIdentifier|adhoc"
  rm -f "$FLAG"
  echo "done"
} >> "$LOG" 2>&1
