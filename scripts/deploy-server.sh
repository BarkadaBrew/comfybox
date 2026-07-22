#!/bin/bash
# Build + restart the ComfyBox warm server (com.barkadabrew.comfybox), signed
# with the STABLE "Developer ID Application" identity so macOS permission
# grants (external volume access, Local Network, etc.) persist across
# rebuilds instead of re-prompting every time the ad-hoc signature changes.
set -e
IDENT="Developer ID Application: Todd Walderman (STHPB624H2)"
BIN=".build/release/ComfyBox"
LABEL="com.barkadabrew.comfybox"

cd "$(dirname "$0")/.."
swift build -c release --product ComfyBox

xattr -cr "$BIN"
if security find-identity -v -p codesigning 2>/dev/null | grep -q "$IDENT"; then
  codesign --force --sign "$IDENT" --identifier "$LABEL" "$BIN"
  echo "Signed with stable identity: $IDENT"
else
  codesign --force --sign - --identifier "$LABEL" "$BIN"
  echo "WARN: stable identity missing — fell back to ad-hoc (permissions will re-prompt)."
fi

# Drain before restart: a SIGTERM landing mid-render tears down MLX while a
# kernel is in flight and the process dies with "mutex lock failed" (observed
# 3x on 2026-07-15 — every "random" warm-server crash that day was actually a
# restart colliding with an active render). Wait for the current job to finish;
# new jobs queue behind it and survive the restart via queue persistence.
DRAIN_TIMEOUT="${DEPLOY_DRAIN_TIMEOUT:-900}"
waited=0
while [ "$waited" -lt "$DRAIN_TIMEOUT" ]; do
  rendering=$(curl -s --max-time 3 "http://127.0.0.1:7870/health" \
    | /usr/bin/python3 -c 'import json,sys; print(json.load(sys.stdin).get("is_rendering", False))' 2>/dev/null || echo "down")
  case "$rendering" in
    False|down) break ;;
    *)
      if [ "$waited" -eq 0 ]; then echo "Render in progress — draining (timeout ${DRAIN_TIMEOUT}s)..."; fi
      sleep 5; waited=$((waited + 5)) ;;
  esac
done
if [ "$waited" -ge "$DRAIN_TIMEOUT" ]; then
  echo "WARN: drain timed out after ${DRAIN_TIMEOUT}s — restarting anyway (render will be killed)."
fi

launchctl kickstart -k "gui/$(id -u)/$LABEL"
echo "Restarted $LABEL"
