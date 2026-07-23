#!/bin/bash
# Waits for the in-progress HEAD build, then Developer-ID re-signs + restarts ComfyBox.
# RUN IN THE MAC GUI TERMINAL, then you can sleep. Claude takes it from there.
cd ~/Projects/zimage.swift
BIN=".build/release/ComfyBox"
IDENT="Developer ID Application: Todd Walderman (STHPB624H2)"
LOG=/tmp/comfybox_head_build.log

echo "Waiting for the HEAD build to finish (up to ~15 min)..."
for i in $(seq 1 180); do
  grep -q BUILD_OK "$LOG" 2>/dev/null && { echo "Build done."; break; }
  grep -q BUILD_FAIL "$LOG" 2>/dev/null && { echo "!! BUILD FAILED — see $LOG"; exit 1; }
  sleep 5
done
grep -q BUILD_OK "$LOG" 2>/dev/null || { echo "!! build did not finish in time — check $LOG"; exit 1; }

echo "Signing with Developer ID (may prompt — click Always Allow)..."
xattr -cr "$BIN"
codesign --force --sign "$IDENT" --identifier com.barkadabrew.comfybox "$BIN" || { echo "!! sign failed"; exit 1; }
codesign -dv "$BIN" 2>&1 | grep -iE "Authority=Developer|TeamIdentifier" || { echo "!! not Developer-ID signed"; exit 1; }

echo "Restarting engine + MCP bridge..."
launchctl kickstart -k "gui/$(id -u)/com.barkadabrew.comfybox"
pkill -f "ComfyBox mcp" 2>/dev/null || true

echo ""
echo "DONE — HEAD deployed, Developer-ID signed, restarted. You can sleep."
echo "Claude will verify the int8 quant loads and work through the night."
