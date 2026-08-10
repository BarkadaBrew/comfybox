#!/bin/bash
# Re-sign ComfyBox with Todd's Developer ID so it regains TCC access to /Volumes/Bolt.
# RUN THIS IN THE MAC GUI TERMINAL (Terminal.app) — NOT over SSH.
# The Developer ID signing key can only be released by the GUI login session.
set -e
cd ~/Projects/zimage.swift
BIN=".build/release/ComfyBox"
IDENT="Developer ID Application: Todd Walderman (STHPB624H2)"

echo "==> 1/5 clearing extended attributes"
xattr -cr "$BIN"

echo "==> 2/5 signing with Developer ID (may prompt to allow keychain access — click Always Allow)"
codesign --force --sign "$IDENT" --identifier com.barkadabrew.comfybox "$BIN"

echo "==> 3/5 verifying signature"
codesign -dv "$BIN" 2>&1 | grep -iE "Authority=Developer|TeamIdentifier" || { echo "!! still not Developer-ID signed"; exit 1; }

echo "==> 4/5 restarting the ComfyBox engine"
launchctl kickstart -k "gui/$(id -u)/com.barkadabrew.comfybox"

echo "==> 5/5 restarting the MCP bridge (daemon respawns it with the new binary)"
# Anchored to the built binary path — a bare "ComfyBox mcp" pattern matches
# any command line containing the words (Kimi review 2026-07-27). This matches
# both the relative and absolute invocation of .build/release/ComfyBox.
pkill -f '\.build/release/ComfyBox mcp' 2>/dev/null || true

echo ""
echo "DONE. ComfyBox re-signed with your Developer ID + restarted."
echo "Tell Claude \"done\" and it will verify Bolt access + fire a test render."
