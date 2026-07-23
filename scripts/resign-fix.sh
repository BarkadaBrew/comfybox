#!/bin/bash
# Fixes codesign errSecInternalComponent by authorizing the signing key for
# codesign, then signs + restarts. RUN IN THE MAC GUI TERMINAL.
cd ~/Projects/zimage.swift
BIN=".build/release/ComfyBox"
IDENT="Developer ID Application: Todd Walderman (STHPB624H2)"
KC="$HOME/Library/Keychains/login.keychain-db"

echo "==> Enter your Mac LOGIN PASSWORD at each prompt (authorizes the signing key):"
security unlock-keychain "$KC"
security set-key-partition-list -S apple-tool:,apple:,codesign: -s "$KC" >/dev/null 2>&1 \
  && echo "   key partition list set" || echo "   (partition-list step returned nonzero — continuing)"

echo "==> Signing..."
xattr -cr "$BIN"
if codesign --force --sign "$IDENT" --identifier com.barkadabrew.comfybox "$BIN"; then
  codesign -dv "$BIN" 2>&1 | grep -iE "TeamIdentifier"
  echo "==> Restarting engine + MCP..."
  launchctl kickstart -k "gui/$(id -u)/com.barkadabrew.comfybox"
  pkill -f "ComfyBox mcp" 2>/dev/null || true
  echo "DONE — signed + restarted. Tell Claude."
else
  echo "!! STILL FAILING. GUI fallback: open Keychain Access -> login keychain ->"
  echo "   your \"Developer ID Application: Todd Walderman\" PRIVATE KEY -> Get Info ->"
  echo "   Access Control -> \"Allow all applications to access this item\" -> Save."
  echo "   Then re-run this script."
  exit 1
fi
