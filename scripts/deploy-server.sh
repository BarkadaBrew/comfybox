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

launchctl kickstart -k "gui/$(id -u)/$LABEL"
echo "Restarted $LABEL"
