#!/bin/bash
# Build + deploy CoffeeShop Desktop, signed with the STABLE identity so macOS
# permissions/Keychain persist. Run setup-signing-identity.sh once first.
set -e
IDENT="CoffeeShop Desktop Signing"
KC="$HOME/Library/Keychains/coffeeshop-signing.keychain-db"
APP="/Applications/CoffeeShop Desktop.app"
cd "$(dirname "$0")/.."
swift build -c release --product ComfyBoxDesktop
pkill -f "CoffeeShop Desktop" 2>/dev/null || true; sleep 1
/bin/cp -f .build/release/ComfyBoxDesktop "$APP/Contents/MacOS/ComfyBoxDesktop"
# Stamp the bundle so the app can say which build it is (Branding.swift reads
# CFBundleShortVersionString): short version = deploy date, build = git sha.
# Done BEFORE codesign so the signature covers the plist.
SHA=$(git rev-parse --short HEAD)
/usr/libexec/PlistBuddy -c "Set :CFBundleShortVersionString $(date +%Y.%-m.%-d)" -c "Set :CFBundleVersion $SHA" "$APP/Contents/Info.plist" \
  && echo "Stamped $(date +%Y.%-m.%-d) ($SHA)"
security unlock-keychain -p coffeeshop-local "$KC" 2>/dev/null || true
if security find-identity "$KC" 2>/dev/null | grep -q "$IDENT"; then
  codesign --force --deep --sign "$IDENT" --keychain "$KC" "$APP"
  echo "Signed with stable identity."
else
  codesign --force --deep --sign - "$APP"
  echo "WARN: stable identity missing — fell back to ad-hoc (run setup-signing-identity.sh)."
fi
touch "$APP"; killall Dock 2>/dev/null || true
echo "Deployed $APP"
