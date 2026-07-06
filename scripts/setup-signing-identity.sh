#!/bin/bash
# Create a STABLE self-signed code-signing identity so the ad-hoc signature
# doesn't change every rebuild (which reprompts Keychain + Local Network perms).
# Idempotent. Run once per machine.
set -e
IDENT="CoffeeShop Desktop Signing"
KC="$HOME/Library/Keychains/coffeeshop-signing.keychain-db"
KCPW="coffeeshop-local"
if security find-identity "$KC" 2>/dev/null | grep -q "$IDENT"; then
  echo "Identity already present."; exit 0
fi
TMP=$(mktemp -d); trap "rm -rf $TMP" EXIT
openssl req -x509 -newkey rsa:2048 -sha256 -days 3650 -nodes \
  -keyout "$TMP/key.pem" -out "$TMP/cert.pem" -subj "/CN=$IDENT" \
  -addext "basicConstraints=critical,CA:false" \
  -addext "keyUsage=critical,digitalSignature" \
  -addext "extendedKeyUsage=critical,codeSigning"
openssl pkcs12 -export -legacy -inkey "$TMP/key.pem" -in "$TMP/cert.pem" \
  -out "$TMP/id.p12" -passout pass:$KCPW -name "$IDENT"
security create-keychain -p "$KCPW" "$KC" 2>/dev/null || true
security set-keychain-settings "$KC"
security unlock-keychain -p "$KCPW" "$KC"
security import "$TMP/id.p12" -k "$KC" -P "$KCPW" -A -T /usr/bin/codesign
security set-key-partition-list -S apple-tool:,apple:,codesign: -s -k "$KCPW" "$KC" >/dev/null
CURR=$(security list-keychains -d user | sed 's/"//g' | xargs)
echo "$CURR" | grep -q coffeeshop-signing || security list-keychains -d user -s $CURR "$KC"
echo "Created signing identity: $IDENT"
