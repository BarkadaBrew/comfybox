#!/bin/bash
# setup-kira-tunnel.sh — durable SSH tunnel to the kira-daemon (comfybox#240)
#
# The kira-daemon's control API listens on 127.0.0.1:3787 ON THE SERVER; the
# Kira tab in CoffeeShop Desktop binds to 127.0.0.1:3787 ON THE MAC. Until the
# Kira Muse migration (Workstream B) moves the daemon onto the Mac, this
# launchd agent keeps a local-forward tunnel up across drops and reboots:
#
#   Mac 127.0.0.1:3787  ──ssh -L──▶  server 127.0.0.1:3787
#
# Self-healing: ExitOnForwardFailure + ServerAlive make a dead tunnel EXIT,
# and launchd KeepAlive respawns it (ThrottleInterval prevents tight loops).
# After the migration, run `uninstall` — the tab's binding needs no change.
#
# Usage: setup-kira-tunnel.sh [install|uninstall|status]
set -euo pipefail

LABEL="com.barkadabrew.kira-tunnel"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
REMOTE="todd@10.0.100.232"
PORT=3787
LOG="/tmp/kira-tunnel.log"

status() {
  if launchctl print "gui/$(id -u)/$LABEL" > /dev/null 2>&1; then
    echo "agent: loaded"
    launchctl print "gui/$(id -u)/$LABEL" | grep -E '^\s+(state|pid)' | head -2
  else
    echo "agent: not loaded"
  fi
  if curl -s --max-time 3 "http://127.0.0.1:$PORT/health" | grep -q '"name":"kira"'; then
    echo "tunnel: kira-daemon reachable on 127.0.0.1:$PORT"
  else
    echo "tunnel: NOT reachable on 127.0.0.1:$PORT"
  fi
}

uninstall() {
  launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
  /bin/rm -f "$PLIST"
  echo "uninstalled $LABEL"
}

install() {
  # A pre-existing manual tunnel would hold the port and make the agent flap.
  pkill -f "ssh.*-L 127.0.0.1:$PORT:127.0.0.1:$PORT" 2>/dev/null || true
  pkill -f "ssh.*-L $PORT:127.0.0.1:$PORT" 2>/dev/null || true

  cat > "$PLIST" <<PLIST_EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>$LABEL</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/bin/ssh</string>
        <string>-N</string>
        <string>-o</string><string>BatchMode=yes</string>
        <string>-o</string><string>ExitOnForwardFailure=yes</string>
        <string>-o</string><string>ServerAliveInterval=15</string>
        <string>-o</string><string>ServerAliveCountMax=3</string>
        <string>-o</string><string>ConnectTimeout=10</string>
        <string>-o</string><string>StrictHostKeyChecking=accept-new</string>
        <string>-L</string><string>127.0.0.1:$PORT:127.0.0.1:$PORT</string>
        <string>$REMOTE</string>
    </array>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key><true/>
    <key>ThrottleInterval</key><integer>10</integer>
    <key>StandardOutPath</key><string>$LOG</string>
    <key>StandardErrorPath</key><string>$LOG</string>
</dict>
</plist>
PLIST_EOF

  launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
  launchctl bootstrap "gui/$(id -u)" "$PLIST"
  echo "installed + started $LABEL"
  sleep 3
  status
}

case "${1:-install}" in
  install) install ;;
  uninstall) uninstall ;;
  status) status ;;
  *) echo "usage: $0 [install|uninstall|status]"; exit 1 ;;
esac
