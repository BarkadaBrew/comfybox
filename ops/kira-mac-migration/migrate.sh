#!/bin/bash
# migrate.sh — Kira daemon: Linux box (10.0.100.232) -> this Mac.  SHAPE 1: same
# codebase, systemd --user unit becomes a launchd user agent.
#
# DEFAULT MODE IS DRY-RUN: prints the plan and runs read-only preflight checks.
# NOTHING destructive happens without --execute, and every destructive step
# additionally asks for an explicit y/N confirmation. Idempotent: safe to re-run;
# completed steps are detected and skipped or re-verified.
#
# Usage:
#   ./migrate.sh                 # dry-run: plan + preflight + rsync --dry-run size report
#   ./migrate.sh --execute       # run the migration, step by step, confirmations on
#   ./migrate.sh --execute --enable-telegram
#                                # FINAL step only: flip telegram back on after the
#                                # box unit is stopped (see Telegram warning below)
#
# Read RUNBOOK.md (same directory) before running with --execute.

set -euo pipefail

# ── Constants ─────────────────────────────────────────────────────────────────
BOX="todd@10.0.100.232"                       # NOTE: box login shell is fish
BOX_KIRA="/home/todd/.kira"
BOX_UNIT="kira-daemon.service"
MAC_KIRA="$HOME/.kira"
CHECKOUT="$HOME/Projects/coffeeshop-server"
NODE_BIN="$HOME/.local/bin/node"              # verified: /usr/bin/node does not exist on Mac
PLIST_SRC="$(cd "$(dirname "$0")" && pwd)/com.barkadabrew.kira-daemon.plist"
PLIST_DST="$HOME/Library/LaunchAgents/com.barkadabrew.kira-daemon.plist"
TUNNEL_PLIST="$HOME/Library/LaunchAgents/com.barkadabrew.kira-tunnel.plist"
DAEMON_LABEL="com.barkadabrew.kira-daemon"
TUNNEL_LABEL="com.barkadabrew.kira-tunnel"
KIRA_PORT=3787                                # final port: what the tunnel/Kira-Suite tab already use
COMFYBOX_BIN="$HOME/Projects/zimage.swift/.build/release/ComfyBox"
UID_N="$(id -u)"

EXECUTE=0
ENABLE_TELEGRAM=0
for a in "$@"; do
  case "$a" in
    --execute) EXECUTE=1 ;;
    --enable-telegram) ENABLE_TELEGRAM=1 ;;
    *) echo "unknown arg: $a"; exit 2 ;;
  esac
done

say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
note() { printf '   %s\n' "$*"; }
fail() { printf '\033[31mFAIL:\033[0m %s\n' "$*"; exit 1; }

confirm() {  # confirm "<prompt>" — returns 0 only in --execute mode after explicit yes
  [ "$EXECUTE" -eq 1 ] || { note "[dry-run] would ask: $1"; return 1; }
  printf '\033[33m?? %s [y/N] \033[0m' "$1"
  read -r ans
  [ "$ans" = "y" ] || [ "$ans" = "Y" ]
}

# ── Plan ──────────────────────────────────────────────────────────────────────
say "PLAN (execute=$EXECUTE)"
cat <<'EOF'
   a. Preflight: node version, checkout present, npm ci + build (dist/kira-daemon.js)
   b. rsync box:/home/todd/.kira -> ~/.kira  (backup existing ~/.kira first;
      --dry-run + size report always shown before the real copy)
   c. Rewrite ~/.kira/config.json for Mac-local operation:
        port -> 3787, hosts.{comfybox,lmStudio} -> 127.0.0.1,
        comfybox.baseUrl -> http://127.0.0.1:7870, comfybox.transport -> "local",
        comfybox.mcpBinary -> absolute ComfyBox path, MCP ssh command -> local spawn,
        /home/todd path strings -> Mac equivalents, telegram.enabled -> false (phase 1)
   e1. Retire the kira-tunnel launchd agent (it squats 127.0.0.1:3787)
   e2. Bootstrap com.barkadabrew.kira-daemon launchd agent
   f. Health verification: /health on :3787, scheduler tick in log, image filed in studio
   d. ONLY after (f) passes: stop + disable the box systemd unit
   g. (--enable-telegram, separate invocation) flip telegram back on + kickstart
   TELEGRAM WARNING: two daemons long-polling one bot token = getUpdates 409 conflict.
   Phase 1 runs the Mac daemon with telegram DISABLED; only after the box unit is
   stopped (d) do we enable telegram (g). Do not reorder.
EOF

# ── (g) enable-telegram fast path ─────────────────────────────────────────────
if [ "$ENABLE_TELEGRAM" -eq 1 ]; then
  say "(g) Enable Telegram on the Mac daemon"
  ssh -o BatchMode=yes -o ConnectTimeout=5 "$BOX" "systemctl --user is-active $BOX_UNIT" 2>/dev/null | grep -q '^active$' \
    && fail "box unit is still ACTIVE — stop it first (step d). Two pollers on one bot token conflict."
  if confirm "Set telegram.enabled=true in $MAC_KIRA/config.json and restart the Mac daemon?"; then
    python3 - "$MAC_KIRA/config.json" <<'PY'
import json,sys
p=sys.argv[1]; c=json.load(open(p))
c.setdefault('telegram',{})['enabled']=True
json.dump(c,open(p,'w'),indent=2)
print('telegram.enabled -> true')
PY
    launchctl kickstart -k "gui/$UID_N/$DAEMON_LABEL"
    note "watch /tmp/kira-daemon.log for Telegram poll start; expect NO 409s"
  fi
  exit 0
fi

# ── (a) Preflight ─────────────────────────────────────────────────────────────
say "(a) Preflight"
[ -x "$NODE_BIN" ] || fail "node not found at $NODE_BIN"
NODE_V="$("$NODE_BIN" --version)"; note "node: $NODE_V ($NODE_BIN)"
case "$NODE_V" in v2[2-9].*|v[3-9]*) : ;; *) fail "node >= 22 required, got $NODE_V" ;; esac
[ -d "$CHECKOUT" ] || fail "checkout missing: $CHECKOUT"
[ -f "$CHECKOUT/systemd/kira-daemon.service" ] || fail "checkout has no systemd/kira-daemon.service — wrong branch?"
[ -x "$COMFYBOX_BIN" ] || fail "ComfyBox binary missing: $COMFYBOX_BIN"
ssh -o BatchMode=yes -o ConnectTimeout=5 "$BOX" "true" || fail "cannot ssh to $BOX (BatchMode)"
note "box reachable over ssh"
ssh -o BatchMode=yes "$BOX" "systemctl --user is-active $BOX_UNIT" | sed 's/^/   box unit: /' || true

if [ ! -f "$CHECKOUT/dist/kira-daemon.js" ]; then
  note "dist/kira-daemon.js MISSING (dist is stale — verified Jun 29 build predates the kira entry)"
  if confirm "Run 'npm ci && npm run build' in $CHECKOUT now?"; then
    (cd "$CHECKOUT" && npm ci && npm run build)
  else
    note "[required before launch] npm ci && npm run build in $CHECKOUT"
  fi
fi
[ -f "$CHECKOUT/dist/kira-daemon.js" ] && note "dist/kira-daemon.js present"

# check the Mac isn't already running port squatters (other than the tunnel we retire in e1)
if lsof -nP -iTCP:$KIRA_PORT -sTCP:LISTEN 2>/dev/null | grep -v '^COMMAND' | grep -qv ssh; then
  note "WARNING: something other than the ssh tunnel is listening on :$KIRA_PORT"
  lsof -nP -iTCP:$KIRA_PORT -sTCP:LISTEN || true
fi

# ── (b) rsync state home ──────────────────────────────────────────────────────
say "(b) Sync $BOX:$BOX_KIRA -> $MAC_KIRA"
note "size on box:"
ssh -o BatchMode=yes "$BOX" "du -sh $BOX_KIRA" || true
note "rsync dry-run (always shown):"
rsync -a --dry-run --stats "$BOX:$BOX_KIRA/" "$MAC_KIRA/" | sed -n '/^Number of files/,/^Total bytes/p' || true

if confirm "Proceed with real rsync? (existing $MAC_KIRA will be backed up first)"; then
  if [ -d "$MAC_KIRA" ]; then
    BK="$HOME/.kira.pre-migration.$(date +%Y%m%d-%H%M%S)"
    note "backing up existing $MAC_KIRA -> $BK"
    note "(the existing ~/.kira is a Jul-17 pre-staged local-test config, incl. its dummy token setup)"
    cp -a "$MAC_KIRA" "$BK"
  fi
  # No --delete: additive sync. Re-runs are cheap and idempotent.
  rsync -a --info=stats2 "$BOX:$BOX_KIRA/" "$MAC_KIRA/"
  note "rsync complete"
fi

# ── (c) Rewrite config.json ───────────────────────────────────────────────────
say "(c) Rewrite $MAC_KIRA/config.json for Mac-local operation"
if [ -f "$MAC_KIRA/config.json" ] && confirm "Rewrite config.json (a .bak is written alongside)?"; then
  python3 - "$MAC_KIRA/config.json" "$KIRA_PORT" "$COMFYBOX_BIN" "$HOME" <<'PY'
import json, sys, shutil
path, port, comfybox_bin, home = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
shutil.copy2(path, path + '.bak')
c = json.load(open(path))

# 1. Generic path/host string rewrite, applied to every string in the tree.
#    10.0.100.134 IS this Mac, so 127.0.0.1 is always a safe substitution here.
SUBS = [
    ('/home/todd/.kira',            home + '/.kira'),
    ('/home/todd/coffeeshop-server', home + '/Projects/coffeeshop-server'),
    ('10.0.100.134',                '127.0.0.1'),
]
def rewrite(o):
    if isinstance(o, dict):  return {k: rewrite(v) for k, v in o.items()}
    if isinstance(o, list):  return [rewrite(v) for v in o]
    if isinstance(o, str):
        for a, b in SUBS: o = o.replace(a, b)
        return o
    return o
c = rewrite(c)

# 2. Structured settings (idempotent sets).
c['port'] = port                       # tunnel is retired; daemon takes :3787 itself
c['bindAddress'] = c.get('bindAddress', '127.0.0.1')
c.setdefault('hosts', {})
c['hosts']['comfybox'] = '127.0.0.1'   # config-hosts.ts resolvers (#1319)
c['hosts']['lmStudio'] = '127.0.0.1'
cb = c.setdefault('comfybox', {})
cb['baseUrl'] = 'http://127.0.0.1:7870'
cb['transport'] = 'local'              # applyComfyBoxTransport (#1318): local MCP spawn,
cb['mcpBinary'] = comfybox_bin         # SCP file-retrieval auto-disabled
cb['mcpPort'] = 7870
cb.pop('sshHost', None)                # unused on the local path; remove to avoid confusion
# 3. Belt-and-suspenders: rewrite the verbatim ssh MCP command too (the ssh path
#    spawns config's command byte-for-byte; transport:'local' already overrides
#    the comfybox entry, but keep config self-describing).
for s in c.get('mcp', {}).get('servers', []):
    if s.get('id') == 'comfybox':
        s['command'] = f'{comfybox_bin} mcp --port 7870'
# 4. TELEGRAM PHASE 1: disabled until the box unit is stopped (step d) —
#    two daemons polling one bot token = getUpdates 409 conflict.
c.setdefault('telegram', {})['enabled'] = False

json.dump(c, open(path, 'w'), indent=2)
print('rewrote', path, '(backup at .bak); telegram DISABLED for phase 1')
PY
  note "verify botTokenPath in config points at a file that now exists under $MAC_KIRA"
  python3 -c "import json;c=json.load(open('$MAC_KIRA/config.json'));import os;[print('   botTokenPath:',b.get('botTokenPath'),'exists=',os.path.exists(os.path.expanduser(str(b.get('botTokenPath'))))) for b in c.get('telegram',{}).get('bots',[])]" || true
fi

# ── (e1) Retire the tunnel ────────────────────────────────────────────────────
say "(e1) Retire kira-tunnel (it forwards Mac 127.0.0.1:$KIRA_PORT -> box; conflicts with local daemon)"
if launchctl print "gui/$UID_N/$TUNNEL_LABEL" >/dev/null 2>&1; then
  if confirm "bootout + remove $TUNNEL_LABEL?"; then
    launchctl bootout "gui/$UID_N/$TUNNEL_LABEL" || true
    mv "$TUNNEL_PLIST" "$TUNNEL_PLIST.retired" 2>/dev/null || true
    note "tunnel retired (plist kept as .retired for rollback)"
  fi
else
  note "tunnel not loaded — nothing to do"
fi

# ── (e2) Bootstrap the Mac launchd agent ──────────────────────────────────────
say "(e2) Bootstrap $DAEMON_LABEL"
if launchctl print "gui/$UID_N/$DAEMON_LABEL" >/dev/null 2>&1; then
  note "already loaded"
  confirm "kickstart -k (restart) the loaded agent?" && launchctl kickstart -k "gui/$UID_N/$DAEMON_LABEL"
elif confirm "Install $PLIST_SRC -> $PLIST_DST and bootstrap?"; then
  cp "$PLIST_SRC" "$PLIST_DST"
  launchctl bootstrap "gui/$UID_N" "$PLIST_DST"
  note "bootstrapped; logs -> /tmp/kira-daemon.log"
fi

# ── (f) Health verification ───────────────────────────────────────────────────
say "(f) Health verification (Mac daemon, telegram still disabled)"
if [ "$EXECUTE" -eq 1 ]; then
  note "waiting for /health on :$KIRA_PORT ..."
  ok=0
  for _ in $(seq 1 30); do
    if curl -fsS -m 2 "http://127.0.0.1:$KIRA_PORT/health" >/dev/null 2>&1; then ok=1; break; fi
    sleep 2
  done
  [ "$ok" -eq 1 ] || fail "/health never came up — see /tmp/kira-daemon.log; box unit left UNTOUCHED"
  curl -fsS "http://127.0.0.1:$KIRA_PORT/health" | head -c 400; echo
  note "MANUAL GATES (do not skip):"
  note " 1. scheduler tick: tail -f /tmp/kira-daemon.log until a scheduler/media-pool cycle logs"
  note "    (first tick fires ~5s after start; content cycles per configured interval)"
  note " 2. one image filed: trigger a render (Kira tab / suggestion box) or wait a cycle, then:"
  note "    find $MAC_KIRA/studio -newer /tmp/kira-daemon.log -type f | head"
  confirm "Both gates green — Mac daemon verified healthy?" || fail "verification not confirmed; stopping before step (d)"
else
  note "[dry-run] would poll http://127.0.0.1:$KIRA_PORT/health, then require manual tick+image gates"
fi

# ── (d) Stop the box unit — LAST, and only after (f) ─────────────────────────
say "(d) Stop + disable the box systemd unit (rollback-friendly: disable, don't delete)"
if confirm "ssh $BOX: systemctl --user stop $BOX_UNIT && systemctl --user disable $BOX_UNIT ?"; then
  ssh -o BatchMode=yes "$BOX" "systemctl --user stop $BOX_UNIT && systemctl --user disable $BOX_UNIT && systemctl --user is-active $BOX_UNIT; true"
  note "box unit stopped+disabled (unit file left in place for rollback)"
  note "NEXT: re-run  ./migrate.sh --execute --enable-telegram  to flip Telegram on (step g)"
else
  note "box unit left running — Mac daemon is live in parallel WITHOUT telegram (safe)"
fi

say "Done. See RUNBOOK.md for post-cutover checks and rollback."
