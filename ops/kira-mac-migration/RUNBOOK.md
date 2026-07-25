# Kira → Mac Migration Runbook (SHAPE 1: move the daemon, unchanged codebase)

Move the `kira-daemon` Node service from the Linux box (`todd@10.0.100.232`,
systemd `--user` unit) to this Mac (launchd user agent), keeping the
coffeeshop-server codebase unchanged. The Mac already hosts everything Kira
talks to (ComfyBox warm server :7870, LM Studio :1234), so post-migration all
of Kira's traffic becomes loopback.

Kit contents (this directory):

| File | Purpose |
|---|---|
| `com.barkadabrew.kira-daemon.plist` | launchd user agent (Mac equivalent of `systemd/kira-daemon.service`) |
| `migrate.sh` | step-by-step migration; **dry-run by default**, destructive steps need `--execute` + per-step y/N |
| `RUNBOOK.md` | this document |

---

## 1. What was VERIFIED vs ASSUMED

Verified by reading the Mac checkout `~/Projects/coffeeshop-server` (HEAD `e3abc20b`):

- **Entrypoint**: `systemd/kira-daemon.service` → `ExecStart=/usr/bin/node %h/coffeeshop-server/dist/kira-daemon.js`, `Environment=BREE_HOME=%h/.kira`, `EnvironmentFile=-/etc/coffeeshop/secrets.env`, `WorkingDirectory=%h/coffeeshop-server`, `Restart=always/RestartSec=5`. `dist/kira-daemon.js` is a dedicated esbuild entry (see `build.mjs`) that **fail-closes unless config role is `companion`** and calls `startDaemon(false)` (no Claude session pool).
- **Mac node**: `/Users/toddwalderman/.local/bin/node` v22.22.3. **`/usr/bin/node` does not exist on the Mac** — the plist uses the absolute `.local/bin` path.
- **Mac `dist/` is stale** (built Jun 29, predates the kira entry): `dist/kira-daemon.js` is **missing**. `npm ci && npm run build` is mandatory (preflight handles it).
- **Migration switches already exist in the code** (`src/config-hosts.ts`, Workstream B #1318/#1319): `hosts.comfybox/lmStudio = "127.0.0.1"` and `comfybox.transport = "local"`. `transport:"local"` rewrites the comfybox MCP entry to a bare local spawn (`<mcpBinary> mcp --port 7870`) and **auto-disables SCP file retrieval** (outputs are already local). With no config the defaults reproduce the box's current remote behavior byte-for-byte.
- **A pre-staged `~/.kira` already exists on the Mac** (Jul 17): a local-test config — port **3799**, loopback URLs, `transport:"local"`, and a **dummy** Telegram token path (`/tmp/kira-mac-dummy-token`). It is NOT the authoritative state. `migrate.sh` backs it up to `~/.kira.pre-migration.<ts>` before rsync.
- **Tunnel agent**: `~/Library/LaunchAgents/com.barkadabrew.kira-tunnel.plist` forwards Mac `127.0.0.1:3787 → box:3787` (ssh -N -L). Verified contents; see §4.
- **ComfyBox binary**: `~/Projects/zimage.swift/.build/release/ComfyBox` (also on PATH at `~/bin/ComfyBox`). The box's MCP command today is the ssh form: `ssh … toddwalderman@10.0.100.134 "cd …/zimage.swift && .build/release/ComfyBox mcp --port 7870"`.

Assumed (could NOT verify without touching the box — check during dry-run):

- **A1 — Box state shape**: `/home/todd/.kira` contains `config.json`, `studio/`, `activity/`, `campaigns/`, `logs/`, ledgers, and a real `botTokenPath` file *inside* `~/.kira` (so rsync carries it). If the token file lives OUTSIDE `/home/todd/.kira`, copy it manually and fix `botTokenPath` after step (c). The rewrite step prints whether the path exists on the Mac — check that line.
- **A2 — Box daemon port is 3787** (tunnel + Kira Suite tab + memory all agree, but the box `config.json` is the truth). If it differs, adjust `KIRA_PORT` in `migrate.sh`.
- **A3 — secrets.env dependency**: the box unit sources `/etc/coffeeshop/secrets.env` (optional, `-` prefixed). Its schema (`secrets.env.example`) is Bree-oriented (bridge token, Bree bot token, Replicate/OpenAI/fal keys). Kira's Telegram token comes from `botTokenPath` in config, **but Kira's voice TTS is `provider: "replicate"`** — if the Replicate key reaches Kira via secrets.env on the box, create `~/.kira/secrets.env` on the Mac with `REPLICATE_API_KEY=…` (the plist wrapper sources it, chmod 600). Key value lives on the box at `~/.bree/config.json` per memory.
- **A4 — Disk space**: studio galleries hold images+videos; check the `du -sh` the dry-run prints before syncing.
- **A5 — The exact scheduler/tick log line text** — verification gate (f) says "watch the log for a scheduler/media-pool cycle" rather than grepping an exact string.

---

## 2. Procedure

```
cd <repo>/ops/kira-mac-migration
./migrate.sh                # DRY-RUN: plan, preflight, rsync --dry-run + size report
./migrate.sh --execute      # steps a,b,c,e1,e2,f then d — each gated by y/N
./migrate.sh --execute --enable-telegram    # step g, ONLY after d
```

Order of operations (and why):

1. **(a) Preflight** — node ≥22 at the `.local/bin` path, checkout present, `npm ci && npm run build` (produces `dist/kira-daemon.js`), ComfyBox binary present, box reachable over ssh, port :3787 audit.
2. **(b) rsync** `box:/home/todd/.kira/ → ~/.kira/` — dry-run + size report first; existing Mac `~/.kira` backed up; additive (`-a`, no `--delete`), so re-runs are idempotent. Run it while the box daemon is still up; rsync again right before cutover if you want the freshest journals (cheap delta).
3. **(c) Config rewrite** (`config.json.bak` kept): `/home/todd` paths → Mac paths; `10.0.100.134` → `127.0.0.1` (the Mac is .134 — always safe); `port → 3787`; `hosts.* → 127.0.0.1`; `comfybox.baseUrl → http://127.0.0.1:7870`; `transport → "local"` + absolute `mcpBinary`; the verbatim ssh MCP command → `<ComfyBox> mcp --port 7870`; **`telegram.enabled → false`** (phase 1 — see §5).
4. **(e1) Retire the tunnel** — must happen before the daemon can bind :3787 (§4).
5. **(e2) Bootstrap** the launchd agent; logs at `/tmp/kira-daemon.log`.
6. **(f) Verify** — `/health` 200 on `127.0.0.1:3787`; one scheduler tick observed in the log (first tick ~5s after start); one image filed under `~/.kira/studio` (trigger a render from the Kira tab or wait a content cycle). Manual y/N gate.
7. **(d) Stop the box** — `systemctl --user stop kira-daemon && systemctl --user disable kira-daemon` (unit file left in place for rollback). Only offered after (f) is confirmed.
8. **(g) Enable Telegram** — separate invocation; refuses to run if the box unit still reports `active`. Watch the log for clean polling (no 409s).

Post-cutover checks:

- Kira Suite tab (⌘K) works against local :3787 — no tunnel involved.
- MCP comfybox tools work (a render lands) — proves the local spawn path.
- Journals/ledgers under `~/.kira` gain new entries with Mac-local timestamps.
- `launchctl print gui/$(id -u)/com.barkadabrew.kira-daemon` shows `state = running` and restarts after `kill <pid>` (KeepAlive).

---

## 3. Rollback

Any point before step (d): just bootout the Mac agent — the box was never touched.

```bash
launchctl bootout gui/$(id -u)/com.barkadabrew.kira-daemon
mv ~/Library/LaunchAgents/com.barkadabrew.kira-tunnel.plist.retired \
   ~/Library/LaunchAgents/com.barkadabrew.kira-tunnel.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.barkadabrew.kira-tunnel.plist
```

After step (d) (full rollback):

```bash
# 1. Stop the Mac daemon FIRST (Telegram single-instance — §5 applies in reverse)
launchctl bootout gui/$(id -u)/com.barkadabrew.kira-daemon
# 2. Re-enable the box unit
ssh todd@10.0.100.232 "systemctl --user enable kira-daemon && systemctl --user start kira-daemon && systemctl --user status kira-daemon --no-pager"
# 3. Restore the tunnel (as above) so the Kira Suite tab reaches the box again
# 4. Config restore is NOT needed on the box (its /home/todd/.kira was only read).
#    State divergence: anything Kira filed on the Mac between cutover and rollback
#    (studio images, journal entries) exists only in Mac ~/.kira — rsync it BACK
#    selectively if it matters:  rsync -an ~/.kira/studio/ todd@10.0.100.232:/home/todd/.kira/studio/  (drop -n after review)
# Mac-side config rollback if you re-try later: ~/.kira/config.json.bak and ~/.kira.pre-migration.<ts>
```

---

## 4. Port implications — the kira-tunnel MUST be retired

`com.barkadabrew.kira-tunnel` (verified) runs `ssh -N -L 127.0.0.1:3787:127.0.0.1:3787 todd@10.0.100.232` with KeepAlive. Consequences:

- While it's loaded, **Mac 127.0.0.1:3787 is taken** — the local daemon cannot bind it (or worse, health checks would silently hit the *box* daemon through the tunnel and "verify" the wrong instance).
- Therefore step (e1) retires the tunnel **before** the local daemon starts, and the local daemon takes :3787 itself so every existing consumer (Kira Suite tab, desktop probes) keeps working unchanged.
- The retirement was already planned: the tunnel memory note says "uninstall after Workstream B migration" — this is that moment. The plist is kept as `*.retired` for rollback.
- The pre-staged Mac test config used port **3799** precisely to coexist with the tunnel; the migration ends that workaround — final port is **3787**.

## 5. Telegram single-instance warning

Telegram long-polling (`getUpdates`) allows **one** consumer per bot token. Two daemons polling Kira's token = constant `409 Conflict`, dropped updates, and both instances flapping. This is why the sequence is strict:

1. Mac daemon comes up with `telegram.enabled=false` (step c does this) → safe parallel running during verification.
2. Box unit is stopped (d) **before** Telegram is enabled on the Mac (g). `--enable-telegram` refuses to proceed while the box unit reports active.
3. Same rule in reverse during rollback: stop the Mac daemon before restarting the box unit.

Also: only ONE outreach scheduler may own `telegram-outreach.json` pacing — parallel running with telegram off also keeps outreach off, so this is covered by the same gate.

## 6. The clock / NTP lesson

Kira's behavior is wall-clock-sensitive: outreach quiet hours (22:00–06:00), rituals (07:00 / 21:30), media-pool build time (02:30), journal ordering, and `rsync -a` preserved mtimes. Cross-host clock skew between box and Mac has bitten before (skew makes quiet-hours/ritual windows fire wrong and makes "which entry is newer" ambiguous across the migration boundary). **(ASSUMED: the original incident details weren't recoverable from this repo — the guidance stands regardless.)** Before cutover:

```bash
# Mac: confirm NTP is on and offset is sane (< 1s)
sudo systemsetup -getusingnetworktime        # should be On
sntp time.apple.com                          # offset printout
# Box (read-only check): timedatectl | grep -E 'synchronized|Time zone'
```

Also confirm **time zones match** (journals store local times): box `timedatectl`, Mac `date +%Z`. If the box was UTC and the Mac is local time, scheduler-facing times in config (quiet hours, rituals, buildTime) will shift by the offset — re-check them after cutover.

---

## 7. Top risks before `--execute`

1. **Telegram double-poll** — enabling telegram on the Mac while the box unit still runs (or restoring the box without stopping the Mac). Mitigated by phase-1 disable + the active-check in `--enable-telegram`, but a manual config edit can bypass it.
2. **Token file doesn't survive the move** — if the box's `botTokenPath` points outside `/home/todd/.kira`, the rsync won't carry it and the rewritten path won't exist; Telegram will fail at (g). Check the `botTokenPath … exists=` line printed by step (c). Same for any Replicate/OpenAI keys Kira gets from `/etc/coffeeshop/secrets.env` (A3).
3. **Verifying the wrong daemon through the tunnel** — if the tunnel isn't retired first, `127.0.0.1:3787` is the *box* daemon; health checks pass against the wrong instance. Step order e1→e2→f exists for this; don't run steps by hand out of order.
4. **Stale build** — the Mac `dist/` predates the `kira-daemon.js` entry; skipping `npm run build` means launchd KeepAlive crash-loops on a missing file (and after any future `git pull`, rebuild before kickstart).
5. **Pre-existing Mac `~/.kira` clobber / state divergence** — the Jul-17 pre-staged tree gets overlaid by box state (backup is taken, but confirm it happened), and anything Kira writes on the Mac after cutover is NOT on the box — rollback after (d) loses it unless synced back (§3).
