# Gallery catalog runbook

The catalog lives at `~/.comfybox/dam.sqlite3` and is served by the
`com.barkadabrew.comfybox-gallery` launchd agent on `127.0.0.1:7871`.

> **The engine is not ours to restart.** `com.barkadabrew.comfybox` (port 7870) is
> the running GPU engine. Never `launchctl` it as part of a catalog operation —
> restarting it orphans in-flight renders. Where a procedure below needs it
> stopped, that is an explicit instruction for the operator, not something to
> script.

---

## Backing up

The database is in **WAL mode** and has two live writers (the gallery service and
the engine's `DAMStore`). A plain `cp dam.sqlite3 backup` is **not** a backup: it
copies the main file and leaves behind everything still sitting in the write-ahead
log, which is routinely megabytes. Restoring such a copy silently loses every
transaction the WAL still held.

Always use SQLite's own snapshot, which checkpoints into a single self-contained
file:

```bash
bash <<'EOF'
set -euo pipefail
DB="$HOME/.comfybox/dam.sqlite3"
BAK="$HOME/.comfybox/dam.sqlite3.bak-$(date +%s)"

sqlite3 "$DB" ".backup $BAK"                    # fails the block if it fails
[ -s "$BAK" ] || { echo "backup was not created — STOP" >&2; exit 1; }
sqlite3 "$BAK" "PRAGMA integrity_check;" | grep -qx ok \
  || { echo "backup is corrupt — STOP" >&2; exit 1; }

N=$(sqlite3 "$BAK" "SELECT COUNT(*) FROM assets;")
echo "backup holds $N assets"
chmod 600 "$BAK"
rm -f "$BAK-wal" "$BAK-shm"
echo "BACKUP OK: $BAK"
EOF
```

Only the final `BACKUP OK:` line means you have a backup. Every step before it
aborts the block on failure, so a `.backup` that fails can no longer be followed
by a cheerful success line for a file that does not exist.

> **`$HOME`, never `~`.** The destination of `.backup` is parsed by the *sqlite3
> shell*, not by bash, so a `~` inside that argument is expanded by neither. The
> command fails with `Error: cannot open "~/.comfybox/…"` and creates nothing.
> Loud rather than silent — but this is the first command of the rollback
> procedure, so get it right.

Expect `ok` and a plausible row count. A backup you have not verified is not a
backup.

The trailing `rm -f` clears the `-wal`/`-shm` sidecars the `sqlite3` CLI creates
next to the backup while reading it; once the file is checkpointed (a 0-byte
`-wal`) they are noise, and leaving them invites someone to mistake them for part
of the backup.

---

## Rollback — restoring a backup

**Copying a backup over `dam.sqlite3` on its own will corrupt the database.** The
live `-wal` and `-shm` belong to the *replaced* database generation. SQLite will
happily replay that WAL onto the restored pages and produce a silently mixed
state — not an error, not a crash, just a database that is quietly part-old and
part-new. The stale files must be removed while **no process holds the database
open**.

**Step 1 — stop the catalog writer.**

```bash
launchctl bootout gui/$(id -u)/com.barkadabrew.comfybox-gallery
```

**Step 2 — stop the engine. Operator action, deliberately not scripted.**

`com.barkadabrew.comfybox` also writes this file via `DAMStore`. Stop it yourself,
when in-flight renders allow. Nothing below is safe until it is down.

**Step 3 — run this block ON ITS OWN and read the result before going further.**
It aborts rather than merely reporting; a gate you can paste past is not a gate.

```bash
if lsof "$HOME/.comfybox/dam.sqlite3" 2>/dev/null | grep -q .; then
  lsof "$HOME/.comfybox/dam.sqlite3"
  echo "STILL OPEN — do not continue. Stop the writers listed above." >&2
else
  echo "clear — safe to restore"
fi
```

**Step 4 — choose the backup, and let the shell name it.** Do not hand-type a
timestamp. List what exists and copy one whole line:

```bash
ls -lt "$HOME"/.comfybox/dam.sqlite3.bak-* "$HOME"/.comfybox/dam.sqlite3.*bak-* 2>/dev/null
```

Then export it, and confirm it is the file you meant:

```bash
export BAK="$HOME/.comfybox/dam.sqlite3.bak-1785454572"   # <-- paste a REAL path
ls -l "$BAK" && sqlite3 "$BAK" "SELECT COUNT(*) FROM assets;"
```

> **Why this is not busywork.** `sqlite3` **creates** a database when handed a
> path that does not exist, and answers `PRAGMA integrity_check` on the new empty
> file with `ok`:
>
> ```
> $ sqlite3 ./nope.sqlite3 "PRAGMA integrity_check;"   # file did not exist
> ok
> $ ls -l ./nope.sqlite3
> -rw-r--r--  0 ./nope.sqlite3
> ```
>
> So an unedited `<TS>` placeholder, or a mistyped timestamp, used to sail
> through an `integrity_check` gate, make the subsequent `cp` succeed, and end
> with an **empty database moved over the live catalog** — the failure only
> surfacing later as `no such table: assets`. Step 5 now gates on existence,
> non-emptiness and an actual row count, in that order.

**Step 5 — restore.** Safe to paste as a whole. It runs in a subshell (`bash
<<'EOF'`) so an abort cannot close your terminal or leave `errexit` set for the
rest of the session, which a bare `set -e` plus `exit 1` would.

```bash
bash <<'EOF'
set -euo pipefail
DB="$HOME/.comfybox/dam.sqlite3"
: "${BAK:?set BAK to the backup you are restoring (see step 4)}"

# --- identity gates: prove the backup is real BEFORE destroying anything ------
[ -e "$BAK" ] || { echo "no such backup: $BAK — aborting" >&2; exit 1; }
[ -s "$BAK" ] || { echo "backup is EMPTY: $BAK — aborting" >&2; exit 1; }
sqlite3 "$BAK" "PRAGMA integrity_check;" | grep -qx ok \
  || { echo "backup fails integrity_check — aborting" >&2; exit 1; }
# The one check an auto-created empty file cannot pass.
N=$(sqlite3 "$BAK" "SELECT COUNT(*) FROM assets;" 2>/dev/null || echo 0)
[ "$N" -ge 100 ] || { echo "backup holds only $N assets — refusing (expected 100+)" >&2; exit 1; }
echo "backup verified: $N assets"

# --- the writers must be down; this is fatal, not advisory --------------------
lsof "$DB" 2>/dev/null | grep -q . && { echo "STILL OPEN — aborting" >&2; exit 1; }

# --- keep the CURRENT database before touching it ----------------------------
# Removing the WAL and overwriting in place are each irreversible on their own,
# so without this a restore that fails partway destroys BOTH generations.
PRE="$DB.pre-rollback-$(date +%s)"
sqlite3 "$DB" ".backup $PRE"
chmod 600 "$PRE"; rm -f "$PRE-wal" "$PRE-shm"
echo "pre-rollback snapshot: $PRE"

# --- stage, verify the COPY, then swap in one step ----------------------------
cp "$BAK" "$DB.incoming"
chmod 600 "$DB.incoming"
sqlite3 "$DB.incoming" "PRAGMA integrity_check;" | grep -qx ok \
  || { rm -f "$DB.incoming"; echo "copy is bad — aborting, live DB untouched" >&2; exit 1; }

rm -f "$DB-wal" "$DB-shm"     # stale WAL belongs to the REPLACED generation
mv "$DB.incoming" "$DB"       # atomic within the filesystem

sqlite3 "$DB" "PRAGMA integrity_check; SELECT COUNT(*) FROM assets;"
echo "RESTORE OK"
EOF
```

**Step 6 — bring the catalog service back.**

```bash
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.barkadabrew.comfybox-gallery.plist
curl -s http://127.0.0.1:7871/healthz     # {"ok":true}
```

**Step 7 — START THE ENGINE AGAIN. Operator action, mirroring step 2.**

You stopped `com.barkadabrew.comfybox` in step 2 and nothing above brings it
back. Start it yourself and confirm it is serving before you walk away:

```bash
launchctl print gui/$(id -u)/com.barkadabrew.comfybox | grep -E 'state|pid'
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:7870/v1/gallery/list?limit=1
```

Expect `state = running` and `200`. **The rollback is not finished until this
step is done** — otherwise the procedure ends with the GPU engine down.

The `rm -f` of `-wal`/`-shm` is the step that is easy to skip and the only one
that loses data silently — which is why it happens *after* both the pre-rollback
snapshot and the verification of the incoming copy, and immediately before the
atomic `mv`.

**The catalog is rebuildable by design.** If a restore is doubtful, the honest
move is to rebuild from disk rather than nurse a suspect file — every fact in it
comes from the media trees, the sidecars and the journals.

---

## Rebuilding from disk

```bash
swift build -c release --product ComfyBoxGallery   # NEVER build the ComfyBox product
./scripts/gallery-backfill.sh
```

The script reads the Mac gallery home plus both studio trees over SMB at
`/Volumes/todd`, and refuses any path containing `Vaults`. An unmounted share is
reported as a warning rather than silently sweeping nothing. Expect **~10–16
minutes**: it hashes ~4,700 files byte-for-byte across the network. That is not a
hang.

Idempotence: a second run reports `indexed: 0` and `edges: 0`. Do **not** expect
`SELECT COUNT(*) FROM assets` to be stable between runs — the engine writes rows
throughout, so the naive count always drifts. `indexed`/`edges` are the signal.

### Re-filing after a rules change

Derived filing runs inside `upsert`, which a re-sweep skips for any asset whose
data has not changed. So new `CollectionRules` file **nothing** on a plain
re-sweep, while still reporting success. After changing the rules:

```bash
./scripts/gallery-backfill.sh --refile
```

`--refile` re-derives filing for every row. It only ever deletes `manual = 0`
memberships, so hand-filings made in the gallery survive.

---

## Reading the report

`edgesUnresolved` and `assetsUnfiled` count **absences**, and are the only
counters that can tell you the sweep covered nothing. A run that reads no
sidecars at all reports `unresolved: 0` — which looks like success and is the
worst possible reading. Sanity-check against the filesystem, not against the
counters alone.

---

## Health checks

```bash
launchctl print gui/$(id -u)/com.barkadabrew.comfybox-gallery | grep -E 'state|pid'
curl -s http://127.0.0.1:7871/healthz
tail ~/.comfybox/gallery.err.log
```

`state = running` matters — a loaded plist whose program exits immediately is not
a running service. The server exits non-zero on a failed bind, so a port conflict
fails visibly rather than pretending to listen.

## Permissions

The catalog holds raw prompt text: `0600` on `dam.sqlite3`, `-wal` and `-shm`,
inside a `0700` `~/.comfybox`. Re-applied on every open. Apply the same `0600` to
any backup you take — it is the same data.
