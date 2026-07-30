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
TS=$(date +%s)
sqlite3 ~/.comfybox/dam.sqlite3 ".backup ~/.comfybox/dam.sqlite3.bak-$TS"
sqlite3 ~/.comfybox/dam.sqlite3.bak-$TS "PRAGMA integrity_check; SELECT COUNT(*) FROM assets;"
chmod 600 ~/.comfybox/dam.sqlite3.bak-$TS
```

Expect `ok` and a plausible row count. A backup you have not verified is not a
backup.

**Housekeeping.** Opening a backup with the `sqlite3` CLI creates `-wal`/`-shm`
sidecars next to it. Delete them once the file is checkpointed (a 0-byte `-wal`
means it is), so nobody later mistakes them for part of the backup:

```bash
rm -f ~/.comfybox/dam.sqlite3.bak-$TS-wal ~/.comfybox/dam.sqlite3.bak-$TS-shm
```

---

## Rollback — restoring a backup

**Copying a backup over `dam.sqlite3` on its own will corrupt the database.** The
live `-wal` and `-shm` belong to the *replaced* database generation. SQLite will
happily replay that WAL onto the restored pages and produce a silently mixed
state — not an error, not a crash, just a database that is quietly part-old and
part-new. The stale files must be removed while **no process holds the database
open**.

```bash
# 1. Stop the catalog writer.
launchctl bootout gui/$(id -u)/com.barkadabrew.comfybox-gallery

# 2. STOP THE ENGINE — operator action, deliberately not scripted here.
#    com.barkadabrew.comfybox also writes to this file via DAMStore. Stop it
#    yourself, when in-flight renders allow. Confirm nothing holds the file:
lsof ~/.comfybox/dam.sqlite3          # must print nothing before continuing

# 3. Remove the stale WAL and shared-memory files.
rm -f ~/.comfybox/dam.sqlite3-wal ~/.comfybox/dam.sqlite3-shm

# 4. Restore, and verify BEFORE letting anything reopen it.
cp ~/.comfybox/dam.sqlite3.bak-<TS> ~/.comfybox/dam.sqlite3
chmod 600 ~/.comfybox/dam.sqlite3
sqlite3 ~/.comfybox/dam.sqlite3 "PRAGMA integrity_check; SELECT COUNT(*) FROM assets;"

# 5. Bring the catalog service back.
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.barkadabrew.comfybox-gallery.plist
curl -s http://127.0.0.1:7871/healthz     # {"ok":true}
```

Step 3 is the one that is easy to skip and the only one that loses data silently.

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
