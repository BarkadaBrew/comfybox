# ComfyBox serve + Gatekeeper — what actually matters (canonical)

**Established 2026-08-15.** Rebuilding `.build/release/ComfyBox` and
restarting the `com.barkadabrew.comfybox` LaunchAgent can get the process
SIGKILL'd by launchd before it logs a single line (`launchd[1] ...
signaled service for aborting trampoline: Killed: 9`), with no crash
report. This is Gatekeeper rejecting the binary's code identity, not a bug
in the binary. `scripts/deploy-server.sh` / `resign-comfybox.sh` already
fix this correctly — this doc exists because it's easy to misdiagnose
which check actually gates launchd and go down a much longer road than
necessary.

## The fix (already implemented, use these scripts)

Sign with the **Developer ID Application** identity (`Developer ID
Application: Todd Walderman (STHPB624H2)`), not ad-hoc (`codesign --sign -`)
and not a self-signed cert. Ad-hoc signatures are content-hash-derived, so
every rebuild gets a NEW TCC identity — launchd trampoline-kills it, and any
previously-granted TCC permission (Local Network, external volume access)
is orphaned too, since it was tied to the old identity.

```
scripts/deploy-server.sh   # build + sign + restart, drains an in-flight render first
scripts/resign-comfybox.sh # re-sign an already-built binary + restart (no rebuild)
```

Both must run in the **Mac GUI Terminal, not over SSH** — the Developer ID
private key only releases from the GUI login session's Keychain.

## The trap: `spctl -a` says "rejected" — that does NOT mean launchd will kill it

```
$ spctl -a -vvv .build/release/ComfyBox
.build/release/ComfyBox: rejected
source=Unnotarized Developer ID
```

This looks damning and leads straight to "I need full Apple notarization"
(App Store Connect API key or app-specific password, `xcrun notarytool
submit`, a submit-and-wait network round trip per build). **Don't chase
that** — verified empirically 2026-08-15: a Developer-ID-signed (NOT
notarized) binary, restarted via plain `launchctl bootout` + `bootstrap`,
runs completely fine under launchd — stable PID, `LastExitStatus = 0` —
**while `spctl -a` on that exact same binary still reports "rejected,
Unnotarized Developer ID" at the same time.**

Conclusion: `spctl -a`'s standalone executable-assessment policy and
launchd's own trampoline check are NOT the same policy. `spctl -a` requires
full notarization; launchd's trampoline check does not — Developer ID
signing alone is sufficient. If you rediscover the "aborting trampoline:
Killed 9" failure, the fix is re-signing with Developer ID, NOT setting up
a notarization pipeline. `spctl -a`'s verdict on this binary will keep
saying "rejected" even after the fix works — that's expected, ignore it.

## Also don't disable Gatekeeper globally

`sudo spctl --master-disable` "solves" this but is a blunt, system-wide
security tradeoff, and on recent macOS additionally needs a System
Settings confirmation step whose location isn't reliably discoverable
(hit this 2026-08-15, gave up after two failed attempts). Not needed —
the Developer ID fix above is scoped to just this binary and doesn't
touch system policy at all.

## Applying launchd plist changes: use bootout+bootstrap, not kickstart

`launchctl kickstart -k` restarts the already-loaded job but does not
reliably re-read `ProgramArguments`/`EnvironmentVariables` changes from
the plist file on disk — confirmed 2026-08-15 (a `--ltx2-gemma` arg
change silently didn't take effect across several `kickstart -k` restarts,
each one still logging the old value). To apply a plist edit:

```
launchctl bootout gui/501/com.barkadabrew.comfybox
launchctl bootstrap gui/501 ~/Library/LaunchAgents/com.barkadabrew.comfybox.plist
```

`kickstart -k` is fine for "restart with no config change" (e.g., picking
up a freshly-signed binary at the same path) — just not for applying
edits to the plist itself.

## Related

- `docs/methods/gemma-encoder-ab-2026-08-15.md` — the `--ltx2-gemma`
  recipe decision (base wins) this plist should reflect.
- `scripts/deploy-server.sh`, `scripts/resign-comfybox.sh`,
  `scripts/resign-when-built.sh` — the actual deploy tooling.
