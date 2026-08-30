# FDD: Headless parity — every UI control reachable by API

**Repo:** `BarkadaBrew/comfybox` (`zimage.swift`, Swift/MLX)
**Worktree/branch:** `~/Projects/zimage-apiparity` @ `feat/ui-api-parity` (base `origin/main` c9dd27d)
**Components:** `Sources/ZImage/Server/WarmServer.swift`, `Sources/ZImage/Server/ComfyBridge/ComfyBridge.swift`, `Sources/ZImage/MCP/*`, `Sources/ZImage/Server/ComfyBoxServerConfig.swift`, `Sources/ComfyBoxDesktop/Views/SettingsView.swift`; cross-repo: `coffeeshop-server/src/tools/`
**PRD:** `docs/PRD-ui-api-parity.md`
**Related:** comfybox#300 (async route starvation), comfybox#217 (actor head-of-line, fixed for `/health` + `GET /v1/queue`), comfybox#218 (unified-memory eviction), comfybox#1479 (LTX-2 preemption), coffeeshop-server#1293
**Author:** Fable (Opus), architect pass — 2026-08-29
**Status:** **v2 — revised after adversarial review (2026-08-29).** v1 shipped 5 blockers, three of them in sections v1 had marked "unverified" and guessed wrong on. Every §-heading changed in v2 is marked **[REVISED v2]**. Phase 0's mechanism question is now **closed by measurement**, not open.

**Scope note (Todd, 2026-08-29):** authentication/authorization is explicitly **out of scope and not a risk on this project** — this stack is never publicly reachable and all callers are trusted local operators or agents. The PRD's §7 "unauthenticated surface grows" risk is **withdrawn**. No token gate, no per-route permission model, no rate limiting. Input *validation* stays (malformed params, out-of-range values, unknown enum members must produce a clean `400`, not a trap) — that is correctness, not security.

---

## 0. v2 changelog [NEW]

What the review overturned, and where it landed:

| # | v1 claim | Reality | Now in |
|---|---|---|---|
| 1 | "Mechanism B is unverified; diagnose first" | **B1 confirmed by live `sample`:** 2964/2972 samples in `__psynch_cvwait` on `com.apple.root.utility-qos.cooperative` — MLX render work blocks *on the cooperative pool*. | §3.1, §4.1 — the executor fix is now **primary and first** |
| 2 | Command mailbox drained "at existing scheduling points" | **No such points exist.** `processLoop` (`:6965–6980`) *exits* (`isProcessing = false; return`) when paused with no `runsWhilePaused` job. A mailbox `resume` would 202 and wedge the queue forever. | §3.1.4 — deltas + explicit wake, `resume` never goes through the mailbox |
| 3 | Serve control reads from `LiveHealthState` | The snapshot is **written only on the actor** (`publishHealth()` `:6729`, all ~15 call sites actor-isolated). Mailbox writes would be invisible for the whole render — AC green, answers wrong. | §3.1.5 — `isPaused` + pending deltas become **authoritative in the lock store** |
| 4 | "Pause takes effect mid-render" | **Undeliverable.** #1479 preemption is LTX-2-only (`LTX2VideoGenerator.swift:288`); it's handoff-and-resume, not an indefinite park (parking pins latents against #218). And `isPaused` is a between-items gate — redefining it breaks a live endpoint. | §3.1.6 — AC dropped; mid-render abort scoped to `interrupt`, family-qualified |
| 5 | Migrate Desktop defaults to preserve client-side behavior | **`DesktopSettings.default{Steps,Guidance,Width,Height}` are write-only UI state** — nothing reads them to build a request. Migrating them would move Bree/MCP/Kira *off* engine defaults. | §3.3 — inverted: seed from the **engine's** fallbacks |
| 6 | Mirror adoption for queue mutations | `pending` has other writers (enqueue `:6929`, `recoverPersistedQueue` `:4310+`); wholesale adoption **drops jobs**. | §3.1.4 — deltas only |
| 7 | D5 pins parser by counting `case (` lines | 73 lines but **76 tuples** — 3 arms carry two (`:1620` pause+resume, `:1654`, `:1680`). A control route would be silently missed. Plus 2 false hits in comments. | §3.5 |
| 8 | Exempt ComfyBridge routes from parity | Can't — they're in a **second switch** (`ComfyBridge.swift:106`) the parser never sees, and include real mutating routes (`POST /queue` `:135`). | §3.5 |
| 9 | `If-Match` mandatory on `PUT /v1/config` | Breaks every current caller; none send it. Also `encode(to:)` `:154–171` writes only enumerated keys, so the rollback story was half-true. | §3.3 |
| 10 | Phases independently shippable | Phase 1's `update_config` would proxy the clobbering `PUT` that Phase 3 then replaces; Phase 4 depends on Phase 3's store. | §4 preamble |

---

## 1. Summary

The Desktop app is already an API client for most engine behavior. The defect is not "UI-only controls" — it is that the control surface is **fragmented across two hosts, split unevenly between HTTP and MCP, and self-describes nowhere.** An agent cannot answer "what can I change, and how?" without reading 9,680 lines of Swift.

| Phase | Delivers | Primary risk |
|---|---|---|
| **0** | Control routes answer during a render (#300) — via a coordinator executor change, then a narrow control-plane carve-out | Concurrency near the render path (now much smaller than v1 assumed) |
| **1** | MCP tool for every mutating warm-server route (minus `update_config`, held to Phase 3) | Low — additive |
| **2** | Kira scheduler controls reachable by agents | Cross-host coupling; a dated topology precondition |
| **3** | Engine defaults become server-side and writable; content modes writable; `ServerConfigStore` + `PATCH` | Changing engine behavior for *non-Desktop* callers |
| **4** | `GET /v1/controls` + generated `docs/api-reference.md` + anti-drift test | Registry rot; parser brittleness |

The through-line is the **ControlRegistry** (§3.4) — a compile-time table that the discovery route, the MCP tool schemas, the generated docs, and the parity test all read. Phases 1–3 populate it; Phase 4 exposes it.

---

## 2. Findings

**2.1 — #300 is two bugs, and the bigger one is now measured. [REVISED v2]** `/v1/queue/pause` hangs because it calls `await coordinator.setPaused()` (`:1619`) and `WarmServerCoordinator` (`:5417`) is an actor occupied for the whole synchronous render (`pipeline.generateFromRequest` is awaited *on* the actor, `:7313`). This is #217, documented in-repo at `:4700–4710`. **Separately**, `/v1/civitai/search` (`:4370`) hangs while its synchronous sibling `/v1/civitai/repo` (`:4425`) is healthy, despite never touching the coordinator. A live `sample` under render settles it: **2964 of 2972 samples sit in `__psynch_cvwait` on `com.apple.root.utility-qos.cooperative`** — the render blocks *on the cooperative pool*, so continuations for every unrelated `await` starve. Call these **Mechanism A** (actor head-of-line) and **Mechanism B1** (pool exhaustion). B1 is the larger and cheaper of the two to fix.

**2.2 — the repo already contains the read-side pattern, but it is actor-authored.** `LiveHealthState` (`:4750+`) is an `NSLock`-guarded snapshot read by `/health` and `GET /v1/queue` (`:2789`) with no actor hop — genuinely torn-read-free (single-lock swap, `:4756–4764`). **But `publishHealth()` (`:6729`) is actor-isolated and called from ~15 actor-isolated sites**; only `setProgress()` is written off-actor. So the snapshot is a *projection of actor state*, not a place off-actor code can record a fact. v1 missed this. Any control write that must be *observable during a render* has to be authoritative in the lock store, with the actor as a reader.

**2.3 — `PUT /v1/config` is worse than "can clobber."** Whole-document decode-and-`save()` (`:811–822`), no version, no in-memory store, and the running server reads most settings from a boot-captured `WarmServerConfiguration`. Two specifics v1 got half-right: `init(from:)` (`:133–151`) **is** tolerant of unknown keys, but `encode(to:)` (`:154–171`) writes **only enumerated keys** — so an older build performing a `GET`→`PUT` round-trip *deletes* any config block added by a newer build. Tolerant decode does not buy forward-compatible round-tripping.

**2.4 — Kira is genuinely cross-host, and the topology has a shelf life. [REVISED v2]** `KiraClient.swift:21–24` targets `127.0.0.1:3787` "reach it via `ssh -N -L`". Verified live on the Linux box: `kira-daemon.service` active; `GET /v1/kira/content-scheduler/status` returns the tier/video-mix document; the write endpoints exist and were confirmed in `kira-api.ts` (`:649, :658, :670, :681, :699, :704, :980, :985`). **This is a dated precondition** — a Kira→Mac migration is in progress, and if the daemon lands on the Mac, D2's co-location argument inverts and Phase 2's host should be revisited (§3.2).

**2.5 — engine defaults are family-dependent. [NEW v2]** `payload.width ?? 1024` / `?? 1024` / `steps ?? defaultSteps` / `guidance ?? defaultGuidance` at `:7413–7416`, but a *different* family path hardcodes `steps ?? 30` at `:7739`, and video is `width ?? preset ?? 704`, `height ?? preset ?? 448` (`:2229–2230`), `steps ?? preset ?? 8` (`:2380`). A single flat `renderDefaults` block would flatten family-aware behavior. This constrains D3's shape.

### 2.6 Route/tool inventory (@ c9dd27d)

- `respond(to:)`: 73 `case (` **lines** but **76 tuples** — `:1620`, `:1654`, `:1680` each carry two — plus ~20 prefix/suffix-matched arms. Two further `case (` occurrences are inside comments (`:271`, `:4891`).
- A **second dispatch switch** in `ComfyBridge.route()` (`ComfyBridge.swift:106`), reached *before* the main one (`WarmServer.swift:615`), containing mutating routes including `POST /queue` (`:135`).
- 44 MCP tools in `MCPToolRegistry.tools`.

**Mutating warm-server routes with no MCP tool (Phase 1 worklist):** `POST /v1/characters` + `PUT /v1/characters` + `DELETE /v1/characters/{id}`; `POST|PUT /v1/presets`, `DELETE /v1/presets/{id}`; `PUT /v1/config` *(held to Phase 3)*; `POST /v1/queue/{id}/move`, `DELETE /v1/queue/{id}`; `POST /v1/loras/import`, `POST /v1/loras/{id}/update`; `POST /v1/civitai/harvest`; `POST /v1/video/traces/{id}/{promote,rating}`; `DELETE /v1/workflows/{id}`; `POST /v1/generate/async`, `POST /v1/video/generate/async`.

**Read-only with no writer (Phase 3):** `GET /v1/content-modes` (`:1687`) — `ContentModeStore` has a working, uncalled `save()` (`ContentModeStore.swift:389`).

---

## 3. Design decisions

### 3.1 — D1: Phase 0 mechanism for #300 **[REVISED v2 — restructured]**

#### 3.1.1 The measurement

Confirmed by live `sample` of `comfybox serve` under a continuous render: **2964/2972 samples in `__psynch_cvwait` on `com.apple.root.utility-qos.cooperative`.** The MLX render, awaited on the coordinator actor, executes on a cooperative-pool worker and blocks it. The pool does not grow past its width, so unrelated continuations — URLSession completions for `/v1/civitai/search`, every other in-flight request task — have nowhere to run. v1 listed this as hypothesis "B1" and gated work on diagnosing it; it is now **fact**, and it reorders the phase.

#### 3.1.2 The decision: fix the executor first, then carve out only what's left

**0.C from v1 is promoted to task 0.A and ships first.** Give `WarmServerCoordinator` a dedicated serial-queue-backed executor:

```swift
private let coordinatorQueue = DispatchQueue(
  label: "z-image.warm-server.coordinator", qos: .userInitiated)
nonisolated var unownedExecutor: UnownedSerialExecutor { coordinatorQueue.asUnownedSerialExecutor() }
```

~20 lines. **Blast radius on the render path is zero**: it changes *which thread* serializes the actor, not *that* it serializes. Mutual exclusion, message ordering, and every invariant #218 (single heavy-model residency) and #1479 (preemption handoff) depend on are properties of actor serialization, which is preserved exactly. It fixes **all** async routes — including ones no classifier would ever have covered — rather than only a hand-listed set. Nothing about it touches MLX, the pipeline, or the queue loop.

Alongside it, one line the review caught: **the per-connection `DispatchQueue` (`:606`) is created with no QoS**, so it defaults low and its work is descheduled behind the render's `.userInitiated` compute. Pin it `.userInitiated`.

**Then re-measure.** After 0.A + the QoS pin, re-run the #300 repro. Everything that stalls afterwards stalls for **Mechanism A only** — a genuine actor hop behind the render. That residue, not v1's speculative list, defines 0.B's scope.

#### 3.1.3 0.B — the residual control-plane carve-out

Expected residue (routes that hop the actor and must answer during a render): `POST /v1/queue/{pause,resume,clear,interrupt}`, `POST /v1/queue/{id}/move`, `DELETE /v1/queue/{id}`, `GET /v1/models`, `GET /v1/stats`. Already actor-free and needing nothing: `/health`, `GET /v1/queue` (`:2789`), `GET /v1/config`, `GET /v1/content-modes`, `GET /v1/audit-log`, `GET /v1/presets*`, `/v1/civitai/repo`.

**Removed from v1's list: `GET /v1/characters*`.** `CharacterStore` is an `actor` (`CharacterStore.swift:201`) and cannot be read synchronously. Converting it to the lock idiom is a separate, unbudgeted change; it is **out of Phase 0** and stays on the normal async path, where 0.A already makes it responsive.

For the residue, classify in `ConnectionHandler.handle()` (`:8200`) before `Task { await respond(...) }` and serve synchronously on the connection's own (now `.userInitiated`) queue. This is belt-and-braces on top of 0.A: it removes the *dependency* on the pool rather than merely relieving pressure on it.

#### 3.1.4 The write path: deltas, not a mirror; and `resume` is special **[REVISED v2 — v1 was broken]**

v1 proposed a mailbox drained "at existing scheduling points" and a pending-queue *mirror* adopted wholesale. Both are wrong:

- **There are no drain points.** `processLoop` (`:6965–6980`) exits outright — `isProcessing = false; return` — when `isPaused` and no `runsWhilePaused` job is pending. Today the *only* thing that restarts it is `setPaused(false)` calling `startProcessingIfNeeded()` (`:6921–6927`). A mailboxed `resume` would return `202` and the loop would never run again: an indefinite creation outage that reports success. This is the worst failure this FDD could have shipped.
- **Wholesale mirror adoption drops jobs.** `pending` has other writers — `enqueue` (`:6929`) and `recoverPersistedQueue` (`:4310+`). Jobs enqueued as actor messages *during* a render, then overwritten by an adopted mirror, vanish while their HTTP callers hang.

Revised design:

1. **`resume` never goes through the mailbox.** It is a plain `Task { await coordinator.setPaused(false) }` fire-and-forget, and the route returns `202` immediately with the *intent*. Correctness is preserved because `setPaused(false)` → `startProcessingIfNeeded()` is exactly the wake the loop needs; only the caller's *acknowledgement* is decoupled, not the effect. Latency until the loop actually restarts is bounded by the in-flight render — which is fine, because when a render is in flight the loop isn't parked anyway.
2. **`pause` writes the authoritative flag in the lock store (§3.1.5), then messages the actor.** The between-items gate reads the lock flag.
3. **Queue mutations are deltas, never snapshots.** `ControlCommandMailbox` holds `.cancel(id)` / `.move(id, direction)` — operations applied against whatever `pending` actually is at drain time, so concurrent enqueues and recovery are unaffected. Drain happens at the top of each `processLoop` iteration **and** in `startProcessingIfNeeded()`, so a delta lands whether the loop is running or parked.
4. **Persistence.** `persistQueueState()` is actor-only, so an off-actor cancel that isn't persisted **resurrects the job on the next bounce**. Undrained deltas are therefore written to a small sidecar (`~/.comfybox/queue-deltas.json`, atomic, same idiom as `QueuePersistence`) and replayed by `recoverPersistedQueue` before it publishes. This is the piece v1 omitted entirely.
5. **`GET /v1/queue` composes** the actor-authored snapshot **plus undrained deltas**, so a cancelled job disappears from the listing immediately rather than after the render.

#### 3.1.5 Snapshot authority **[NEW v2]**

`LiveHealthState` is a projection of actor state (§2.2), so off-actor control writes are invisible to it. For the two facts a control caller must see immediately, **invert the ownership**: `isPaused` and the undrained-delta list become **authoritative in the lock store**, and the actor becomes a *reader* of `isPaused` (its between-items gate) rather than its owner. `publishHealth()` stops writing `isPaused` into the snapshot; the read path composes `lockStore.isPaused` with the actor-authored remainder. Everything else in `HealthSnapshot` keeps its current, correct actor-authored semantics.

This is the minimum inversion that makes the AC honest. Without it, Phase 0's "returns within 2s" passes while `/v1/queue` reports `is_paused: false` for the entire render.

#### 3.1.6 What Phase 0 does *not* deliver **[NEW v2 — v1 promised this and was wrong]**

**"Pause takes effect mid-render" is withdrawn.** Three independent reasons:

- The #1479 precedent is **LTX-2 only** (`LTX2VideoGenerator.swift:288`). Image families have no preemption plumbing at all, so there is nothing to reuse.
- #1479 is a **handoff-and-resume** — yield, run the preemptor, resume — not an indefinite park. Parking a render would pin materialized latents in unified memory precisely against #218's eviction logic.
- `isPaused` is a **between-items gate** on a live endpoint. Redefining it as "stops the current render" is a breaking semantic change for every existing caller.

**Mid-render abort is scoped to `interrupt`, and family-qualified.** `interrupt` already works mid-render for ZImage, ZImageControl, Flux2 and Fibo; **Krea2 and Chroma have zero `checkCancellation` sites** and will not abort until their sampling loops gain them. The AC says exactly that, rather than implying uniform behavior.

#### 3.1.7 Alternatives rejected

- *Yield points in the render loop* — highest blast radius in the codebase, and 0.A makes it unnecessary.
- *A second `NWListener`* — the stall is at the executor layer; each connection already has its own queue.
- *`Task.detached` for the render* — changes the residency/preemption invariants #218 and #1479 depend on. 0.A achieves the thread-move without touching them.
- *v1's plan (classifier-first, executor-second)* — inverted: it would have hand-fixed a listed subset while leaving every unlisted async route broken, at higher risk, for more work.

### 3.2 — D2: Phase 2 cross-daemon strategy **[REVISED v2 — precondition + AC]**

**Decision unchanged: option (c) — document the Kira daemon as a first-class agent surface and register control-plane tools in coffeeshop-server's built-in tool layer, on the Linux box.** No warm-server proxy routes.

Justification: the daemon is loopback-only on the Linux box and reachable from the Mac only through a hand-managed `ssh -N -L` tunnel (`KiraClient.swift:21`). A warm-server proxy (option a) would make ComfyBox's control plane depend on that tunnel and fail looking like "ComfyBox is broken"; worse, a warm-server stall — the bug Phase 0 fixes — would also block Kira control. A second MCP server (option b) inherits the same tunnel from the Mac and is redundant from the Linux box, where coffeeshop-server already has ~60 `src/tools/*-tools.ts` modules with direct precedent (`comfybox-http-video-executor.ts`, `content-mode-tools.ts`). Option (c) is one new file over loopback: better failure isolation in both directions, and no Kira *behavior* moves anywhere.

**Dated precondition [NEW v2].** This decision rests on *today's* topology: Kira daemon on Linux, warm server on Mac. A Kira→Mac migration is in progress. **If the daemon moves to the Mac, re-open D2** — co-location would then favour hosting the tools next to the warm server, and the tunnel argument disappears. Record the assumption in `docs/kira-control-api.md` with its date so the next reader doesn't inherit it as timeless.

**Isolation.** Tools are namespaced and allowlisted to control-plane only — `kira_scheduler_status`, `kira_scheduler_pause`, `kira_scheduler_resume`, `kira_scheduler_run_now`, `kira_scheduler_policy`, `kira_stream_mode`. Nothing touching conversation, memory, persona, Telegram, or media generation. Gated on `role !== 'companion'`, with one added boundary-ratchet assertion that the namespace exposes nothing outside the list.

**Federation cost.** Kira controls appear in `/v1/controls` with `host: "kira-daemon"` — **advertised, not proxied**. An agent on the Mac cannot reach them. Documented and machine-readable rather than papered over.

**Strengthened AC [REVISED v2].** `PUT /v1/kira/content-scheduler/policy` (`kira-api.ts:704`) persists and *best-effort* live-applies without failing the response — so "returns 200" is not proof. The AC asserts **live application**: change a tier's `imageCount`, then observe the change reflected in `GET .../status` **and** in the next scheduler tick's behavior, with no daemon restart.

### 3.3 — D3: Phase 3 settings model **[REVISED v2 — premise inverted]**

**Write path: `PATCH /v1/config` (RFC 7386 JSON Merge Patch) as primary; `PUT` retained as full replace; both through a new lock-serialized `ServerConfigStore`. No per-key routes.**

Per-key routes are the drift engine Phase 4 exists to kill. Merge-patch gives "change one knob" with one route and one tool, and composes with discovery: a descriptor's write action is `{PATCH, /v1/config, pointer: /renderDefaults/steps}` — machine-executable without a URL per control. Today's `PUT` makes omission mean *deletion* from the client's perspective (§2.3); merge-patch makes it mean "unchanged," which is what every caller already assumes.

**Concurrency.** `ServerConfigStore` is a `final class: @unchecked Sendable` with an `NSLock` (the `PromptRepositoryStore` idiom — chosen over an actor precisely because §3.1 makes actor hops the enemy). Every write is `lock → apply patch to the in-memory document → validate → atomic write (temp + rename) → publish → unlock`. Because the merge happens *inside* the lock against the current document, two agents patching different pointers cannot conflict at all — no retry loop.

**`If-Match` is advisory, not mandatory [REVISED v2].** v1 made it required on `PUT`. No current caller sends it; requiring it breaks the Desktop and every script on day one. Instead: `GET` returns an `ETag`; `PUT`/`PATCH` **honour** `If-Match` when present (`409` on mismatch) and proceed without it otherwise; a `Warning` header marks unconditional `PUT` as deprecated. Revisit making it mandatory once callers have migrated.

**Rollback, stated correctly [REVISED v2].** `init(from:)` (`:133–151`) tolerates unknown keys, so an older build can *read* a newer config. But `encode(to:)` (`:154–171`) writes only enumerated keys, so an older build doing `GET`→`PUT` **silently deletes** the new blocks. The rollback story is therefore: reverting the server is safe for reading; the hazard is an old *client* round-tripping the document. Mitigation: after Phase 3, clients use `PATCH` (which never round-trips unknown keys), and the deprecation `Warning` on `PUT` exists for exactly this.

#### The migration, inverted **[REVISED v2 — v1's premise was false]**

v1 asserted that `DesktopSettings.default{Steps,Guidance,Width,Height}` were applied client-side and had to be preserved. **They are write-only UI state** — the only readers are the `SettingsView` declaration, its defaults (`:108–111`) and its own `TextField` bindings (`:357–389`). Nothing builds a generate request from them. (`EngineService.swift:171,503` has same-named fields, but those come from the *server's* model descriptor dict, unrelated.) So:

- **There is no client-side default application to preserve.** The value-preservation test v1 specified would have tested a behavior that does not exist.
- **Migrating those values would be a regression**, not a preservation: it would take one user's stale UI-form numbers and impose them on **every non-Desktop caller** — Bree, MCP, the Kira scheduler — which today get the engine's own defaults.

**Revised rule: seed `renderDefaults` from the engine's existing fallbacks, not from `desktop-config.json`.** Those fallbacks are `payload.width ?? 1024`, `height ?? 1024`, `steps ?? defaultSteps`, `guidance ?? defaultGuidance` (`:7413–7416`), with a family-specific `steps ?? 30` elsewhere (`:7739`); video is `?? 704 × 448` (`:2229–2230`) and `steps ?? 8` (`:2380`) — *after* preset resolution.

**Family-awareness [NEW v2].** Because the current defaults are family-dependent (§2.5), a flat `renderDefaults` would flatten real behavior. Shape it as `renderDefaults: { default: {...}, byFamily: { "krea2": {...}, ... } }`, resolution order **`request → preset → config.byFamily[family] → config.default → engine constant`**. The config layer slots *above* the hardcoded constant and *below* everything that exists today, so with an empty config the resolution is bit-identical to current behavior.

**Rewritten value-preservation test:** with `config.json` freshly migrated and no user edits, assert that for a matrix of (family × request-with-fields-omitted) the resolved render parameters equal the **pre-migration engine** values. The baseline is the engine, not the Desktop.

**What actually migrates from `desktop-config.json`:** only `videoWidth/Height/Frames` — and only because `MotionView.swift:393–398` genuinely reads them to seed its initial control state, so a user has meaningful values there. They land in `videoDefaults` and the Motion tab reads them from the server afterwards. **`videoDefaults.backend` is dropped — no such field exists in `DesktopSettings`** (v1 carried it over from the PRD without checking). The API keys (`civitaiApiKey`, `replicateApiKey`, `falApiKey`) still de-duplicate, server-wins.

**Stays local** (presentation / Class D): `serverHost`, `serverPort`, `autoConnect`, `serverHealthEndpoint`, `thumbnailSize`, `gallerySortDefault`, `uiScale`, `archiveRoots`, `watchedServices`, and `default{Steps,Guidance,Width,Height}` — which, being pure UI form state, have no business on the server at all. `outputDirectory` stays local as the Desktop's save location; the server's `allowedOutputDirectory` is a containment boundary and merging the two would weaken it.

**First run:** if `config.json` lacks `renderDefaults`, write the engine-derived seed; if it lacks `videoDefaults` and `desktop-config.json` has video values, import those. Log each imported value to the audit log (`config.migrate.*`). `desktop-config.json` is never deleted.

**Hot-apply:** defaults are read from `ServerConfigStore` at request-decode time (`decodedGeneratePayload`, `:4289`) and in the video prep path — a lock read, off-actor, Phase-0-compatible. `port`/`host` are marked `requiresRestart: true`.

**Content modes (Class E):** `PUT /v1/content-modes/{mode}` writing `guidanceBoost`, `promptHint`, `negativePromptAdditions`, `styleVariant` via `ContentModeStore.save()`, with range validation and `400` on unknown `styleVariant`. `DELETE` reverts a mode to its built-in definition rather than removing it.

### 3.4 — D4: Phase 4 discovery surface

**`GET /v1/controls`, generated from a compile-time `ControlRegistry`.**

```swift
public struct ControlDescriptor: Codable, Sendable {
  public let id: String            // "render.defaults.steps" — stable dotted handle
  public let title: String
  public let summary: String
  public let scope: ControlScope   // .engine .queue .creative .provider .model .kira
  public let type: ControlType     // .int .double .bool .string .enum .object .action
  public let range: ClosedRange<Double>?
  public let allowed: [String]?
  public let unit: String?
  public let defaultValue: JSONValue?
  public let read: ActionRef?      // { host, method, path, pointer }
  public let write: ActionRef?
  public let mcpTool: String?
  public let host: ControlHost     // .comfybox | .kiraDaemon  (federated, §3.2)
  public let mutatesEngine: Bool
  public let requiresRestart: Bool
  public let since: String
}
```

The one rule that stops it becoming a third truth: **the registry declares where a value lives and never caches a copy.** Values are resolved per-request by dereferencing `read.pointer` against `ServerConfigStore` / `ContentModeStore` / the live control state.

Anti-drift by construction: the registry is load-bearing in three consumers — (1) the route has no list of its own; (2) config-shaped MCP tools derive JSON Schema `properties` from descriptor `type`/`range`/`allowed`, so adding a control widens the tool automatically; (3) a new `comfybox docs generate` subcommand emits `docs/api-reference.md` from the registry plus the parsed route table, with CI asserting the checked-in file byte-matches a fresh generation.

### 3.5 — D5: the anti-drift test **[REVISED v2 — parser corrected]**

**Parse the dispatch switches from source as ground truth for routes; compare against the compile-time registries.** Swift has no runtime reflection over a `switch`, and refactoring `respond(to:)`'s arms into a data table would be a large risky edit to a file Phases 0 and 3 are already rewriting. Parsing is honest; the design's job is making its failures loud. Test: `Tests/ZImageTests/ControlSurfaceParityTests.swift`.

**Both switches, not one [REVISED v2].** `ComfyBridge.route()` (`ComfyBridge.swift:106`) is a *second* dispatch switch reached **before** the main one (`WarmServer.swift:615`), and it contains real mutating routes (`POST /queue`, `:135`). v1 proposed exempting bridge routes — impossible, since the parser never sees them, so an exemption list would silently accept anything. **Decision: parse both files.** Bridge routes are tagged `surface: .comfyUICompat` and held to a *declared* policy — they need no MCP tool (they exist for ComfyUI/Krita clients, which have their own protocol) but they must be **enumerated**, so adding one is visible in review rather than invisible.

**Parser rules, corrected:**

1. **Strip comments first.** Two `case (` occurrences live in comments (`:271`, `:4891`) and would otherwise inflate the count.
2. **Count tuples, not lines.** 73 `case (` lines yield **76 tuples**: `:1620` (`pause`, `resume`), `:1654` (`POST`, `PUT /v1/characters`), `:1680` (`POST`, `PUT /v1/presets`). v1's line-count pin would have let a **control route be silently missed** — `/v1/queue/resume`, of all things.
3. **Accept both orderings** of `hasPrefix`/`hasSuffix` in `where` clauses, and the `case _ where request.method == …` form the bridge uses (`ComfyBridge.swift:126`).

**Five assertions:**

1. Extract `Set<RouteRef>` from both files.
2. **Pin the parser:** parsed tuple count equals a checked-in constant **per file**, and every non-comment `case (` occurrence was consumed by a recognizer — an unrecognized arm fails as "unparsed dispatch arm at `<file>:<line>`" rather than being skipped. This is the assertion that makes the approach trustworthy.
3. Every mutating `surface: .v1` route is claimed by ≥1 MCP tool via a new `routes: [RouteRef]` field on `MCPToolDefinition` (populated in Phase 1), or is listed in `ParityExemptions.swift` with a non-empty reason.
4. Every `.comfybox`-hosted descriptor's `write.route` and `mcpTool` resolve to real entries. Kira-hosted descriptors are covered by a coffeeshop-server contract test instead.
5. Encode a default `ComfyBoxServerConfig` to JSON, walk it to key pointers, assert each has a descriptor or is in `nonControlKeys`. This catches "someone added a config field and no descriptor" — the most likely future drift.

---

## 4. Phases

**Shippability, corrected [REVISED v2].** v1 claimed all five phases were independently shippable. Two dependencies are real: **`update_config` must not ship in Phase 1** (it would proxy the clobbering whole-doc `PUT` that Phase 3 replaces, breaking its callers) — it moves to Phase 3 alongside `PATCH`. And **Phase 4 depends on Phase 3's `ServerConfigStore`** for value resolution. Order: 0 → 1 → 3 → 4, with 2 parallel to any of them (different repo).

### 4.1 Phase 0 — control routes answer during a render (#300) **[REVISED v2]**

**0.A (first, primary).** `unownedExecutor` on `WarmServerCoordinator` backed by a dedicated `.userInitiated` serial queue; QoS pin on the per-connection queue (`:606`). ~20 lines, no render-path edits.

**0.A′ — re-measure.** Re-run the #300 repro plus a fresh `sample`. Record which routes still stall; those are Mechanism A and define 0.B.

**0.B — residual carve-out.** Classifier in `ConnectionHandler.handle()`; `isPaused` + queue deltas authoritative in a lock store (§3.1.5); `ControlCommandMailbox` of **deltas** with drain at the top of `processLoop` **and** in `startProcessingIfNeeded()`; `resume` bypasses the mailbox entirely (§3.1.4); delta sidecar persisted and replayed by `recoverPersistedQueue`; `GET /v1/queue` composes snapshot + undrained deltas. **`GET /v1/characters*` excluded** (actor-backed store, §3.1.3).

**Files.** `WarmServer.swift` (executor + QoS + classifier + queue arms `:1610–1646` + `processLoop` `:6965` + `startProcessingIfNeeded` `:6921` + `recoverPersistedQueue` `:4310`), new `Sources/ZImage/Server/ControlPlane.swift`.

**Test seam [NEW v2].** The integration AC needs a long-running operation that occupies the coordinator without a GPU. `QueuedOperation` has no injectable synthetic case today. **Budget it explicitly:** add a `#if DEBUG` `.synthetic(durationMs:)` case plus the `runsWhilePaused` arm and `processLoop` handling. Without this seam the AC is untestable in CI and would degrade into a manual check.

**Tests.** Delta mailbox: ordering, cap, eviction, and — critically — a **`resume`-while-parked** test asserting the loop restarts (the v1 blocker, pinned). Enqueue-during-render + concurrent cancel: neither job is lost. Cancel → bounce → job stays cancelled (sidecar replay). Integration: with a synthetic 60s operation active, every residual control route returns `< 2s`, and `GET /v1/queue` reports `is_paused: true` **during** the operation, not after.

**ACs.** (1) With a render in flight, every control route returns within 2s. (2) `GET /v1/queue` reflects a pause/cancel issued during that render, in that render. (3) `interrupt` aborts mid-render **for ZImage, ZImageControl, Flux2 and Fibo**; Krea2 and Chroma abort at the next item boundary until `checkCancellation` sites are added (tracked separately). (4) **No** mid-render pause AC (§3.1.6).

**Rollback.** 0.A reverts as one commit (removing an executor restores the default one; no data). 0.B reverts behind `COMFYBOX_CONTROL_PLANE_SYNC=0`, with the delta sidecar drained on next boot regardless of the flag so no cancel is lost across the toggle.

### 4.2 Phase 1 — MCP parity for existing routes **[REVISED v2]**

**Scope.** One tool per unmapped mutating route (§2.6), **excluding `update_config`** (→ Phase 3). Add `routes: [RouteRef]` + `surface` to `MCPToolDefinition`, populated for all 44 existing tools — the metadata D5 needs.

**New tools.** `upsert_character`, `delete_character`, `create_preset`, `delete_preset`, `move_queue_job`, `cancel_queue_job`, `import_loras`, `update_lora_triggers`, `civitai_harvest`, `promote_video_trace`, `rate_video_trace`, `delete_workflow`, `generate_image_async`, `generate_video_async`.

**Files.** `MCPToolRegistry.swift`, `MCPToolExecutor.swift` (all proxy via `WarmServerClient` — no new transport), `MCPTypes.swift` (`RouteRef`, `RouteSurface`), new `ParityExemptions.swift`.

**Tests.** `ControlSurfaceParityTests` steps 1–3 land here, including the two-file parse and the tuple-count pin. Per-tool executor tests against a stub client asserting method/path/body and clean errors on bad params.

**AC.** Every mutating `.v1` route has a tool or a reasoned exemption; a new route without one fails CI; a new *bridge* route is enumerated and visible in review.

**Rollback.** Additive; revert the commit.

### 4.3 Phase 2 — Kira controls as an agent surface **[REVISED v2]**

**Scope.** comfybox: `docs/kira-control-api.md` (including the **dated topology precondition**, §3.2) + federated descriptors staged for Phase 4. coffeeshop-server: `src/tools/kira-control-tools.ts` — six namespaced tools over loopback to `127.0.0.1:3787`, tolerant parsing, clean errors when the daemon is down.

**Tests.** coffeeshop-server: tool tests against a stubbed daemon; boundary-ratchet assertion that the namespace exposes nothing outside the allowlist and is unavailable to `role === 'companion'`. comfybox: doc-contract test that each federated descriptor's path appears in `docs/kira-control-api.md`.

**AC [strengthened].** An agent reads and sets 24/7 on/off, tier config, stream override and video mix with no file edits and no restart — **and the change is verified live**: a tier `imageCount` change is reflected in `GET .../status` *and* in the next scheduler tick's behavior (because `PUT .../policy`, `kira-api.ts:704`, live-applies best-effort and a 200 alone proves nothing).

**Rollback.** Remove the module from the coffeeshop-server registration list. Nothing in comfybox executes against Kira.

### 4.4 Phase 3 — server-side settings **[REVISED v2]**

**Scope.** `ServerConfigStore`; `PATCH /v1/config`; advisory `ETag`/`If-Match` + `PUT` deprecation warning; **family-aware** `renderDefaults` seeded from **engine** fallbacks; `videoDefaults` (no `backend`) importing the three Motion values; hot-apply at request-decode time; `PUT|DELETE /v1/content-modes/{mode}`; the `update_config` MCP tool (moved from Phase 1); `SettingsView` Motion tab rebound to the server, Generation tab's four fields left local (they're UI state).

**Files.** New `ServerConfigStore.swift`; `ComfyBoxServerConfig.swift` (+2 blocks, +`encode(to:)` extended, +migration); `WarmServer.swift` (`:802–822`, content-mode arms, `:4289` and the video prep default resolution); `ContentModeStore.swift`; `SettingsView.swift`; `MotionView.swift` (`:393–398` reads server values).

**Tests.**
- **Engine-baseline value-preservation:** for a matrix of (family × request-with-omitted-fields), resolved parameters after a fresh migration equal the **pre-migration engine** values. Explicitly covers the family split (`:7415` vs `:7739`) and the video path.
- Resolution order: `request → preset → byFamily → default → engine constant`, each layer verified to override only the one below.
- Merge-patch semantics: omitted unchanged, explicit `null` deletes, nested merge not replace.
- Concurrency: N threads patching N distinct pointers → all N present, one valid file, no torn write.
- `If-Match` present-and-stale → `409`; absent → proceeds with a deprecation `Warning`.
- Validation: out-of-range `guidanceBoost`, unknown `styleVariant`, negative `steps` → `400` naming the field.
- Desktop: migrated video keys no longer persisted locally; unreachable-server shows cached values read-only.

**AC.** Changing a default via API changes what the engine produces for **all** callers; Desktop reflects it after refresh; migration is engine-behavior-neutral (asserted) and every imported value is in the audit log.

**Rollback.** Revert restores boot-captured config reads. `desktop-config.json` is never deleted. Hazard is an old client round-tripping `PUT` and dropping the new blocks (§3.3) — mitigated by `PATCH` and the deprecation warning.

### 4.5 Phase 4 — discovery **[REVISED v2 — dependency stated]**

**Depends on Phase 3** (`ServerConfigStore` is the value-resolution backend). Not shippable before it.

**Scope.** `ControlRegistry` + `ControlDescriptor`; `GET /v1/controls` (control-plane classified); `comfybox docs generate`; regenerated `docs/api-reference.md`; parity steps 4–5.

**Files.** New `Sources/ZImage/Server/ControlRegistry.swift`, `Sources/ZImage/Server/ControlDescriptor.swift`; `WarmServer.swift` (one arm); new `Sources/ComfyBox/DocsGenerateCommand.swift`; `docs/api-reference.md` (generated).

**Tests.** No dangling `read.pointer`; `docs generate` idempotent and byte-matching; parity steps 4–5.

**AC.** One call answers "what can I change and how"; `docs/api-reference.md` regenerates from the same source and CI fails on a stale checked-in copy.

**Rollback.** Additive; `api-reference.md` reverts to hand-maintained.

---

## 5. Risks **[REVISED v2]**

**R1 — queue wedge from the Phase 0 write path (was: render regression).** The v1 design would have wedged the queue on `resume`. The revised design avoids it structurally (`resume` bypasses the mailbox) and pins it with a named test. Residual risk: a *future* mailbox command type added without a corresponding wake path. *Mitigation:* `ControlCommand` carries a `requiresWake: Bool` and the drain asserts it — a command that parks the loop without waking it fails in test.

**R1′ — render-path regression from 0.A.** Now small: the executor change preserves actor serialization exactly, so #218/#1479 invariants are structurally untouched. *Mitigation:* soak one full 24/7 cycle; watch for jetsam events and preemption-episode timings in `ltx2EvictMean`/`ltx2ReloadMean`.

**R2 — lost queue mutations.** Off-actor cancels that aren't persisted resurrect on bounce; wholesale mirror adoption drops concurrent enqueues. Both were live defects in v1. *Mitigation:* deltas + sidecar + replay (§3.1.4), with tests for enqueue-during-render and cancel-then-bounce.

**R3 — Phase 3 changes engine behavior for non-Desktop callers.** The real risk, and the inverse of what v1 named. Seeding from engine fallbacks and layering config *below* request/preset keeps an empty config bit-identical to today; the engine-baseline test pins it. Residual: a family path missed in the sweep. *Mitigation:* enumerate every `?? <constant>` default site in the generate/video paths in the PR description.

**R4 — Kira topology moves.** D2 rests on a dated precondition (§3.2). *Mitigation:* the date and the trigger condition are written into `docs/kira-control-api.md`.

**R5 — Kira isolation erosion.** Boundary-ratchet assertion on the namespace allowlist.

**R6 — registry rot.** D5 catches missing descriptors and missing tools, not a descriptor whose `range` or `summary` is wrong. Accepted.

**R7 — parser brittleness.** Mitigated by per-file tuple-count pins, comment stripping, and consumption assertions (§3.5). The v1 line-count version would have silently dropped `/v1/queue/resume`.

**R8 — bridge surface invisible to parity.** Now parsed rather than exempted; bridge routes are enumerated under a declared no-MCP policy.

**R9 — `WarmServer.swift` contention.** Phases 0 and 3 both edit it heavily. Sequence them; no parallel builders on this file.

---

## 6. What I could not verify **[REVISED v2]**

v1's list is superseded — most of it was resolved during review, three items **against** what v1 assumed. Remaining:

- **The post-0.A residue.** Which routes still stall after the executor fix is a prediction (§3.1.3), not a measurement. Task 0.A′ exists to replace it; 0.B's scope should be rewritten from that data, not from this list.
- **`ControlCommandMailbox` drain placement.** `processLoop`'s top and `startProcessingIfNeeded()` are the right hooks based on reading `:6921–6980`, but I have not traced every path that mutates `pending` (`recoverPersistedQueue` `:4310+` in particular is read only at its call site). Confirm before implementing the sidecar replay ordering.
- **Krea2/Chroma cancellation.** "Zero `checkCancellation` hits" comes from the review's grep, which I did not re-run. The AC is written conservatively either way.
- **coffeeshop-server tool registration mechanics.** Confirmed `src/tools/*-tools.ts` is the pattern (~60 modules); did not read the registration entry point, so Phase 2's "one new file" remains an estimate.
- **Kira daemon behavior under `PUT .../policy`.** Endpoint existence and persistence confirmed (`kira-api.ts:704`); "best-effort live-apply" is from the review's reading of that handler, not mine. The strengthened AC tests the behavior directly rather than trusting the reading.
- **Whether `#if DEBUG` synthetic operations are acceptable** in `QueuedOperation` given the deploy procedure builds release. If not, the Phase 0 integration AC needs a different seam (e.g. an env-gated sleep in a real op) — budgeted either way, but the shape is unconfirmed.

---

## 7. Appendix — verified file/line references (@ c9dd27d)

| Location | What |
|---|---|
| `WarmServer.swift:200` | `listenerQueue` — serial queue for listener + timers + pressure source |
| `WarmServer.swift:606` | per-connection `DispatchQueue` — **no QoS** (Phase 0 pins `.userInitiated`) |
| `WarmServer.swift:615` | `await comfyBridge.route(request)` — first hop; `ComfyBridge` is a `final class` |
| `WarmServer.swift:620–632` | `/health` from the lock snapshot + the #217 explanation |
| `WarmServer.swift:802–822` | `GET`/`PUT /v1/config` — load-from-disk / decode-and-save |
| `WarmServer.swift:1610–1646` | queue mutation arms — all `await coordinator.…` |
| `WarmServer.swift:1620, 1654, 1680` | **multi-tuple `case` arms** — why D5 counts tuples, not lines |
| `WarmServer.swift:271, 4891` | `case (` inside comments — why D5 strips comments |
| `WarmServer.swift:2229–2230, 2380` | video engine fallbacks `704×448`, `steps ?? 8` |
| `WarmServer.swift:2789` | `GET /v1/queue` — already actor-free, lock snapshot |
| `WarmServer.swift:4289` | `decodedGeneratePayload` — Phase 3's default-resolution site |
| `WarmServer.swift:4310+` | `recoverPersistedQueue` — another `pending` writer (R2) |
| `WarmServer.swift:4700–4710` | `HealthSnapshot` doc comment — authoritative #217 statement |
| `WarmServer.swift:4756–4764` | `LiveHealthState` single-lock swap — no torn reads |
| `WarmServer.swift:5417` | `private actor WarmServerCoordinator` — 0.A's target |
| `WarmServer.swift:6729` | `publishHealth()` — **actor-isolated**, ~15 actor-only call sites (§2.2) |
| `WarmServer.swift:6889, 6921–6927` | `setPaused` → `startProcessingIfNeeded` — the only wake for a parked loop |
| `WarmServer.swift:6955–6980` | `runsWhilePaused` + `processLoop` — **exits when paused** (the v1 blocker) |
| `WarmServer.swift:6929` | `enqueue` — another `pending` writer |
| `WarmServer.swift:7313` | `await pipeline.generateFromRequest(...)` on the actor — the blocking render |
| `WarmServer.swift:7413–7416, 7739` | image engine fallbacks — **family-dependent** (§2.5) |
| `WarmServer.swift:8049, 8200` | `ConnectionHandler`; `Task { await server.respond(...) }` — classifier site |
| `ComfyBridge.swift:106, 126, 135` | **second dispatch switch**; `case _ where` form; `POST /queue` (mutating) |
| `LTX2VideoGenerator.swift:288` | #1479 preemption — **LTX-2 only** (§3.1.6) |
| `ComfyBoxServerConfig.swift:133–151 / 154–171` | tolerant `init(from:)` / **enumerated-keys-only `encode(to:)`** (§2.3) |
| `ContentModeStore.swift:65–79, 389` | `ContentModeDefinition`; unused `save()` |
| `PromptRepositoryStore.swift` | `NSLock` + cap + eviction — pattern for the mailbox and config store |
| `CharacterStore.swift:201` | `public actor CharacterStore` — why `/v1/characters` left the control plane |
| `MCP/MCPToolRegistry.swift:13–52` | 44 static tool defs |
| `MCP/WarmServerClient.swift:30–73` | `get`/`post`/`put`/`delete` — all Phase 1 needs |
| `ComfyBoxDesktop/Views/SettingsView.swift:60–90, 108–111, 357–389` | `DesktopSettings`; **`default*` are write-only UI state** (§3.3) |
| `ComfyBoxDesktop/Views/MotionView.swift:393–398` | the only genuine reader of `videoWidth/Height/Frames` |
| `ComfyBoxDesktop/Kira/KiraClient.swift:21–29` | `127.0.0.1:3787` + `ssh -N -L` — the cross-host fact behind D2 |
| `coffeeshop-server` `kira-api.ts:649,658,670,681,699,704,980,985` | Kira write endpoints (verified live) |
| Live `sample` (2026-08-29) | 2964/2972 in `__psynch_cvwait` on `com.apple.root.utility-qos.cooperative` — **B1 confirmed** |
