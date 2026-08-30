# FDD: Headless parity — every UI control reachable by API

**Repo:** `BarkadaBrew/comfybox` (`zimage.swift`, Swift/MLX)
**Worktree/branch:** `~/Projects/zimage-apiparity` @ `feat/ui-api-parity` (base `origin/main` c9dd27d)
**Components:** `Sources/ZImage/Server/WarmServer.swift`, `Sources/ZImage/MCP/*`, `Sources/ZImage/Server/ComfyBoxServerConfig.swift`, `Sources/ComfyBoxDesktop/Views/SettingsView.swift`; cross-repo: `coffeeshop-server/src/tools/`
**PRD:** `docs/PRD-ui-api-parity.md`
**Related:** comfybox#300 (async route starvation), comfybox#217 (the same class of bug, already fixed for `/health`), comfybox#1479 (preemption — the lock-based write path this design reuses), coffeeshop-server#1293
**Author:** Fable (Opus), architect pass — 2026-08-29
**Status:** v1 design. Not implemented. Phase 0 carries an unresolved diagnostic (§3.1.2) that must be closed by measurement before its second half is built.

**Scope note (Todd, 2026-08-29):** authentication/authorization is explicitly **out of scope and not a risk on this project** — this stack is never publicly reachable and all callers are trusted local operators or agents. The PRD's §7 "unauthenticated surface grows" risk is **withdrawn**. No token gate, no per-route permission model, no rate limiting. Input *validation* stays (malformed params, out-of-range values, unknown enum members must produce a clean `400`, not a trap) — that is correctness, not security.

---

## 1. Summary

The Desktop app is already an API client for most engine behavior. The defect is not "UI-only controls" — it is that the control surface is **fragmented across two hosts, split between HTTP and MCP unevenly, and self-describes nowhere.** An agent cannot answer "what can I change, and how?" without reading 9,680 lines of Swift.

This FDD delivers five phases, each independently shippable:

| Phase | Delivers | Primary risk |
|---|---|---|
| **0** | Control-plane routes answer during a render (#300) | Touches concurrency near the render path |
| **1** | MCP tool for every mutating warm-server route | Low — additive |
| **2** | Kira scheduler controls reachable by agents | Cross-host coupling; Kira isolation |
| **3** | Engine-affecting defaults move server-side, content modes become writable | Migration silently changing render output |
| **4** | `GET /v1/controls` + generated `docs/api-reference.md` + anti-drift test | Registry becomes hand-maintained and rots |

The through-line is a single new construct — the **ControlRegistry** (§3.4) — a compile-time table of control descriptors that the discovery route, the MCP tool definitions, the generated docs, and the parity test all read. Phases 1–3 are written so their outputs *land in that registry*, which is why Phase 4 is last rather than first: the registry is populated by the work, not ahead of it.

---

## 2. Findings that change the PRD's picture

The PRD's audit table stands. Four things the code review adds or corrects:

**2.1 — #300 is #217 recurring, but only partly.** `/v1/queue/pause` hangs for a provable, already-diagnosed reason: it calls `await coordinator.setPaused(paused)` (`WarmServer.swift:1619`), and `WarmServerCoordinator` (`:5417`) is a `private actor` whose executor is occupied for the **entire synchronous GPU render**. This is documented verbatim in-repo at `:4700–4710`:

> *"The actor is blocked for the full duration of a synchronous GPU render (seconds to minutes). Routing /health through `await coordinator.health()` made the endpoint queue behind the render and return nothing (HTTP 000) for the render's whole duration…"*

Every failing route in the #300 report that touches the coordinator is explained by this. **`/v1/civitai/search` is not** (§3.1.2) — it never touches the coordinator, which means #300 is at least two bugs wearing one issue number.

**2.2 — the repo already contains both halves of the fix pattern.** `LiveHealthState` (`:4750+`) is a lock-based snapshot the coordinator publishes to, read by `/health` with no actor hop. `PreemptionSignal`, `LTX2StepPosition`, `LTX2PhaseTelemetry`, `PendingPreemptorBox` (`:253–278`) are lock-based channels that an *off-actor route handler writes* and the *in-flight render loop reads* — explicitly "read inside the render loop with no actor hop." Phase 0 does not need a novel concurrency mechanism; it needs these two patterns generalized and applied to the remaining control routes.

**2.3 — `PUT /v1/config` is worse than "can clobber."** It is a whole-document decode-and-`save()` (`:811–822`) with no read-back, no version, and no in-memory store — and the *running* server reads most of its own settings from a separate `WarmServerConfiguration` captured at boot. So today a `PUT` both (a) silently discards any key a client didn't round-trip and (b) mostly doesn't affect the running engine. Phase 3 cannot be built on it as-is.

**2.4 — Kira is genuinely cross-host, over an SSH tunnel.** `KiraClient` (`Sources/ComfyBoxDesktop/Kira/KiraClient.swift:21–24`) defaults to `127.0.0.1:3787` with the comment *"127.0.0.1-only on the server (reach it via `ssh -N -L 3787:127.0.0.1:3787`)"*. Verified live: `kira-daemon.service` is running on the Linux box and `GET /v1/kira/content-scheduler/status` returns the tier/video-mix document the PRD wants controllable. The warm server is on the Mac. Any warm-server proxy would be a Mac→Linux hop over a hand-managed tunnel. This is the single fact that decides Phase 2.

### 2.5 Route/tool inventory (as of c9dd27d)

- 55 literal `case ("METHOD", "/path")` arms in `respond(to:)` plus ~20 prefix/suffix-matched arms (`:685`, `:978`, `:1081`, `:1106`, `:1123`, `:1215`, `:1222`, `:1372`, `:1398`, `:1623`, `:1635`, `:1653`, `:1656`, `:1679`, `:1682`).
- 31 literal mutating (`POST`/`PUT`/`DELETE`) routes + 8 prefix-matched mutating routes.
- 44 MCP tools in `MCPToolRegistry.tools`.

**Mutating routes with no MCP tool (Phase 1 worklist):**

| Route | Control it gates |
|---|---|
| `POST /v1/characters`, `DELETE /v1/characters/{id}` | character registry write |
| `POST /v1/presets`, `DELETE /v1/presets/{id}` | preset create/delete |
| `PUT /v1/config` | providers, content-mode default map, krea2 model map, model spec ("Set as Warm") |
| `POST /v1/queue/{id}/move`, `DELETE /v1/queue/{id}` | queue reorder / per-job cancel |
| `POST /v1/loras/import`, `POST /v1/loras/{id}/update` | LoRA import; trigger-word edit |
| `POST /v1/civitai/harvest` | prompt-repo harvest |
| `POST /v1/video/traces/{id}/promote`, `/rating` | trace curation |
| `DELETE /v1/workflows/{id}` | workflow delete |
| `POST /v1/generate/async`, `POST /v1/video/generate/async` | submit-and-poll (tools are sync-only today) |
| `POST /v1/nearline/*` | *(has tools — listed for completeness)* |

**Read-only with no writer (Phase 3 worklist):** `GET /v1/content-modes` (`:1687`) — `ContentModeStore` is a `Codable` value type with a working `save()` (`ContentModeStore.swift:389`) and no route that calls it.

---

## 3. Design decisions

### 3.1 — D1: Phase 0 mechanism for #300

#### 3.1.1 Mechanism A (verified): coordinator-actor head-of-line blocking

The request path is: `NWListener` (serial `listenerQueue`) → `accept()` makes a **fresh serial `DispatchQueue` per connection** (`:605–612`) → `ConnectionHandler.handle()` → `Task { await server.respond(to: request) }` (`:8200`) on the **global cooperative pool**. So connection I/O is genuinely parallel; the bottleneck is entirely downstream.

`respond(to:)` then dispatches a `switch`. Any arm that says `await coordinator.…` enqueues a message on an actor whose executor thread is inside a synchronous MLX render (`pipeline.generateFromRequest` is awaited *on the actor*, `:7313`, not detached). Actor messages are FIFO-ish and the render never suspends, so the message is not delivered until the render returns. `/v1/queue/pause`, `/v1/queue/clear`, `/v1/queue/interrupt`, `/v1/queue/{id}/move`, `DELETE /v1/queue/{id}`, `/v1/models`, `/v1/stats` are all in this class. **`/health` and `GET /v1/queue` already escaped it** via `LiveHealthState`.

This half is proven by the code and by #217's precedent. No further diagnosis needed.

#### 3.1.2 Mechanism B (NOT verified): coordinator-free async routes also hang

`/v1/civitai/search` (`:4370`) reaches the coordinator **nowhere**. It resolves a key, validates a host against the allowlist, and `await`s `CivitAIClient.searchModels` over URLSession. Its sibling `/v1/civitai/repo` (`:4425`) is a **synchronous** function and, per the audit, responds fine under load. The one structural difference between them is *the presence of an await*. That points at the Swift cooperative thread pool, not at the actor.

Two candidate causes, both plausible, neither confirmed:

- **B1 — cooperative pool exhaustion.** The render occupies a cooperative worker for minutes. If concurrent in-flight request `Task`s (Desktop polling, ComfyBridge clients, Bree/MCP) each park a worker in blocking work, the pool — which will not grow past its width — has nothing left to resume a URLSession continuation on.
- **B2 — CPU starvation.** MLX saturates every core during eval. Cooperative workers are not priority-elevated, so a continuation can sit runnable-but-unscheduled. This would produce *severe latency* rather than a hard 120s zero, so B2 alone is a weaker fit and probably a contributor rather than the cause.

**Phase 0 therefore opens with a measurement, not a patch** (§4.1, task 0.A). The discipline is the one that worked in `FDD-ltx2-temporal-motion.md`: rule out hypotheses by measurement before touching the render path.

#### 3.1.3 The decision

**Take the control plane off the cooperative pool entirely, and off the coordinator actor entirely — reusing the two patterns already in the file. Do not insert yield points into the render loop, and do not add a second listener.**

Concretely, three changes:

1. **A control-plane classifier in `ConnectionHandler.handle()`.** Before `Task { await respond(…) }`, test the request against `ControlPlaneRoutes.matches(method:path:)`. On a hit, call a new **synchronous** `WarmServer.respondControlPlane(to:) -> RoutedResponse` directly on the connection's own serial `DispatchQueue` and `finish()` from there. Control-plane handlers are lock-bounded and microseconds long, so this is safe on a network queue. Crucially it makes the control plane **structurally independent of the cooperative pool**, which means it is correct whether Mechanism B turns out to be B1, B2, or something we haven't named. That robustness-under-uncertainty is the main reason to prefer it.

2. **Reads served from lock-based snapshots** — generalize `LiveHealthState` into `LiveControlState` (same `NSLock` + value-snapshot shape, same `publish()`-at-every-transition contract). Queue listing, pause state, active job, model state, stats already have snapshot fields; extend the struct rather than adding parallel classes.

3. **Writes delivered through a lock-based command mailbox**, not an actor call. New `ControlCommandMailbox` (`NSLock`-guarded, bounded, oldest-evicting — modelled directly on `PromptRepositoryStore`'s serialization + cap + eviction, and on `PendingPreemptorBox`). The route handler appends a command and returns `202 Accepted` with the recorded intent plus the *observed* state from `LiveControlState`; the coordinator drains the mailbox at every scheduling point it already has (between queue items, and — for `pause` specifically — the render loop reads the paused flag directly with no actor hop, exactly as it reads `ltx2PreemptionSignal` today). `pause` and `interrupt` thus take effect *mid-render* rather than after it, which is the behavior the operator always assumed they had.

   For commands whose result the caller genuinely needs (`move`, per-job `DELETE`), the mailbox operation mutates a **lock-guarded pending-queue mirror** owned outside the actor, and the coordinator adopts the mirror when it next picks work. `202` + mirror state is honest; a synchronous `200` would be a lie under a render.

**Alternatives rejected:**

- *Yield points in the render loop.* Highest blast radius in the codebase — MLX eval ordering and the LTX-2 preemption checkpointing are both sensitive, and per `FDD-ltx2-temporal-motion.md` this pipeline has already produced one silent quality regression from an innocuous-looking numeric change. It also doesn't help: actor reentrancy at a yield point would let control messages interleave with half-mutated render state, trading a hang for a correctness hazard.
- *A dedicated custom executor for `WarmServerCoordinator`* (`nonisolated var unownedExecutor` backed by a private serial queue). Attractive and small — it moves the render's blocking off the cooperative pool, which would fix B1 directly. But it does **not** fix Mechanism A (the actor is still serialized behind the render, so `pause` still queues). Keep it as **task 0.C, conditional**: adopt only if 0.A confirms B1, as a cheap defence-in-depth for every *non*-control async route (`/v1/civitai/*`, `/v1/enhance`, workflow runs) that the classifier deliberately leaves on the normal path.
- *A second `NWListener` on another port for control routes.* Fixes nothing — the starvation is at the executor layer, not the accept layer (each connection already has its own queue) — and it splits the client contract across two ports for no benefit.
- *Making the coordinator's render `Task.detached`.* Changes model-residency and preemption invariants that `#218`/`#1479` depend on. Out of proportion to the problem.

**Route classification (the control plane):** `/health`, `GET /v1/queue`, `POST /v1/queue/{pause,resume,clear,interrupt}`, `POST /v1/queue/{id}/move`, `DELETE /v1/queue/{id}`, `GET /v1/models`, `GET /v1/stats`, `GET /v1/memory`, `GET|PUT|PATCH /v1/config`, `GET|PUT /v1/content-modes`, `GET /v1/presets*`, `GET /v1/characters*`, `GET /v1/audit-log`, `GET /v1/controls`. Everything that *does work* (render, upscale, video, workflow run, CivitAI network calls) stays exactly where it is. **The render path is not modified in Phase 0** except for the paused/interrupt flag read, which reuses an existing mechanism.

### 3.2 — D2: Phase 2 cross-daemon strategy

**Decision: option (c) — formally document the Kira daemon API as a first-class agent surface, and register its control tools in the coffeeshop-server agent-tool layer, co-located on the Linux box. The ComfyBox warm server gains no Kira routes and no Kira knowledge beyond a federated *descriptor* entry in `/v1/controls` (§3.4).**

Justification, in the order the constraints bind:

- **Topology decides it.** The Kira daemon is loopback-bound on the Linux box; the warm server is on the Mac and reaches it only through a manually-established `ssh -N -L` tunnel (`KiraClient.swift:21`). Option (a) — proxy routes on the warm server — would make ComfyBox's control plane depend on a hand-managed cross-host tunnel, and would fail in a way that looks like "ComfyBox is broken" when the tunnel drops. Option (b) — a second MCP server pointed at the Kira daemon — is topologically fine but *from the Mac* inherits the same tunnel; from the Linux box it is redundant with infrastructure that already exists.
- **The infrastructure already exists and is co-located.** Verified on the Linux box: `kira-daemon.service` is active and answering on `127.0.0.1:3787`, and coffeeshop-server has an established built-in agent-tool pattern (`src/tools/*-tools.ts`, ~60 modules, with direct precedent in `comfybox-http-video-executor.ts` and `content-mode-tools.ts`). Phase 2 is therefore one new file, `src/tools/kira-control-tools.ts`, doing loopback HTTP — no tunnel, no new process, no new transport.
- **Failure isolation is strictly better.** Warm server down → Kira controls still work. Kira daemon down → ComfyBox unaffected, and the tool returns a clean error instead of a proxy timeout. Under (a), a warm-server render stall (the very bug Phase 0 fixes) would also block Kira control.
- **Kira isolation survives.** The regime forbids companion logic entering shared cores and forbids companion-role agents acquiring general surfaces. Nothing about Kira's *behavior* moves anywhere: the daemon keeps owning it. To keep the boundary ratchet meaningful, the tool set is **control-plane only and namespaced** — `kira_scheduler_status`, `kira_scheduler_pause`, `kira_scheduler_resume`, `kira_scheduler_run_now`, `kira_scheduler_policy` (tiers: `activeHours`, `imageCount`, `videoCount`, `tierRotation`, `intervalMinutes`, `videoMode`, `videoI2vRatio`, `clipSeconds`), `kira_stream_mode`. **Explicitly excluded:** anything touching conversation, memory, persona, Telegram, or media generation. Registration is gated the same way every other Kira-adjacent surface is (`role !== 'companion'`), and the existing boundary-ratchet test gets one added assertion: the `kira_control` namespace exposes no tool outside that allowlist.
- **Trust ownership is unambiguous.** Both processes are on the same host, on loopback, under the same operator. There is no cross-trust question to answer (and per the scope note, no authn to design).

**Cost, stated plainly:** the parity guarantee becomes federated rather than unified. An agent still makes one call to enumerate controls (`GET /v1/controls`), but Kira's entries carry `host: "kira-daemon"` and are *advertised, not proxied* — the agent must be able to reach the Linux box. For Bree that is trivially true (she runs there). For an agent running only on the Mac it is not, and that is an accepted, documented limitation rather than a tunnel we pretend is reliable.

**Deliverables split:** comfybox repo → `docs/kira-control-api.md` (the daemon's control surface, documented as a supported contract) + federated descriptors in the ControlRegistry. coffeeshop-server repo → `src/tools/kira-control-tools.ts` + tests + one boundary-ratchet assertion.

### 3.3 — D3: Phase 3 settings model

**Decision: keep one config document; add `PATCH /v1/config` with RFC 7386 JSON Merge Patch semantics as the primary write path; keep `PUT /v1/config` as explicit full replace. Both go through a new lock-serialized `ServerConfigStore`. Concurrency safety by document version (`If-Match`), advisory on `PUT`, enforced on nothing else. No per-key routes.**

- **Why merge-patch over per-key routes.** Per-key routes are the drift engine Phase 4 exists to kill: every new setting would need a route, a tool, a doc line, and a test, hand-maintained. Merge-patch gives "change one knob" semantics with **one** route and **one** MCP tool (`update_config`), and the *knobs* are described by the ControlRegistry rather than by URL space. It also composes with §3.4: a control descriptor's write action is `{method: "PATCH", path: "/v1/config", pointer: "/renderDefaults/steps"}`, which is machine-executable without inventing a route per control.
- **Why not keep whole-doc PUT as primary.** Today's `PUT` is a read-modify-write from the client's perspective, and clients do not round-trip keys they don't know about — a Desktop build one version behind will silently delete a config block added by the server. Merge-patch makes omission mean "unchanged," which is the semantics every caller already assumes.
- **Concurrency.** `ServerConfigStore` is a `final class: @unchecked Sendable` with an `NSLock` (the `PromptRepositoryStore` idiom, chosen over an actor precisely because §3.1 forbids actor hops on the control plane). It loads once at boot, holds the decoded document in memory, and every write is `lock → apply patch to in-memory doc → validate → atomic write (temp + `rename`) → publish → unlock`. Because the merge happens *inside* the lock against the current document, two agents patching different keys cannot clobber each other at all — no retry loop needed. `GET /v1/config` returns an `ETag` (SHA-256 of the canonical encoding); `PUT` **requires** `If-Match` and returns `409` on mismatch (full replace is the only genuinely destructive operation); `PATCH` accepts `If-Match` optionally and only fails when the *same pointers* changed underneath it.
- **Hot-apply.** The live server currently reads from a boot-captured `WarmServerConfiguration`, so a config write mostly doesn't reach the engine (§2.3). Fix the specific keys this FDD needs: render/video defaults are read from `ServerConfigStore` **at request-decode time** in `decodedGeneratePayload` (`:4289`) and the video prep path — a lock read, off-actor, ~nanoseconds, Phase-0-compatible. Keys that genuinely cannot hot-apply (`port`, `host`) are marked `requiresRestart: true` in their descriptor and the response says so.

**Migration (`desktop-config.json` → server config).**

*Moves* (engine-affecting), into two new blocks:

```
renderDefaults: { steps, guidance, width, height }        ← defaultSteps, defaultGuidance, defaultWidth, defaultHeight
videoDefaults:  { width, height, frames, steps, backend } ← videoWidth, videoHeight, videoFrames, videoSteps, (backend: see §6)
```
plus de-duplication of `civitaiApiKey` / `replicateApiKey` / `falApiKey`, which exist in **both** documents today — server wins after migration, Desktop stops persisting them.

*Stays local* (presentation / Class D, per PRD non-goals): `serverHost`, `serverPort`, `autoConnect`, `serverHealthEndpoint`, `thumbnailSize`, `gallerySortDefault`, `uiScale`, `archiveRoots`, `watchedServices`. `outputDirectory` stays local as the Desktop's *save* location; the server keeps `allowedOutputDirectory` as its containment boundary — they are different concepts and merging them would weaken path containment.

*First-run mechanics.* On boot, if `config.json` lacks `renderDefaults` **and** `~/.comfybox/desktop-config.json` exists → copy the local values verbatim into the new blocks, write once, append `config.migrate.desktopDefaults` to the audit log with the imported key/value pairs, and **leave `desktop-config.json` untouched** (it is the rollback artifact). If the local file is absent, seed from the constants the render path uses today (`448×768`, and the existing `defaultSteps`/`defaultGuidance` defaults) — *not* from new opinions.

*Value-preservation invariant.* Server-side defaults are applied **only where the incoming request omits the field**, which is precisely what the Desktop did client-side before. A request carrying explicit `steps` is bit-identical pre- and post-migration. This is the property that makes "migration can silently change render behavior" (PRD §7) testable rather than hoped-for, and §4.4 pins it with a test.

*After migration the Desktop* reads defaults from `GET /v1/config` and writes them with `PATCH /v1/config`; the Generation and Motion tabs of `SettingsView` bind to the server document. If the server is unreachable it shows the last-known cached server values **read-only** with a "server unreachable" note, rather than silently falling back to stale local values — divergence is the disease, not the symptom.

**Content modes (Class E)** get `PUT /v1/content-modes/{mode}` writing `guidanceBoost`, `promptHint`, `negativePromptAdditions`, `styleVariant` through `ContentModeStore.save()`, with range validation (`guidanceBoost` clamped to a documented band, unknown `styleVariant` → `400`). Built-ins remain in code as the reset source; `DELETE /v1/content-modes/{mode}` reverts a mode to its built-in definition rather than deleting it.

### 3.4 — D4: Phase 4 discovery surface

**Decision: `GET /v1/controls`, generated from a compile-time `ControlRegistry` that is the *same* source the MCP tool definitions and the generated `docs/api-reference.md` read. Name kept as the PRD's `/v1/controls` — it is the noun the operator uses.**

```swift
public struct ControlDescriptor: Codable, Sendable {
  public let id: String            // "render.defaults.steps" — stable, dotted, the agent's handle
  public let title: String
  public let summary: String
  public let scope: ControlScope   // .engine .queue .creative .provider .model .kira
  public let type: ControlType     // .int .double .bool .string .enum .object .action
  public let range: ClosedRange<Double>?
  public let allowed: [String]?    // enum members
  public let unit: String?
  public let defaultValue: JSONValue?
  public let read: ActionRef?      // { host, method, path, pointer }
  public let write: ActionRef?     // { host, method, path, pointer }
  public let mcpTool: String?
  public let host: ControlHost     // .comfybox | .kiraDaemon   (federated, §3.2)
  public let mutatesEngine: Bool
  public let requiresRestart: Bool
  public let since: String
}
```

The route returns `{controls: [...], generatedAt, serverVersion}`; `?scope=`/`?host=`/`?mutatesEngine=` filter. **Current values are resolved at request time** by dereferencing each descriptor's `read.pointer` against the live documents (`ServerConfigStore`, `ContentModeStore`, `LiveControlState`) — the registry declares *where* the value lives, it never caches a copy. That single rule is what keeps `/v1/controls` from becoming a third truth.

**Anti-drift by construction, not by discipline.** The registry is consumed in three places, and two of them are *load-bearing*:

1. `GET /v1/controls` — the route has no control list of its own.
2. `MCPToolRegistry` — config-shaped tools (`update_config`, `set_content_mode`, `set_render_default`) derive their JSON Schema `properties` **from the descriptors' `type`/`range`/`allowed`**, so a control with a bad range cannot produce a valid tool schema, and adding a control automatically widens the tool.
3. `comfybox docs generate` (new subcommand under `Sources/ComfyBox/`) writes `docs/api-reference.md` from the registry + parsed route table. CI asserts the checked-in file byte-matches a fresh generation. Docs cannot rot without failing the build.

Hand-maintenance is not eliminated — someone still writes the descriptor. What is eliminated is *silent* hand-maintenance: §3.5 makes the omission fail CI.

### 3.5 — D5: the anti-drift test

**Decision: a hybrid — parse the dispatch switch from source as ground truth for routes, compare against the compile-time registries, and pin the parser's own yield so a parse miss also fails.** Test target `Tests/ZImageTests/ControlSurfaceParityTests.swift`.

Swift offers no runtime reflection over a `switch`, and refactoring `respond(to:)`'s ~75 arms into a data-driven table would be a large, risky edit to a 9,680-line file that is simultaneously being changed by Phases 0–3. Parsing is the honest option; the design's job is to make the parser's failure modes loud.

The test does five things:

1. **Extract the route table.** Read `WarmServer.swift` (path resolved from `#filePath`, so it works from any checkout) and scan for both dispatch forms: `case ("METHOD", "/literal")` and `case ("METHOD", _) where request.path.hasPrefix("…")` (with optional `hasSuffix`). Produce `Set<RouteRef>` of `(method, pattern)`.
2. **Pin the parser.** Assert `parsed.count == expectedRouteCount` (a checked-in constant) **and** that every `case (` occurrence in the switch body was consumed by one of the two recognizers. A route written in a shape the parser doesn't know therefore fails CI as "unparsed dispatch arm at line N" instead of being silently skipped. This is the assertion that makes the whole approach trustworthy.
3. **Route → tool.** `MCPToolDefinition` gains a `routes: [RouteRef]` field (populated during Phase 1 — it is the only new metadata the phase needs). Assert every **mutating** parsed route is claimed by at least one tool, or appears in `ParityExemptions.swift` with a non-empty `reason` string. Exemptions are expected for the ComfyUI-bridge compatibility routes and `/v1/shutdown`-adjacent lifecycle paths; each is one line with a written justification, reviewed like code.
4. **Descriptor → route/tool.** Assert every `ControlDescriptor.write?.route` with `host == .comfybox` resolves to a parsed route, and every `mcpTool` resolves to a registered tool. Kira-hosted descriptors are skipped here and covered by a contract test in coffeeshop-server instead.
5. **Config key → descriptor.** Encode a default `ComfyBoxServerConfig` to JSON, walk it into a set of key pointers, and assert every pointer either has a descriptor or is in a `nonControlKeys` allowlist. This is the direction that catches "someone added a config field and no descriptor" — the most likely future drift.

Failure messages name the offender and the fix ("`POST /v1/loras/{id}/update` has no MCP tool; add one to `MCPToolRegistry.tools` with `routes: [...]`, or exempt it in `ParityExemptions.swift` with a reason").

---

## 4. Phases

### 4.1 Phase 0 — control routes answer during a render (#300)

**Tasks.**
- **0.A — diagnose Mechanism B (blocking, do first).** Under a continuous render, capture `sample`/`spindump` of the `comfybox serve` process while curling `/v1/civitai/search`, and log cooperative-pool width and in-flight request-task count. Record which of B1/B2 (§3.1.2) holds, or a third cause. Deliverable: a findings note appended to this FDD's §3.1.2. **0.C is gated on this.**
- **0.B — the control plane.** New `ControlPlaneRoutes` (classifier), `LiveControlState` (generalized from `LiveHealthState`), `ControlCommandMailbox` (`NSLock`, bounded, oldest-evicting). Synchronous `respondControlPlane(to:)`. Rewire the queue-mutation arms (`:1610–1646`) off `await coordinator.…`. Coordinator drains the mailbox at existing scheduling points; render loop reads the paused/interrupt flags with no actor hop.
- **0.C — conditional.** If 0.A confirms B1, give `WarmServerCoordinator` a dedicated serial-queue-backed `unownedExecutor` so the render never consumes a cooperative worker, protecting the *non*-control async routes.

**Files.** `Sources/ZImage/Server/WarmServer.swift` (classifier + `ConnectionHandler.handle`; `respond`/`respondControlPlane` split; queue arms; coordinator drain points), new `Sources/ZImage/Server/ControlPlane.swift` (classifier + mailbox + `LiveControlState`).

**Tests.** Unit: mailbox ordering/cap/eviction; classifier coverage (every route in §3.1.3's list classifies as control-plane, and no render route does). Integration (`ZImageIntegrationTests`): with a long synthetic render occupying the coordinator, assert every control route returns `< 2s`; assert `pause` takes effect **mid-render**, not after.

**AC.** PRD Phase 0 AC — with a render in flight, every control route returns within 2s. Plus: `POST /v1/queue/pause` during a 5-minute render pauses before the render completes.

**Rollback.** `COMFYBOX_CONTROL_PLANE_SYNC=0` env restores the old `Task { await respond(…) }` path for every route (the classifier returns empty). Single-flag, no data migration. 0.C reverts independently.

### 4.2 Phase 1 — MCP parity for existing routes

**Scope.** One MCP tool per unmapped mutating route (§2.5 table). Add `routes: [RouteRef]` to `MCPToolDefinition` and populate it for **all** tools, existing ones included — this is the metadata D5 depends on.

**New tools.** `upsert_character`, `delete_character`, `create_preset`, `delete_preset`, `update_config`, `move_queue_job`, `cancel_queue_job`, `import_loras`, `update_lora_triggers`, `civitai_harvest`, `promote_video_trace`, `rate_video_trace`, `delete_workflow`, `generate_image_async`, `generate_video_async`.

**Files.** `Sources/ZImage/MCP/MCPToolRegistry.swift` (defs + `routes:`), `MCPToolExecutor.swift` (`switch name` arms — all proxy through `WarmServerClient`, no new transport), `MCPTypes.swift` (`RouteRef`), new `Sources/ZImage/MCP/ParityExemptions.swift`.

**Tests.** `ControlSurfaceParityTests` steps 1–3 (§3.5) land here — the parity assertion ships *with* the phase that satisfies it. Per-tool executor tests against a stub `WarmServerClient` asserting method + path + body shape, and that bad params yield a clean tool error rather than a trap.

**AC.** PRD Phase 1 AC — the diff-test asserts every mutating `/v1/*` route has a tool; a new route without one fails CI.

**Rollback.** Additive; revert the commit. No behavior change to existing tools.

### 4.3 Phase 2 — Kira controls as an agent surface

**Scope.** Per §3.2. comfybox: `docs/kira-control-api.md` + federated `ControlDescriptor`s (`host: .kiraDaemon`) staged for Phase 4. coffeeshop-server: `src/tools/kira-control-tools.ts` — the seven namespaced tools, loopback HTTP to `127.0.0.1:3787`, tolerant parsing (the daemon's payloads are JSON dictionaries, and `KiraClient` already models them defensively), clean error text when the daemon is down.

**Tests.** coffeeshop-server: tool-level tests against a stubbed daemon; one added boundary-ratchet assertion that `kira_control` exposes nothing outside the allowlist and is unavailable to `role === 'companion'`. comfybox: a doc-contract test asserting each federated descriptor's `read.path` appears in `docs/kira-control-api.md` (weak, but it catches doc/registry divergence).

**AC.** PRD Phase 2 AC — an agent reads and sets 24/7 on/off, tier config, stream override, and video mix with no file edits and no service restart. Verify live against the running daemon: `run-now` produces a render; `policy` change is visible in `content-scheduler/status`.

**Rollback.** Remove the tool module from the coffeeshop-server registration list. Nothing in comfybox executes against Kira, so there is nothing to unwind on the Mac.

### 4.4 Phase 3 — server-side settings

**Scope.** `ServerConfigStore`; `PATCH /v1/config`; `If-Match`/`ETag` on `GET`/`PUT`; `renderDefaults` + `videoDefaults` blocks; the `desktop-config.json` migration; hot-apply of defaults at request-decode time; `PUT|DELETE /v1/content-modes/{mode}`; `SettingsView` Generation + Motion tabs rebound to the server document.

**Files.** New `Sources/ZImage/Server/ServerConfigStore.swift`; `ComfyBoxServerConfig.swift` (+2 blocks, +migration); `WarmServer.swift` (`:802–822` config arms, content-mode write arms, `decodedGeneratePayload` default resolution); `ContentModeStore.swift` (validation helpers); `Sources/ComfyBoxDesktop/Views/SettingsView.swift` (`DesktopSettings` shrinks; tabs rebind).

**Tests.**
- **The value-preservation test (the important one):** given a synthetic `desktop-config.json` with non-default values, run migration and assert the resulting effective render parameters for (i) a request omitting every field and (ii) a request specifying every field are **identical** to the pre-migration effective parameters. This is the direct guard against "migration silently changes render behavior."
- Merge-patch semantics: omitted key unchanged, explicit `null` deletes, nested object merges not replaces.
- Concurrency: N threads patching N distinct pointers → all N present, document valid, one file, no torn write.
- `If-Match` mismatch on `PUT` → `409`.
- Validation: out-of-range `guidanceBoost`, unknown `styleVariant`, negative `steps` → `400` with a message naming the field.
- Desktop: `DesktopSettingsTests` updated — migrated keys no longer persisted locally; unreachable-server path shows cached values read-only.

**AC.** PRD Phase 3 AC — changing a default via API changes what the engine produces; Desktop reflects it after refresh; migration preserves existing local values on first run (asserted, plus an audit-log line naming each imported value).

**Rollback.** `desktop-config.json` is never deleted. Revert restores the Desktop's local reads; the added server config blocks are ignored by older builds (decoding is already tolerant of unknown keys — **verify** before shipping, §6).

### 4.5 Phase 4 — discovery

**Scope.** `ControlRegistry` + `ControlDescriptor` + `GET /v1/controls` (control-plane classified, so it answers during renders); `comfybox docs generate`; `docs/api-reference.md` regenerated; `ControlSurfaceParityTests` steps 4–5.

**Files.** New `Sources/ZImage/Server/ControlRegistry.swift`, `Sources/ZImage/Server/ControlDescriptor.swift`; `WarmServer.swift` (one route arm); new `Sources/ComfyBox/DocsGenerateCommand.swift`; `docs/api-reference.md` (now generated).

**Tests.** Every descriptor's `read.pointer` dereferences successfully against a live default state (no dangling pointers). `docs generate` is idempotent and byte-matches the checked-in file. Parity steps 4–5.

**AC.** PRD Phase 4 AC — one call answers "what can I change and how"; `docs/api-reference.md` regenerates from the same source and CI fails if the checked-in copy is stale.

**Rollback.** Additive; revert. `api-reference.md` reverts to hand-maintained.

---

## 5. Risks

**R1 — render-path regression from Phase 0 (highest).** The paused/interrupt flags become readable mid-render. If a flag is read at a point where the pipeline holds partially-updated state, an interrupt could leave a model or the LTX-2 preemption checkpoint inconsistent. *Mitigation:* read the flags **only at the existing `#1479` checkpoint sites**, which are already proven safe for exactly this kind of mid-render observation — do not introduce new observation points. Ship 0.B behind `COMFYBOX_CONTROL_PLANE_SYNC` and soak it for a full 24/7 cycle before removing the flag.

**R2 — Mechanism B is neither B1 nor B2.** Then 0.C is wasted and some async routes still stall. *Mitigation:* 0.A is blocking and 0.C is explicitly conditional on it; 0.B's value does not depend on the answer, because taking the control plane off the pool entirely is correct under any cause.

**R3 — migration changes render behavior (Phase 3).** Covered by the value-preservation test (§4.4) and by the omit-only application rule (§3.3). Residual risk: a code path that reads a Desktop default *indirectly* (e.g. the Motion tab computing an LTX-2 parameter before submitting) and is missed in the sweep. *Mitigation:* grep every `DesktopSettings` field read in `Sources/ComfyBoxDesktop/` before the cut, and enumerate them in the PR description.

**R4 — cross-host coupling (Phase 2).** Accepted, not eliminated: an agent on the Mac cannot reach Kira controls. Documented in `docs/kira-control-api.md` and encoded in the descriptor's `host` field so the limitation is machine-readable rather than folklore.

**R5 — Kira isolation erosion.** A future contributor adds a "convenient" Kira tool outside the control-plane allowlist. *Mitigation:* the boundary-ratchet assertion in §4.3 fails on any tool in the namespace that isn't on the list.

**R6 — the registry rots anyway.** D5 catches *missing* descriptors and *missing* tools. It cannot catch a descriptor whose `range` or `summary` is wrong. Accepted; the pointer-dereference test at least catches a descriptor pointing at a key that no longer exists.

**R7 — parser brittleness (D5).** Mitigated by the count/consumption pin (§3.5 step 2), which converts a silent parse miss into a named CI failure.

**R8 — Phase 0 and Phase 3 both edit `WarmServer.swift` heavily.** Sequence them; do not run parallel builders against this file (established practice). Phase 1 and Phase 2 can run concurrently with either, since they touch `MCP/` and another repo.

---

## 6. What I could not verify

Stated plainly, so nothing here is taken as established:

- **Why `/v1/civitai/search` starves.** I confirmed it never touches the coordinator and that its synchronous sibling `/v1/civitai/repo` is reported healthy — so the actor explanation cannot cover it. B1/B2 (§3.1.2) are hypotheses derived from the dispatch structure, **not measurements**. I did not run the repro. Task 0.A exists for exactly this and gates 0.C.
- **The #300 symptom report itself** (HTTP 000 at 120s, which routes pass/fail) is taken from the task brief and the PRD. I did not reproduce it.
- **The LTX-2 `backend` default.** The PRD names a "backend" key among the LTX-2 Desktop defaults. In `DesktopSettings` (`SettingsView.swift:60–90`) I found `videoWidth`, `videoHeight`, `videoFrames`, `videoSteps` and **no** backend field. It may live in the Motion tab's local state, in `AppConfig`, or be a naming difference. Confirm before writing the `videoDefaults` block.
- **`ComfyBoxServerConfig` decoding tolerance of unknown keys.** It has a custom `init(from:)`/`encode(to:)` (`:134`, `:154`), which usually implies tolerance, but I did not read the bodies. Phase 3's rollback story depends on it — verify.
- **Whether the coordinator has clean drain points** for the command mailbox between queue items. I read the actor's declaration and the render call sites, not its full queue loop. If no natural point exists, 0.B's write path needs one added, which raises R1's blast radius slightly.
- **`GET /v1/queue`'s exact current implementation.** The comment at `:4735` says it was moved onto the health snapshot; I did not read the route arm to confirm it no longer hops.
- **The Kira daemon's write endpoints.** I verified `GET /v1/kira/content-scheduler/status` live and read the shape it returns. The `PUT .../policy`, `POST .../{pause,resume,run-now}` and `PUT /v1/kira/stream-mode` contracts come from `KiraClient.swift`'s client-side expectations and the task brief, **not** from the daemon's source. Confirm against the daemon before writing `docs/kira-control-api.md`.
- **coffeeshop-server tool registration mechanics.** I confirmed `src/tools/*-tools.ts` is the established pattern (~60 modules) and that `src/mcp/` exists, but did not read the registration entry point, so "one new file" is an estimate.

---

## 7. Appendix — verified file/line references (@ c9dd27d)

| Location | What |
|---|---|
| `WarmServer.swift:200` | `listenerQueue` — single serial queue for listener + timers + pressure source |
| `WarmServer.swift:253–278` | `ltx2PreemptionSignal` / `LTX2StepPosition` / `PendingPreemptorBox` — lock-based, "read inside the render loop with no actor hop" (the Phase 0 write-path precedent) |
| `WarmServer.swift:284` | `liveHealth` — "served without hopping onto the actor… stays responsive during a render (#217)" (the read-path precedent) |
| `WarmServer.swift:468` / `:605–612` | `NWListener` on `listenerQueue`; per-connection serial `DispatchQueue` |
| `WarmServer.swift:606` | `respond(to:)` — the dispatch switch |
| `WarmServer.swift:615` | `await comfyBridge.route(request)` — first hop for every request; `ComfyBridge` is a `final class`, not an actor |
| `WarmServer.swift:620–632` | `/health` served from the lock snapshot, with the #217 explanation |
| `WarmServer.swift:802–822` | `GET`/`PUT /v1/config` — load-from-disk / decode-and-`save()`, no version, no in-memory store |
| `WarmServer.swift:1610–1646` | queue mutation arms — all `await coordinator.…` (the Phase 0 rewire target) |
| `WarmServer.swift:1687` | `GET /v1/content-modes` — read-only, no writer |
| `WarmServer.swift:4370` / `:4425` | `civitaiSearchRoute` (async, no coordinator) vs `civitaiRepoRoute` (synchronous) — the Mechanism B pair |
| `WarmServer.swift:4273` | `Task.detached(priority:)` for montage — existing precedent for moving CPU-bound work off the request executor |
| `WarmServer.swift:5417` | `private actor WarmServerCoordinator` |
| `WarmServer.swift:7313` | `await pipeline.generateFromRequest(...)` — awaited **on the actor**; the blocking render |
| `WarmServer.swift:4700–4710` | `HealthSnapshot` doc comment — the authoritative #217 root-cause statement |
| `WarmServer.swift:4750+` | `LiveHealthState` — `NSLock` publisher (generalize to `LiveControlState`) |
| `WarmServer.swift:8049` / `:8200` | `ConnectionHandler`; `Task { await server.respond(to: request) }` — the cooperative-pool entry point and Phase 0's classifier insertion site |
| `ComfyBoxServerConfig.swift:80–96` | config document fields; `:134`/`:154` custom coding; `:197` `loadOrMigrate`; `:218` `save` |
| `ContentModeStore.swift:65–79` | `ContentModeDefinition` (`guidanceBoost`, `styleVariant`, `promptHint`, `negativePromptAdditions`); `:389` unused `save()` |
| `PromptRepositoryStore.swift` | `NSLock` + cap + oldest-eviction — the pattern for `ControlCommandMailbox` and `ServerConfigStore` |
| `MCP/MCPToolRegistry.swift:13–52` | `tools: [MCPToolDefinition]` — 44 static defs |
| `MCP/MCPToolExecutor.swift:22+` | `switch name` → `WarmServerClient` proxy |
| `MCP/WarmServerClient.swift:30–73` | `get`/`post`/`put`/`delete` — all Phase 1 needs |
| `ComfyBoxDesktop/Views/SettingsView.swift:60–90` | `DesktopSettings` fields; `:146` `configPath`; `:351` Generation tab; `:514` Motion tab |
| `ComfyBoxDesktop/Kira/KiraClient.swift:21–29` | `127.0.0.1:3787`, "reach it via `ssh -N -L`" — the cross-host fact behind D2 |
| Linux box, live | `kira-daemon.service` active; `GET /v1/kira/content-scheduler/status` returns `{enabled, intervalMinutes, videoMode, videoI2vRatio, clipSeconds, tiers{…}, tierRotation, activeHours}` |
| `coffeeshop-server/src/tools/` | ~60 `*-tools.ts` modules incl. `comfybox-http-video-executor.ts`, `content-mode-tools.ts` — the Phase 2 host |
