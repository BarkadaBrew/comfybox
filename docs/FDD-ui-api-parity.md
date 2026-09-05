# FDD: Headless parity — every UI control reachable by API

**Repo:** `BarkadaBrew/comfybox` (`zimage.swift`, Swift/MLX)
**Worktree/branch:** `~/Projects/zimage-apiparity` @ `feat/ui-api-parity` (base `origin/main` c9dd27d)
**Components:** `Sources/ZImage/Server/WarmServer.swift`, `Sources/ZImage/Server/ComfyBridge/ComfyBridge.swift`, `Sources/ZImage/MCP/*`, `Sources/ZImage/Server/ComfyBoxServerConfig.swift`, `Sources/ComfyBoxDesktop/Views/SettingsView.swift`; cross-repo: `coffeeshop-server/src/tools/`
**PRD:** `docs/PRD-ui-api-parity.md`
**Related:** comfybox#300 (async route starvation), comfybox#217 (actor head-of-line, fixed for `/health` + `GET /v1/queue`), comfybox#218 (unified-memory eviction), comfybox#1479 (LTX-2 preemption), coffeeshop-server#1293
**Author:** Fable (Opus), architect pass — 2026-08-29; v2.3 rework — 2026-08-30
**Status:** **v2.3 — 0.B-1 reworked after its production failure (2026-08-30).** v1 shipped 5 blockers, corrected in v2; v2.2 folded in the 0.A re-measure. **v2.3 reworks §3.1.3/§4.1's 0.B-1 after it crashed production:** the v2.2 render-executor design deployed (`30e2757`, default-on) and **killed LTX video renders** — `MLX/ErrorHandler.swift:343: mutex lock failed: Invalid argument` at `mlx-c/transforms.cpp:73`, 52 crash-restart cycles in ~1h; reverted via `COMFYBOX_RENDER_TASK_EXECUTOR=0`. Meanwhile **0.B-2 shipped** (`f134d64`, PR#321) and now owns the control-plane AC live (~1ms during renders), so 0.B-1's remaining scope narrowed to async-internals routes only. v2.3 inverts the 0.B-1 design accordingly. v2.3 changes are marked **[REVISED v2.3]**; superseded v2.2 text on 0.B-1 is retained struck-through-in-prose where it carries the lesson.

**Scope note (Todd, 2026-08-29):** authentication/authorization is explicitly **out of scope and not a risk on this project** — this stack is never publicly reachable and all callers are trusted local operators or agents. The PRD's §7 "unauthenticated surface grows" risk is **withdrawn**. No token gate, no per-route permission model, no rate limiting. Input *validation* stays (malformed params, out-of-range values, unknown enum members must produce a clean `400`, not a trap) — that is correctness, not security.

---

## 0. v2 changelog [NEW]

What the review overturned, and where it landed:

| # | v1 claim | Reality | Now in |
|---|---|---|---|
| 1 | "Mechanism B is unverified; diagnose first" | **B1 confirmed by live `sample`:** 2964/2972 samples in `__psynch_cvwait` on `com.apple.root.utility-qos.cooperative` — MLX render work blocks *on the cooperative pool*. | §3.1, §4.1 — the executor fix is now **primary and first** |
| 2 | Command mailbox drained "at existing scheduling points" | **No such points exist.** `processLoop` (`:6965–6980`) *exits* (`isProcessing = false; return`) when paused with no `runsWhilePaused` job. A mailbox `resume` would 202 and wedge the queue forever. | §3.1.4a — deltas + explicit wake, `resume` never goes through the mailbox |
| 3 | Serve control reads from `LiveHealthState` | The snapshot is **written only on the actor** (`publishHealth()` `:6729`, all ~15 call sites actor-isolated). Mailbox writes would be invisible for the whole render — AC green, answers wrong. | §3.1.5 — `isPaused` + pending deltas become **authoritative in the lock store** |
| 4 | "Pause takes effect mid-render" | **Undeliverable.** #1479 preemption is LTX-2-only (`LTX2VideoGenerator.swift:288`); it's handoff-and-resume, not an indefinite park (parking pins latents against #218). And `isPaused` is a between-items gate — redefining it breaks a live endpoint. | §3.1.6 — AC dropped; mid-render abort scoped to `interrupt`, family-qualified |
| 5 | Migrate Desktop defaults to preserve client-side behavior | **`DesktopSettings.default{Steps,Guidance,Width,Height}` are write-only UI state** — nothing reads them to build a request. Migrating them would move Bree/MCP/Kira *off* engine defaults. | §3.3 — inverted: seed from the **engine's** fallbacks |
| 6 | Mirror adoption for queue mutations | `pending` has other writers (enqueue `:6929`, `recoverPersistedQueue` `:4310+`); wholesale adoption **drops jobs**. | §3.1.4a — deltas only |
| 7 | D5 pins parser by counting `case (` lines | 73 lines but **76 tuples** — 3 arms carry two (`:1620` pause+resume, `:1654`, `:1680`). A control route would be silently missed. Plus 2 false hits in comments. | §3.5 |
| 8 | Exempt ComfyBridge routes from parity | Can't — they're in a **second switch** (`ComfyBridge.swift:106`) the parser never sees, and include real mutating routes (`POST /queue` `:135`). | §3.5 |
| 9 | `If-Match` mandatory on `PUT /v1/config` | Breaks every current caller; none send it. Also `encode(to:)` `:154–171` writes only enumerated keys, so the rollback story was half-true. | §3.3 |
| 10 | Phases independently shippable | Phase 1's `update_config` would proxy the clobbering `PUT` that Phase 3 then replaces; Phase 4 depends on Phase 3's store. | §4 preamble |

### 0.1 v2.2 changelog — the 0.A′ re-measure [NEW v2.2]

| # | v2 claim | Reality (measured post-0.A) | Now in |
|---|---|---|---|
| 11 | 0.A (actor executor) fixes all async routes; residue is Mechanism A only | 0.A deployed and verified (coordinator on `DispatchQueue_78: z-image.warm-server.coordinator`) — **and all async routes still HTTP 000.** New sample: multiple threads on `com.apple.root.user-initiated-qos.cooperative` blocked in `@isolated(any)` async thunks, one in `__psynch_mutexwait` (MLX mutex). **Actor executor ≠ task executor** (SE-0338): nonisolated async pipeline code hops off the actor's queue at every await and blocks on the global pool. The QoS pin merely moved the exhausted tier from utility to user-initiated. | §3.1.2 |
| 12 | "Mechanism A" (actor head-of-line) is a distinct residue | Largely dissolved: with the pool exhausted, `Task { await respond(…) }` (`:8219` in the 0.A tree) cannot even *start*, so every async route fails identically whether or not it touches the actor. What remains is one mechanism — cooperative-pool exhaustion by blocking nonisolated async render code — plus the structural argument for a pool-independent control plane. | §3.1.2, §3.1.4 |
| 13 | (coordinator's working assumption) SE-0417 task executors "likely NOT available" on this toolchain | **Wrong — verified available.** Host is macOS **27.0**, toolchain Swift **6.4** (`swift-driver 1.168.5`). SE-0417 APIs are gated `@available(macOS 15+)`; the package floor is macOS 14, so usage needs an `#available` guard, but the production Mac clears it by 12 major versions. | §3.1.3 |

---

## 1. Summary

The Desktop app is already an API client for most engine behavior. The defect is not "UI-only controls" — it is that the control surface is **fragmented across two hosts, split unevenly between HTTP and MCP, and self-describes nowhere.** An agent cannot answer "what can I change, and how?" without reading 9,680 lines of Swift.

| Phase | Delivers | Primary risk |
|---|---|---|
| **0** | All routes answer during a render (#300) — actor executor (0.A, shipped) + pool-independent control plane (0.B-2, **shipped**, ~1ms live) + async-internals **route** executor (0.B-1, reworked v2.3 after the render-executor crash) | Low (v2.3) — render untouched; route executor is MLX-free, worst case = today's starvation. [was: render thread-affinity crash, realized in v2.2 and designed out] |
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

### 3.1 — D1: Phase 0 mechanism for #300 **[REVISED v2.2 — finalized against the 0.A′ re-measure]**

#### 3.1.1 The measurements **[REVISED v2.2]**

**Sample 1 (pre-0.A):** 2964/2972 samples in `__psynch_cvwait` on `com.apple.root.utility-qos.cooperative` — MLX render work blocks a cooperative-pool worker; the pool does not grow; every unrelated continuation starves.

**0.A deployed** (`fd08ba9`, live as build `ba7d525`): `WarmServerCoordinator` on a dedicated serial-queue executor + per-connection queues pinned `.userInitiated`. **Verified working as designed** — the post-deploy sample shows `DispatchQueue_78: z-image.warm-server.coordinator (serial)` active during a render, and the utility cooperative pool no longer appears.

**Sample 2 (post-0.A, during render + in-flight `/v1/civitai/search` probe):** all async routes **still** HTTP 000 (SSRF-guard 400 path, real search, queue pause/resume — 15–30s timeouts). Multiple threads on `com.apple.root.user-initiated-qos.cooperative` blocked in compiler-generated async thunks (`specialized thunk for @escaping @isolated(any) … @async`), at least one in `__psynch_mutexwait` — MLX's internal mutex.

#### 3.1.2 The mechanism, fully understood **[REVISED v2.2]**

**Actor executor ≠ task executor.** 0.A pins *actor-isolated* code to the coordinator's queue. But per SE-0338, **nonisolated async functions run on the global concurrent executor, not the caller's** — so the moment the coordinator awaits into the pipeline (`await pipeline.generateFromRequest`, a nonisolated async call), execution hops *off* the coordinator's queue and back onto the cooperative pool, where the pipeline's async code blocks on MLX's mutex while holding a cooperative worker. The QoS pin didn't change the mechanism; it moved the exhausted tier from utility to user-initiated.

This also dissolves v2's "Mechanism A residue" as a separate category: with the pool exhausted, `Task { await server.respond(...) }` (`:8219`, 0.A tree) **cannot even start** — there is no free worker to run the first instruction of any async route, actor-touching or not. That is why the SSRF-guard *400 path* (pure validation, no I/O, no actor) also times out: one mechanism, uniform failure.

Corollary worth recording: v1/v2's reading of #217 ("the actor is blocked for the render") was mechanically imprecise. The actor's *task* blocks inside nonisolated async code that never suspends; pre-0.A the effect was indistinguishable from a blocked actor. 0.A was still correct and necessary — it removes the actor's serialization from the contended pool and is the substrate 0.B-2 needs — but it could never have been sufficient, and the FDD said otherwise. §0.1 row 11 owns that.

#### 3.1.3 0.B-1 — inverted: move the *victims* off the pool, not the render **[REVISED v2.3]**

**Decision: apply the SE-0417 executor preference to the starved async-internals ROUTES (`/v1/civitai/search|harvest`, `/v1/enhance`, workflow runs), and leave the render on the cooperative pool exactly as it is today.** This is candidate (c) from the rework brief — the inversion of v2.2. v2.2 moved the *blocker* (the render) off the pool; that crashed. v2.3 moves the *victims* (the starved routes) off the pool instead, touching no render code and no MLX threading. It is the same move 0.B-2 already made for the control plane — *don't fix the pool, avoid it* — applied to the async remainder.

**Why v2.2 failed (the root cause, with evidence). [REVISED v2.3]** v2.2's `RenderTaskExecutor` (committed `d293527`, deployed in `30e2757` default-on) is a `TaskExecutor` backed by `DispatchQueue(attributes: .concurrent)` gated by `DispatchSemaphore(value: 2)`; its `enqueue` does `queue.async { semaphore.wait(); job.runSynchronously(...) }`. Two properties of that implementation are fatal to MLX:

1. **A `.concurrent` GCD queue does not preserve OS-thread identity across a task's suspension points.** Under SE-0417, *every* job segment of the render task — each continuation resumed after an `await` — is a *separate* `enqueue` → `queue.async`, and a concurrent queue services each from an arbitrary worker thread in GCD's pool. So a *single* render's evaluation calls migrate across OS threads at every await boundary. The `width: 2` semaphore additionally admits two render segments running on two threads *truly* concurrently.
2. **MLX forbids exactly this.** mlx-swift documents the invariant directly: `Source/MLX/State.swift:22–23` — *"do not evaluate these values or expressions that depend on them across multiple threads,"* and `Device.swift:154–155` flags the cross-thread device path as *"isn't thread safe or really usable across tasks/threads."* The native eval scheduler (`mlx-c/transforms.cpp`) carries per-stream synchronization state with OS-thread affinity; driving `eval` from a migrating/second thread corrupts it, surfacing as `pthread_mutex_lock` → `EINVAL` (*"mutex lock failed: Invalid argument"*) at `transforms.cpp:73` — the observed crash.

   *Note:* Swift's default-device selection (`Device._tlDefaultDevice`) is `@TaskLocal`, so it correctly follows the *task* across threads — that is not the bug. The bug is one layer down, in the native C++ eval machinery's thread-affine locks, which `@TaskLocal` does nothing for.

**Why Krea2 survived and LTX did not — and why "just serialize" doesn't fix it.** Pre-0.B-1 (and today, post-revert) the render runs on the cooperative pool, where the heavy MLX work executes as *long synchronous stretches with no `await` inside the eval critical section* — so the runtime never re-enqueues mid-eval and the computation naturally pins to one thread. That is precisely why the render *blocks* the pool (§3.1.2), and precisely why it never crashed there. Krea2 (image) has short eval stretches and few suspension points, so it cleared the deploy smoke even on the executor. LTX (larger graphs, more eval calls, a distinct audio branch, and #1479 model eviction/reload) interleaves `await`s *through* its eval sequence — so on the `.concurrent` queue its post-await continuations resume on different OS threads while native stream state is live → reliable crash (52 cycles/hr). **This is why candidate (a), a width-1 *serial* `DispatchQueue`, is also unsafe:** a GCD serial queue guarantees mutual exclusion but *not* thread identity — it, too, may service successive submissions (successive continuations) from different worker threads. Serializing removes the two-concurrent-evals case but not the single-render cross-thread-migration case, which is sufficient on its own to crash LTX. Only *pinning all MLX work to one persistent OS thread* (candidate (b)) would make a render-side executor safe — and that is strictly more code and more render-path risk than the problem now warrants.

**Why the render no longer needs to move at all. [REVISED v2.3]** v2.2 moved the render to free the pool *for the routes*. **0.B-2 shipped** (`f134d64`, PR#321) and made the entire control-plane set answer in ~1ms during renders by serving it on connection queues before Swift concurrency is ever involved — so the promise-bearing AC is already met and *structural*. The only routes still starving during a render are the async-internals set: `/v1/civitai/search|harvest`, `/v1/enhance`, and workflow runs. Those are exactly the victims we can lift off the pool directly, without perturbing the render.

**Verified safe to move: the victim routes are MLX-free.** Traced at `@ 30e2757`:
- `/v1/enhance` (`WarmServer.swift:2863` `enhancePromptResponse`) constructs a `PromptOptimizer` and `await optimizer.optimize(...)` — an OpenAI-style HTTP call to the configured ollama/LM-Studio endpoint. Network + one `characterStore` actor read; **no MLX**.
- `/v1/civitai/search` (`:4390`) → `CivitAIClient.searchModels` — HTTPS to the CivitAI conduit. `/v1/civitai/harvest` (`:4416`) → `CivitAIHarvestRunner.run` — paged HTTP fetch + repo upsert to disk. **No MLX.** (`Weights/CivitAICheckpoint.swift` *does* import MLX, but that is weight *decode* invoked at model-activation time on the render path, never inside these route handlers.)

Because these handlers only ever `await` on network/disk/actor I/O, thread migration is harmless to them; a plain concurrent executor is correct and safe.

**Mechanics. [REVISED v2.3]** One `RouteTaskExecutor` — a `TaskExecutor` backed by a dedicated queue, used to lift the async-internals route handlers off the cooperative pool. Preferred shape: a small **width-N concurrent** executor (thread migration is a non-issue here, so `.concurrent` is fine and lets independent harvests/enhances proceed in parallel), or simply per-route serial queues; either is acceptable because none of this code touches MLX. Apply via `withTaskExecutorPreference(routeExecutor) { await handler() }` wrapping the three async-internals dispatch arms (`:1600` enhance, `:1709` civitai/search, `:1712` civitai/harvest) — the 0.B-0 spike confirmed the preference carries into the whole nonisolated async callee chain, so the entire URLSession/optimizer await chain runs off the cooperative pool. Even while a render saturates the cooperative pool, these route continuations resume on the route executor and return. **The render spawn sites (`:6947/:7030/:7037`) are reverted to plain unstructured `Task {}` — the pre-0.B-1, currently-live behavior.** `RenderTaskExecutor.swift` and its three attachment points are deleted.

**The flag, repurposed.** Ship behind the *existing* `COMFYBOX_RENDER_TASK_EXECUTOR` env var (renamed in prose to the route-executor toggle; keep the same variable name to avoid a plist/ops change), `0` = today's all-on-the-pool behavior, unset/`1` = routes on the executor. Deploys **flag-off**; soaks with **LTX + Krea2 + audio renders concurrently with live harvest/enhance traffic** before flipping. The failure mode this guards is now inverted and far smaller: worst case a route executor misbehaves and those three routes regress to today's starvation — renders, which never move, cannot be affected.

**What this does *not* recover.** v2.2 claimed #1479 image/video render *coexistence* as a benefit of the width-2 render executor. That benefit was always thin: the coordinator's single `processLoop` renders queue items one at a time, and #1479 "coexistence" is *sequential preemption handoff* (a parked video model evicted so an image can run, then reloaded) — the parked render is suspended, not evaluating, so no genuine concurrent eval was happening or is lost by not having a render executor. v2.3 gives that up explicitly, and it costs nothing real.

#### 3.1.4 0.B-2 — the control-plane guarantee **[SHIPPED — `f134d64`, PR#321; REVISED v2.3]**

**Shipped and live-verified (2026-08-30).** With a render in flight: `GET /v1/queue` 1.1ms, pause 1.9ms, resume 1.0ms, stats 0.9ms, config 0.7ms — from eternal HTTP 000 to ~1ms. The v2 design shipped as specified — classifier in the connection handler before the `Task`, synchronous service on the connection's own queue, lock-store authority for `isPaused` + deltas, fire-and-forget `resume`, WAL-ordered delta sidecar. Its justification and scope:

- **It is the guarantee, and it no longer depends on 0.B-1.** The pool was exhausted twice, on two QoS tiers, by two builds; then v2.2's 0.B-1 *crashed* trying to empty it. 0.B-2 makes the promise structural without touching the pool at all: the control set requires **zero free cooperative threads**, so pool exhaustion cannot starve it — proven live. This is why v2.3 can afford to narrow 0.B-1 to a best-effort route lift: the promise-bearing AC is already banked here.
- **Scope: the sync-servable set only** — `POST /v1/queue/{pause,resume,clear,interrupt}`, `POST /v1/queue/{id}/move`, `DELETE /v1/queue/{id}`, `GET /v1/queue`, `GET /v1/queue/lifecycle` (comfybox#283), `GET /v1/models` (from the snapshot), `GET /v1/model/family` (comfybox#359 — pure file-existence detection), `GET /v1/stats`, `GET /v1/config`, and `GET /health` (comfybox#217 — every input is the lock-based `LiveHealthState` snapshot or an immutable value, so the WHOLE payload is servable here and the Desktop's progress/queue polling cannot be starved by a render). Genuinely-async routes (civitai search/harvest, enhance, workflow runs) are **explicitly not classified** — they are 0.B-1's job; a synchronous classifier cannot serve network I/O and should not try. `GET /v1/characters*` stays excluded (actor-backed store, §3.1.4-v2 note below).

The write-path design retained from v2 (deltas not mirrors, `resume` bypassing the mailbox, sidecar persistence) follows in the next section unchanged.

(`GET /v1/characters*` exclusion, restated from v2: `CharacterStore` is an `actor`, `CharacterStore.swift:201`, and cannot be read synchronously; converting it to the lock idiom is separate, unbudgeted work. It stays on the normal async path, which 0.B-1 makes responsive.)

#### 3.1.4a The write path: deltas, not a mirror; and `resume` is special **[REVISED v2 — v1 was broken; unchanged in v2.2]**

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

#### 3.1.7 Alternatives rejected **[REVISED v2.3]**

- *Render-side executor — move the render off the pool (v2.2's chosen design).* **Tried and reverted.** Crashed LTX with a native MLX mutex `EINVAL` because a `.concurrent`/width-2 executor migrates the render's eval across OS threads across await boundaries, violating MLX's documented single-thread eval invariant (§3.1.3). Not fixable by serializing (a serial GCD queue still migrates threads across submissions); only a single-dedicated-thread executor (candidate (b)) would be safe, at more render-path risk than the now-narrowed win justifies.
- *Candidate (b): pin all MLX work to one persistent OS thread.* The only render-side design that would actually be safe. Rejected not because it's wrong but because it's disproportionate: 0.B-2 already delivered the control-plane guarantee, so the residual win is only three I/O-bound routes, which candidate (c) reaches without any render-path or MLX-threading risk at all.
- *Yield points in the render loop* — highest blast radius in the codebase; unnecessary once the victims, not the render, are moved.
- *A second `NWListener`* — the stall is at the executor layer; each connection already has its own queue.
- *`Task.detached` for the render* — loses the retained-task structure `/interrupt` cancellation depends on (`:7030/:7037`) and still lands on the global pool. Moot in v2.3: the render is not moved.
- *v1's plan (classifier-first, executor-second)*, *v2's plan (actor executor as sufficient)*, and *v2.2's plan (render-executor as root-cause fix)* — inverted, insufficient, and crash-inducing respectively; superseded by 0.B-2 (control plane, shipped) as the guarantee + v2.3's 0.B-1 (route executor) for the async remainder.

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

### 4.1 Phase 0 — control routes answer during a render (#300) **[REVISED v2.2]**

**0.A — DONE.** `unownedExecutor` on `WarmServerCoordinator` + per-connection QoS pin. Shipped `fd08ba9`, live as build `ba7d525`. Verified behaving as designed (coordinator on its own queue; utility pool clear). **Necessary, not sufficient** (§3.1.2). Overnight soak on `ba7d525` in progress (jetsam + preemption timings, per R1′).

**0.A′ — DONE.** Re-measure complete; data in §3.1.1 sample 2. Residue: **all async routes**, one mechanism (nonisolated async render code blocking the user-initiated cooperative pool). Defines 0.B as follows.

**0.B-0 — spike. DONE (`d293527`).** SE-0417 semantics confirmed on this toolchain: `#available(macOS 15, *)` compiles clean under swift-tools 5.9; `Task(executorPreference:)` demonstrably carries nonisolated async callees onto the preferred executor; the preference does **not** survive an inner unstructured `Task {}` boundary without re-attachment; plain `DispatchQueue` conforms to `TaskExecutor` directly. **These findings survive the v2.2 failure and are exactly what v2.3's route executor relies on** — the mechanism (preference lifts an async chain off the pool) works; it was only MLX-on-migrating-threads that broke, and v2.3's routes don't touch MLX.

**0.B-1 — async-internals routes off the pool (v2.3 rework).** **Delete** `RenderTaskExecutor.swift` and revert the three render spawn sites (`:6947/:7030/:7037`) to plain `Task {}`. Add a `RouteTaskExecutor` (concurrent or per-route serial; MLX-free code, so migration is a non-issue) and wrap the three async-internals dispatch arms — `/v1/enhance` (`:1600`), `/v1/civitai/search` (`:1709`), `/v1/civitai/harvest` (`:1712`) — in `withTaskExecutorPreference(routeExecutor)`. Keep the `COMFYBOX_RENDER_TASK_EXECUTOR` env var name (now the route-executor toggle) to avoid a plist change. **Deploy flag-off; soak (0.B-1′) with LTX + Krea2 + audio renders under concurrent harvest/enhance load** before flipping. Re-measure: the three async routes must return `<2s` during a live LTX render, with zero render crashes across the soak.

**0.B-2 — control-plane guarantee. SHIPPED (`f134d64`, PR#321).** Classifier serves the sync-servable set (§3.1.4) on connection queues before Swift concurrency; live-verified ~1ms during renders (queue/pause/resume/stats/config). This is the promise-bearing AC and it is now met and structural, independent of 0.B-1's fate. `GET /v1/characters*` excluded (actor-backed store), served by the normal async path — which 0.B-1 does *not* cover in v2.3 (it was never in the starving async-internals set that harvest/enhance are; if it regresses under load it joins the route-executor set, tracked separately).

**Files. [REVISED v2.3]** `WarmServer.swift` (revert spawn sites `:6947/:7030/:7037`; wrap route arms `:1600/:1709/:1712`), new `Sources/ZImage/Server/RouteTaskExecutor.swift`, **delete** `Sources/ZImage/Server/RenderTaskExecutor.swift`. (`ControlPlane.swift` already shipped with 0.B-2.)

**Test seam [NEW v2].** The integration AC needs a long-running operation that occupies the render path without a GPU. `QueuedOperation` has no injectable synthetic case today. **Budget it explicitly:** a `#if DEBUG` `.synthetic(durationMs:)` case (plus `runsWhilePaused` arm and `processLoop` handling) whose body *blocks* its thread — so it exercises exactly the pool-exhaustion mechanism 0.B-1 fixes. Without this seam the AC is untestable in CI.

**Tests. [REVISED v2.3]** 0.B-0 spike assertions retained (executor-affinity on the toy function). Delta mailbox tests unchanged (shipped with 0.B-2). New for v2.3: integration — with a synthetic blocking 60s operation active, (a) every classified control route returns `< 2s` regardless of the flag (0.B-2, shipped); (b) `/v1/enhance`, `/v1/civitai/search`, `/v1/civitai/harvest` return `< 2s` **with `COMFYBOX_RENDER_TASK_EXECUTOR=1`** and time out with `=0` (proving the route executor). **The load-bearing gate is the soak, not CI:** the failure v2.2 shipped was invisible to the test suite (1581 green) and only appeared under a real LTX render — so 0.B-1′ requires a live LTX+Krea2+audio render soak under concurrent harvest/enhance traffic with **zero render crashes** before the flag flips. CI cannot exercise MLX; the render is deliberately untouched precisely so CI coverage of it stays valid.

**ACs. [REVISED v2.3]** (1) With a render in flight, every `/v1` route — control *and* async — returns within 2s (control via 0.B-2, shipped; async-internals via 0.B-1's route executor). (2) The classified control set meets (1) even with the route-executor flag off (0.B-2 independence — live-verified). (3) `GET /v1/queue` reflects a pause/cancel issued during that render, in that render (0.B-2). (4) `interrupt` aborts mid-render **for ZImage, ZImageControl, Flux2 and Fibo**; Krea2 and Chroma abort at the next item boundary (tracked separately). (5) **No** mid-render pause AC (§3.1.6). (6) **[NEW v2.3]** With the route-executor flag on, an LTX render (video + audio) completes without an MLX/native crash across the full soak — the render path is byte-for-byte its pre-0.B-1 self.

**Rollback. [REVISED v2.3]** 0.A: shipped, soaked. 0.B-1: `COMFYBOX_RENDER_TASK_EXECUTOR=0` reverts the route arms to the plain async path at runtime; **the render path has no flag because it is not modified** — it is already pinned to its proven pre-0.B-1 behavior in code, so the v2.2 crash class cannot recur. 0.B-2: `COMFYBOX_CONTROL_PLANE_SYNC=0`, delta sidecar drained on next boot regardless. Flags independent.

**Sequencing note (per coordinator, 2026-08-29 late):** design-only tonight; 0.B builds tomorrow against clean overnight soak data from `ba7d525`. The 0.B-1 soak is a *separate, subsequent* cycle — do not conflate the two in the ledger.

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

**R1′ — render-path regression [REVISED v2.3 — the risk materialized, then was designed out].** v2.2's 0.B-1 moved the render onto a dedicated executor and **crashed LTX** (native MLX mutex `EINVAL`; §3.1.3) — R1′ was not hypothetical, it fired. The absence-of-evidence hedge in v2.2 ("MLX has tolerated thread migration so far") was wrong: MLX tolerated migration only because the render's *synchronous eval stretches never suspended*, keeping each eval on one thread; a `.concurrent`/width-2 executor broke that. **v2.3 removes the risk at its root by not moving the render at all** — the render path reverts to its proven pre-0.B-1 code, and MLX only ever sees the same-thread synchronous stretches it always has. The residual risk transfers to the *route* executor, where it is benign: those handlers are MLX-free (verified, §3.1.3), so thread migration cannot corrupt native state; worst case the three routes regress to today's starvation. *Mitigation:* flag-off deploy; an LTX+Krea2+audio soak under concurrent harvest/enhance load with a zero-render-crash gate (AC 6) before flip.

**R2 — lost queue mutations.** Off-actor cancels that aren't persisted resurrect on bounce; wholesale mirror adoption drops concurrent enqueues. Both were live defects in v1. *Mitigation:* deltas + sidecar + replay (§3.1.4a), with tests for enqueue-during-render and cancel-then-bounce.

**R3 — Phase 3 changes engine behavior for non-Desktop callers.** The real risk, and the inverse of what v1 named. Seeding from engine fallbacks and layering config *below* request/preset keeps an empty config bit-identical to today; the engine-baseline test pins it. Residual: a family path missed in the sweep. *Mitigation:* enumerate every `?? <constant>` default site in the generate/video paths in the PR description.

**R4 — Kira topology moves.** D2 rests on a dated precondition (§3.2). *Mitigation:* the date and the trigger condition are written into `docs/kira-control-api.md`.

**R5 — Kira isolation erosion.** Boundary-ratchet assertion on the namespace allowlist.

**R6 — registry rot.** D5 catches missing descriptors and missing tools, not a descriptor whose `range` or `summary` is wrong. Accepted.

**R7 — parser brittleness.** Mitigated by per-file tuple-count pins, comment stripping, and consumption assertions (§3.5). The v1 line-count version would have silently dropped `/v1/queue/resume`.

**R8 — bridge surface invisible to parity.** Now parsed rather than exempted; bridge routes are enumerated under a declared no-MCP policy.

**R9 — `WarmServer.swift` contention.** Phases 0 and 3 both edit it heavily. Sequence them; no parallel builders on this file.

---

## 6. What I could not verify **[REVISED v2.2]**

v1's list is superseded; v2's "post-0.A residue" item is now **measured and closed** (§3.1.1 sample 2 — the answer was "everything," against v2's prediction). Remaining:

- **SE-0417 semantics — now CONFIRMED (0.B-0 spike, `d293527`).** `#available`-guarded usage compiles under swift-tools 5.9; the preference carries into nonisolated async callees; it does not survive an inner unstructured `Task {}` without re-attachment; plain `DispatchQueue` conforms to `TaskExecutor`. v2.3's route executor rests on the first two of these, both verified.
- **[NEW v2.3] That the route executor keeps the victim routes responsive while the render still saturates the cooperative pool.** The 0.B-1 probe proved the *converse* (moving the render off the pool freed the routes); it did not directly prove routes-off-a-private-executor stay responsive while the render *holds* the pool. Both rest on the same verified SE-0417 primitive (preference lifts a whole async chain onto the target executor), and URLSession does its network wait on its own threads, so the mechanism is sound — but the specific direction is unproven until the 0.B-1′ soak measures the three routes under a live render. This is the one open empirical question in v2.3.
- **[NEW v2.3] Whether `CivitAIHarvestRunner`/`PromptOptimizer` reach any MLX under an uncommon path.** Traced the direct handler bodies (all network/disk/actor); did not exhaustively walk every callee of the harvest runner. If a harvest path ever triggered weight decode (`CivitAICheckpoint`, which imports MLX), the route executor's thread migration would reintroduce the crash class. Confirm no MLX reachable from these three handlers before the flag flips — cheap grep gate in 0.B-1.
- **[RESOLVED v2.3] The exact native mechanism of `transforms.cpp:73`.** The crash is a `pthread_mutex_lock` `EINVAL` in MLX's eval scheduler, driven cross-thread; mlx-swift documents the "no cross-thread eval" invariant (`State.swift:22`, `Device.swift:154`). I did not read the mlx-c C++ source to name the exact mutex, but the invariant + the executor's `.concurrent` queue are sufficient to attribute it; v2.3 avoids the whole regime rather than depending on the precise line.
- **`ControlCommandMailbox` drain placement.** `processLoop`'s top and `startProcessingIfNeeded()` are the right hooks based on reading `:6921–6980` (@ c9dd27d), but I have not traced every path that mutates `pending` (`recoverPersistedQueue` `:4310+` in particular is read only at its call site). Confirm before implementing the sidecar replay ordering.
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
| `WarmServer.swift:5417` | `private actor WarmServerCoordinator` — 0.A's target (0.A now shipped, `fd08ba9`) |
| `WarmServer.swift:6947, 7030, 7037` (0.A tree) | the three unstructured render `Task {}` spawn sites — v2.2 attached the render executor here and it crashed LTX; **v2.3 reverts them to plain `Task {}`** (render stays on the pool). `:7030/:7037` are the retained tasks `/interrupt` cancels |
| `WarmServer.swift:1600, 1709, 1712` | `/v1/enhance`, `/v1/civitai/search`, `/v1/civitai/harvest` dispatch arms — v2.3's `withTaskExecutorPreference(routeExecutor)` wrap sites (MLX-free handlers) |
| `WarmServer.swift:8219` (0.A tree) | `Task { await server.respond(...) }` — cannot start when the pool is exhausted (§3.1.2) |
| `Package.swift:1, 6` | swift-tools 5.9; platform floor macOS 14 — why the route executor needs `#available(macOS 15, *)` |
| `.build/checkouts/mlx-swift/Source/MLX/State.swift:22`, `Device.swift:154` | MLX's documented **no-cross-thread-eval** invariant — the constraint v2.2's `.concurrent` render executor violated |
| Mac host (verified 2026-08-29) | macOS **27.0**, Swift **6.4** (`swift-driver 1.168.5`) — SE-0417 available |
| Live `sample` #2 (post-0.A, 2026-08-29) | threads on `com.apple.root.user-initiated-qos.cooperative` in `@isolated(any)` async thunks, one in `__psynch_mutexwait` (MLX mutex); coordinator visible on `DispatchQueue_78` — 0.A working, insufficient |
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
