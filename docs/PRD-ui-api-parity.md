# PRD — Headless Parity: every UI control reachable by API

**Status:** draft for FDD
**Author:** Claude session 2026-08-29, from Todd's directive: *"Anything the ui can do should be exposed via api or similar."*
**Repo:** comfybox (`zimage.swift`) — warm server + Desktop app
**Related:** comfybox#300 (async route starvation), coffeeshop-server#1293 (agent-driven creation)

---

## 1. Why

Todd operates this stack through agents as much as through the Desktop app. When a control exists only in the UI — or only on an API the agents can't reach — automation silently loses a capability, and the operator (or an agent) burns time inventing workarounds for something that was one field away.

**Concrete cost, 2026-08-29:** an agent session spent ~45 minutes building idle-detection waiters to schedule a deploy around the 24/7 render loop, on the false belief that the loop was a Desktop-UI-only toggle. It was config plus a service restart. The wrong belief was then written into a memory file and repeated in status reports. A discoverable, uniform control surface would have made that a single call.

**The real problem is not "UI-only controls."** The audit (§2) shows the Desktop app is already an API client for most engine behavior. The problem is **fragmentation and discoverability**:

- Controls are split across **two servers** — the ComfyBox warm server (`:7870`) and the Kira daemon (`:3787`) — and agents reach only the first (via MCP).
- **HTTP-without-MCP**: a dozen controls have a route but no tool, so agents can't use them.
- **Genuinely local**: a small set of settings live only in `desktop-config.json` and never reach the engine.
- **Nothing declares the mapping.** There's no way to ask "what can I control, and how?"

## 2. Findings (audit 2026-08-29, full inventory in FDD)

| Class | Examples | State |
|---|---|---|
| **A. Reachable, wrong door** | 24/7 toggle, run-now, per-tier windows/counts, stream override, video mode + i2v ratio | Kira daemon API only; **no MCP tool**, not on warm server |
| **B. HTTP exists, no MCP tool** | queue move, LoRA trigger-words, preset create/delete, "Set as Warm", content-mode default map, provider config | Agents blind to them |
| **C. Desktop-local only** | default render params, LTX-2 defaults (W/H/frames/steps/backend) | `~/.comfybox/desktop-config.json`; never reaches engine |
| **D. Local side-effects** | gallery maintenance, prompt library, smart tabs, canvases, archives, service start/stop | Desktop stores / launchd shell-outs |
| **E. No editor anywhere** | content-mode guidance boost / hint / negatives | `GET /v1/content-modes` is read-only |

Corrections this audit forced: the "creation task checkbox" referenced in prior notes **does not exist** in ComfyBox Desktop; and the 24/7 control was never UI-only.

## 3. Goals

1. **Every engine-affecting control is callable headlessly** — one documented HTTP route and one MCP tool.
2. **One discoverable surface.** An agent can enumerate available controls and their current values without reading Swift.
3. **UI becomes a client, not a source of truth.** Any setting that changes what the engine renders lives server-side; the Desktop app reads and writes it through the same API an agent uses.
4. **No silent divergence.** A control added to the UI without an API path should fail review — enforced by a test, not vigilance.

## 4. Non-goals

- Authentication/authorization. The warm server is unauthenticated on loopback today; adding authn is a separate decision (noted as a risk in §7, not solved here).
- Migrating **Class D** local concerns (DAM/gallery/canvas/archives, launchd service control). They don't change engine behavior; parity there is not worth the coupling.
- Rewriting the Desktop UI's look or navigation. Rewiring is limited to where state moves server-side.
- Merging the Kira daemon into the warm server. They stay separate services.

## 5. Scope & acceptance criteria

### Phase 0 — prerequisite: comfybox#300
Control-plane routes must answer **while the engine renders**. Today async routes (`/v1/queue/pause`, `/v1/civitai/*`) time out under continuous load — verified: 120s timeout, HTTP 000.
**AC:** with a render in flight, every control route returns within 2s.
*Rationale: shipping new knobs that jam under load reproduces the exact failure this PRD exists to fix.*

### Phase 1 — MCP parity for existing routes (Class B)
Add MCP tools for every warm-server route that mutates engine behavior and lacks one.
**AC:** a diff-test asserts every mutating `/v1/*` route has a corresponding MCP tool; new routes without tools fail CI.

### Phase 2 — cross-daemon reach (Class A)
Expose Kira scheduler controls to agents. Design choice for the FDD: thin proxy routes on the warm server vs. a second MCP server pointed at the Kira daemon vs. documenting the Kira API as a first-class agent surface. **The FDD picks one and justifies it**; the constraint is that Kira isolation must not be violated (Kira's daemon stays separate; no Kira behavior moves into warm-server core).
**AC:** an agent can read and set 24/7 on/off, tier config, stream override, and video mix without shelling into config files.

### Phase 3 — server-side settings (Class C + E)
Move engine-affecting defaults out of `desktop-config.json` into server config; add a write path for content-mode definitions.
**AC:** changing a default render param or LTX-2 default via API changes what the engine produces, with the Desktop UI reflecting it after refresh. Migration preserves existing local values on first run.

### Phase 4 — discoverability
A `GET /v1/controls` (name TBD in FDD) enumerating controllable settings: id, current value, type/range, route, MCP tool.
**AC:** an agent can answer "what can I change and how" from one call; `docs/api-reference.md` regenerates from the same source.

## 6. Success measures

- An agent performs a full creation-mode change (tier, counts, window, video mix) with zero file edits and zero service restarts.
- Control routes respond during active renders (Phase 0 proof).
- No control-plane workaround scripts (idle-waiters, config-poking) are needed for routine operations.

## 7. Risks & open questions

- **Security/authn is explicitly OUT OF SCOPE** (Todd, 2026-08-29): this stack is never publicly accessible, so no token gate, permission model, or auth middleware is to be designed, and "unauthenticated routes" is not a risk to raise. Input validation still applies as a correctness concern (malformed params → clean 400, not a trap), as do resource bounds that protect the render engine and anything that would send credentials outbound.
- **Kira isolation.** Phase 2 must not pull companion logic into warm-server core (established regime; boundary ratchet test exists on the coffeeshop-server side).
- **Config migration** (Phase 3) can silently change render behavior if server defaults differ from a user's local values — migration must be value-preserving and logged.
- **Whole-document `PUT /v1/config`** is the current settings idiom; concurrent writers can clobber. FDD should decide patch-vs-replace.
- **Scope creep into Class D.** Explicitly out; revisit only on evidence.

## 8. Rollout

Phased, each independently shippable and separately deployable. Phase 0 first (it gates the value of everything after). Deploys follow the engine-deploy procedure: `--no-pause`, timed to job rotation, `bootout` confirm-gone before `bootstrap`.
