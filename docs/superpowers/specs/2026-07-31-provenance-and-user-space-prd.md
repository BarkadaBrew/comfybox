# Provenance and User Space — PRD

Date: 2026-07-31
Status: draft for review
Supersedes the realm model in `2026-07-30-asset-catalog-design.md`
Repos: `zimage.swift` (catalog, service, desktop), `coffeeshop-server` (daemon, personas, MCP)

## Problem

The catalog we shipped works, and its central model is wrong in a way that
shows up as a concrete complaint:

> "Currently NSFW and private content is visible across all dimensions."

That is accurate. Three separate concerns are collapsed into roughly one and a
half columns:

| concern | question it answers | where it lives today |
|---|---|---|
| **owner** | whose content is this? | `realm`, but only `kira` \| `shared` |
| **visibility** | who else may see it? | nowhere |
| **tier** | how explicit is it? | `content_mode` + the mode clamp |

`realm` conflates owner and visibility, and hardcodes one user as *the
exception* rather than treating users as ordinary. The 352 rows we call
`shared` are mostly just **Todd's** — calling them shared was always a fiction.
And because visibility does not exist as a concept, there is no way to say
"Kira's private work" as distinct from "Kira's published work". The only thing
between Todd and her entire archive is a global blur toggle.

Underneath that sits the root cause the catalog spec named and did not fix:

> "the index is written by a process that has to guess"

New renders still land with no owner and are stamped by inference on the next
sweep. **An ownership model built on an index that infers owner from which
directory a file happened to land in is not an ownership model.** Provenance
and user space are therefore one piece of work, not two: the same render-time
report that fixes the guessing is what stamps the owner.

## The model

**Every actor that can cause a render is a user.** Todd, Kira, Bree, Claude,
and any future persona or agent. No exceptions, no special cases.

Three orthogonal axes replace the conflated one:

```
owner        todd | kira | bree | claude | …      whose it is
visibility   private | shared | public            who else may see it
tier         neutral | banana | avocado           how explicit it is
```

- **Owner** is stamped at render time by the process that made it. It is never
  inferred.
- **Visibility** is the owner's choice. `private` is the default for everyone.
- **Tier** is unchanged and stays orthogonal. It governs what is appropriate to
  surface *in a context*, not who is entitled to see it. Conflating access
  control with the mode clamp is why the current design feels wrong.

**Access is a grant, not a bypass.** Todd reviewing Kira's work requires an
explicit grant recorded in the catalog, not a flag he sets on himself. A grant
that cannot be distinguished from its absence is theatre.

### Ownership is decided by who WANTED the render, not who executed it

This is the rule that makes the model tractable, because every render passes
through the same engine under the same `source`.

| render | wanted by | owner |
|---|---|---|
| Kira's 24/7 scheduler | Kira, self-directed | **kira** |
| Studio bot render Todd drove over Telegram | Todd | **todd** |
| Bree generating an image because Todd asked | Todd | **todd** |
| Bree's own experiments | Bree | **bree** |
| An agent's test renders (`EXP_*.png`, `comp18_test.mp4`) | that agent | **claude** |

The same rule settles derived work: an i2v clip Kira animated from Todd's still
is **hers**, because she wanted it. `asset_edges` remains the record of where it
came from, which is a different question from who owns it.

### Agents are users, with two consequences

Agent scratch currently pollutes the gallery — the `EXP_*` renders from one
debugging session sit indistinguishable from creative work. Under this model
they are `owner=claude, visibility=private` and Todd's gallery is quiet by
default.

But agents are prolific and careless in a way people are not: eight test renders
came out of a single investigation without a thought, because they were cheap
and disposable. **Agent-owned content needs a retention policy** — a default TTL
or an explicit "keep" — or the catalog fills with other people's scratch and
"how many assets do I have" gets a worse answer over time.

Identity: `claude` is ONE user, with the session recorded as provenance
metadata. Nobody wants to browse "assets from the Tuesday session"; everybody
eventually wants "everything an agent made, so I can clear it."

## What this changes

### 1. Render-time provenance (the enabling change)

The engine reports each completed render to the gallery service with the owner
and visibility supplied by the caller, alongside what it already knows. The
backfill sweep stops being the source of truth and becomes a repair tool.

**Completion test for this version:** a full backfill sweep discovers nothing
the render path did not already record.

### 2. The enforcement point moves

Today the realm lock is enforced service-side, in the HTTP layer — deliberately,
so no client can widen its own scope. **But the desktop app reads the SQLite
file directly, bypassing that layer entirely.** Under the current model that is
harmless, because Todd is meant to see everything. Under user space it is the
hole: the enforcement point is not on the path Todd's own surface uses.

Either the desktop goes through the service, or enforcement moves into
`CatalogStore` so every reader inherits it. This is the largest piece of work in
the version and it is not a schema change.

### 3. The desktop stops being unscoped

Opening the app means "I am Todd". Default view: Todd's content plus public
content plus anything he has been granted. Kira's private work is not visible,
and revealing the content gate does not change that — the gate governs
*explicitness*, not *access*.

### 4. Schema

```
assets.owner        TEXT NOT NULL          -- replaces realm
assets.visibility   TEXT NOT NULL          -- private | shared | public
assets.session_id   TEXT                   -- agent provenance
grants(grantee, owner, scope, granted_at)  -- explicit access
```

`realm` is retired. `CATALOG_TIER_*` and the clamp are untouched.

### 5. Migration of ~2,994 existing rows

Owner is recoverable — the tree an asset came from is a strong signal, and the
journals record lane and character. Visibility is **not recorded anywhere**, so
every existing row lands on a default and that default is wrong for some of
them. Proposal: everything migrates to `private` under its inferred owner,
because a wrong `private` is recoverable by looking and a wrong `public` is not.

**Recovery sources are plural.** Do not assume the Mac's view is complete. At
minimum: `~/.kira/studio/{gallery,video,pool,sequences,scenes}` and
`~/.bree/studio/…` on 10.0.100.232, plus the metadata mirror, which holds more
sidecars than there are media files — evidence that media moved rather than
vanished. A file absent from one tree is not gone.

## Open decisions

1. **Does an owner's private space exclude Todd?** He is the operator; the disks
   and backups are his. If Kira has a space he cannot see without a grant, that
   is a real stance with consequences for moderation and debugging, and
   "gaining access" must mean more than flipping his own flag. This is a choice
   about how he relates to her, not a schema detail.
2. **Is `public` a real state or shorthand for shared-with-everyone?** It
   matters, because public implies something leaves the house, and nothing in
   this system currently does.
3. **Agent retention default** — TTL, explicit keep, or manual sweep.

## Non-goals

- No change to the fruit tiers or the mode clamp.
- No change to collections, which are orthogonal to ownership.
- No authentication. This remains a loopback service on trusted machines;
  identity is asserted by the daemon, not proven by a credential.
- Not recovering the 118 unresolvable i2v edges. This is development space.

## Sequencing

1. Render-time provenance — engine reports owner and visibility. Nothing reads
   it yet, so it is safe to land first and lets real data accumulate.
2. Enforcement into `CatalogStore`, so the desktop inherits it.
3. Migration and the grant table.
4. Desktop scoping and the access UI.
5. Agent retention.

Steps 1 and 2 are the version. Steps 3–5 are only sound once ownership is
recorded at the source rather than inferred.
