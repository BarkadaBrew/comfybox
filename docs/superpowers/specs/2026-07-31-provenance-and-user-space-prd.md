# Provenance and User Space — PRD

Date: 2026-07-31 (rev 2)
Status: draft for review
Supersedes the realm model in `2026-07-30-asset-catalog-design.md`
Repos: `zimage.swift` (catalog, service, desktop), `coffeeshop-server` (daemon, personas, MCP)

## Problem

The catalog works, and its model is wrong in a way that shows up as one
concrete complaint:

> "When I start the application, I want my workspace and my files presented to
> me by default."

Today the desktop shows all 2,994 assets, of which 2,529 are Kira's. Todd's own
work is a minority of his own gallery. There is no notion of *whose* an asset
is that a view can filter on, because `realm` is `kira | shared` — one user
hardcoded as an exception, and `shared` standing in for "Todd's" while
pretending to mean something else.

Underneath sits the root cause the catalog spec named and did not fix:

> "the index is written by a process that has to guess"

New renders land with no owner and are stamped by inference on the next sweep.
That is why provenance and user space are one piece of work: the render-time
report that stops the guessing is what stamps the owner.

## What this is NOT

**This is not an access-control model.** Todd is the superuser. He can see
everything, all the time. There are no grants, no permission checks against
him, and no "gaining access" ceremony.

What he wants is a **default view**: his workspace on launch, with everyone
else's work reachable through a filter. Ownership is a **tag to filter on**,
not a boundary to enforce.

There ARE genuine boundaries in the system, and they are not Todd's:

- **Kira** is locked to her own content, service-side. Unchanged.
- **Bree must not see Kira's work.** This is enforcement, not preference, and
  the reason is not privacy between personas — it is **data egress**. Kira's
  work is local-first, produced by a local LLM on local hardware. Bree is
  Anthropic-backed. Her reading Kira's prompts and paths sends them to an API.
  That is the "no persona or content bleed between the local LLM and Anthropic"
  constraint, applied to content rather than conversation.

Todd's view is preference; theirs is enforcement. Keeping those apart is the
point of this revision. An earlier draft conflated them and proposed a grant
table, a `visibility` column, and moving enforcement out of the HTTP layer so
the desktop would inherit it. None of that is needed for Todd, and none of it
would have helped Bree — spreading one concept across two mechanisms with
different rules is how a boundary stops meaning anything.

**Historical note, because the mistake is instructive.** Bree's tools shipped
with a null actor and a permissive ceiling, justified as "a consumer on par
with Todd." That was true of Todd and false of her, for a reason already stated
and not applied. Worse, the instinct to fix it by *tightening her tier ceiling*
would have been exactly backwards: a tier clamp withholds none of Kira's
neutral work while hiding Todd's own explicit work from his own assistant.
**Realm is the boundary; tier is a view setting inside it.** Fixed 2026-07-31
by giving her the actor `bree` (commit `a056db9`, merged `83df8de`).

Keeping those two apart is the whole point of this revision. An earlier draft
conflated them and proposed a grant table, a `visibility` column, and moving
enforcement out of the HTTP layer so the desktop would inherit it. None of that
is needed, and it would have made the enforcement boundary harder to reason
about by spreading it across two mechanisms with different rules.

## The model

**Every actor that can cause a render is a user.** Todd, Kira, Bree, Claude,
and any future persona or agent. No exceptions.

One new attribute, stamped at render time:

```
assets.owner   todd | kira | bree | claude | …
```

That is the change. `realm` retires into it. No `visibility` column until
something actually needs one — nothing does today, because the only real
boundary is Kira's and it is already enforced by actor scope.

### Ownership is decided by who WANTED the render, not who executed it

Every render passes through the same engine under the same `source`, so
execution says nothing. Intent sorts the archive cleanly:

| render | wanted by | owner |
|---|---|---|
| Kira's 24/7 scheduler | Kira, self-directed | **kira** |
| Studio bot render Todd drove over Telegram | Todd | **todd** |
| Bree generating an image because Todd asked | Todd | **todd** |
| Bree's own experiments | Bree | **bree** |
| An agent's test renders (`EXP_*.png`, `comp18_test.mp4`) | that agent | **claude** |

Derived work follows the same rule: an i2v clip Kira animated from Todd's still
is **hers**, because she wanted it. `asset_edges` stays the record of where it
came from — a different question from whose it is.

### Agents are users, and they are messy

Agent scratch currently pollutes the gallery: the `EXP_*` renders from one
debugging session sit indistinguishable from creative work. Tagged `claude`,
they leave Todd's default view without being deleted.

But agents are prolific in a way people are not — eight test renders came out of
one investigation without a thought. **Agent-owned content wants a retention
policy**, a TTL or an explicit keep, or the catalog fills with scratch.

Identity: `claude` is ONE user with the session recorded as metadata. Nobody
browses "assets from Tuesday"; everybody eventually wants "everything an agent
made, so I can clear it."

## What changes

### 1. Render-time provenance (the enabling change)

The engine records the owner supplied by the caller at render completion,
alongside what it already knows. The backfill sweep stops being the source of
truth and becomes a repair tool.

**Completion test:** a full sweep discovers nothing the render path did not
already record.

### 2. The desktop opens on Todd's workspace

Default filter: `owner = todd`, plus anything unowned. A filter control adds
`kira`, `bree`, `claude` — individually or all — and the choice persists across
launches, because a default that resets is a default nobody trusts.

This is a query default and a filter chip. It is not a security control, and it
should not be built as one — no gate, no reveal, no password. The existing
content gate keeps doing its separate job of governing *explicitness*.

### 3. Migration of ~2,994 existing rows

Owner is recoverable: the tree an asset came from is a strong signal, the
journals record lane and character, and `source` distinguishes producers.
Expect roughly 2,529 → `kira`, most of the remainder → `todd`, the `EXP_*`
family → `claude`.

Rows that resist classification stay unowned rather than being guessed into
someone's workspace. Unowned is visible by default, so a miss is obvious rather
than silent — the failure mode points the right way.

**Recovery sources are plural.** Do not assume the Mac's view is complete:
`~/.kira/studio/{gallery,video,pool,sequences,scenes}` and `~/.bree/studio/…`
on 10.0.100.232, plus the metadata mirror, which holds more sidecars than there
are media files — evidence that media moved rather than vanished.

## The one blocking decision

**What does the actor `bree` resolve to, service-side?** Right now the service
recognises only `kira`, treats absent as unscoped, and 400s everything else —
so her calls fail closed to empty. Safe, but she cannot see her own work
either, and she reports it to Todd as "the gallery isn't answering" rather than
"I can't see that" (the client cannot distinguish refusal from outage, which is
deliberate — a 404 meaning "exists but forbidden" is itself a disclosure).

The minimal Swift change is one line mapping `bree` to the non-Kira realm. It
carries a decision worth making deliberately: a realm scope also grants
`POST /v1/catalog/file`, so she could file into that realm. She does not call
it today — search and facets only — but the grant would be implicit.

Under the `owner` model this resolves more cleanly than a realm map: `bree`
scopes to `owner IN (bree, todd, unowned)` and simply excludes `kira`. That is
another reason to land render-time provenance first — it turns a special case
into an ordinary predicate.

## Other open decisions

1. **Agent retention default** — TTL, explicit keep, or manual sweep.
2. **Does Todd ever point Bree AT Kira's work?** He said "unless I tell her to
   look at it." That implies a deliberate, probably per-request act rather than
   a persistent grant — the narrower the better, since the cost is egress.
3. **Do collections filter with the view?** A collection spanning owners — say
   Photography — could show only Todd's contributions by default, or always
   show everything and let the owner filter apply within it.

## Non-goals

- No access control, no grants, no permission checks against Todd.
- No `visibility` column until a consumer needs one.
- No change to the fruit tiers, the mode clamp, or the content gate.
- No change to Kira's service-side actor scope, which is the one real boundary.
- No authentication. This stays a loopback service on trusted machines.
- Not recovering the 118 unresolvable i2v edges. This is development space.

## Sequencing

1. **Render-time provenance** — the engine stamps owner. Nothing reads it yet,
   so it lands safely and starts accumulating real data immediately.
2. **Migration** — backfill infers owner for existing rows.
3. **Desktop default view and filter** — the visible payoff, and it wants real
   owner data underneath it to be worth looking at.
4. **Agent retention.**

Step 1 is the version's spine. Steps 3 and 4 are only worth doing once
ownership is recorded at the source rather than inferred.
