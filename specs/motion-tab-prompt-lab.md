# SPEC: Motion Tab Redesign + Prompt Lab

**Status:** design approved 2026-08-02 · **Depends on:** #9 (parameter externalization)
**Defers to own specs:** v2v engine support · optimizer fine-tuning (see `prd-prompt-optimizer-finetuning.md`)

## 1. Why

Two problems, one design.

**The Motion tab is stale and under-powered.** Mode is implicit (a reference image
happens to be attached → i2v), resolution presets predate the two-stage
"dims = final" convention, frame options cap at 121 while production renders
193/289 single-chunk, and none of the parameters that actually decide quality
(guidance, guidance_rescale, img_compression, two-stage, sigmas) are reachable.

**Prompt optimization is invisible everywhere.** `PromptOptimizer` already runs
server-side for every video render on every surface — desktop, Telegram, and
Kira via MCP. No surface shows what it produced, lets you correct it, or records
it. Consequences seen in production: a double-enhancement path (daemon optimizes,
engine optimizes again) that nobody could see, and an optimizer refusal that the
reference author's own workflow rendered *as the prompt*.

Nothing today captures the one thing worth keeping — **which prompts produced good
renders, and how a human corrected the optimizer.**

## 2. Scope

**In:** Motion tab redesign (i2v + t2v), three-version prompt capture, engine-side
trace logging, Gallery rating, Prompt Lab browsing, exemplar feedback into the
optimizer, training-data export.

**Out:** v2v (own spec; designed-for, not built). Actual fine-tuning (own PRD).
Image render logging ships as a follow-on that reuses this schema unchanged.

## 3. Data model

### Prompt trace — engine-written, immutable
`~/.comfybox/prompt-traces.jsonl`, append-only, one JSON object per line;
rotates to `prompt-traces-YYYYMM.jsonl` past a size threshold.

```
id, timestamp, source (motion-tab | kira | telegram | mcp)
media_kind (video | image)          # image from day one; video implemented first
mode (t2v | i2v | v2v)
content_mode, character
intent            # what the human/daemon asked for
optimized         # what the optimizer returned (null if not optimized)
final             # what was actually rendered
edited            # final != optimized
resolved_config   # full parameter set + provenance, from #9
seed, model, loras
output_path, status (succeeded | failed), failure_reason
```

JSONL, not a JSON array: appends are crash-safe with no read-modify-write, Kira
can write hundreds a day without contention, and it IS the export format.

Failed renders ARE logged — a prompt that produced a crash or a black clip is data.

### Ratings — desktop-written sidecar
`~/.comfybox/prompt-ratings.json`, keyed by trace id:
`{adherence, anatomy, look}` (1–5 each) + optional free-text note + timestamp.

Separate file because traces are facts (engine is sole writer) and ratings are
opinions added later (desktop is sole writer). No contention, and a rating edit
can never corrupt render history.

### Preference pairs — desktop-written sidecar
`~/.comfybox/prompt-preferences.json`:
`{winner_trace_id, loser_trace_id, batch_id, axis, timestamp}`.
A preference is a relation between two traces, not a property of one.

### Exemplars — desktop-written, engine-read
`~/.comfybox/prompt-exemplars.json`: curated intent→final pairs per
(mode, content_mode), hard-capped at 3–6 per bucket with a per-example length cap.

### Retention
Prune unrated traces by age/count (configurable). Rated traces and any trace in a
preference pair are exempt, permanently.

### Separate from `prompt-library.json`
The existing saved-prompts store keeps its current purpose and lifecycle. Merging
would muddle both.

## 4. Engine-side logging

**Write at terminal state, not at submit** — `output_path`, resolved config, and
success/failure only exist once the job finishes. Prompt data rides on the job
record in between.

**Three-version capture requires two request fields.** The desktop's flow is:
1. `POST /v1/enhance` → optimized text
2. human edits it
3. render submit with `prompt` = final, plus optional `intent` and `optimized`,
   plus `enhance: false`

Without `intent`/`optimized` on the submit, the engine only ever sees the final
string and the correction signal — the most valuable data — is lost. The
`enhance: false` flag (shipped 2026-08-02) stops the engine re-optimizing text the
human already approved, which also closes the double-enhancement hole.

Automatic callers (Kira, Telegram) are unchanged: engine optimizes inline and logs
`intent` = incoming, `optimized` = result, `final` = same, `edited: false`.

**Two non-negotiable rules:**
1. Logging NEVER fails a render. Every write wrapped; I/O error degrades to a warning.
2. `log_trace: false` skips the trace entirely — the daemon's sealed-turn contract
   must not be silently violated by prompt persistence.

**Privacy note:** this file accumulates every prompt in plaintext on disk. That is
the point (it is the dataset), but it should be a deliberate choice whether it
lives somewhere excluded from backups.

## 5. Motion tab

Left control column, three regions; preview stays on the right.

**Mode** — explicit segmented control: *Text → Video* / *Image → Video*, with
*Video → Video* present-but-disabled so the layout doesn't shift when v2v lands.
Mode drives the optimizer template and which controls apply (reference image and
strength are i2v-only).

**Prompt, two-stage** — intent box → *Optimize* → optimized text in a second
editable box with changes visible → your final pass → *Render*. Skipping
*Optimize* submits raw with `enhance: false`; no silent rewriting, which is the
current behavior's chief sin. A *regenerate* control re-rolls a rewrite.

**Recipe + Advanced** — recipe picker (Portrait / Action / T2V solo /
Reference-faithful), backed by the existing video `PresetStore` so desktop and
daemon share one vocabulary. An *Advanced* disclosure exposes Tier A parameters
(#9) pre-filled from the recipe, each marked when overridden.

Recipes-first because these parameters interact: cfg 3.5 is correct with
guidance_rescale ~0.6 and catastrophic without it; compression is the motion lever
but costs anatomical fidelity. A flat slider panel invites individually-reasonable,
jointly-broken combinations — which is most of what went wrong the week of
2026-07-28.

**Fixed while in there:** resolution presets become final-output sizes under the
two-stage convention; frame options gain 193/289.

**No rating UI in this tab.** It renders and hands off; rating is Gallery-only.

## 6. Gallery rating

Rating lives in exactly one place: the Gallery, via `AssetDetailView`.

Rationale: Kira's output lands in the Gallery, not the Motion tab. Rating only in
Motion would limit labels to hand-authored clips and leave the bulk stream — the
volume that makes a dataset viable — unlabeled.

- **Linkage** by `output_path`, which the trace already carries.
- **Detail pane** shows the prompt trio with both diffs (optimizer's changes, then
  the human's), resolved config, and seed.
- **Rating control** is one reusable component: three axes (adherence, anatomy,
  look) + note. Compact enough for a detail pane.
- **A/B batches** additionally offer winner-picking, writing a preference pair.
- **Assets with no trace** (pre-ship renders, sealed renders) show no panel.

Applies to images identically once image logging ships.

## 7. Prompt Lab

Extends `PromptLibraryView` into "Prompts" with three segments — *Saved*
(unchanged), *Traces*, *Exemplars*.

**Traces**: rows show media-kind/mode badges, timestamp, truncated intent, rating
summary, edited marker. Detail shows the full trio with diffs, resolved config,
seed, thumbnail linking to the Gallery asset.

**Filters** (required for Kira-scale volume): media kind, mode, content mode,
source, rated/unrated, edited/not, date range, full-text across all three
versions. Default view is rated + own-authored; one toggle reveals the full stream.

**Actions**: promote to exemplar · copy any version into the Motion tab · open
asset in Gallery · export filtered selection as JSONL.

**Exemplars**: per (mode, content_mode), ordered, capped. Only traces that are
`edited: true` or strongly rated are promotable — an unedited unrated trace is
"the optimizer did something and nobody objected," a weak thing to teach from.

## 8. Feedback loop

**Now (a) — exemplar injection.** `selectSystemPrompt` gains an examples block
appended from the exemplar store, filtered by mode + content mode, rendered as
intent → good-rewrite pairs. Count and per-example length hard-capped: these are
user strings entering a system prompt, and unbounded growth blows context and
drowns the rules. Engine reloads on file change — promoting an exemplar takes
effect next render, no restart.

**Proving it works:** run one intent with exemplars on and off, same seed, judge
the pair in the Gallery. Reuses the preference machinery, pointed at the optimizer
instead of at render parameters. Without this, "exemplars help" stays a belief.

**Later (b) — export only.** A command reading traces + both sidecars, emitting
supervised pairs `{system, intent, final}` (edited-or-well-rated only) and
preference pairs `{prompt, chosen, rejected}`. Export is in scope here; training
is not — see the PRD.

## 9. Build order

1. Trace schema + engine logging + `intent`/`optimized`/`log_trace` request fields
2. Motion tab redesign (mode, two-stage prompt flow, recipes + advanced)
3. Gallery rating component + trace linkage
4. Prompt Lab (traces list, filters, exemplars)
5. Exemplar injection into the optimizer + on/off A/B validation
6. Export command
7. *Follow-on:* image render logging (schema unchanged, one hook)

Steps 1–2 deliver value alone. Step 3 unlocks the dataset. Steps 5–6 close the loop.

## 10. Risks

- **Volume**: Kira can generate hundreds of traces/day. Mitigated by filters,
  retention, and default-filtered views.
- **Exemplar drift**: bad exemplars degrade every subsequent render. Mitigated by
  the promotion filter, the cap, and on/off A/B.
- **Scope creep into training**: explicitly fenced into the PRD.
- **Plaintext prompt archive**: flagged in §4; a deliberate decision, not a default.
