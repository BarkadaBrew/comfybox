# PREPARED (not applied): prompt-template lighting fixes

**Status:** ready to apply after the 2026-08-02 soak · **Task:** #15
**File:** `Sources/ZImage/Telegram/PromptOptimizer.swift`

Deliberately NOT applied while Kira soaks — editing prompt templates mid-validation
would inject a variable into exactly what's being measured.

## Why

Kira's 18:00 t2v measured **brightness 26.4 / 255** — near-black, and darkness
dominates every other quality signal (a dark frame also measures low sharpness,
which is what made her output read as "diffuse" beyond the stage-1 floor issue).

Root cause is a contradiction inside the prompt we send:

- shared `ltxRules` **rule 5**: *"the subject must be WELL-LIT with the face clearly
  visible … NEVER stage the subject as a backlit silhouette"*
- `systemPromptVideoAvocado` mode section ends: *"available-light amateur aesthetic"*

Two conflicting instructions in one prompt. The mode-specific line is more concrete
and later in the text, so it tends to win. Rule 5 also sits as clause three of a
compound rule that covers environment AND lens AND lighting — easy to skim past.

`systemPromptVideoBanana` has no lighting instruction at all, relying entirely on
rule 5 surviving contact with the scene description.

## Change 1 — remove the contradicting aesthetic line (avocado)

```diff
   ## EXPLICIT MODE — HARDCORE
   GRAPHIC adult content. Full nudity, sex acts, anatomy in MOTION — describe the movement of the act explicitly. Direct anatomical language, never euphemism. Default to NUDE when no clothing specified; available-light amateur aesthetic.
+  GRAPHIC adult content. Full nudity, sex acts, anatomy in MOTION — describe the movement of the act explicitly. Direct anatomical language, never euphemism. Default to NUDE when no clothing specified. Natural, unstyled realism — but the scene is still LIT (see the lighting rule); "available light" never means dark.
```

Rationale: the phrase was presumably there for an unpolished realism look. Keep the
intent, remove the licence to render an unlit scene.

## Change 2 — give banana an explicit lighting line

```diff
   ## SUGGESTIVE MODE
   Suggestive, not explicit. Lingerie, partial nudity, sensual movement, intimate framing. No genitalia or explicit acts. Lean into motion and tension.
+  Suggestive, not explicit. Lingerie, partial nudity, sensual movement, intimate framing. No genitalia or explicit acts. Lean into motion and tension. "Intimate" refers to FRAMING, not darkness — keep the subject clearly lit (see the lighting rule).
```

Rationale: "intimate framing" reads as a cue toward dim scenes; name the distinction
rather than hoping rule 5 wins.

## Change 3 — SPLIT ltxRules; do NOT just make rule 5 louder

**Revised 2026-08-02 after checking the routing.** `selectSystemPrompt` sends
`video-i2v` to `systemPromptVideoI2V`, everything else video to the
neutral/banana/avocado templates — so Changes 1 and 2 are **t2v-only**.

But `systemPromptVideoI2V` also embeds `\(ltxRules)`, so i2v inherits rule 5's
"describe the LIGHTING, state it falls on her face" — two lines above its own
section saying *"Do NOT re-describe the subject's looks, body, clothing, or the
environment — those come from the image and any contradiction corrupts the result."*

Promoting lighting to a LOUDER standalone rule would therefore make i2v WORSE:
shouting an instruction that i2v must specifically not follow.

The shared block is not shared-safe. Split it:

- **universal core** (applies to every mode): motion coherence, camera move,
  performance cues, character canon, prose formatting, anatomy grounding, figure count
- **t2v-only scene block** (NOT included by the i2v template): environment, LIGHTING,
  lens/film feel

Then the strengthened lighting rule lives in the t2v block only, and i2v stops
receiving instructions that fight its own purpose.

```diff
-  5. Then add ENVIRONMENT, LIGHTING, and lens/film feel briefly (golden hour, 85mm, shallow
-     depth of field, low angle). LIGHTING RULE: the subject must be WELL-LIT with the face
-     clearly visible — state it explicitly ("warm light illuminating her face and body").
-     NEVER stage the subject as a backlit silhouette against a bright sky unless the user
-     explicitly asks for a silhouette.
+  5. LIGHTING — THE SUBJECT MUST BE CLEARLY LIT. State the light source and that it falls
+     ON the subject's face and body ("warm morning light across her face"). This OVERRIDES
+     any mood, aesthetic or genre cue in the request: dim, candlelit, available-light and
+     intimate all still mean the subject is VISIBLE. Never a backlit silhouette, never an
+     underexposed frame, unless the user explicitly asks for one. LTX renders dark scenes
+     as mud — an unlit clip is a failed clip regardless of everything else.
+  6. Then add ENVIRONMENT and lens/film feel briefly (golden hour, 85mm, shallow depth of
+     field, low angle).
```

## Verification when applied

Matched seed, same scene, before/after: **mean brightness** is the primary metric
(target ≥ 80 / 255; Kira's failing clip was 26.4, a good manual render ~100–110).
Watch that saturation and the cyan-blown fraction do not regress — the metric that
matters is cyan-hued high-saturation pixels as a fraction of ALL pixels, not of
high-saturation pixels (that ratio false-positives on soft pastel scenes).

**Mode scope:** Changes 1–2 are t2v-only (content-mode templates). Change 3 touches
the shared block and therefore both — which is precisely why it must SPLIT rather than
amplify. `systemPromptVideoI2V`'s own section is correct as written; the problem is
what it inherits.

**Brightness is a t2v criterion only.** In i2v the brightness comes from the init
image, so a dark i2v clip means a dark SEED, not a prompt fault — gating i2v on
brightness would misattribute it.

## Sequencing

Ideally lands AFTER task #15 externalizes the templates, so the change is a file edit
with a recorded hash rather than a rebuild — and so the before/after is attributable
in the trace. If applied to source first, it costs a build + resign + bounce.
