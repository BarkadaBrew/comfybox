# PRD: the Creative Library

Status: draft v0.1 (2026-09-17). Nothing here is built.
Requested by Todd after reading a third-party library pack: "we should integrate all the good
ideas into Desktop app… it can be a wholistic improvement throughout."
Reference pack studied: `SickOllie_LibraryStarterPack.soslibrary` (format
`sickollie-creative-library-pack` v3, 145 MB, 1,466 files — 240 templates, 266 prompts, 520 scene
components, 336 wardrobe items, 12 outfits, 101 recipes, 1,450 thumbnails).

## 1. Problem

Everything needed to make a good render already exists somewhere in ComfyBox, and none of it is
reusable as *material*.

- A prompt that worked lives as text in the Prompt Library (positive only), or in a render's
  sidecar, or in a preset's prose field. Nothing carries its negative, its LoRA stack, its style
  or its seed together.
- Slot templates exist, but only inside Studio Packs, which are read-only, hand-written JSON,
  desktop-only, and invisible to Kira and Bree (`docs/prd-comfybox-studio-packs.md` FR-9 unbuilt).
- Fifteen prompt style presets ship in the engine (`ComfyBridgeStylePresets`, `GET /v1/styles`)
  and **no screen in the app reads them**.
- Camera, lighting and shot phrases are one-shot inserts. A look you liked cannot be saved.
- Characters carry `defaultLoras`, `triggerWords` and `negativePrompt` that **no render path
  consumes**, and there is no wardrobe concept anywhere.
- The gallery knows ratings, favourites and 22 catalog columns, but you cannot turn a render you
  liked back into something you can use again.

The result: every session starts from a blank prompt box, and taste accumulated over hundreds of
renders is not compounding.

## 2. What the pack got right (the ideas worth taking)

| Idea | In the pack | We have |
|---|---|---|
| Slot templates | 240 templates with literal `OUTFIT` / `NAME` / `BRAND` slots and a recorded `placeholder_signature` | Studio Pack `{slotId}` templates, read-only, in one built-in pack |
| Faceted taxonomy | every prompt tagged across cast / concept / content / mood / scene / shot / structure / style / time / subcategory | catalog facets over *renders*, nothing over material |
| Components | 520 scene fragments + 12 outfits, each with preview, rating, collections | nothing |
| Wardrobe | 336 garments in category → subtype, typed `piece` / `set` / `styling` | nothing |
| Recipes from images | 101 setups imported by reading image metadata: LoRA name+hash+strength, seed and seed source, node graph | `/v1/presets/import-legacy` (from another app's presets), CivitAI prompt repository |
| Collections | coloured, nested, with a saved display order per kind | smart collections over renders (`CollectionRules.swift`), rating + favourite |
| Pack merge policy | pack-provided membership tracked separately from yours, tombstones for deletions, reconcile-not-overwrite on reimport | `.cbarchive` for assets only |
| Scoped export | pick scopes, with/without thumbnails, dependency-aware | none for material |
| Previews everywhere | 1,450 thumbnails keyed by content hash, each recording its source pack | thumbnails for assets only |

## 3. Goals

1. **One library of reusable material**, shared by the engine and every client, covering
   templates, components, wardrobe, outfits, looks and recipes.
2. **Holistic, not a silo.** The library is a layer: Generate, Director, Gallery, the assistant,
   Kira and Bree all read and write it. The Library tab is one surface, not the point.
3. **Taste compounds.** Anything you rate, favourite or reuse feeds back. Any render can give
   back its recipe; anything you liked becomes material in one action.
4. **Provenance both ways.** A render records the template, components and recipe that made it. A
   library item records what produced it and how often it has been used.
5. **Packs.** Import and export curated sets, including third-party packs, without losing local
   edits.

### Non-goals

- Not a replacement for presets. A preset stays the render recipe the engine resolves.
- Not a new asset store. Renders stay in the DAM/catalog; the library references them.
- No content filtering or moderation features.
- No cloud sync or marketplace.
- Not a rewrite of Studio Packs in v1 — they become a pack type that imports into the library.

## 4. Users

- **Todd, at the desktop.** Browsing, curating, assembling a shot.
- **The assistant (Glimmer).** Drafting from real material instead of from nothing.
- **Bree.** Todd's assistant, with full write access: she curates on his behalf, files what he
  likes, and can build a collection from a conversation.
- **Kira.** Reads the library so her scheduled and requested renders use the same wardrobe, looks
  and templates Todd curates. Her own additions are proposals awaiting approval.
- **Director / Programs (Phase 2–3).** Shot prompts, wardrobe continuity and the bible come from
  the library.

## 5. Data model

One engine-side store (`~/.comfybox/library/`), one JSON file per kind plus a thumbnail
directory, served over HTTP and MCP so every client shares it.

**Common to every item**: `id`, `kind`, `name`, `value` (the text it contributes), `facets`
(multi-valued taxonomy), `collections`, `rating` (0–5), `favorite`, `archived`, `preview_ref`
(content-hashed thumbnail), `source` (`local` | `pack:<id>` | `render:<assetId>` | `import:<app>`),
`created_at`, `updated_at`, `use_count`, `last_used_at`, `notes`.

| Kind | Adds |
|---|---|
| `template` | `slots: [{id, label, placeholder, default, options[]}]`, `signature` (the slot ids it uses), `body` with `{slot}` markers. Superset of `StudioPackTemplate`. |
| `component` | `component_kind` (scene, pose, lighting, camera, expression, styling, negative), free `value` |
| `wardrobe_item` | `category`, `subtype`, `item_type` (piece / set / styling) |
| `outfit` | ordered `items: [wardrobe_item_id]` plus optional free text; renders to one phrase |
| `look` | prompt prefix/suffix, negative, LoRA stack refs, sampler hints — the saveable camera/lighting/style combination; seeded from `ComfyBoxStylePresets` and `StylePack` |
| `recipe` | a full render setup: preset id or inline preset snapshot, seed and seed source, model, LoRA stack with hashes, dims/steps/guidance, and the prompt as authored |
| `collection` | `color`, `parent_id`, `display_order`, `membership` (manual / derived / pack) |

**Facets** are a controlled vocabulary per axis, editable in the app, seeded from the pack's axes
(cast, concept, content, mood, scene, shot, style, time) plus ours (tier, character, model family).

**Relations**: `character → default outfit / look / wardrobe`, `template → recommended look`,
`recipe → preset`, `item → renders that used it` (the provenance edge, stored in the catalog's
existing `asset_edges` table as `used_item`).

## 6. Surfaces — the holistic part

**Library tab (⌘L)** — the front door. Source list of kinds and collections (coloured, nested,
reorderable). Grid of cards with previews, rating, use count. Facet filter rail. Full-text search.
Multi-select for collection membership, rating, export. Every card's primary action is **Use**,
never "view".

**Generate.** The prompt box gains slot awareness: choose a template, and its slots become fields
with pickers (wardrobe for `OUTFIT`, character for `NAME`, component for `SCENE`). A Looks row
replaces today's invisible style presets. "Save as…" on any render offers template, component,
wardrobe item, look or recipe.

**Director.** Shot prompts come from templates; the bible's wardrobe and scene entries are library
items, so continuity is by reference, not by retyping. The picker panel (FDD
`docs/FDD-director-projects-pickers.md`) gains Library as a source.

**Gallery.** Any asset → "Make into…" (component, look, recipe, template) and "Open recipe".
Ratings and favourites on a render propagate to the items that made it (a soft signal used for
ranking, never an automatic edit).

**Assistant.** The drafter is given the library index (names, facets, slots — not full prose) and
must cite the item ids it used. Its `AgentAction` gains `libraryRefs`, and applying an action no
longer drops negative prompt, LoRAs or model (survey gap 15).

**Bree and Kira.** `library_*` MCP tools plus HTTP. Bree writes (create, edit, rate, file,
collect); Kira reads and queues proposals. Kira's authored prompts fill slots from the library
rather than inventing wardrobe, so a "blue slip dress" means the same one every time.

**Canvas.** A board can be saved as a mood collection; items dropped on a board keep their ids.

## 7. Packs

**Our format** (`.cblibrary`, zip): `manifest.json` (id, name, creator, version, counts,
capabilities, selection, merge policy) + `library/*.json` + `previews/`. Deliberately modelled on
the reference pack, which got the hard parts right:

- **Scoped export.** Choose kinds, collections, with or without thumbnails; dependencies resolved
  (an outfit exports the wardrobe items it names).
- **Reconcile, don't overwrite.** Pack-provided collection membership is tracked separately from
  yours. Reimporting a newer pack updates pack membership and leaves your edits, ratings and
  collections alone.
- **Tombstones.** Deletions are recorded so a reimport does not resurrect what you removed.
- **Content-hashed previews** with a recorded source pack.
- **Dry-run first.** Import shows what will be added, updated, skipped and conflicting, before
  anything is written.

**Third-party import.** A reader for `sickollie-creative-library-pack` v3 maps templates,
prompts, components, wardrobe, recipes, collections, facets and previews onto our model. Their
recipes carry ComfyUI node graphs and LoRA hashes: we import the parts we can resolve (prompt,
seed, LoRA names + strengths, dims) and keep the raw payload for reference rather than pretending
to reproduce a foreign graph.

**Studio Packs** import as a pack: templates become `template` items, `prompt_prefix`/`suffix`
plus QA rules become a `look`, and the LoRA stack becomes a `recipe`. The built-in pack ships as a
library pack, and the Studio Packs disclosure group in Generate is replaced by the Looks and
Templates rows once the migration lands.

## 7b. Ideas from rgthree-comfy

Todd 2026-09-17: rgthree-comfy is named in the pack's CivitAI listing. Its recipes carry no
rgthree nodes (they use Sick Ollie's own `SOPromptLogEngineStudio` /
`SOGenerationPipelineStudio`), but rgthree is the clearest example of the *conveniences* that
make a heavy library usable. What is worth copying, and where it lands here:

| rgthree | What it solves | Where it lands |
|---|---|---|
| **Power Lora Loader** — many LoRAs, each a row with an on/off toggle and a strength, in one compact block | Editing a stack without adding and removing entries | The preset and recipe LoRA editors (`PresetView`): per-row enable toggle, inline strength, drag to reorder. A disabled row is remembered, not deleted |
| **Drag-and-drop widget import** — drop a previous generation onto the graph to reload its settings | Getting back a setup you already made | Drop any render (or a foreign PNG) on Generate or the Library tab to create a `recipe` from its metadata. This is the same move the pack makes with `imported_from_image` |
| **Power Prompt** — inline dropdowns for LoRAs and embeddings inside the prompt box | Not memorising trigger words and filenames | Generate's prompt field: `@` opens a library picker (component, wardrobe, look), `#` opens LoRA triggers, which the LoRA chips already know |
| **Auto-nested combos** — long dropdowns become hierarchical menus | 336 wardrobe items in one flat list is unusable | Every library picker groups by category → subtype, the pack's own structure |
| **Seed node** — random/fixed with the value kept in metadata | Reproducing one image out of a batch | Recipes store seed *and* seed source, as the pack does; Generate gets an explicit fixed/random control beside the seed field |
| **Image Comparer** — side-by-side with a slider | Judging two variants honestly | The Compare tab exists (`BatchSeedSweep` → Compare); add a slider mode and "compare against the item's best render" |
| **Fast Muter / Bypasser, group toggles** | Turning parts of a setup off without dismantling it | Toggle a look, a wardrobe layer or stage 2 off for one render without editing the preset |
| **Bookmarks** | Returning to a place in a large workspace | Saved library views (facet + collection + sort), which SmartTabs already do for the gallery |
| **Progress bar across the top** | Knowing what the queue is doing without switching tabs | A persistent queue strip in the desktop chrome, fed by `/v1/queue` |
| **Copy image to clipboard** | Getting a render into another app fast | Gallery and Generate context menus |

Not taken: node-graph plumbing (reroute, context, switches, puter). ComfyBox is not a graph editor,
and the Director timeline is the composition surface here.

## 8. Work packages

Order matters: the store and the API first, then the surfaces that make it compound.

| WP | Scope | Size |
|---|---|---|
| **L1** | Library store + schema: kinds, facets, collections, previews, use counts, provenance; atomic JSON persistence; migration of Prompt Library entries into `component`/`template` | M |
| **L2** | HTTP + MCP surface: `/v1/library/items` (CRUD, query by kind/facet/collection/text), `/v1/library/collections`, `/v1/library/facets`, `/v1/library/render` (fill a template's slots), MCP `library_search` / `library_get` / `library_upsert` / `library_fill_template` | M |
| **L3** | Library tab: source list, cards, facets, search, multi-select, ratings, collections, reorder | L |
| **L4** | Generate integration: template slot fields, wardrobe/outfit/component pickers, Looks row (surfacing `ComfyBoxStylePresets` + `StylePack` at last), "Save as…" from any render, `@`/`#` inline pickers in the prompt field, fixed/random seed control (§7b) | M |
| **L5** | Provenance loop: renders record the library items used (`asset_edges.used_item`), library items show their renders, use counts and "top rated" ordering | M |
| **L6** | Packs: `.cblibrary` export with scoping and dependencies, import with dry-run, reconcile, tombstones | M |
| **L7** | Third-party import: `.soslibrary` v3 reader with a mapping report; drop-an-image-to-make-a-recipe from our own sidecars and from foreign PNG/mp4 metadata (§7b) | M |
| **L7b** | Convenience pass (§7b): per-row LoRA toggles with remembered disabled rows, nested category pickers, saved library views, persistent queue strip, Compare slider mode, copy-image | M |
| **L8** | Studio Pack migration: import built-in + user packs, retire the Generate disclosure group | S |
| **L9** | Director + assistant integration: Library as a picker source, drafter citations, `AgentAction.libraryRefs`, and the apply-drops-fields fix | M |
| **L10** | Character links: default outfit / look / wardrobe, and honouring `defaultLoras` / `triggerWords` / `negativePrompt` on image renders (survey gap 10) | S |

## 9. Acceptance

- **L1–L2:** a wardrobe item created in the app is visible to `library_search` from Bree within
  one second, and filling a template through `/v1/library/render` substitutes every slot, leaving
  unknown markers visible (the Studio Pack contract).
- **L3:** 1,400 items browse and filter at 60 fps; facet filtering narrows 520 scene components to
  a working set in under three interactions.
- **L4:** a full prompt can be assembled from library material with no free typing: template +
  outfit + scene + look, and the resulting render matches what the same text typed by hand
  produces (same seed, byte-identical prompt).
- **L5:** open any render from the last week and see the items it used; open any item and see its
  renders.
- **L6:** export a collection, delete an item locally, reimport, and confirm the deletion stands
  (tombstone) while a locally added collection membership survives.
- **L7:** the reference pack imports with counts matching its manifest, with a report naming
  anything skipped.
- **L10:** a render for a character with a default outfit uses that outfit's wardrobe items
  verbatim.

## 10. Risks

- **A museum, not a workshop.** A big browsable library nobody generates from. Mitigation: every
  card's primary action is Use; the tab ships *after* the Generate integration in the same release.
- **Taxonomy sprawl.** Ten facet axes invite inconsistent tagging. Mitigation: controlled
  vocabulary, bulk retag, and facets suggested by the assistant from an item's text.
- **Two sources of truth with presets.** A `recipe` that drifts from its preset. Mitigation: a
  recipe references a preset id where one exists and snapshots only what the preset does not hold.
- **Foreign recipes.** Imported ComfyUI graphs cannot be replayed here. Mitigation: import the
  resolvable fields, keep the raw payload, and label those recipes "reference only".
- **Prose in library files.** Library JSON will contain prompt text, including explicit material.
  It follows the existing rule for the preset store: never committed, never in fixtures or
  tickets, ids only in logs.
- **Scope.** Ten work packages is a programme, not a sprint. L1, L2 and L4 alone deliver most of
  the daily value; the rest can follow.

## 11. Decisions (Todd, 2026-09-17)

1. **Generate first, tab second.** L1, L2 and L4 ship as release one: slot fields, wardrobe,
   outfit and component pickers, and the Looks row. The Library tab (L3) follows. A tab with
   nothing generating from it is a museum.
2. **Studio Packs migrate and retire in v1** (L8 moves into release one's tail). There is one
   built-in pack and no in-app editing, so two template systems cost more than they protect.
3. **Import the reference pack's wardrobe, then prune.** Its 336 garments and 10 categories are
   the starting vocabulary; Todd deletes what does not fit. An empty wardrobe never gets filled.
4. **Bree writes, Kira proposes.** Bree has full write access — she is Todd's assistant and acts
   on his behalf, so she can create, edit, rate and file items directly. Kira reads, and may
   *propose* additions from her own good renders, queued for Todd's approval. The distinction is
   agency, not trust: Kira is a persona whose taste is the subject of the library, not its
   curator.

**Release one = L1 + L2 + L4 + L7 (import) + L8 (Studio Pack migration).**
Release two = L3 (tab) + L5 (provenance loop) + L6 (packs out) + L7b + L9 + L10.
## 12. Questions answered above (kept for history)

1. **Ship order.** L1+L2+L4 (slots and pickers in Generate, no tab) first, with the tab in a
   second release, or all three together?
2. **Studio Packs.** Migrate and retire in v1 (L8), or keep both surfaces for a release?
3. **Wardrobe depth.** Import the reference pack's 336 items and 10 categories as the starting
   vocabulary, or start from your own renders and grow it?
4. **Kira's authority.** May Kira *add* to the library (new components from her own good renders),
   or only read it?
