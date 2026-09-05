# MCP Tool Surface Consolidation — Plan for #291

Status: **PROPOSAL — not a decision.** This document changes nothing by itself. It
is the input for Todd to approve, adjust, or reject before any code lands.
Nothing here silently changes the daemon contract (intent.md): every existing
tool name keeps working, unchanged, for at least one release after any merge
ships.

## 0. Scope and method

- **Surface counted:** the 56 tools on `origin/main` (`Sources/ZImage/MCP/MCPToolRegistry.swift`,
  commit `b505065`, which already includes #367's `get_job`) plus `nearline_anchor`,
  which is implemented and reviewed on the in-flight `feat/anchor-to-internal-273`
  branch but not yet merged to main. **57 tools total** — matching the count in
  the task brief.
- **Daemon call-site evidence:** `~/Projects/coffeeshop-server/src` is the only
  process with a standing MCP connection to ComfyBox (Bree; Kira runs inside the
  same daemon process and shares its tool-calling code). ComfyBox registers its
  tools to that client as `mcp_comfybox__<tool_name>` (confirmed in
  `docs/mcp-reference.md`: "Tools are registered as `mcp_comfybox__<tool_name>`").
  Call-site counts below are `grep -rw` hits on the literal string
  `mcp_comfybox__<tool>` across `src/**/*.ts`, excluding `__tests__` and
  `*.test.*`. This catches actual invocations (`callDaemonTool(...)`,
  `executor.execute(...)`, a `case` in a dispatch switch) as well as static
  registration in `src/tool-categories.ts` and `src/resilience/tool-resilience.ts`
  — both counted as "the daemon knows about this tool," which is the relevant
  signal for a consolidation plan even where the literal call is behind a
  generic dispatcher.
  - **Caveat:** a naive `grep -w <tool_name>` (no `mcp_comfybox__` prefix) is
    noisy for tools whose name is also an ordinary English word or a common
    field name (`upscale`, `get_config`, `list_characters`, `extend_video`,
    `list_workflows` all collide with unrelated local functions/params). The
    prefixed grep avoids that; it is the number reported in the table.
  - **Important finding:** `src/tool-categories.ts` — the list Bree's own
    conversational LLM is actually offered to pick from — contains only 21 of
    the 57 tools (the original tool set, roughly through 2026-07). Everything
    added since (video jobs, winner actions, workflows, nearline, CivitAI,
    presets/characters CRUD, config, `get_job`, `cancel_job`, `queue_list`,
    `interrupt_render`, `move_queue_job`, `enhance_prompt`, `repair_image`,
    `update_lora_triggerwords`) is invoked only by **direct, non-LLM-mediated
    code** in the daemon (button handlers, scheduler code, Telegram command
    handlers) — never by an LLM choosing among them. So the actual
    "small-model-picks-a-tool-by-name" exposure that #291 is worried about is
    narrower than 57: it's the full `tools/list` response any bare MCP client
    sees (Claude Code's `claude mcp add comfybox`, and any future Kira
    component that talks to ComfyBox's MCP server directly instead of through
    Bree's curated dispatcher) — but the fix has to be at the server, since
    ComfyBox has no way to know which kind of client asked.
- **Desktop/other callers:** the ComfyBox Desktop app (`Sources/ComfyBoxDesktop`)
  never uses MCP — it calls the same HTTP routes directly
  (`EngineService.swift`, `SettingsView.swift`, `CharactersView.swift`,
  `PresetView.swift`, `MotionView.swift` hit `/v1/config`, `/v1/presets`,
  `/v1/characters`, `/v1/nearline`). **This means every MCP-only merge in this
  plan is invisible to the Desktop app — it is not a caller of any of these
  tools and cannot be broken by consolidating them.** Where a proposed merge
  would also need the underlying HTTP route to change, that's flagged
  separately (none of the merges below touch routes).
- Not grepped: Todd's own direct Claude Code/Desktop MCP usage of ComfyBox
  (invisible from `coffeeshop-server`). Assume it exists for any tool until
  Todd says otherwise — this is called out per-tool below only where the
  daemon evidence is zero, since zero-daemon-evidence is not zero-evidence.

## 1. Full tool inventory (57)

Annotation legend: RO = readOnly, ADD = additive, DES = destructive (from #297).
Routes column is abbreviated (see `MCPToolRegistry.swift` for exact `RouteRef`s;
`*` = comfyUICompat surface, i.e. the ComfyUI-bridge dispatcher, not WarmServer's
native `/v1` surface). Call sites = `mcp_comfybox__<tool>` hits in
`coffeeshop-server/src`.

| # | Tool | Annot. | Routes | Daemon call sites | Other callers | Verdict |
|---|------|--------|--------|---:|---|---|
| 1 | `generate_image` | ADD | `POST /v1/generate`, `/v1/generate/async`, `GET /v1/queue` | 39 | Desktop: no (REST direct). Todd/Claude: likely. | **KEEP** |
| 2 | `repair_image` | ADD | `POST /v1/generate` | 1 | — | **KEEP** |
| 3 | `swap_loras` | ADD | `POST /v1/lora/swap` | 6 | Desktop: yes, via REST directly (not this tool) | **KEEP** |
| 4 | `list_models` | RO | (none declared) | 7 | — | **KEEP** |
| 5 | `list_styles` | RO | (none declared) | 6 | — | **MERGE → `style`** |
| 6 | `server_health` | RO | (none declared) | 3 | — | **KEEP** (absorbs `system_stats`) |
| 7 | `queue_status` | RO | (none declared) | 6 | — | **KEEP** (absorbs `queue_list`) |
| 8 | `clear_queue` | DES | `POST /queue` * (comfyUICompat, not native `/v1/queue/clear`) | 3 | — | **MERGE → `queue_control`** |
| 9 | `pause_queue` | ADD | `POST /v1/queue/pause` | 0 | — | **MERGE → `queue_control`** |
| 10 | `resume_queue` | ADD | `POST /v1/queue/resume` | 0 | — | **MERGE → `queue_control`** |
| 11 | `list_loras` | RO | (none declared) | 6 | — | **MERGE → `lora_library`** |
| 12 | `shutdown_server` | DES | `POST /v1/shutdown` | 1 | — | **KEEP** (safety-gated, do not fold into anything) |
| 13 | `system_stats` | RO | (none declared) | 3 | — | **MERGE → `server_health`** |
| 14 | `apply_style` | RO | (none declared) | 2 | — | **MERGE → `style`** |
| 15 | `lora_library` | RO | (none declared) | 3 | Desktop: no | **KEEP** (absorbs `list_loras`) |
| 16 | `lora_scan` | ADD | `POST /v1/loras/scan` | 6 | `style-installer.ts` hardcodes this tool name as its default LoRA-registration path | **MERGE → `lora_manage`** |
| 17 | `lora_quarantine` | DES | `POST`/`DELETE /v1/loras/{id}/quarantine` | 1 | — | **MERGE → `lora_manage`** |
| 18 | `load_model` | ADD | `POST /v1/model/load` | 3 | `image-bot.ts` direct call, 30s timeout hardcoded | **MERGE → `model_manage`** |
| 19 | `switch_model` | ADD | `POST /v1/model/activate` | 1 | — | **MERGE → `model_manage`** |
| 20 | `model_pool` | RO | (none declared) | 3 | `kira-api.ts` polls this in a status loop | **MERGE → `model_manage`** |
| 21 | `unload_model` | ADD | `POST /v1/model/unload` | 1 | — | **MERGE → `model_manage`** |
| 22 | `generate_video` | ADD | `POST /v1/video/generate/async` | 70 | — | **KEEP** |
| 23 | `video_status` | RO | (none declared) | 49 | — | **DEPRECATE → alias of `get_job`** (see Risks — vocabulary mismatch) |
| 24 | `get_job` | RO | `GET /v1/generate/status/{id}`, `/v1/video/status/{id}`, `/v1/queue` | 0 (shipped today, daemon not migrated yet) | — | **KEEP** (target of all job-poll aliases) |
| 25 | `compose_montage` | ADD | `POST /v1/montage/compose` | 2 | — | **KEEP**¹ |
| 26 | `render_storyboard` | ADD | `POST /v1/storyboard/render` | 0 | — | **KEEP** |
| 27 | `rerender_video` | ADD | `POST /v1/video/rerender` | 5 | — | **KEEP** |
| 28 | `extend_video` | ADD | `POST /v1/video/extend` | 5 | — | **KEEP** |
| 29 | `import_workflow` | ADD | `POST /v1/workflows/import` | 0 | Possible: Krita bridge / ComfyUI-compat clients (unconfirmed — those talk to the emulated ComfyUI HTTP protocol, not necessarily this MCP tool) | **MERGE → `workflow`** |
| 30 | `list_workflows` | RO | (none declared) | 0 | same as above | **MERGE → `workflow`** |
| 31 | `run_workflow` | ADD | `POST /v1/workflows/{id}/run` | 0 | same as above | **MERGE → `workflow`** |
| 32 | `workflow_run_status` | RO | (none declared) | 0 | same as above | **MERGE → `workflow`** (action `status`) |
| 33 | `upscale` | ADD | `POST /v1/upscale` | 10 | — | **KEEP** |
| 34 | `enhance_prompt` | ADD | `POST /v1/enhance` | 0 | `generate_video`'s own `enhance:true` default means the server already self-enhances; this standalone tool may be Todd/Claude-only | **KEEP** |
| 35 | `list_characters` | RO | (none declared) | 0 | — | **MERGE → `character_manage`** |
| 36 | `list_presets` | RO | (none declared) | 0 | Desktop: yes, via REST (`PresetView.swift`) | **MERGE → `preset_manage`** |
| 37 | `import_legacy_presets` | ADD | `POST /v1/presets/import-legacy` | 0 | One-time migration utility | **MERGE → `preset_manage`** |
| 38 | `queue_list` | RO | (none declared) | 0 | — | **MERGE → `queue_status`** |
| 39 | `interrupt_render` | DES | `POST /v1/queue/interrupt` | 0 | — | **MERGE → `queue_control`** |
| 40 | `cancel_job` | DES | `DELETE /v1/queue/{id}` | 1 | — | **KEEP** (3rd queue-domain tool per the issue's own "status/control/cancel" framing) |
| 41 | `nearline_list` | RO | (none declared) | 0 | Desktop: yes, via REST (`EngineService.swift`) | **MERGE → `nearline_manage`** |
| 42 | `nearline_scan` | ADD | `POST /v1/nearline/scan` | 0 | Desktop: yes, via REST | **MERGE → `nearline_manage`** |
| 43 | `nearline_stage` | ADD | `POST /v1/nearline/stage` | 0 | Desktop: yes, via REST | **MERGE → `nearline_manage`** |
| 44 | `nearline_evict` | DES | `POST /v1/nearline/evict` | 0 | Desktop: yes, via REST | **MERGE → `nearline_manage`** |
| 45 | `nearline_anchor` | ADD | `POST /v1/nearline/anchor` (from #273, unmerged) | 0 | Desktop: yes, via REST (#273 also adds a Desktop UI toggle) | **MERGE → `nearline_manage`** |
| 46 | `civitai_search` | RO | (none declared) | 0 | Desktop: per tool description, gated on "the Desktop app's saved key" | **MERGE → `civitai`** |
| 47 | `civitai_prompts` | ADD | `POST /v1/civitai/harvest` | 0 | same | **MERGE → `civitai`** |
| 48 | `move_queue_job` | ADD | `POST /v1/queue/{id}/move` | 0 | — | **MERGE → `queue_control`** |
| 49 | `update_lora_triggerwords` | ADD | `POST /v1/loras/{id}/update` | 0 | — | **MERGE → `lora_manage`** |
| 50 | `create_preset` | ADD | `POST`/`PUT /v1/presets` | 0 | Desktop: likely, presets editor | **MERGE → `preset_manage`** |
| 51 | `delete_preset` | DES | `DELETE /v1/presets/{id}` | 0 | Desktop: likely | **MERGE → `preset_manage`** |
| 52 | `set_warm_preset` | ADD | `POST /v1/model/activate`+`/load`, `GET`/`PUT /v1/config` | 0 | Desktop: yes — description says it "mirrors the Desktop app's Preset 'Set as Warm' action exactly" (Desktop has its own code path; this tool exists for non-Desktop callers) | **MERGE → `preset_manage`** |
| 53 | `create_character` | ADD | `POST`/`PUT /v1/characters` | 0 | Desktop: likely (`CharactersView.swift`) | **MERGE → `character_manage`** |
| 54 | `delete_character` | DES | `DELETE /v1/characters/{id}` | 0 | Desktop: likely | **MERGE → `character_manage`** |
| 55 | `get_config` | RO | `GET /v1/config` | 0 | Desktop: yes (`SettingsView.swift`) | **MERGE → `config`** |
| 56 | `patch_config` | ADD | `PATCH /v1/config` | 0 | — | **MERGE → `config`** |
| 57 | `update_config` | ADD | `PUT /v1/config` | 0 | Desktop: yes (`SettingsView.swift`) | **MERGE → `config`** |

¹ `compose_montage` has real daemon usage and a distinct purpose (assembling
stills/clips into one file, no diffusion model). It stays a standalone tool,
filed under the `video` domain in §2.

**Tally:** 57 tools → 39 marked MERGE-into-something, 1 marked DEPRECATE
(`video_status` folds into `get_job` directly rather than into a same-domain
tool — the whole point of #367's job model — so it's tagged DEPRECATE rather
than MERGE; `workflow_run_status` is tagged MERGE since it folds into a new
same-domain tool's `status` action alongside three siblings), 17 marked KEEP
as independent tools (three of which absorb a sibling's data with no schema
change: `server_health` gains `system_stats`'s fields, `lora_library` gains
`list_loras`'s data, `queue_status` gains `queue_list`'s data).

**Daemon call-site summary:** 26 of 57 tools have at least one
`mcp_comfybox__` hit in `coffeeshop-server/src` (i.e., live daemon usage);
31 have zero (deprecated/merged with no visible internal caller, but see the
Desktop-REST and Todd-direct caveats above before treating "zero" as "safe to
delete outright" — this plan never deletes, only aliases, precisely because
of that uncertainty).

## 2. Target surface (~25, this proposal lands at 27)

Grouped as the issue suggested, plus a `system` bucket for the two tools that
don't fit generate/jobs/models/nearline/presets/config/queue/video (server
health and shutdown have no natural home in those nine).

### generate (4)
`generate_image`, `repair_image`, `upscale`, `enhance_prompt`

### jobs (1)
`get_job` — sole poll tool. `video_status` and `workflow_run_status` become
thin deprecated aliases (§3); no new tool needed since `get_job` already
covers image/video/storyboard.

### models / LoRAs (6)
`list_models`, `model_manage`, `swap_loras`, `lora_library`, `lora_manage`, `civitai`

### nearline (1)
`nearline_manage`

### presets (3)
`preset_manage`, `character_manage`, `style`

### config (1)
`config`

### queue (3)
`queue_status`, `queue_control`, `cancel_job`

### video (6)
`generate_video`, `compose_montage`, `render_storyboard`, `rerender_video`, `extend_video`, `workflow`

### system (2)
`server_health`, `shutdown_server`

### gallery (0)
No MCP tools exist for gallery browsing/search today (out of scope for #291 —
flagging the gap, not proposing to fill it here).

**Total: 4+1+6+1+3+1+3+6+2 = 27.** This is 2 above the issue's "~25," not 25
exactly. The two soft spots, left for Todd to call:

- **`workflow`** (4 old tools → 1) has **zero** confirmed daemon callers and
  unconfirmed Desktop/Krita usage. If Todd confirms nothing uses ComfyUI
  workflow import/run today, this entire cluster could be **DEPRECATE**
  outright (remove after one release with no replacement) instead of
  MERGE-and-keep, which would land the total at exactly 25 without inventing
  a replacement nobody asked for.
- Alternatively, **`shutdown_server`** could fold into `queue_control` as
  `action: "shutdown"` — but it's the one tool in the whole surface with a
  hand-written safety guard (`confirm: true` required) for an operation nothing
  else in `queue_control` shares (server lifecycle vs. render lifecycle). This
  plan recommends keeping it separate on safety grounds, at the cost of one
  tool over target.

### Merged-tool schemas

Each follows #367's `get_job` pattern: one discriminated field selects the
old tool's behavior; other fields are that old tool's own parameters,
documented as "used when `action` is X."

**`nearline_manage`** (replaces `nearline_list`, `nearline_scan`, `nearline_stage`, `nearline_evict`, `nearline_anchor`)
```jsonc
{
  "action": { "enum": ["list", "scan", "stage", "evict", "anchor"], "required": true },
  "item":   { "type": "string", "description": "Filename from a prior `list`. Required for stage/evict/anchor." },
  "kind":   { "enum": ["lora", "model"], "description": "Required for action=anchor only." },
  "anchored": { "type": "boolean", "description": "Required for action=anchor only: true to pin, false to unpin." }
}
```
Old→new: `nearline_list()` → `{action:"list"}`; `nearline_scan()` →
`{action:"scan"}`; `nearline_stage({name})` → `{action:"stage", item:name}`;
`nearline_evict({name})` → `{action:"evict", item:name}`;
`nearline_anchor({kind,id,anchored})` → `{action:"anchor", item:id, kind, anchored}`.

**`queue_control`** (replaces `pause_queue`, `resume_queue`, `clear_queue`, `interrupt_render`, `move_queue_job`)
```jsonc
{
  "action":    { "enum": ["pause", "resume", "clear", "interrupt", "move"], "required": true },
  "id":        { "type": "string", "description": "Pending job id. Required for action=move only." },
  "direction": { "enum": ["top", "up", "down"], "description": "Required for action=move only." }
}
```
`clear_queue`'s route is the ComfyUI-bridge `POST /queue {clear:true}`
(comfyUICompat surface), deliberately NOT the native `/v1/queue/clear`
(documented in-source as "DECLARED REALITY," pinned by a parity test). The
merged tool's `action="clear"` handler must keep dispatching to that same
route — this is a routing detail to preserve exactly, not a behavior to
"clean up" during the merge.

**`queue_status`** (replaces `queue_list`; response is `queue_status`'s existing summary superset with `queue_list`'s per-job ids folded in — no schema change, both took no params)

**`model_manage`** (replaces `load_model`, `switch_model`, `model_pool`, `unload_model`)
```jsonc
{
  "action":       { "enum": ["list", "load", "activate", "unload"], "required": true },
  "model":        { "type": "string", "description": "Required for load/activate/unload." },
  "quantization": { "type": "string", "description": "load only." },
  "activate":     { "type": "boolean", "description": "load only, default true." },
  "wait":         { "type": "boolean", "description": "load only, default true." }
}
```
`switch_model` becomes `action="activate"`.

**`lora_manage`** (replaces `lora_scan`, `lora_quarantine`, `update_lora_triggerwords`)
```jsonc
{
  "action":        { "enum": ["scan", "set_quarantine", "set_triggerwords"], "required": true },
  "force":         { "type": "boolean", "description": "scan only, default false." },
  "id":            { "type": "string", "description": "Required for set_quarantine/set_triggerwords." },
  "quarantine":    { "type": "boolean", "description": "set_quarantine only." },
  "reason":        { "type": "string", "description": "set_quarantine only." },
  "triggerwords":  { "type": "array", "items": {"type":"string"}, "description": "set_triggerwords only." }
}
```

**`civitai`** (replaces `civitai_search`, `civitai_prompts`)
```jsonc
{
  "action": { "enum": ["search", "prompts"], "required": true },
  // shared search params (both actions): query, types, base_model, sort, period, nsfw, limit, site
  "harvest": { "type": "boolean", "description": "prompts only: run a fresh harvest before querying." },
  "filter_base_model": {}, "filter_act": {}, "filter_tag": {}, "keyword": {}, "max_entries": {}
  // (all prompts-only query filters, unchanged from civitai_prompts)
}
```

**`config`** (replaces `get_config`, `patch_config`, `update_config`)
```jsonc
{
  "action": { "enum": ["get", "patch", "replace"], "required": true },
  "patch":  { "type": "object", "description": "RFC 7386 JSON Merge Patch. Required for action=patch." },
  "config": { "type": "object", "description": "Full document. Required for action=replace." }
}
```

**`style`** (replaces `list_styles`, `apply_style`)
```jsonc
{
  "action":          { "enum": ["list", "apply"], "required": true },
  "style_id":        { "type": "string", "description": "apply only, required." },
  "prompt":          { "type": "string", "description": "apply only, required." },
  "negative_prompt": { "type": "string", "description": "apply only." }
}
```

**`preset_manage`** (replaces `list_presets`, `create_preset`, `delete_preset`, `set_warm_preset`, `import_legacy_presets`)
```jsonc
{
  "action": { "enum": ["list", "save", "delete", "set_warm", "import_legacy"], "required": true },
  "id": {}, "name": {}, "description": {}, "media_kind": {}, "provider": {}, "engine": {},
  "model": {}, "prompt": {}, "negative_prompt": {}, "prompt_prefix": {}, "prompt_suffix": {},
  "steps": {}, "guidance": {}, "seed": {}, "width": {}, "height": {}, "scheduler": {}, "loras": {}
  // ^ all "save only" (id+name required), same fields as today's create_preset.
  // "delete": needs only id. "set_warm": needs only model. "list"/"import_legacy": no params.
}
```

**`character_manage`** (replaces `list_characters`, `create_character`, `delete_character`)
```jsonc
{
  "action": { "enum": ["list", "save", "delete"], "required": true },
  "id": {}, "name": {}, "kind": {}, "description": {}, "base": {}, "banana": {}, "avocado": {},
  "default_loras": {}, "prompt_snippet": {}, "negative_prompt": {}
  // save: name required (id defaults to slug). delete: id required. list: no params.
}
```

**`workflow`** (replaces `import_workflow`, `list_workflows`, `run_workflow`, `workflow_run_status`)
```jsonc
{
  "action": { "enum": ["import", "list", "run", "status"], "required": true },
  "name": {}, "workflow_json": {},           // import only (workflow_json required)
  "workflow_id": {}, "prompt": {}, "negative_prompt": {}, "seed": {}, "output_path": {}, // run only (workflow_id required)
  "run_id": {}                                // status only, required
}
```

**`server_health`** (absorbs `system_stats` — no schema change; response gains device/VRAM fields alongside the existing model/LoRA/memory/queue fields; both took no params today)

**`lora_library`** (absorbs `list_loras` — no schema change; `list_loras`'s bare-filename list becomes redundant with `lora_library`'s existing unfiltered call)

## 3. Compatibility plan

For one full release after any merge ships:

- Every one of the 33 old tool names in §1 marked MERGE or DEPRECATE **stays
  registered** in `MCPToolRegistry.tools`, as a thin wrapper that translates
  its own call into the new tool's `action` and forwards. Byte-for-byte
  identical request/response shape to today, except:
  - `video_status`'s response CANNOT be a pure passthrough of `get_job` (see
    Risks §4) — its alias must keep constructing `video_status`'s own
    `{status, output_path, duration}` shape from whatever `get_job` returns
    internally, not simply proxy `get_job`'s `{state, progress, ...}` object.
  - Every other alias (`queue_list`, `pause_queue`, `system_stats`, `list_loras`,
    etc.) is response-transparent — old tool called old route/logic under the
    hood, so its output doesn't change at all, just its internal
    implementation routes through the new consolidated tool's action handler.
- Each alias's `description` gets a prefix: `"[DEPRECATED — use <new_tool>
  with action=<x>. Will be removed after <release>.]"` followed by its
  existing description text, unchanged.
- Each alias's annotation gains `deprecated: true`. This is a **new field**
  on `MCPToolAnnotations` / the `tools/list` response — it doesn't exist in
  `MCPTypes.swift` today (checked: no `deprecated` anywhere in
  `Sources/ZImage/MCP/`). Adding it is itself a small additive schema change,
  scoped to the first implementation PR (Phase 1 below), not this doc.
- **One removal ticket per alias**, filed when the alias PR merges (Phase 1),
  titled `MCP: remove deprecated alias <tool_name> (superseded by <new_tool>)`,
  referencing #291, not to be closed until:
  1. the daemon migration PR (Phase 2) has removed the last
     `mcp_comfybox__<old_name>` reference in `coffeeshop-server/src`, confirmed
     by the same grep methodology as §0, and
  2. Todd confirms no other client (Desktop, Claude Code, Krita bridge) still
     calls it.
- Tools with **zero** daemon call sites today (31 of 33 merge/deprecate
  candidates) get the same one-release alias treatment as the 2 with heavy
  usage (`video_status`, and to a lesser extent `queue_status`/`clear_queue`/
  `lora_scan`/`swap_loras`-adjacent tools) — zero evidence in one repo is not
  proof of zero callers (see §0 caveat on Todd/Claude Desktop/Krita usage).

## 4. Risks — tools with live daemon call sites whose semantics would change

1. **`video_status` → `get_job` (49 call sites — the single largest dependency
   in this entire plan).** The vocabularies genuinely differ:
   `video_status.status` ∈ `{queued, processing, succeeded, failed}` vs.
   `get_job.state` ∈ `{queued, running, completed, failed, unknown}`. Three of
   `coffeeshop-server`'s files pattern-match on the literal strings
   (`src/tools/video-tools.ts` ×3, `src/tools/comfybox-http-video-executor.ts`,
   `src/resilience/tool-resilience.ts`'s idempotency profile keyed on the tool
   name). The alias must translate `get_job`'s internal state into
   `video_status`'s exact old vocabulary on the way out — it cannot just
   forward `get_job`'s payload. Get this translation wrong and every in-flight
   video poll in production breaks silently (jobs that finished successfully
   would report as `unknown`/never-terminal to the daemon's poll loop).
2. **`clear_queue` → `queue_control(action="clear")` (3 call sites).** This
   tool's route is *already* a documented special case — it hits the
   ComfyUI-bridge `/queue` endpoint, not the native `/v1/queue/clear`, per an
   explicit in-source comment warning future maintainers not to "fix" this.
   Folding it into a generic `queue_control` action makes it easy to lose that
   special case (e.g., if the new tool's dispatcher defaults every action to
   the native `/v1` surface and only `clear` needs the comfyUICompat one). The
   merge must preserve the surface distinction per-action, not just per-tool.
3. **`model_pool` / `load_model` (3 + 3 call sites, both polled/called from
   `kira-api.ts` and `image-bot.ts` respectively).** `kira-api.ts` polls
   `model_pool` in a status loop with its own timing assumptions;
   `image-bot.ts` calls `load_model` with a hardcoded 30s timeout. Folding
   both into `model_manage`'s `action` field changes the *tool name* every
   caller must send — this isn't a response-shape risk like #1, but every one
   of these call sites needs a coordinated one-line edit in Phase 2, and
   missing even one means that code path silently starts calling a
   now-nonexistent-in-spirit... no — it keeps working via the alias, so the
   actual risk here is lower than #1/#2 *as long as the alias lands first*.
   Called out because it's the highest-call-site merge after `video_status`/
   `clear_queue`, not because the alias is unsafe.
4. **`lora_scan` (6 call sites, including `style-installer.ts`'s hardcoded
   default LoRA-registration path).** Same shape as #3 — safe under the alias
   plan, flagged because it's a real production install-time dependency, not
   a theoretical one.
5. **Anything with zero grep hits is not proven safe.** Per §0/§3, Desktop
   (`get_config`, `update_config`, `create_preset`, `delete_preset`,
   `set_warm_preset`, `create_character`, `delete_character`, all five
   `nearline_*`) calls the *routes* these tools wrap directly over REST, not
   through MCP — so the MCP merge is safe with respect to Desktop specifically.
   But Todd's own direct Claude Code/Desktop MCP usage of any zero-hit tool
   is invisible to this analysis and is exactly why every merge — even the
   zero-evidence ones — gets the same alias treatment, not a fast-tracked
   removal.

## 5. Phased PR list

**Phase 0 (this PR):** this document only. No code, no schema. `Refs #291`.

**Phase 1 — Add consolidated tools + aliases (additive only, one PR per domain
recommended, all against `main`):**
1. `config` (new) + `get_config`/`patch_config`/`update_config` become aliases.
2. `style` (new) + `list_styles`/`apply_style` become aliases.
3. `character_manage` (new) + `list_characters`/`create_character`/`delete_character` become aliases.
4. `preset_manage` (new) + `list_presets`/`create_preset`/`delete_preset`/`set_warm_preset`/`import_legacy_presets` become aliases.
5. `nearline_manage` (new, depends on #273 landing first) + all 5 `nearline_*` become aliases.
6. `civitai` (new) + `civitai_search`/`civitai_prompts` become aliases.
7. `workflow` (new) + `import_workflow`/`list_workflows`/`run_workflow`/`workflow_run_status` become aliases. **(Pending Todd's call in §2 — could instead be a straight deprecation with no replacement.)**
8. `lora_manage` (new) + `lora_scan`/`lora_quarantine`/`update_lora_triggerwords` become aliases; `lora_library` absorbs `list_loras`.
9. `model_manage` (new) + `load_model`/`switch_model`/`model_pool`/`unload_model` become aliases.
10. `queue_status` absorbs `queue_list`; `queue_control` (new) + `pause_queue`/`resume_queue`/`clear_queue`/`interrupt_render`/`move_queue_job` become aliases (preserve the `clear` comfyUICompat routing — Risk #2).
11. `server_health` absorbs `system_stats`.
12. `video_status` becomes a `get_job`-backed alias with response-shape translation (Risk #1) — do this one alone, last, with its own PR and its own test pass, given the call-site count.

Each Phase 1 PR: adds the `deprecated: true` annotation field to
`MCPToolAnnotations`/`MCPTypes.swift` (first PR only), updates
`docs/mcp-reference.md`, files the removal tickets for its aliases, runs
`-only-testing:ZImageTests`.

**Phase 2 — Daemon migration (in `coffeeshop-server`, separate repo, separate
PRs, one per domain matching Phase 1):** update every `mcp_comfybox__<old_name>`
call site found in §0/§4 to call the new consolidated tool with the right
`action`. Highest priority: `video_status` (49 sites) and `clear_queue`
(3 sites, verify the comfyUICompat route survived). Re-run the same grep
methodology as §0 after each PR to confirm zero remaining references to the
migrated old name.

**Phase 3 — Removal (one release later, per alias, only after both Phase 1's
alias and Phase 2's daemon migration are confirmed and Todd has verified no
other client depends on the old name):** delete each alias from
`MCPToolRegistry.swift`, close its removal ticket, update
`docs/mcp-reference.md`'s tool count.
