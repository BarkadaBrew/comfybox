# Swift-Qwen3.8-27b vs Muse-Glimmer-30B — text-workload evaluation (2026-09-16)

**Setup.** Both served by the same mlx-serve 26.8.10 on the Mac (:11234), GPU idle (ComfyBox idle, admission local, no video jobs). Glimmer = `Muse-Glimmer-30B-heretic-MLX-Q6` (6-bit, 23.4 GB, vision). Swift = `andosen/Swift-Qwen3.8-27b-mlx-5Bit` (5-bit, 17.2 GB, text-only) + the stock `mlx-community/Qwen3.8-27B-MTP-4bit` head grafted as `mtp/weights.safetensors` (speculation on). Authoring tests ran through the **production code path** (`optimizeT2VPrompt` / `optimizePrompt` from coffeeshop-server, real system prompts, temp 0.4, `reasoning_budget 256`) with Kira's engine identity block; tool tests used 42 tools from Bree's live catalog (56 KB); long-context = a 900-line synthetic ledger with one needle. Raw outputs: session scratchpad `eval-vs-glimmer.json`, `eval-t2v-more.json`. Avocado register not exercised (Todd's purview).

## Results

| Test | Glimmer | Swift | Read |
|---|---|---|---|
| t2v authoring (apple/neutral), 6 seeds | 12.2–18.8 s; decode 15–20 tok/s; **BEATS line 6/6** | 5.6–12.0 s; decode 33 tok/s; **BEATS line 0/6** | Prose quality equivalent; Swift ignores the beats instruction as placed today → scheduler falls back to even splits (#1819 regression). Fixable: with the instruction restated at the END of the user message Swift emits BEATS (probe 1/1). Thinking ON makes Swift spend its whole budget reasoning and return empty content — keep thinking off, as production does. |
| Image authoring (neutral), 2 seeds | 1 timed out at the route's 25 s budget, 1 completed in 37 s (over budget → skipped) | 1 timed out, 1 completed in 18.7 s (inside budget) | **Operational finding:** the 25 s gpu-idle route budget clips Glimmer's image authoring at ~300 output tokens; Swift's decode fits. Swift's prose was strong (85 mm portrait, off-centre, tungsten lantern light). |
| Tool calling (Bree catalog, 42 tools) | correct tools both asks; 76 s / 13.7 s; 209–486 chars of reasoning despite thinking off | same tool choices; 70 s / **1.8 s** (prefix-cache hit); 0 reasoning | Equal correctness; Swift added a valid `visible_to` arg. The 70 s is 14k tokens of tool-schema prefill for both — the prefix cache is what makes turn 2 fast, which argues for the larger cache in WP1. |
| Long context (needle at line 612) | correct; 17.2k tokens; prefill 194 tok/s; **decode 10.9 tok/s** at that length | correct; **22.2k tokens** (tokenizer is ~29% less efficient on the same text); prefill 199 tok/s; decode 39.7 tok/s | Both find the needle. Swift pays ~30% more prefill on identical text but decodes 3.6× faster at long context — for Bree's 17k-token turns that is a wash on prefill and a big win on the reply. |
| Vision | yes (captioning, i2v enrichment, defect QA) | **no** in this build | Glimmer stays for every vision path regardless. |

## Verdict

Not a clean win. Swift is 1.7–3.6× faster on decode, equal on tool calls and needle recall, and fits the image-authoring budget Glimmer currently misses — but it fails the t2v beats contract as the prompt is written today and has no vision. Recommendation: **split by capability**, not replace:

- **Keep Glimmer** for t2v/i2v authoring (beats contract honoured 6/6) and all vision.
- **Move to Swift**: Bree's conversational/tool turns (reply decode 3.6× at 17k context; `enable_thinking` off, `reasoning_budget 256`) and image authoring (fits the 25 s route budget). Both are config changes on .232 (`backend` model id / `promptOptimizer` block).
- **To make Swift eligible for t2v**: restate `BEATS_BLOCK_INSTRUCTION` at the end of the user message in `optimizeT2VPrompt` (one-line change, verified by probe), then re-run the 6-seed check.

## Quantizing our own (Todd's condition)

The bf16 source `ukisai/Swift-Qwen3.8-27b` (55.6 GB) **keeps both the vision tower (333 tensors) and the fine-tune's own MTP head (15 `mtp.*` tensors)**; the andosen 5-bit dropped both. `mlx_lm.convert` (Homebrew python3.12, mlx-lm 0.31.3) cannot produce what we want: its `qwen3_5` loader explicitly discards `mtp.*` and `vision_tower`/`model.visual` keys (`mlx_lm/models/qwen3_5.py` sanitize), which is exactly how the andosen build lost them. A faithful 5-bit needs mlx-vlm (not installed; qwen3_5 support unverified) for the vision tower plus a custom step to carry the 15 MTP tensors into `mtp/weights.safetensors` (stdlib-feasible: slice by safetensors header offsets from the bf16 shards). Alternatively `Yanun/Swift-Qwen3.8-27b-oQ4e-mtp` already ships vision + matched head at 4-bit class (on disk, 16 GB; lost the speed A/B at 31 vs 35 tok/s). Decision deferred until Swift earns a role beyond Bree's text turns.
