# FDD: mlx-serve — use the Glimmer server to its potential, and give Kira and Bree local voices

**Status:** v1 — draft for Codex review · WP1/WP2/WP3/WP5-A DONE (2026-09-16); open: WP4 vision path (in progress 2026-09-17), WP5-B clips, WP6
**Author:** Fable (technical architect)
**Date:** 2026-09-15
**Origin:** Todd's question "are we using mlx-serve to its full potential?" plus "I would like Kira and Bree to have voices" (same day)
**Repos:** `BarkadaBrew/comfybox` (engine + desktop, Swift/MLX) · `BarkadaBrew/coffeeshop-server` (Bree/Kira daemons, TypeScript) · Mac host config (launchd plist, not in any repo today)
**Evidence:** `~/.mlx-serve/logs/mlx-serve-11234.log` (93,958 lines, 2026-08-27 → 2026-09-15), `~/Library/LaunchAgents/com.barkadabrew.mlx-serve.plist`, `~/.comfybox/config.json`, `/home/todd/.bree/config.json` + `/home/todd/.kira/config.json` on 10.0.100.232, `mlx-serve serve --help` (26.8.11), https://mlxserve.com docs, HF `ddalcu/Muse-Glimmer-30B-MLX-Serve-8bit`

Every number in §0 was read from disk or the live server in the session that produced this document. Anything I could not measure is labelled **predicted**.

---

## 0. Problem & context (measured)

### 0.1 What mlx-serve is, and what it is for us

mlx-serve (ddalcu, Zig + `mlx-c`, MIT, no Python, single binary) is an Apple-Silicon inference server speaking the OpenAI, Anthropic and Ollama HTTP APIs. It also ships image (FLUX.2-klein, Krea-2-Turbo), video (LTX-Video 2.3/2.5), speech (Qwen3-TTS with zero-shot cloning, Kokoro), music and 3D endpoints. It satisfies `intent.md`'s "zero Python at runtime" rule, and it is already the LLM tier of the stack:

| Consumer | Endpoint | Purpose |
|---|---|---|
| Bree daemon (.232) | `10.0.100.134:11234/v1` | `avocadoTextBackend`, `vision` |
| Kira daemon (.232) | `10.0.100.134:11234/v1` | Kira persona backend, `vision`, prompt authoring, `avocadoTextBackend` |
| ComfyBox engine + desktop | `127.0.0.1:11234/v1` | `providers.vision`, `providers.captioning` |

Ollama (`:11434`, dolphin3 8B GGUF + nomic embeddings) is **deliberately** kept as the CPU-only backend for when the GPU lease is held by a render (Todd, 2026-09-15). It is not in scope to consolidate. LM Studio (`:1234`) is running as a login item with nothing loaded; ComfyBox's code default for `promptOptimization` still points at it, but the config file overrides that to Ollama.

### 0.2 How it is launched today

`com.barkadabrew.mlx-serve` (RunAtLoad, KeepAlive, ThrottleInterval 30):

```
~/mlx-serve/mlx-serve-macos-arm64/mlx-serve serve
  --model ~/LocalModels/Muse-Glimmer-30B-heretic-MLX-Q6
  --model-dir ~/.mlx-serve/models          # empty dir; startup logs "scan failed: FileNotFound"
  --max-resident-models 3
  --idle-evict-secs 300
  --port 11234 --host 0.0.0.0
```

The launchd process is the tarball binary, **26.8.10** (MLX 0.32.0; the log banner confirms it on all 20 MLX restarts). `MLX Core.app` 26.8.11 is also installed in `/Applications` and `~/.local/bin/mlx-serve` points at *its* embedded copy, so `mlx-serve --version` at a shell reports 26.8.11 while the server runs 26.8.10. Latest release is 26.9.2 (2026-09-09): per-model settings, `--prefix-cache-disk`, constrained JSON decoding. Todd (2026-09-15): there should be only one binary. §0.7 covers what the app adds and why the two collided.

### 0.3 What the log says (19 days, 10,043 chat requests)

| Measurement | Value | Source line shape |
|---|---|---|
| Requests that are 1-token keep-warm probes | **7,986 / 10,043 (80%)** | `max_tokens=1 … user=2b` ("ok") |
| Probe wall time while a render holds the GPU | 1.1–7.6 s each | `<- 60+1 tokens (7564ms)` |
| PLD speculation disabled at runtime | **1,595 of 1,786 spec-stat lines (89%)** | `runtime_disabled=true` |
| Drafted tokens accepted, total | 29,185 over 1,786 requests (~16/request) | `accepts=N` |
| Drafter / MTP loaded | **none** (`drafter_loaded:false`, `[args] drafter: <none>` ×20 restarts) | `/v1/models` |
| Prefix-cache budget | 2 GB default, 13–32 entries, `resident=2028/2048 MB` | `[hot-cache]` |
| Prefix-cache LRU evictions | **2,351** | `evicted LRU entry (byte budget …)` |
| Prompts > 4k tokens (Bree tool calls, ~17k) — cached-token share | 0.80 (917/931 reuse something; 844 reuse > 80%) | `(N cached / M total)` |
| Prompts 500–4k tokens (Kira authoring, ~2k) — cached-token share | **0.37** (345/978 reuse > 80%) | same |
| Big calls (> 4k), last 100: wall time | 51 s avg, 5,118 s total | `(Nms)` |
| Decode, all real generations | median 9.7 tok/s (p10 2.1, p90 15.3) | `decode: N tok/s` |
| Decode, last 8 real calls (LTX render active, 63 GB engine footprint, 60 min) | **1.5–1.7 tok/s**, 265–297 s per call | same |
| Prefill, uncached | median 179 tok/s, p90 233 | `prefill: N tok/s` |
| Restarts in the log | 25 (20 MLX, 5 GGUF-engine experiments) | `mlx-serve 26.8.x (…)` |
| llama.cpp `SessionCreateFailed` | 460, all from past GGUF experiments (dflash Q4_K_M, mlasli Q6_K) | `[llama] session_create_kv_quant failed` |
| `--metrics`, `--kv-quant`, `--prefix-cache-disk`, `--api-key` | all off | `[args]` |

Residency: `/v1/models` reports `bytes_resident: 5537792` (5.5 MB). That number is wrong. `vmmap --summary` shows a 23.4 GB physical footprint, 23.1 GB of it `IOAccelerator (graphics)`. The model is resident; the slow probes are GPU contention, not cold loads.

### 0.4 The three things fighting each other

1. **Keep-warm vs idle-evict.** `coffeeshop-server/src/backend-keep-warm.ts` fires a 1-token generation at every local backend every 180 s (`KEEP_WARM_INTERVAL_MS`), written for LM Studio's JIT unload ("~97 s cold reload", Todd 2026-07-08). The plist's `--idle-evict-secs 300` is what it defeats. Net effect: Glimmer is always resident *and* 80% of traffic is no-op GPU work that queues in mlx-serve's scheduler ahead of real calls during renders.
2. **Speculation with nothing to speculate with.** PLD (n-gram lookup) is the only engine active and it self-disables on creative prose. ddalcu's own `Muse-Glimmer-30B-MLX-Serve-8bit` ships a 2.72 GB DFlash `drafter/` sidecar (`config.json` + `model.safetensors`); the model card reports 41.4 vs 15.9 tok/s on code (2.61×) and ~35% draft acceptance on prose, on an M4 Max. Our heretic Q6 checkpoint has no `drafter/` folder. `--no-drafter`'s help text confirms a checkpoint-shipped `drafter/` subdir auto-loads.
3. **A 2 GB prefix cache under a 128 GB machine.** Each entry costs ~61 KB/token measured (`156 MB` for 2,558 tokens; the bf16 KV arithmetic, 52 layers × 2 KV heads × 128 dim × K+V × 2 bytes, gives 53 KB, the rest is entry overhead), so a 17k-token Bree prefix is ~1 GB and the budget holds two of them. The cache churns (2,351 evictions) and dies on every restart.

### 0.5 ComfyBox-side defects found on the way

- `Sources/ZImage/MCP/MCPToolExecutor.swift:127-160` (`diagnoseDefects`, used by `repair_image`) resolves its vision endpoint from `COMFYBOX_VISION_URL` or the hard-coded LM Studio `http://127.0.0.1:1234/v1`. It never reads `providers.vision`, even though the warm server exposes it at `GET /v1/config`. Today that call goes to LM Studio, which has no vision model loaded, so the tool silently returns `nil` and repair runs blind.
- The same call sends `max_tokens: 300` with no thinking control. coffeeshop-server documented the failure on this exact model (`src/vision/vision-text.ts:7-9`): `max_tokens 900` → HTTP 200 with empty `content` and 948 reasoning tokens; `3000` answered in 28 s. `Sources/ComfyBoxDesktop/VisionService.swift:49` sends 320 with the same exposure.
- `SettingsView.swift:98` seeds "LM Studio" as a default watched service; mlx-serve is not watched at all.

### 0.6a CLI vs MLX Core: what collided, and the one real gap

**Same server, two launchers.** The app bundles the identical `mlx-serve` binary and launches it headless as a child process with flags rendered from its own `ServerOptions` (stored in `~/Library/Preferences/com.dalcu.mlx-core.plist`, key `serverOptions`). Decoded from disk today: `port 11234`, `host 0.0.0.0`, `prefixCacheMem 2GB`, `prefixCacheDisk 10GB`, `enableMetrics true`, `maxConcurrent 1`, `defaultTemperature 0.8`, `defaultMaxTokens 16384`, and `selectedModelPath` = the **dflash GGUF Q4_K_M** Glimmer under `~/.cache/huggingface`. So when the app runs, it starts a *second* server on the same port with a different model and different defaults. Whichever binds first wins; the loser logs `Port 11234 is already in use — another mlx-serve instance may be running` (6 occurrences in the launchd stderr log, each a KeepAlive retry 30 s apart). When the app won, Bree and Kira were silently talking to the app's GGUF Q4 Glimmer with the app's sampling defaults, and its model unloaded when the app quit. That is the "they do not work together" Todd hit, and it is by construction, not a bug we can configure away: the app has no documented "attach to an existing server" mode in 26.8.x (a string `allowLocalEndpointReuse` exists in the app binary but is not in the saved options; 26.9.2's "chat provider" integration may be that feature, unverified).

**What the app adds that the CLI does not:** chat UI, agent mode with 10 built-in tools and a sandbox VM, Telegram bridge, scheduled tasks, voice mode, quick launcher, folder RAG and memory, a model browser, per-model settings (26.9.2, GUI only), and the media windows. Every one of those except the last is a consumer duplicate of something Bree already owns. Not a gap for us.

**The one real gap: media model acquisition.** Kokoro, Qwen3-TTS, Krea-2, LTX-2.5 and the rest are downloaded through the app's windows; the docs' gotchas say `mlx-serve pull` has no short names for them. The catalog is in the app binary, though, and they are ordinary HF repos: `ddalcu/Kokoro-82M-MLX-Serve`, `mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit`, `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit` (bf16 variants exist; the app itself notes a "broken Qwen3-TTS bf16 dir"), `ddalcu/Krea-2-Turbo-MLX-Serve-mixed-4-8`, `ddalcu/LTX-2.5-MLX-Serve-{4,8}bit`. They land under `~/.mlx-serve/models/<org>/<repo>`, and **serving them is server-side**: the Zig binary discovers media packs through `--model-dir` (`classifyModelPath`, "incomplete media pack" checks, `[kokoro] ready`, `[image] Krea-2 models + tokenizer ready` are all strings in the server, not the app). The plist already passes `--model-dir ~/.mlx-serve/models`. So the CLI server can serve TTS once the packs are on disk; only the *download* path is app-flavoured. **Verified 2026-09-15 (OQ-2 approved and executed):** `mlx-serve pull <org>/<repo>` fetches media packs, but **flat**: it pulled 3 of the Kokoro repo's 8 files and 8 of the Qwen3-TTS repo's 14, skipping every subfolder (`g2p/` with the three misaki pronunciation tables; `speech_tokenizer/` with the 682 MB speaker-encoder weights). With the subfolders fetched by hand from the same repos, both packs load and synthesize on a **CLI-only** server: Kokoro `[kokoro-g2p] 183561 pronunciations … ready (0.33 GB resident)`, Qwen3-TTS `speaker encoder (ECAPA-TDNN) loaded … voice cloning ready (1.85 GB resident)`. Two gotchas found on the way: (1) the server decides a pack is complete at **discovery time** and caches the verdict per model id; adding the missing files and calling `POST /v1/models/rescan` does not clear it (`Model load failed: FileNotFound` persists even after `[kokoro] ready`), a **restart** does; (2) the `model` field wants the full discovered id (`ddalcu/Kokoro-82M-MLX-Serve`); the README's short id `kokoro` returns 503 "No default model configured". So the app is not needed for anything; the download step is a pull plus a subfolder sweep (WP5 script). The app is not installed on the serving Mac (D5).

### 0.6b Accent control (Todd 2026-09-15 evening: "can you give her a filipina accent?")

A zero-shot clone reproduces whatever the reference sounds like, so accent is a property of the reference clip, not a request field. Probed on 26.8.11 (temporary server on :11235): `mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit` (pulled, 2.9 GB with `speech_tokenizer/`) loads and speaks, but the server ignores every description field — `instructions`, `voice`, upstream's `instruct`, none, and an "angry old man" control all produce **byte-identical** WAVs (md5 `d3d363fc…`). VoiceDesign is therefore unusable for accent through mlx-serve today. Todd chose route (1) and three accented Kira takes were submitted directly to the engine (seeds 2001–2003, preset `kira-video-apple`; the engine's optimizer kept the accent and the passage), then **stopped at 20:38 to reclaim the GPU — none finished; re-submit when the GPU is free** (see the pickup memory). Routes that remain: (1) **LTX** — name the accent in the Audio clause of the reference render (D11's own path; needs the engine out of local mode with the video backend loaded); (2) a one-off `mlx-audio` run of VoiceDesign with `instruct` to mint the reference wav, then clone it with Base as usual — tooling, not a serving path, and only if (1) is too slow to reach. Kokoro has no Filipino-accented English voice.

### 0.6 Voices today

Both daemons already carry `voice.tts.provider: "replicate"` with Kokoro preset names (`af_bella`, `af_sarah`, `nicole`) and `voiceSynthesis.voices.{bree,kira}`. `generate_voice` (`src/tools/voice-tools.ts`) runs `jaaari/kokoro-82m` on Replicate, downloads a `.wav`/`.mp3`, and hands it to `send_telegram_voice`; Kira's `async-envelope.ts` already routes a `voice` envelope on the `immediate` lane. So "voices" exist, but they are cloud-rendered, generic presets, and Replicate is the last cloud dependency in the media stack. Todd's rule, stated 2026-09-15: **local first for everything except cognition; no Replicate.** mlx-serve's `POST /v1/audio/speech` takes `input` plus an optional base64 `ref_audio` clip (6–8 s, no transcript) for Qwen3-TTS cloning, or a Kokoro preset voice; Kokoro runs "~17× faster than realtime". Response format, streaming and RTF for Qwen3-TTS on an M3 Max are **not documented** and are measured in WP5.

---

## 1. Target architecture

```
                 ┌──────────── Mac (10.0.100.134, 128 GB) ────────────┐
Bree/Kira ──────►│ mlx-serve :11234  Glimmer Q6 + DFlash drafter        │
(.232)           │   prefix cache 6 GB RAM + 20 GB SSD, no idle-evict   │
                 │   /metrics.json on · Qwen3-TTS + Kokoro on demand    │
ComfyBox ───────►│   (vision/captioning via /v1/config, thinking-safe)  │
desktop + MCP    │                                                      │
Bree/Kira ──────►│ Ollama :11434  dolphin3 8B GGUF on CPU (lease-free)  │  ← unchanged
                 │ ComfyBox :7870 Krea2/LTX (GPU lease owner)           │  ← unchanged
                 └──────────────────────────────────────────────────────┘
```

Nothing changes in who talks to whom. What changes: the server is configured for the workload it actually has, the probe traffic disappears, ComfyBox's vision calls go where the config says, and voice notes render on the Mac.

---

## 2. Key design decisions

**D1 — Stop evicting, stop probing; don't replace one with the other.** Remove `--idle-evict-secs` from the plist *and* exclude mlx-serve backends from the keep-warm pool. Removing only the flag leaves 8k/19d no-op requests; removing only the probes reintroduces the cold reload after 5 idle minutes. mlx-serve without the flag keeps a loaded model resident until `--max-resident-models`/`--max-resident-mem` pressure, which with one LLM never triggers. Alternative rejected: keep-warm via `GET /health` — it proves the process is up, not that the model is loaded, and with eviction off it is redundant.

**D2 — Keep-warm exclusion is explicit config, not URL sniffing.** `BackendConfig.keepWarm?: boolean` (default `true`); `selectKeepWarmBackends` drops `false`. Bree and Kira set it on their four `:11234` backends. Detecting "is this mlx-serve?" by probing `/api/version` was rejected: it adds a network dependency to a pure, tested selector, and Ollama also answers `/api/version`.

**D3 — Prefix cache sized to the workload, persisted to SSD.** `--prefix-cache-mem 6GB` holds ~6 Bree-sized prefixes or ~40 Kira-sized ones; `--prefix-cache-disk 20GB` survives restarts (25 in 19 days). Memory budget: Glimmer 23.4 + drafter 2.7 + cache 6 = 32 GB, against ComfyBox's measured 63 GB LTX peak, inside 128 GB with ~30 GB headroom. 6 GB rather than 16 GB because ComfyBox's video admission gate reads free memory; growing the cache further is a ComfyBox-side decision.

**D4 — Drafter is an experiment with a kill switch, not a commitment.** The sidecar was trained against the base 8-bit checkpoint; ours is a heretic (abliterated) Q6. Speculative decoding verifies every draft against the target, so output is exact regardless (byte-identical at `temp=0`); the only risk is *slower* if acceptance collapses, plus 2.7 GB. Rollback is `--no-drafter` or deleting the folder. Accept only on measured ≥1.2× decode on our prompt shapes (AC-9).

**D5 — One binary, upgraded to 26.9.2 in the same window.** Todd's call: a single mlx-serve on the machine. Layout mirrors the engine's `~/.comfybox/bin/current` convention: `~/mlx-serve/26.9.2/mlx-serve` with `~/mlx-serve/current -> 26.9.2`; the plist and `~/.local/bin/mlx-serve` both resolve through `current`. `MLX Core.app` is uninstalled (Todd does the delete; it is the menu-bar chat app, and its auto-updater would otherwise change the server under the daemons). Rollback is retargeting `current` to the previous version directory, which is kept until the next upgrade, then removed so there is never a third copy.

**D6 — The plist becomes a repo artifact.** `ops/launchd/com.barkadabrew.mlx-serve.plist` in comfybox, installed by `scripts/deploy-mlx-serve.sh` (bootout → copy → bootstrap → verify). Today the only copy is the live file. This mirrors `deploy-serve.sh` for the engine.

**D7 — ComfyBox vision resolution: config first, env override second, LM Studio never.** `diagnoseDefects` fetches `GET /v1/config` through `WarmServerClient` and uses `providers.vision ?? providers.captioning`; `COMFYBOX_VISION_URL`/`_MODEL` remain as explicit overrides for tests. No provider configured → return `nil` with a logged reason (today's silent path, now named). The request body is built by one shared function in `ZImage` used by both MCP and the desktop `VisionService`, sending `max_tokens: 1200` and `reasoning_budget: 256` — the pair coffeeshop-server measured on this model — and treating "empty `content` with non-empty `reasoning_content`" as its own error, not `nil`.

**D8 — Voices: local only, no cloud fallback.** Todd's rule (2026-09-15): local first for everything except cognition; Replicate is not used. `VoiceConfig.tts.provider` becomes `'mlx-serve'` (the `'replicate'` value is removed, and the Replicate branch in `voice-tools.ts` is deleted with its client import); per-character config gains `refAudio` (path to a 6–8 s clip) *or* a Kokoro `voice`. Phase A ships Kokoro-local (same voice names as today, zero behaviour change for the LLM prompt); Phase B adds cloned voices from reference clips **rendered by LTX** (Todd's call, D11). A failed local call surfaces as a tool error to the LLM, fail-closed, exactly like a failed render; nothing retries in the cloud.

**D9 — Voice notes do not take the GPU lease.** Kokoro at 17× realtime for a ≤2000-char note is seconds of light GPU work; Kira's envelope already routes voice `immediate`. If Qwen3-TTS cloning measures > 10 s per note under a render (AC-14), Phase B routes it through `withGpuPriority('voice')` — decided by measurement, not now.

**D11 — Voice identity comes from LTX.** The persona's reference clip is extracted from an LTX-2 render of that persona speaking (the video pipeline already produces synced dialogue audio), so the cloned voice in a Telegram note is the same voice the persona has in her videos. The chosen 6–8 s clip becomes canon: it is stored server-side (`~/.kira/voice/kira-ref.wav`, `~/.bree/voice/bree-ref.wav`, where `voice-tools.ts` runs and base64-encodes it), versioned by hash in the sidecar of every note, and only replaced deliberately. Re-rendering it changes her voice.

**D12 — Voice is a rendering of a reply that already exists, never a second reply channel.** Constraint from Bree (relayed by Todd, 2026-09-15): "the voice path doesn't become a second reply channel. We spent real time killing the 'second voice' problem. If Codex wires TTS, it should render the text I already sent, not generate its own." Today's `generate_voice` tool takes free `text` and its description *invites* composing something new ("a whispered goodnight, a laugh, a teasing tone"), which is precisely the second-voice shape. Under this FDD: (a) `generate_voice` renders verbatim text only; no prompt optimizer, no rewrite, no LLM call sits between the text and the synthesizer; (b) the daemon rejects a voice request whose text is not the content of a message the persona has sent or is sending in the same turn (exact match after whitespace normalisation; a `message_ref` parameter is the preferred way to say which one); (c) nothing renders voice autonomously — no scheduler, muse tick or arc may call it without a sent message to render; (d) the note's sidecar records the source message id and the SHA-256 of the rendered text, so a note can always be traced to the reply it voices. The tool description is rewritten to say "read a message you already sent aloud" and drops the composing language.

**D10 — Not touched:** Ollama (Todd's CPU fallback), `--host 0.0.0.0` (LAN clients), `--max-resident-models 3` (Glimmer + TTS + one spare), `--kv-quant` (docs give no muse_glimmer support statement; revisit after D4 with metrics on), `--api-key` (Kira already sends `Bearer mlx-serve`, ComfyBox sends nothing; OQ-4).

---

## 3. Component design

### 3.1 WP1 — mlx-serve host configuration (Mac, comfybox `ops/`)

**Files:** `ops/launchd/com.barkadabrew.mlx-serve.plist` (new), `scripts/deploy-mlx-serve.sh` (new), `docs/ops/mlx-serve.md` (new, runbook).

Plist `ProgramArguments` after the change:

```
~/mlx-serve/current/mlx-serve serve      # current -> 26.9.2
  --model ~/LocalModels/Muse-Glimmer-30B-heretic-MLX-Q6
  --model-dir ~/.mlx-serve/models
  --max-resident-models 3
  --prefix-cache-mem 6GB
  --prefix-cache-disk 20GB
  --metrics
  --port 11234 --host 0.0.0.0
```

Removed: `--idle-evict-secs 300`. `--model-dir` stays but the directory is created so the startup warning stops (`mkdir -p ~/.mlx-serve/models`).

`deploy-mlx-serve.sh <version>`:
0. Unpack the release tarball into `~/mlx-serve/<version>/`, refuse unless `mlx-serve --version` there prints `<version>`, then repoint `~/mlx-serve/current`. Verify `~/.local/bin/mlx-serve` resolves to the same file and that no other `mlx-serve` binary exists on `$PATH` or in `/Applications` (prints a warning naming any extra copy; Todd removes it).
2. `cp` the live plist to `~/Library/LaunchAgents/com.barkadabrew.mlx-serve.plist.bak-<date>`.
3. `launchctl bootout gui/501/com.barkadabrew.mlx-serve` → copy plist → `launchctl bootstrap gui/501 <plist>`.
4. Poll `GET /health` (≤ 90 s), then `GET /v1/models` and assert `loaded:true`; print the `[args]` lines from the log.
5. On failure, restore the `.bak` and bootstrap it. Exit non-zero.

Outage per run: the Glimmer load (24.6 GB from internal SSD; the log's `Batch eval all weights: 5542ms` plus shard reads — **predicted** 30–60 s). Bree/Kira calls during the window fail-open per their own timeouts; Kira's scheduler retries.

### 3.2 WP2 — keep-warm exclusion (coffeeshop-server)

**Files:** `src/llm-router.ts` (`BackendConfig.keepWarm?: boolean`), `src/backend-keep-warm.ts` (`selectKeepWarmBackends` skips `keepWarm === false`; header comment rewritten — it still says "LM Studio"), `src/backend-keep-warm.test.ts` (+2 cases), `src/config-types.ts` (surface the field on the backend config types that feed `telegramBackends` and the fallback backend), `docs/examples/kira-companion-config.json`.

Live config change (Todd, on .232): `"keepWarm": false` on the four `:11234` backend blocks in `~/.bree/config.json` and `~/.kira/config.json`. Ollama backends keep the default and keep being pinged — Ollama *does* unload on idle.

Not a behaviour change for anything else: the selector is pure, and the daemon log line `[keep-warm] keeping N local backend(s) resident` will show N decrease by the excluded count (AC-4).

### 3.3 WP3 — DFlash drafter trial (Mac)

1. Download only `drafter/config.json` + `drafter/model.safetensors` from `ddalcu/Muse-Glimmer-30B-MLX-Serve-8bit` into `~/LocalModels/Muse-Glimmer-30B-heretic-MLX-Q6/drafter/` (2.72 GB). Verify safetensors header offsets against file size before use (intent.md: "verify weights before blaming recipes").
2. Restart via `deploy-mlx-serve.sh`. Expect `/v1/models … "drafter_loaded": true` and `[args] drafter:` naming the folder. If the loader rejects the sidecar (quant/arch mismatch is plausible — the target is Q6, the sidecar was built beside 8-bit), delete the folder, restart, record the log line in the runbook, and WP3 closes as "not viable".
3. Benchmark with `scripts/mlx-serve-bench.sh` (new): 10 Kira-shaped prompts (~2k tokens, temp 1.0) and 5 Bree-shaped prompts (~17k, tools, `thinking=true`) captured from the log's request shapes with synthetic content; run with drafter and with `--no-drafter`, GPU idle (ComfyBox queue paused per the standing pause-for-deploys OK, then resumed). Record `decode tok/s`, `[spec-stats] accepts`, wall time. Exactness check: 3 prompts at `temp=0`, both modes, diff the text.

### 3.4 WP4 — ComfyBox vision path (comfybox)

**Files:**
- `Sources/ZImage/Vision/VisionRequest.swift` (new): `VisionRequest.body(model:base64PNG:question:maxTokens:)` returns the `[String: Any]` payload with `temperature 0.2`, `max_tokens 1200`, `reasoning_budget 256`; `VisionRequest.parse(_ data: Data) -> Result<String, VisionReplyError>` distinguishing `.emptyContentAfterReasoning(chars:)`, `.malformed`, `.ok(text)`.
- `Sources/ZImage/MCP/MCPToolExecutor.swift`: `diagnoseDefects` → `resolveVisionEndpoint()` (env override → `GET /v1/config` `providers.vision ?? providers.captioning` → `nil`), then `VisionRequest`. Logs the chosen base URL and model once per process.
- `Sources/ComfyBoxDesktop/VisionService.swift`: `requestBody` delegates to `VisionRequest.body`; `parseDescription` runs on `VisionRequest.parse`'s `.ok` text only.
- `Sources/ComfyBoxDesktop/Views/SettingsView.swift:98`: default watched services gain `mlx-serve` → `http://127.0.0.1:11234/health`; "LM Studio" stays (it is still running) but moves below.
- Tests: `Tests/ZImageTests/MCP/MCPVisionResolutionTests.swift` (new), `Tests/ZImageTests/Vision/VisionRequestTests.swift` (new).

No wire-format change for any daemon; `GET /v1/config` already exists.

### 3.5 WP5 — Local voices for Kira and Bree (coffeeshop-server + Mac)

**Config (`src/config-types.ts`):**
```ts
export interface VoiceConfig {
  enabled: boolean;
  tts: {
    provider: 'mlx-serve';           // 'replicate' removed — local only
    baseUrl?: string;            // mlx-serve: http://10.0.100.134:11234/v1
    model?: string;              // mlx-serve: 'kokoro' | 'qwen3-tts' (server model id, OQ-2)
    defaultVoice: string;
    defaultSpeed: number;
  };
}
// voiceSynthesis.voices.<character>: { voice?: string; speed?: number; refAudio?: string }
```

**`src/tools/voice-tools.ts`:** provider branch. For `mlx-serve`: `POST {baseUrl}/audio/speech` with `{ model, input: text, voice, speed, response_format: 'wav', ref_audio?: <base64 of refAudio file> }`; write the body to the same output dir/sidecar the Replicate branch uses; on non-2xx or timeout (30 s), return `{ error: 'voice synthesis failed: <reason>' }` and log `[voice] mlx-serve failed reason=<…>`. The Replicate branch, the `providers/replicate/client` import in this file and the `config.replicate` test hooks are removed; the daemon's `generate_voice` capability line (`daemon.ts:9405`) stops keying on `imageProviders.replicate.apiKey`. The tool description stays as-is (the LLM-facing contract is unchanged).

**`src/tools/voice-tools.test.ts`:** provider selection, request body (with and without `ref_audio`), failure surfaced as a tool error, and the D12 gate: text matching a sent message renders; text that matches none is rejected with a reason; the request body's `input` is byte-equal to the message content (no rewrite).

**Mac (done 2026-09-15):** both packs are on disk under `~/.mlx-serve/models/` (Kokoro 348 MB, Qwen3-TTS 0.6B 8-bit 1.9 GB with `speech_tokenizer/`), subfolders included, verified end-to-end on a temporary CLI server (§0.6a). `scripts/mlx-serve-pull-pack.sh <org>/<repo>` (new, comfybox) wraps `mlx-serve pull` and then walks the HF file listing (`/api/models/<org>/<repo>` → `siblings`) to fetch every path `pull` skipped, so the next pack (the 1.7B Qwen3-TTS, or LTX 2.5 for A/B) does not repeat the hand step. **The live launchd server must be restarted once** before it serves either model — it cached both as incomplete during the first attempt; fold that into the WP1 window. They load on demand and count against `--max-resident-models 3`. The TTS `model` field takes the full discovered id (`ddalcu/Kokoro-82M-MLX-Serve`, `mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit`); output is 24 kHz mono WAV.

**Measured on the temporary server, with an LTX render holding the GPU:** Kokoro `af_bella`, 26 chars → 2.8 s audio, and a two-sentence 100-char note → 7.7 s audio in **2.6 s** wall (AC-14's ≤ 5 s target met even under a render); Qwen3-TTS clone of Kira from the interim clip (0.7–9.65 s window, 8.95 s mono 24 kHz), 74 chars → 5.0 s audio in **31.4 s** (27.7 s generate, 2.7 s codec decode, 1.0 s speaker embedding, cached for reuse). **Idle-GPU numbers (AC-14, taken 2026-09-15 evening after a reboot, GPU free):** Kokoro two-sentence note 7.7 s audio in **0.52 s**; Qwen3-TTS clone of the 24-word passage 6.8 s audio in **1.44 s** (Kira interim ref) and 6.4 s in **2.43 s** (Bree ref, first load). Cloning under a render was 31 s; idle it is under 3 s, so D9 stands: no GPU lease for voice notes. Both samples were sent to Todd for the by-ear check.

**Bree's reference (Todd, 2026-09-15 evening): `ltx2-C5183982-B1B7-437E-AC2A-5CC99A5E2848.mp4`** (a ladder take, seed 771144, 241 f; its visual prompt is Kira's — the voice is what is taken). Extracted with `scripts/voice-ref-from-ltx.sh`: 6.94 s, ~4.07 s voiced, sha256 `09ba2cb3…`. Todd's note on the earlier Kira interim clone: "the audio was not clear" — the Kira monologue render stays the plan for her canon sample. The three Kira takes queued that evening were lost to a machine restart (daemon log: `[orphaned] … no output path`); re-queue when the engine leaves local mode and its video backend is loaded (after the reboot `/health` showed `admission_mode: local`, `video.available: false`).

**Phase A** = Kokoro-local with today's voice names; **Phase B** = `refAudio` clips per persona, produced by `scripts/voice-ref-from-ltx.sh <clip.mp4> <out.wav>` (new, comfybox): `ffmpeg -vn -ac 1 -ar 24000`, silence-trim, pick the longest clean speech span of 6–8 s, print duration and peak. Source clips (both chosen by Todd, 2026-09-15 evening): **Bree = `ltx2-C5183982-B1B7-437E-AC2A-5CC99A5E2848.mp4`**, **Kira = `ltx2-3D976330-D08B-4A76-BF41-BB4C2A938D87.mp4`** (ladder take `a6-euler-10step`, seed 771144, 241 f; extracted 7.39 s / ~4.76 s voiced, sha256 `1630eea2…`; clone of the passage sent to Todd, 6.0 s in 1.3 s idle) — replacing the earlier interim **`ltx2-9DCD3240-2CC4-424C-8563-BEADFD0A3144.mp4`** (Todd, 2026-09-15; `~/Pictures/ComfyBox/`, sha256 `f4182efb3caae65d…`, 241 frames / 10.04 s, 576×896, AAC 48 kHz stereo, transformer-distilled recipe with audio). Measured with `silencedetect -35dB`: five voiced spans totalling ~4.7 s (0.72–2.33, 3.86–5.35, 6.95–8.12 s and two short ones), mean −24 dB, peak −2.1 dB. That is under the docs' "6–8 s is plenty" for voiced content, so this clip is the **interim** reference: the extraction keeps it whole, silence-trimmed at both ends (pauses are natural speech). The **canon** sample is a purpose-rendered Kira monologue (Todd, 2026-09-15: "we can generate a monologue for a greater sample when we are ready and the GPU is free"): one authored spoken passage, no music or ambience in the prompt, rendered through the normal LTX path at 10 s (241 f, the same recipe as this clip) so it yields ~8 s of continuous voiced speech; scheduled by hand in a GPU-idle window, never through the 24/7 scheduler. Its extraction replaces the interim file and bumps the reference hash; every note after that carries the new hash. **Bree gets the same treatment** (Todd: "do the same for bree"). She has no clip today, but she is not blocked: `bree` exists in ComfyBox's character store (`GET /v1/characters`) alongside `kira`, and the video route already weaves a named character's description into the authored prompt, so a Bree monologue renders through exactly the same path with `character: "bree"`. No interim clip for Bree: she keeps her Kokoro-local voice (Phase A) until her monologue is rendered GPU-idle and approved, then her reference hash is set for the first time. Todd approves each clip by ear before it becomes canon. Kira's envelope path and `send_telegram_voice` are untouched: they already accept the `.wav` the tool returns.

### 3.6 WP6 — Observability hook (comfybox desktop, small)

Health tab: the `mlx-serve` watched service (WP4) plus one card reading `GET /metrics.json` — resident models, prefix-cache bytes, requests in flight. Fits the existing "Datadog, but local" PRD; no new backend. Ships only if it stays under ~150 lines; otherwise it is a follow-up ticket.

---

## 4. Technical acceptance criteria

| # | WP | Criterion | How verified |
|---|---|---|---|
| AC-1 | 1 | After deploy, `GET /v1/models` shows Glimmer `loaded:true`, `mlx-serve --version` = 26.9.2, log `[args]` shows no idle-evict, `prefix-cache-mem 6GB`, `prefix-cache-disk 20GB`, metrics on | `deploy-mlx-serve.sh` prints them |
| AC-2 | 1 | `GET /metrics.json` returns 200 with a resident-model count | curl |
| AC-3 | 1 | Rollback path exercised once: deploy the `.bak`, Glimmer back within 90 s | runbook step, recorded in `docs/ops/mlx-serve.md` |
| AC-4 | 2 | `selectKeepWarmBackends` excludes `keepWarm:false`, includes default; daemon log shows the reduced pool count after the config change | unit test + `[keep-warm]` log line on .232 |
| AC-5 | 1+2 | 24 h after both land: `max_tokens=1` request count in the mlx-serve log = 0 for `:11234`; Ollama keep-warm still ticks in the daemon log | log grep |
| AC-6 | 1 | 24 h after: cached-token share on 500–4k-token prompts ≥ 0.60 (from 0.37) | the §0.3 awk over the new log window |
| AC-7 | 1 | After a deliberate restart, the first Kira-shaped request reports `cached > 0` (disk tier restored the prefix) | log line |
| AC-8 | 3 | Drafter either loads (`drafter_loaded:true`) or is rejected with the log line recorded and the folder removed | `/v1/models` + runbook |
| AC-9 | 3 | Keep the drafter only if median decode on the 10 Kira-shaped prompts ≥ 1.2× the `--no-drafter` run **and** the 3 `temp=0` outputs are byte-identical across modes | `mlx-serve-bench.sh` output committed under `docs/ops/` |
| AC-10 | 4 | `diagnoseDefects` uses `providers.vision` from `/v1/config`; with `COMFYBOX_VISION_URL` set, the env wins; with neither, returns `nil` and logs "no vision provider" | unit tests with a stubbed `WarmServerClient` |
| AC-11 | 4 | `VisionRequest.body` carries `max_tokens 1200`, `reasoning_budget 256`; `parse` yields `.emptyContentAfterReasoning` for `{content:"", reasoning_content:"…"}` | unit test |
| AC-12 | 4 | Live: 5 real ComfyBox renders through `repair_image` produce a non-nil diagnosis from Glimmer (memory rule: validate gates against live Glimmer, replies saved to JSONL under `docs/ops/`) | Todd or Fable with the engine running |
| AC-13 | 5 | `generate_voice` for `bree` and `kira` returns a playable `.wav` produced by mlx-serve, and `send_telegram_voice` delivers it; `grep -ri replicate src/tools/voice-tools.ts` is empty | live check on Telegram, both personas + grep |
| AC-14 | 5 | Kokoro-local latency for a 2-sentence note ≤ 5 s GPU-idle; Qwen3-TTS cloned note latency measured idle and under a render, numbers recorded before Phase B is enabled | timed curl, recorded in the runbook |
| AC-15 | 5 | A failed local synthesis returns a tool error naming the reason; no network call leaves the LAN (unit test asserts the only fetch target is the configured `baseUrl`) | unit test |
| AC-17 | 5B | For each persona, a cloned note and her reference clip (Kira: interim `ltx2-9DCD3240-…`, then her monologue; Bree: her monologue) played back to back are judged the same voice by Todd; the note's sidecar records the reference hash in use | live check + sidecar read |
| AC-18 | 5B | Each monologue render (Kira, Bree) is done GPU-idle (ComfyBox `/health` `is_rendering:false`, queue paused), uses the persona's ComfyBox character entry, yields ≥ 6 s voiced per `silencedetect`, and Todd approves it by ear before the reference hash is set or changed | health check + silencedetect + Todd |
| AC-19 | 5 | D12 gate: `generate_voice` with text equal to a sent message succeeds; with novel text it returns an error naming the rule; the synthesizer request `input` is byte-equal to the message; the sidecar carries the message id and text hash; no code path calls the synthesizer without a sent message (grep + unit test) | unit tests + code review |
| AC-16 | all | comfybox unit suite (`-only-testing:ZImageTests`) and coffeeshop-server `merge_gate` green | CI |

---

## 5. Test plan

- **Unit (comfybox):** `VisionRequestTests` (body fields, parse branches), `MCPVisionResolutionTests` (env > config > nil; stubbed client returning a fixture `/v1/config`). Agents run `-only-testing:ZImageTests` only.
- **Unit (coffeeshop-server):** `backend-keep-warm.test.ts` (+2), `voice-tools.test.ts` (provider branch, fallback matrix), config-type decode for the new fields.
- **Bench (Mac, Todd's window):** `scripts/mlx-serve-bench.sh` for WP3; the same script doubles as the AC-14 timer with `--tts`.
- **Soak (24 h):** AC-5/6/7 from the live log. No synthetic load — the daemons are the load.
- **Live (Telegram):** AC-12, AC-13, per the calibrate-claims rule: "done" needs the JSONL/log evidence attached to the PR.

---

## 6. Work packages & sequencing

| WP | Repo | Size | Depends on | Owner |
|---|---|---|---|---|
| WP2 keep-warm flag | coffeeshop-server | S | **DONE 2026-09-16:** PR #1864 (`keepWarm:false` opt-out) + #1865 (admission probe moved off `tick()` after the gate exposed that scheduler tests reached the live engine) → main d7ca12a6, gated 1015/1015, deployed 12:36 EDT; Kira's `:11234` bot backend carries `keepWarm:false` live | Fable |
| WP1 plist + deploy script + runbook + 26.9.2 | comfybox `ops/` + Mac | S | **DONE 2026-09-16; upgraded to 26.9.4 on 2026-09-17** (drafter loads, SSD prefix cache restores across restarts, Kokoro OK; rollback = `~/mlx-serve/26.8.10`): `ops/launchd/com.barkadabrew.mlx-serve.plist` + `scripts/deploy-mlx-serve.sh` (versioned `~/mlx-serve/current`), no idle-evict, prefix cache 6 GB + 20 GB disk, `--metrics`, 4 resident slots; Todd ran it through me; first bootstrap raced the bootout (script now waits). MLX Core.app still installed — Todd's delete. | Fable |
| WP4 vision path | comfybox | M | — | Fable |
| WP3 drafter trial | Mac | S + bench time | **DONE 2026-09-16: KEEP.** ddalcu's DFlash `drafter/` (2.72 GB, 5 layers, targets 1/13/25/37/49) dropped into `~/LocalModels/Muse-Glimmer-30B-heretic-MLX-Q6/drafter/`; auto-loads with the checkpoint (`drafter_loaded: true`, mode=dflash). Idle-GPU A/B, engine verified idle before/after each run: plain prose 22.5–24.3 vs 14.9–15.4 tok/s (1.5×, 31% acceptance); Kira t2v authoring through the production path 28.8–35.7 vs 14.8–15.3 tok/s (2.0×, 48–62% acceptance). AC-8 ✓ AC-9 ✓ (≥1.2×). First two calls after a load run at ~3 tok/s while the DFlash cost table calibrates — expected, not contention. Rollback = delete the folder or `--no-drafter`. | Fable |
| WP5 voices Phase A | coffeeshop-server + Mac | M | **DEPLOYED 2026-09-15 18:18 EDT: coffeeshop-server #1859 + gate fix #1860 → main 91b2cc34**, both daemons restarted; mlx-serve :11234 restarted (14 s) and serves Kokoro live; Bree's tool index shows the D12 description. AC-13 live Telegram check still to run | Fable → Todd merges |
| WP5 voices Phase B | same + comfybox script | S | Code path shipped in #1859 (`refAudio` per character, hard error if unreadable, sha256 in sidecar, `scripts/voice-audition.mjs`); remaining = the clips: Kira interim chosen + monologue when the GPU is free, Bree monologue when the GPU is free (§3.5) | Fable renders/extracts, Todd approves by ear |
| WP6 metrics card | comfybox desktop | S | WP1 | optional |

Order: WP2 → WP1 (one maintenance window) → WP4 in parallel → WP3 → WP5A → WP5B.

---

## 7. Rollout

1. **WP2 merge**, then Todd edits the two live configs on .232 (`keepWarm:false` ×4) and restarts the daemons through the normal deploy path. Verify AC-4.
2. **WP1 window** (Todd runs it; warn-before-blocking rule): pause the ComfyBox queue (standing OK), `deploy-mlx-serve.sh 26.9.2`, verify AC-1/2, resume the queue. Rollback = `deploy-mlx-serve.sh --rollback`.
3. **24 h soak**, then AC-5/6/7 from the log; post the numbers in the PR.
4. **WP3** in the next window; keep or drop by AC-9.
5. **WP4** ships on its own PR; AC-12 live check before merge.
6. **WP5A**: model pulls (~1.9 GB), config flip on .232 (`provider: 'mlx-serve'`), AC-13/14. The Replicate voice path is gone in the same PR; the Replicate key and `imageProviders.replicate` block are left in config for Todd to remove separately (they are not read by voice after this).

---

## 8. Risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| `mlx-serve pull` silently skips subfolders in media packs (verified: `g2p/`, `speech_tokenizer/`) and the server caches the incomplete verdict until restart | certain | `mlx-serve-pull-pack.sh` sweeps the HF file listing after `pull`; restart, don't rescan, after patching a pack; report upstream (issue: pull is flat) |
| 26.9.2 changes a default under the daemons (e.g. thinking or sampling) | medium | Read release notes on the day; AC-12's JSONL captures replies before/after; rollback is a directory swap |
| Drafter incompatible with the heretic Q6 target | medium–high | D4: kill switch, AC-8/9 gate; nothing else depends on it |
| Larger prefix cache tips ComfyBox's video admission gate | low | 6 GB not 16; watch `/health` `memory_usage_bytes` during the first LTX render after deploy |
| Removing eviction keeps Glimmer resident forever, starving a future second LLM | low | `--max-resident-models 3` + `--max-resident-mem` (89.6 GB default) still evict under real pressure |
| Qwen3-TTS response format / model id differ from the docs | medium | WP5 starts with a curl probe (OQ-2) before code; Kokoro-local is Phase A |
| Voice becomes a second reply channel (Bree's "second voice" regression) | medium without D12 | D12 gate in the daemon, not in the prompt: the tool cannot synthesize text that was not sent; AC-19 |
| Local TTS outage means no voice notes at all (no cloud fallback by rule) | medium | Fail-closed tool error the LLM can see and say; Kokoro (345 MB, on-demand load) is the resilient tier, Qwen3-TTS the quality tier |
| GPU contention still makes live Glimmer calls slow during renders | certain, out of scope | §10; the lease routing of live calls is its own ticket |

---

## 9. Open questions for Todd

- ~~OQ-1~~ **Resolved 2026-09-15:** one binary (D5). Todd uninstalls `MLX Core.app` in the WP1 window.
- ~~OQ-2~~ **Resolved 2026-09-15:** approved and done; both packs pulled, patched with their subfolders, and verified on a CLI-only server (§0.6a, §3.5). Live server restart pending (WP1 window).
- ~~OQ-3~~ **Resolved 2026-09-15:** reference clips come from LTX renders (D11). Kira: interim `ltx2-9DCD3240-…`, then a monologue render. Bree: a monologue render from her existing ComfyBox character entry, no interim. Both renders wait for a free GPU (§3.5).
- **OQ-6** With voice off Replicate, nothing in the media stack reads the Replicate key (memory: it was kept for TTS only). Remove the key and the `imageProviders.replicate` block from both daemon configs in the same window?
- **OQ-4** `--api-key mlx-serve` now that the server is on the LAN? Kira/Bree already send that bearer; ComfyBox's two provider blocks would need `apiKey` added. Not included in WP1 by default.
- **OQ-5** Accept the ~1 minute Glimmer outage per WP1/WP3 restart in a normal evening window, or schedule inside the overnight window?

---

## 10. Out of scope

- Routing *live* (non-pregen) Glimmer calls through the GPU lease so they don't run at 1.6 tok/s under a render — server-side scheduling ticket, needs its own measurement.
- Consolidating Ollama or retiring LM Studio (Todd: Ollama is the CPU fallback).
- Using mlx-serve's Krea-2 / LTX endpoints as a serving path (intent.md: ComfyBox is the engine; mlx-serve stays an A/B oracle and an LTX 2.5 port reference at most).
- `--kv-quant`, `--max-concurrent`, Neural Engine prefill offload: no documented muse_glimmer support; revisit with metrics on.
- Any content-moderation or age-gating logic in the vision prompt (Todd's purview, not Claude's).
