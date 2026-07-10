# chroma-generate Review: ComfyBox Tickets

Date: 2026-07-10
Reviewed source: `/Users/toddwalderman/Projects/chroma-generate`
Target repo: `BarkadaBrew/comfybox`

## Created Tickets

1. [#210 chroma-generate: benchmarked quality/speed preset tiers for sampler and sigma settings](https://github.com/BarkadaBrew/comfybox/issues/210)
   - Productizes ComfyBox's sampler/sigma support into validated user-facing tiers.
   - Inspired by chroma-generate's Flash/Fast/Balanced/Quality/Premium/Ultimate vocabulary.

2. [#211 chroma-generate: vision QC scorecards for generated assets](https://github.com/BarkadaBrew/comfybox/issues/211)
   - Adds optional VLM-backed asset scorecards for quality, prompt adherence, technical artifacts, and pack-specific criteria.
   - Stores results with DAM assets rather than creating a separate history database.

3. [#212 chroma-generate: two-stage draft QC refine workflow](https://github.com/BarkadaBrew/comfybox/issues/212)
   - Adds a reusable draft -> QC -> refine/upscale -> compare/promote workflow.
   - Builds on ComfyBox's variation board, DAM, img2img, SeedVR2, comparison grid, and #211.

4. [#213 chroma-generate: research naturalness controls for guidance decay and texture realism](https://github.com/BarkadaBrew/comfybox/issues/213)
   - Validates dynamic guidance and naturalness ideas before exposing controls.
   - Explicitly avoids direct FreeU porting unless a backend architecture supports it.

5. [#214 chroma-generate: DAM lineage for refinement chains and derived assets](https://github.com/BarkadaBrew/comfybox/issues/214)
   - Adds explicit source/derived lineage across draft, QC, img2img, upscale, Kontext, SVG, and final assets.
   - Supports workflow auditability and project bundle export/import.

6. [#215 chroma-generate: lightweight queue/status CLI for ComfyBox operators](https://github.com/BarkadaBrew/comfybox/issues/215)
   - Adds terminal queue/status/control commands for long-running local generation without opening Desktop.
   - Inspired by chroma-generate's TUI/operator flows, scoped to ComfyBox's existing queue/server.

## Already Covered Or Partially Covered

- Sampler expansion is already partially implemented in `SchedulerFactory` and documented in `docs/prd-sampler-expansion.md`; #210 focuses on benchmarked presets, not another raw sampler port.
- Prompt history, sidecar metadata, DAM search, favorites, ratings, and Send to Generate already exist in ComfyBox.
- Batch seed sweeps and checkpointed batch runner already exist; #212 and #215 cover workflow/operator polish rather than basic batch generation.
- VLM captioning is already tracked by [#68](https://github.com/BarkadaBrew/comfybox/issues/68) and [#71](https://github.com/BarkadaBrew/comfybox/issues/71); #211 uses that capability for output QC.
- Studio Pack prompt/template work is already tracked by [#195](https://github.com/BarkadaBrew/comfybox/issues/195), [#198](https://github.com/BarkadaBrew/comfybox/issues/198), and [#201](https://github.com/BarkadaBrew/comfybox/issues/201).

## Not Ported

- chroma-generate's repository structure is a research scratchpad and should not be copied.
- chroma-generate's exact sampler rankings need ComfyBox validation before becoming defaults.
- FreeU is U-Net-specific; ComfyBox should only expose analogous controls after backend-specific validation.
- Domain-specific prompt/content systems from chroma-generate are intentionally excluded from this handoff.
