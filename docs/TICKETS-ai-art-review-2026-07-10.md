# AI_Art Review: ComfyBox Tickets

Date: 2026-07-10
Reviewed source: `/Users/toddwalderman/Projects/AI_Art`
Target repo: `BarkadaBrew/comfybox`

## Created Tickets

1. [#204 AI_Art: guided workflow wizards for portal cards, training illustrations, and quick sketch](https://github.com/BarkadaBrew/comfybox/issues/204)
   - Adds an end-to-end wizard layer inspired by AI_Art's SwiftUI wizard framework and portal-card flow.
   - Depends on Studio Packs, templates, variation board, projects, and local animation/export.

2. [#205 AI_Art: local still-image animation, title overlays, and GIF/MP4 export](https://github.com/BarkadaBrew/comfybox/issues/205)
   - Tracks local Ken Burns/pan/pulse/fade style animation and FFmpeg export.
   - Complements existing LTX-2 MotionView instead of replacing generative video.

3. [#206 AI_Art: native SVG editor for vtracer output](https://github.com/BarkadaBrew/comfybox/issues/206)
   - Tracks in-app SVG selection, transform, edit, save, and round-trip.
   - Separate from #108, which is an external Inkscape conduit.

4. [#207 AI_Art: posable mannequin control primitive for layout and ControlNet](https://github.com/BarkadaBrew/comfybox/issues/207)
   - Adds an interactive articulated pose primitive for Canvas/control workflows.
   - Extends #200 beyond static templates.

5. [#208 AI_Art: portable project bundle export/import](https://github.com/BarkadaBrew/comfybox/issues/208)
   - Tracks portable bundle export/import with manifest, assets, sidecars, snapshots, recipes, and export presets.
   - Builds on #197 project workspaces.

6. [#209 AI_Art: expose mflux Kontext image editing in Desktop](https://github.com/BarkadaBrew/comfybox/issues/209)
   - Adds Desktop-only support for `mflux-generate-kontext`.
   - Existing mflux UI supports generation/img2img, training, and save/quantize, but not dedicated Kontext editing.

## Covered By Existing Work

- Prompt/template studio concepts are already tracked by [#195](https://github.com/BarkadaBrew/comfybox/issues/195) and [#198](https://github.com/BarkadaBrew/comfybox/issues/198).
- Vector-first generation and SVG review are already tracked by [#196](https://github.com/BarkadaBrew/comfybox/issues/196); #206 adds native element editing.
- Batch queue/variation comparison is already tracked by [#202](https://github.com/BarkadaBrew/comfybox/issues/202), and ComfyBox already has QueueView/QueuePanel primitives.
- Control template categories are already tracked by [#200](https://github.com/BarkadaBrew/comfybox/issues/200); #207 adds the missing interactive mannequin primitive.
- Project organization is already tracked by [#197](https://github.com/BarkadaBrew/comfybox/issues/197); #208 adds archive/import/export packaging.
- VLM/image captioning is already tracked by [#68](https://github.com/BarkadaBrew/comfybox/issues/68) and [#71](https://github.com/BarkadaBrew/comfybox/issues/71).

## No New Ticket

- AI_Art generation queue concepts overlap with ComfyBox's existing queue views and #202.
- AI_Art prompt library/enhancer concepts overlap with Studio Packs and the pack-aware composer.
- AI_Art LoRA discovery overlaps with existing ComfyBox LoRA scanner/library work.
- AI_Art comparison grid overlaps with existing ComfyBox comparison grid and #202.
