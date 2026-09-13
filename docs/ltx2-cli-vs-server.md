# LTX-2 video: the warm server is the recipe, the CLI is not

Decision (Todd, 2026-07; reaffirmed 2026-09-13 after the Codex review of build 25f639d):
production video renders go through the warm server (`ComfyBox serve`, port 7870)
and its MCP bridge. The `ComfyBox video` CLI builds a separate pipeline: it
rejects the monolithic PinkCherry checkpoint the server runs, truncates prompts
at 128 Gemma tokens where the server uses 1024, and does not share the
server's config resolver, recipe fingerprint, sidecar, or quality gates.

Use the CLI only for engine bring-up experiments. Anything that must match a
production clip (A/Bs, replays, parity checks) is submitted to the server —
`POST /v1/video/generate/async` — with the sidecar's recipe. Routing the CLI
through `LTX2VideoGenerator` remains an open item; it is not scheduled.
