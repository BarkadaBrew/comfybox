# ComfyBox Server — API Notes (hand-maintained)

Operational notes and body schemas that complement the generated
[`api-reference.md`](api-reference.md). That file is regenerated wholesale by
`comfybox docs generate` (and byte-checked in CI); THIS file is hand-maintained
and never touched by the generator — put prose, schemas and examples here.
Content below carried verbatim from the pre-Phase-4 hand-maintained reference
(`docs/api-reference.md` @ 3654f8b4).

## Video generation (LTX-2 / Replicate)

`POST /v1/video/generate`: Video generation. **Local LTX-2** (T2V + I2V) when
the server is started with `--ltx2-weights` + `--ltx2-gemma` (runs through the
render queue, returns 200 with the MP4 path); otherwise the Replicate cloud
proxy (202, job-based). `GET /v1/video/status/{id}` reports video job status
(Replicate proxy).

Local LTX-2 body (snake_case): `{prompt, negative_prompt?, image_path?, width?, height?, frames? (1+8k), steps?, seed?, strength?, extend_to_seconds?, fps?, output_path?}`
→ `{success, output_path, frame_count, duration_seconds, elapsed_seconds, backend: "ltx2-local"}`.
`image_path` present = image-to-video; absent = text-to-video. Output is
contained to the server's allowed output directory.

## Prompt enhancement

`POST /v1/enhance` body: `{prompt, character?, character_description?, content_mode?}`
→ `{success, prompt, enhanced, note?}`.

## LoRA roles (`POST /v1/lora/swap`)

`POST /v1/lora/swap` accepts an optional semantic role on every entry:

```json
{
  "loras": [
    {
      "path": "krea2_turbo_distill_r256.safetensors",
      "scale": 0.6,
      "role": "accel"
    }
  ]
}
```

Valid roles are `kroma`, `accel`, `bypass`, and `control`; omit `role` for an
ordinary style/character LoRA. Roles are declarations, not filename guesses.
In particular, Krea-2 distillation files such as
`krea2_turbo_distill_r256.safetensors` must declare `"role":"accel"` when
they fill the accelerator slot. Auto-staging may change `path`, but preserves
`role`.

## Preset LoRA references (Krea-2)

Preset LoRA references use `filename` rather than `path` and preserve the
same optional `role`:

```json
{
  "loras": [
    {
      "filename": "krea2_turbo_distill_r256.safetensors",
      "scale": 0.6,
      "role": "accel"
    }
  ]
}
```

For Krea-2 presets, Kroma belongs in the structured `kroma` object and must
not be duplicated in `loras[]`. See
[Krea-2 Raw + r256 preset stack](methods/krea2-r256-preset-stack.md).

## Startup imports

- Character + preset legacy imports also run **once at server startup**
  (idempotent), merging from `~/.coffeeshop/image-service/`.
