# Krea-2 Raw + r256 preset stack

**Established:** 2026-08-23  
**ComfyBox implementation:** `501acda` (`fix/krea2-r256-stack`)  
**Purpose:** make a Krea-2 Raw preset carry an adjustable r256 distillation
adapter without relying on its filename to identify it as the accelerator.

## Canonical preset shape

The deployed `krea-kira` configuration is:

```json
{
  "id": "krea-kira",
  "model": "krea2-raw",
  "checkpoint_family": "raw-accel",
  "steps": 12,
  "guidance": 1,
  "loras": [
    {
      "filename": "krea2_turbo_distill_r256.safetensors",
      "scale": 0.6,
      "role": "accel"
    }
  ],
  "kroma": {
    "file": "kroma-v0.3-base-lora-rank-384-fro-0985.safetensors",
    "strength": 0.6
  }
}
```

The two adjustable weights are deliberately represented differently:

- `kroma` is a structured policy field. It must not also appear in `loras[]`.
- r256 is an ordinary visible LoRA row with an editable `scale`, plus the
  semantic declaration `role: "accel"`.

This is not a filename convention. `krea2_turbo_distill_r256.safetensors`
does not contain `turbo_lora`, and callers must not infer its purpose from
spelling. A Raw-accelerated preset declares the role explicitly.

## Role contract

`LoraReference`, the desktop mirrors, image metadata, `/v1/lora/swap`, and
nearline staging preserve the optional role end to end. The server accepts:

| Role | Meaning |
|---|---|
| `accel` | Distillation/step-reduction adapter for a Raw checkpoint |
| `bypass` | Filter-bypass conditioning adapter |
| `control` | Control adapter |
| `kroma` | Kroma engine slot; saved Krea-2 presets should use structured `kroma` instead |
| absent | Ordinary character, realism, or style LoRA |

Unknown roles are rejected when a preset is saved. Nearline staging may
replace the storage path, but it must not remove the role.

## Desktop editing

In **Presets → Edit Preset**:

1. Keep Kroma in its own row and adjust its strength there.
2. Add or select `krea2_turbo_distill_r256.safetensors` in the LoRA list.
3. Set its role menu to **Accelerator**.
4. Adjust the r256 scale normally; `0.6` is the current Krea-Kira value.
5. Save. Reopening the preset must still show **Accelerator**.

The other role choices are **Style**, **Bypass**, and **Control**. **Style**
means no role is emitted.

## Guidance and negative prompts

`guidance: 1` is the normal distilled Krea-2 Raw path and does not run the
classifier-free-guidance pair. The accelerator role does not require a
different guidance value.

To make a negative prompt active, use guidance around `2`. That intentionally
runs positive and negative conditioning and approximately doubles model
evaluations. Treat it as a quality/control experiment, not as the r256
default.

## Verification without rendering

Read the stored preset:

```bash
curl -sS http://127.0.0.1:7870/v1/presets/krea-kira
```

Resolve it onto server defaults:

```bash
curl -sS -H 'Content-Type: application/json' \
  --data '{"id":"krea-kira"}' \
  http://127.0.0.1:7870/v1/presets/resolve
```

Both responses must contain exactly one generic r256 entry with
`"role":"accel"`, plus the separate `kroma` object. A healthy server may
still report `"loras":[]` in `/health` immediately after restart: that is the
active GPU stack, not the saved preset. Applying or rendering the preset is
what activates its stack.

## Implementation coverage

The role is preserved through:

- canonical preset persistence and resolution;
- server-to-desktop and desktop-to-server preset conversion;
- the Preset editor and Generate view;
- image sidecar metadata and Send to Generate;
- image/video LoRA request bodies; and
- nearline path staging.

Focused persistence/configuration tests and the complete `ZImageTests` suite
cover the server contract. On the 2026-08-23 deployment, 1,470 tests passed,
40 environment-dependent/heavy tests were skipped, and none failed. Rendering
remains a manual/integration verification.

## Deployment record and rollback

Build `501acda` was installed as
`~/.comfybox/bin/ComfyBox-501acda`; `/health.build_sha` must report `501acda`.
The matching CoffeeShop Desktop executable was installed and the app
re-signed.

Pre-deployment copies are recoverable from:

- `/private/tmp/ComfyBox-before-r256-role-20260823-0852`
- `/private/tmp/ComfyBoxDesktop-before-r256-role-20260823-0852`
- `/private/tmp/comfybox-presets-before-r256-role-20260823-0852.json`

