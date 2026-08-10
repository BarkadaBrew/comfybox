#!/usr/bin/env python3
"""Golden exporter for ONE BasicAVTransformerBlock (task #21 wire 1 oracle).

Loads DiT block 0 from the BF16 v16b monolith (plain weights both Swift and
torch can read), runs the REFERENCE block on CPU fp32 with seeded inputs,
and dumps every input + both outputs. pe=None in v1 — RoPE application gets
model-level fixtures separately.

Usage: ComfyUI/.venv/bin/python export_av_block_goldens.py <out_dir>
"""
import hashlib, json, subprocess, sys

COMFY = '/Volumes/Bolt/ComfyUI-validate/ComfyUI'
CKPT = '/Volumes/Bolt/ComfyUI-validate/downloads/SexGod_PinkCherry_dev_bf16_LTX23_v16b.safetensors'
sys.path.insert(0, COMFY)

import torch
import safetensors
import safetensors.torch as st

torch.manual_seed(31337)
out_dir = sys.argv[1] if len(sys.argv) > 1 else '.'

import comfy.ops
from comfy.ldm.lightricks.av_model import BasicAVTransformerBlock

PREFIX = 'model.diffusion_model.transformer_blocks.0.'
sd = {}
with safetensors.safe_open(CKPT, framework='pt') as f:
    for k in f.keys():
        if k.startswith(PREFIX):
            sd[k[len(PREFIX):]] = f.get_tensor(k)
print(f"block-0: {len(sd)} tensors from v16b")

block = BasicAVTransformerBlock(
    v_dim=4096, a_dim=2048, v_heads=32, a_heads=32, vd_head=128, ad_head=64,
    v_context_dim=4096, a_context_dim=2048,
    cross_attention_adaln=True,  # 9-row tables + prompt tables (checkpoint layout)
    apply_gated_attention=True,  # checkpoint has to_gate_logits on every attention
    operations=comfy.ops.disable_weight_init,
)
missing, unexpected = block.load_state_dict(sd, strict=False)
print(f"state_dict: missing={len(missing)} unexpected={len(unexpected)}")
if missing: print("  missing e.g.", missing[:6])
if unexpected: print("  unexpected e.g.", unexpected[:6])
block = block.float().eval()

B, TV, TA, TCTX = 1, 64, 16, 32
g = lambda *shape: torch.randn(*shape, dtype=torch.float32) * 0.5
inputs = {
    'vx': g(B, TV, 4096),
    'ax': g(B, TA, 2048),
    'v_context': g(B, TCTX, 4096),
    'a_context': g(B, TCTX, 2048),
    'v_timestep': g(B, 1, 9 * 4096),
    'a_timestep': g(B, 1, 9 * 2048),
    'v_cross_scale_shift_timestep': g(B, 1, 4 * 4096),  # table[:4] -> 4 params
    'a_cross_scale_shift_timestep': g(B, 1, 4 * 2048),
    'v_cross_gate_timestep': g(B, 1, 1 * 4096),    # table[4:] -> gate
    'a_cross_gate_timestep': g(B, 1, 1 * 2048),
    'v_prompt_timestep': g(B, 1, 2 * 4096),
    'a_prompt_timestep': g(B, 1, 2 * 2048),
}

pristine = {k: v.clone() for k, v in inputs.items()}
with torch.no_grad():
    vx_out, ax_out = block(
        (inputs['vx'].clone(), inputs['ax'].clone()),
        v_context=inputs['v_context'], a_context=inputs['a_context'],
        v_timestep=inputs['v_timestep'], a_timestep=inputs['a_timestep'],
        v_pe=None, a_pe=None, v_cross_pe=None, a_cross_pe=None,
        v_cross_scale_shift_timestep=inputs['v_cross_scale_shift_timestep'],
        a_cross_scale_shift_timestep=inputs['a_cross_scale_shift_timestep'],
        v_cross_gate_timestep=inputs['v_cross_gate_timestep'],
        a_cross_gate_timestep=inputs['a_cross_gate_timestep'],
        v_prompt_timestep=inputs['v_prompt_timestep'],
        a_prompt_timestep=inputs['a_prompt_timestep'],
        transformer_options={},
    )

fixtures = dict(pristine)
fixtures['vx_out'] = vx_out.clone()
fixtures['ax_out'] = ax_out.clone()
st.save_file({k: v.contiguous() for k, v in fixtures.items()},
             out_dir + '/av_block0_goldens.safetensors')

with open(CKPT, 'rb') as fh:
    head = fh.read(1 << 20)
manifest = {
    'seed': 31337, 'checkpoint': CKPT,
    'checkpoint_head_sha256': hashlib.sha256(head).hexdigest(),
    'comfy_rev': subprocess.run(['git', '-C', COMFY, 'rev-parse', 'HEAD'],
                                capture_output=True, text=True).stdout.strip(),
    'dims': {'v_dim': 4096, 'a_dim': 2048, 'v_heads': 32, 'a_heads': 32,
             'vd_head': 128, 'ad_head': 64, 'TV': TV, 'TA': TA, 'TCTX': TCTX},
    'pe': 'None (v1 — RoPE fixtures come from model-level export)',
    'shapes': {k: list(v.shape) for k, v in fixtures.items()},
}
with open(out_dir + '/av_block0_manifest.json', 'w') as f:
    json.dump(manifest, f, indent=1)
print(json.dumps({k: list(v.shape) for k, v in dict(vx_out=vx_out, ax_out=ax_out).items()}))
print("block goldens written to", out_dir)
