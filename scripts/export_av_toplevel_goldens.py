#!/usr/bin/env python3
"""Golden exporter for the LTXAV TOP-LEVEL audio plumbing (task #21 wire 1).

Covers everything the block oracle does NOT: per-modality timestep
production (audio adaln, prompt adaln, the four av_ca adaln singles with
their crossed gate inputs and 1/1000 av_ca factor), audio patchify+proj
input path, real-time cross-modal RoPE (video seconds + audio start/end
coords), audio self RoPE, and the audio output processing
(norm -> table+embedded modulation -> proj -> unpatchify).

The 48-block loop itself is covered by the block oracle; video timestep /
video RoPE paths are proven by the shipping video engine.

Distinct sigmas (v=0.7, a=0.35) so any swapped wiring shows up.

Usage: ComfyUI/.venv/bin/python export_av_toplevel_goldens.py <out_dir>
Also writes the small weight extract to ~/.comfybox/reference/v16b_av_toplevel.safetensors
"""
import hashlib, json, os, sys, types

COMFY = '/Volumes/Bolt/ComfyUI-validate/ComfyUI'
CKPT = '/Volumes/Bolt/ComfyUI-validate/downloads/SexGod_PinkCherry_dev_bf16_LTX23_v16b.safetensors'
sys.path.insert(0, COMFY)

import torch
import safetensors
import safetensors.torch as st

torch.manual_seed(7331)
out_dir = sys.argv[1] if len(sys.argv) > 1 else '.'

import comfy.ops
from comfy.ldm.lightricks.model import (
    AdaLayerNormSingle, LTXRopeType, generate_freq_grid_np, generate_freq_grid_pytorch,
)
import comfy.ldm.lightricks.model as ltx_model
from comfy.ldm.lightricks.symmetric_patchifier import AudioPatchifier

ops = comfy.ops.disable_weight_init

PREFIXES = [
    'audio_adaln_single.', 'audio_prompt_adaln_single.',
    'av_ca_video_scale_shift_adaln_single.', 'av_ca_audio_scale_shift_adaln_single.',
    'av_ca_a2v_gate_adaln_single.', 'av_ca_v2a_gate_adaln_single.',
    'audio_patchify_proj.', 'audio_proj_out.', 'audio_scale_shift_table',
]
DIT = 'model.diffusion_model.'

sd = {}
with safetensors.safe_open(CKPT, framework='pt') as f:
    for k in f.keys():
        if not k.startswith(DIT):
            continue
        short = k[len(DIT):]
        if any(short.startswith(p) for p in PREFIXES):
            sd[short] = f.get_tensor(k)
print(f'extracted {len(sd)} top-level tensors')

# Persist the extract for the Swift side (bf16, same keys).
ref_dir = os.path.expanduser('~/.comfybox/reference')
os.makedirs(ref_dir, exist_ok=True)
st.save_file(sd, os.path.join(ref_dir, 'v16b_av_toplevel.safetensors'))

def build_adaln(prefix, dim, coeff):
    m = AdaLayerNormSingle(dim, embedding_coefficient=coeff,
                           use_additional_conditions=False, operations=ops)
    sub = {k[len(prefix):]: v.float() for k, v in sd.items() if k.startswith(prefix)}
    missing, unexpected = m.load_state_dict(sub, strict=True), None
    m = m.float().eval()
    return m

audio_adaln = build_adaln('audio_adaln_single.', 2048, 9)
audio_prompt_adaln = build_adaln('audio_prompt_adaln_single.', 2048, 2)
av_v_ss = build_adaln('av_ca_video_scale_shift_adaln_single.', 4096, 4)
av_a_ss = build_adaln('av_ca_audio_scale_shift_adaln_single.', 2048, 4)
av_a2v_gate = build_adaln('av_ca_a2v_gate_adaln_single.', 4096, 1)
av_v2a_gate = build_adaln('av_ca_v2a_gate_adaln_single.', 2048, 1)

patchify_proj = ops.Linear(128, 2048, bias=True)
patchify_proj.load_state_dict({'weight': sd['audio_patchify_proj.weight'].float(),
                               'bias': sd['audio_patchify_proj.bias'].float()})
proj_out = ops.Linear(2048, 128, bias=True)
proj_out.load_state_dict({'weight': sd['audio_proj_out.weight'].float(),
                          'bias': sd['audio_proj_out.bias'].float()})
audio_scale_shift_table = sd['audio_scale_shift_table'].float()
norm_out = torch.nn.LayerNorm(2048, elementwise_affine=False, eps=1e-6)

B, TA, NV = 1, 12, 6
V_SIGMA, A_SIGMA = 0.7, 0.35
TS_MULT, AV_CA_MULT = 1000.0, 1.0
FRAME_RATE = 25.0

g = {}

# ---- audio input path ----
ax_latents = torch.randn(B, 8, TA, 16, dtype=torch.float32)
patchifier = AudioPatchifier(1, start_end=True)
ax_tokens, a_latent_coords = patchifier.patchify(ax_latents)
g['ax_latents'] = ax_latents
g['a_latent_coords'] = a_latent_coords.float()
g['a_tokens_projected'] = patchify_proj(ax_tokens)

# ---- timesteps (scalar per batch) ----
v_ts_scaled = torch.tensor([V_SIGMA * TS_MULT])
a_ts_scaled = torch.tensor([A_SIGMA * TS_MULT])
cond = {"resolution": None, "aspect_ratio": None}

a_emb, a_embedded = audio_adaln(a_ts_scaled, cond, batch_size=B, hidden_dtype=torch.float32)
g['a_timestep_emb'] = a_emb.view(B, 1, -1)
g['a_embedded_timestep'] = a_embedded.view(B, 1, -1)

ap_emb, _ = audio_prompt_adaln(a_ts_scaled, cond, batch_size=B, hidden_dtype=torch.float32)
g['a_prompt_timestep'] = ap_emb.view(B, 1, -1)

vss, _ = av_v_ss(v_ts_scaled, cond, batch_size=B, hidden_dtype=torch.float32)
g['av_ca_video_ss_timestep'] = vss.view(B, 1, -1)
ass, _ = av_a_ss(a_ts_scaled, cond, batch_size=B, hidden_dtype=torch.float32)
g['av_ca_audio_ss_timestep'] = ass.view(B, 1, -1)

# Gates are CROSSED and use the av_ca factor (1/1000 -> raw sigma):
av_factor = AV_CA_MULT / TS_MULT
# a2v gate input = max AUDIO sigma x av_factor (raw sigma); v2a mirrors with video.
a2v_gate, _ = av_a2v_gate(a_ts_scaled.max().reshape(1) * av_factor, cond, batch_size=B, hidden_dtype=torch.float32)
g['av_ca_a2v_gate_timestep'] = a2v_gate.view(B, 1, -1)
v2a_gate, _ = av_v2a_gate(v_ts_scaled.max().reshape(1) * av_factor, cond, batch_size=B, hidden_dtype=torch.float32)
g['av_ca_v2a_gate_timestep'] = v2a_gate.view(B, 1, -1)

# ---- RoPE (shim exposing the model's _precompute_freqs_cis) ----
class Shim: pass
shim = Shim()
shim.split_positional_embedding = LTXRopeType.SPLIT
shim.freq_grid_generator = generate_freq_grid_np
shim._precompute_freqs_cis = types.MethodType(ltx_model.LTXVModel._precompute_freqs_cis, shim)

# Synthetic video pixel coords (B, 3, Nv, 2) start/end pairs:
# frame idx spaced 8 (end = start + 8), h/w spaced 32.
starts = torch.tensor([[[0., 8, 16, 24, 32, 40],
                        [0, 32, 64, 96, 128, 160],
                        [0, 32, 64, 96, 128, 160]]], dtype=torch.float32)
ends = starts + torch.tensor([8., 32, 32]).view(1, 3, 1)
v_pixel_coords = torch.stack([starts, ends], dim=-1)
g['v_pixel_coords'] = v_pixel_coords

max_pos = 20  # max(video temporal 20, audio 20)
v_secs = v_pixel_coords.clone()
v_secs[:, 0] = v_secs[:, 0] * (1.0 / FRAME_RATE)
def pe_cos_sin(ret):
    rot, _split = ret  # (B, T, H, half, 2, 2): [0,0]=cos, [1,0]=sin
    cos, sin = rot[..., 0, 0], rot[..., 1, 0]
    B, T, H, half = cos.shape
    return cos.reshape(B, T, H * half).float(), sin.reshape(B, T, H * half).float()

cross_v = shim._precompute_freqs_cis(
    v_secs[:, 0:1, :], dim=2048, out_dtype=torch.float32, max_pos=[max_pos],
    use_middle_indices_grid=True, num_attention_heads=32)
g['cross_v_pe_cos'], g['cross_v_pe_sin'] = pe_cos_sin(cross_v)

cross_a = shim._precompute_freqs_cis(
    a_latent_coords[:, 0:1, :], dim=2048, out_dtype=torch.float32, max_pos=[max_pos],
    use_middle_indices_grid=True, num_attention_heads=32)
g['cross_a_pe_cos'], g['cross_a_pe_sin'] = pe_cos_sin(cross_a)

# Audio self RoPE: middle-grid True and False variants (config-dependent).
for flag in (True, False):
    pe = shim._precompute_freqs_cis(
        a_latent_coords, dim=2048, out_dtype=torch.float32, max_pos=[20],
        use_middle_indices_grid=flag, num_attention_heads=32)
    key = 'a_pe_mid' if flag else 'a_pe_nomid'
    g[f'{key}_cos'], g[f'{key}_sin'] = pe_cos_sin(pe)

# ---- audio output processing ----
ax_hidden = torch.randn(B, TA, 2048, dtype=torch.float32)
g['ax_hidden'] = ax_hidden
ssv = audio_scale_shift_table[None, None] + g['a_embedded_timestep'][:, :, None]
a_shift, a_scale = ssv[:, :, 0], ssv[:, :, 1]
ax_out = norm_out(ax_hidden)
ax_out = ax_out * (1 + a_scale) + a_shift
ax_out = proj_out(ax_out)
g['ax_out_latents'] = patchifier.unpatchify(ax_out, channels=8, freq=16)

os.makedirs(out_dir, exist_ok=True)
st.save_file({k: v.contiguous() for k, v in g.items()},
             os.path.join(out_dir, 'av_toplevel_goldens.safetensors'))

manifest = {
    'seed': 7331, 'B': B, 'TA': TA, 'NV': NV,
    'v_sigma': V_SIGMA, 'a_sigma': A_SIGMA,
    'timestep_scale_multiplier': TS_MULT, 'av_ca_timestep_scale_multiplier': AV_CA_MULT,
    'frame_rate': FRAME_RATE, 'rope': 'split + float64 grid', 'cross_max_pos': max_pos,
    'gate_inputs': 'a2v gate <- max audio sigma raw; v2a gate <- max video sigma raw',
    'checkpoint_sha256_first16': hashlib.sha256(open(CKPT, 'rb').read(1 << 20)).hexdigest()[:16],
    'shapes': {k: list(v.shape) for k, v in g.items()},
}
with open(os.path.join(out_dir, 'av_toplevel_manifest.json'), 'w') as f:
    json.dump(manifest, f, indent=1)
print(json.dumps(manifest['shapes'], indent=1))
print('done')
