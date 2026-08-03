#!/usr/bin/env python3
"""Golden-tensor exporter for the LTX-2.3 audio codec chain (task #21).

Runs the REFERENCE implementation (ComfyUI's comfy.ldm.lightricks) on CPU
float32 with a fixed seed and dumps every stage the Swift port must match:

  z_normalized  -> [normalizer denorm] -> z_denorm
                -> [causal VAE decode] -> decoded (2ch mel-domain, 4T-3 crop)
                -> [base vocoder]      -> wav16k stereo
                -> [BWE + resample]    -> wav48k stereo

Usage: ComfyUI/.venv/bin/python export_audio_codec_goldens.py <out_dir>
Pin: checkpoint sha256 + comfy git rev recorded in manifest.json.
"""
import hashlib, json, subprocess, sys

COMFY = '/Volumes/Bolt/ComfyUI-validate/ComfyUI'
CKPT = COMFY + '/models/vae/LTX23_audio_vae_bf16.safetensors'
sys.path.insert(0, COMFY)

import torch
import safetensors.torch as st

torch.manual_seed(4242)
out_dir = sys.argv[1] if len(sys.argv) > 1 else '.'

import comfy.ldm.lightricks.vae.audio_vae as av

with open(CKPT, 'rb') as f:
    sha = hashlib.sha256(f.read()).hexdigest()

sd = st.load_file(CKPT)
import safetensors
with safetensors.safe_open(CKPT, framework='pt') as f:
    metadata = dict(f.metadata() or {})

vae = av.AudioVAE(metadata)
missing, unexpected = vae.load_state_dict(
    {k.replace('audio_vae.', 'autoencoder.').replace('vocoder.', 'vocoder.'): v
     for k, v in sd.items()}, strict=False)
# Try the naive mapping first; report and fall back to raw keys if bad.
if len(missing) > len(sd) // 2:
    missing, unexpected = vae.load_state_dict(sd, strict=False)
print(f"state_dict: missing={len(missing)} unexpected={len(unexpected)}")
if missing[:5]: print("  missing e.g.", missing[:5])
if unexpected[:5]: print("  unexpected e.g.", unexpected[:5])

vae = vae.float().eval()

C = vae.latent_channels
F = vae.latent_frequency_bins
T = 32
z_norm = torch.randn(1, C, T, F, dtype=torch.float32)

with torch.no_grad():
    # Mirror AudioVAE.decode's exact sub-steps to capture each stage.
    z_denorm = vae.normalizer.denormalize(z_norm)
    target_shape = vae.target_shape_from_latents(z_norm.shape)
    mel = vae.autoencoder.decode(z_denorm, target_shape=target_shape)
    wav_final = vae.run_vocoder(mel)
    # One-call parity check: whole-chain decode must equal the staged path.
    wav_whole = vae.decode(z_norm)
    assert torch.allclose(wav_final, wav_whole), "staged path diverges from decode()"

fixtures = {'z_normalized': z_norm, 'z_denorm': z_denorm, 'mel': mel, 'wav_final': wav_final}
st.save_file({k: v.contiguous() for k, v in fixtures.items()}, out_dir + '/codec_goldens.safetensors')

comfy_rev = subprocess.run(['git', '-C', COMFY, 'rev-parse', 'HEAD'],
                           capture_output=True, text=True).stdout.strip()
manifest = {
    'seed': 4242, 'T': T, 'latent_channels': C, 'latent_frequency_bins': F,
    'checkpoint_sha256': sha, 'comfy_rev': comfy_rev,
    'dtype': 'float32', 'device': 'cpu',
    'shapes': {k: list(v.shape) for k, v in fixtures.items()},
    'target_shape': list(target_shape),
    'sample_rate_internal': metadata.get('sampling_rate', 'see-metadata'),
    'metadata_keys': sorted(metadata.keys()),
}
with open(out_dir + '/manifest.json', 'w') as f:
    json.dump(manifest, f, indent=1)
print(json.dumps(manifest['shapes'], indent=1))
print("goldens written to", out_dir)
