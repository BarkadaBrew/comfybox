#!/bin/zsh
# Reject accidental drift from the validated PinkCherry production recipe.
# A deliberate, time-boxed experiment may set LTX2_QUALITY_SOAK_UNTIL_EPOCH
# in the launchd EnvironmentVariables dictionary to a future Unix timestamp.
set -euo pipefail

PLIST_PATH=${1:-${COMFYBOX_PLIST_PATH:-$HOME/Library/LaunchAgents/com.barkadabrew.comfybox.plist}}
[[ -f "$PLIST_PATH" ]] || { print -u2 "video recipe preflight: plist not found: $PLIST_PATH"; exit 1; }

# 2026-09-15: the engine resolves configFile > env > builtin, and the
# production recipe now lives in ~/.comfybox/config.json `video` (registry
# names, e.g. stg_scale) so it can change without a launchd re-bootstrap. A key
# present there wins; otherwise fall back to the plist EnvironmentVariables.
CONFIG_JSON=${COMFYBOX_CONFIG_JSON:-$HOME/.comfybox/config.json}
read_value() {
  local key=$1 name v
  name=${${key#LTX2_}:l}
  if [[ -f "$CONFIG_JSON" ]]; then
    v=$(python3 -c 'import json,sys
try:
  d=json.load(open(sys.argv[1])).get("video",{})
  x=d.get(sys.argv[2]); print("" if x is None else x)
except Exception: print("")' "$CONFIG_JSON" "$name" 2>/dev/null)
    [[ -n "$v" ]] && { print -r -- "$v"; return; }
  fi
  /usr/libexec/PlistBuddy -c "Print :EnvironmentVariables:$key" "$PLIST_PATH" 2>/dev/null || true
}

soak_until=$(read_value LTX2_QUALITY_SOAK_UNTIL_EPOCH)
if [[ -n "$soak_until" ]]; then
  [[ "$soak_until" == <-> ]] || {
    print -u2 "video recipe preflight: LTX2_QUALITY_SOAK_UNTIL_EPOCH must be a Unix timestamp"
    exit 1
  }
  now=$(date +%s)
  if (( now < soak_until )); then
    print "video recipe preflight: explicit quality soak active until epoch $soak_until"
    exit 0
  fi
  print -u2 "video recipe preflight: quality soak marker expired at epoch $soak_until; remove it or restore the recipe"
  exit 1
fi

failed=0
expect() {
  local key=$1 expected=$2 actual
  actual=$(read_value "$key")
  if [[ "$actual" != "$expected" ]]; then
    print -u2 "video recipe preflight: $key=${actual:-<missing>}; expected $expected"
    failed=1
  fi
}

expect LTX2_TWO_STAGE 0
expect LTX2_AUDIO_REFINE 0
# 2026-09-15 (Todd's same-seed reads, seed 771144, PinkCherry v1.8 distill06
# int8 + heretic Gemma): the validated recipe is the author's 19-step phase-1
# schedule, single pass, cfg++ euler, NAG 5, STG 0.3 run FLAT (stg_head_boost 0
# — the +1.0/+0.5 head burned/posterized while STG itself fixed the motion
# ghost), and color_anchor 1.0 (removed the last-third color/shadow float;
# the 2026-09-11 "veiled ghosts" were seen under the since-stripped 3-LoRA
# motion stack — re-judge on the avocado soak). Two-stage is OUT: it drifts
# composition and loses lip sync on the refine re-denoise.
expect LTX2_SAMPLER euler_cfg_pp
expect LTX2_STAGE1_SIGMAS 1.0,0.998,0.995,0.99,0.982,0.97,0.94,0.89,0.82,0.73,0.62,0.50,0.38,0.27,0.18,0.11,0.06,0.03,0.01,0.0
expect LTX2_REFINE_SIGMAS 0.85,0.7250,0.4219,0.0
expect LTX2_NAG_SCALE 5
expect LTX2_NAG_ALPHA 0.25
expect LTX2_NAG_TAU 2.5
expect LTX2_STG_SCALE 0.3
expect LTX2_STG_HEAD_BOOST 0
expect LTX2_I2V_COMPRESSION 22
expect LTX2_COLOR_ANCHOR 1
expect LTX2_AUDIO_TARGET_DB -24

upsampler=$(read_value LTX2_UPSAMPLER_PATH)
if [[ "$upsampler" != *ltx-2.3-spatial-upscaler-x2-1.1-official.safetensors ]]; then
  print -u2 "video recipe preflight: LTX2_UPSAMPLER_PATH is not the official 1.1 upsampler"
  failed=1
fi

(( failed == 0 )) || exit 1
print "video recipe preflight: validated production recipe"
