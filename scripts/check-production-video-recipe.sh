#!/bin/zsh
# Reject accidental drift from the validated PinkCherry production recipe.
# A deliberate, time-boxed experiment may set LTX2_QUALITY_SOAK_UNTIL_EPOCH
# in the launchd EnvironmentVariables dictionary to a future Unix timestamp.
set -euo pipefail

PLIST_PATH=${1:-${COMFYBOX_PLIST_PATH:-$HOME/Library/LaunchAgents/com.barkadabrew.comfybox.plist}}
[[ -f "$PLIST_PATH" ]] || { print -u2 "video recipe preflight: plist not found: $PLIST_PATH"; exit 1; }

read_value() {
  /usr/libexec/PlistBuddy -c "Print :EnvironmentVariables:$1" "$PLIST_PATH" 2>/dev/null || true
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
expect LTX2_SAMPLER euler_ancestral_cfg_pp
expect LTX2_STAGE1_SIGMAS 1,0.9953,0.9836,0.949,0.848,0.675,0.452,0.243,0.1,0.028,0
expect LTX2_REFINE_SIGMAS 0.85,0.7250,0.4219,0.0
expect LTX2_NAG_SCALE 11.0
expect LTX2_NAG_ALPHA 0.25
expect LTX2_NAG_TAU 2.5
expect LTX2_I2V_COMPRESSION 22
expect LTX2_COLOR_ANCHOR 1.0
expect LTX2_AUDIO_TARGET_DB -24

upsampler=$(read_value LTX2_UPSAMPLER_PATH)
if [[ "$upsampler" != *ltx-2.3-spatial-upscaler-x2-1.1-official.safetensors ]]; then
  print -u2 "video recipe preflight: LTX2_UPSAMPLER_PATH is not the official 1.1 upsampler"
  failed=1
fi

(( failed == 0 )) || exit 1
print "video recipe preflight: validated production recipe"
