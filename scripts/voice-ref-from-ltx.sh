#!/usr/bin/env bash
# voice-ref-from-ltx.sh — extract a voice-cloning reference wav from an LTX clip.
#
# FDD-mlx-serve-tuning §3.5 / D11: a persona's reference voice comes from an
# LTX render of her speaking. This pulls the audio track as mono 24 kHz PCM
# (what Qwen3-TTS expects), trims leading/trailing silence, and reports the
# voiced seconds + SHA-256 so the sidecar can name which reference produced a
# note. Optional --from/--to select a window (seconds) before trimming.
#
# Usage: scripts/voice-ref-from-ltx.sh <clip.mp4> <out.wav> [--from S] [--to S] [--threshold -35dB]
set -euo pipefail
in=${1:?clip.mp4}; out=${2:?out.wav}; shift 2
from=""; to=""; thr="-35dB"
while [ $# -gt 0 ]; do case "$1" in
  --from) from=$2; shift 2;; --to) to=$2; shift 2;; --threshold) thr=$2; shift 2;;
  *) echo "unknown arg $1" >&2; exit 2;; esac; done
win=(); [ -n "$from" ] && win+=(-ss "$from"); [ -n "$to" ] && win+=(-to "$to")
# Trim silence at both ends only (reverse trick), never internal pauses — they are speech.
ffmpeg -y -v error ${win[@]+"${win[@]}"} -i "$in" -vn -ac 1 -ar 24000 \
  -af "silenceremove=start_periods=1:start_threshold=${thr},areverse,silenceremove=start_periods=1:start_threshold=${thr},areverse" \
  -c:a pcm_s16le "$out"
dur=$(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$out")
# Voiced seconds = duration minus detected internal silences.
sil=$(ffmpeg -hide_banner -nostats -i "$out" -af "silencedetect=noise=${thr}:d=0.3" -f null - 2>&1 | grep -oE 'silence_duration: [0-9.]+' | awk '{s+=$2} END {printf "%.2f", s}')
voiced=$(python3 -c "print(round(float('$dur')-float('${sil:-0}'),2))")
sha=$(shasum -a 256 "$out" | cut -c1-64)
printf 'out=%s\nduration_s=%s\nvoiced_s=%s\nsha256=%s\n' "$out" "$dur" "$voiced" "$sha"
python3 - "$out" "$in" "$dur" "$voiced" "$sha" <<'PY'
import sys,json,datetime,os
out,src,dur,voiced,sha=sys.argv[1:]
json.dump({"reference_wav":os.path.abspath(out),"source_clip":os.path.abspath(src),"duration_s":float(dur),"voiced_s":float(voiced),"sha256":sha,"extracted_at":datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")},open(out.rsplit('.',1)[0]+'.json','w'),indent=2)
PY
