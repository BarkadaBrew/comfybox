#!/bin/bash
# Enable the Krita AI Diffusion plugin and point it at ComfyBox.
#
# Krita's Python Plugin Manager stores enabled plugins in kritarc under a
# [python] section (enable_<id>=true). If that section is lost, the plugin
# silently disappears from the UI even though its files are intact. This script
# restores it. Krita MUST be closed — KConfig rewrites kritarc on exit and would
# clobber a live edit.
#
# Idempotent: safe to run repeatedly. Also verifies the plugin's server_url
# points at the ComfyBox endpoint (default 127.0.0.1:7870).
set -euo pipefail

KR="$HOME/Library/Application Support/krita"
RC="$KR/kritarc"
SETTINGS="$KR/ai_diffusion/settings.json"
ENDPOINT="${1:-127.0.0.1:7870}"

if pgrep -x krita >/dev/null; then
  echo "✗ Krita is running. Quit Krita completely (handle any save dialog), then re-run." >&2
  exit 1
fi
[ -d "$KR/pykrita/ai_diffusion" ] || { echo "✗ ai_diffusion plugin not installed at $KR/pykrita" >&2; exit 2; }
[ -f "$RC" ] || { echo "✗ kritarc not found at $RC" >&2; exit 3; }

cp "$RC" "$RC.bak-$(date +%s)"

# Ensure a [python] section exists with the plugin enables (idempotent).
python3 - "$RC" <<'PY'
import sys, re
path = sys.argv[1]
text = open(path).read()
wanted = {"enable_ai_diffusion": "true", "enable_kritamcp": "true"}
lines = text.splitlines()
# Find [python] section bounds.
start = next((i for i, l in enumerate(lines) if l.strip() == "[python]"), None)
if start is None:
    if lines and lines[-1].strip() != "":
        lines.append("")
    lines.append("[python]")
    for k, v in wanted.items():
        lines.append(f"{k}={v}")
else:
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("[")), len(lines))
    section = lines[start:end]
    for k, v in wanted.items():
        if not any(re.match(rf"\s*{k}\s*=", l) for l in section):
            section.append(f"{k}={v}")
        else:
            section = [re.sub(rf"^(\s*{k}\s*=).*", rf"\g<1>{v}", l) for l in section]
    lines = lines[:start] + section + lines[end:]
open(path, "w").write("\n".join(lines) + "\n")
print("✓ [python] enable_ai_diffusion=true, enable_kritamcp=true")
PY

# Point the plugin at ComfyBox (external ComfyUI server) if settings exist.
if [ -f "$SETTINGS" ]; then
  python3 - "$SETTINGS" "$ENDPOINT" <<'PY'
import sys, json
path, endpoint = sys.argv[1], sys.argv[2]
s = json.load(open(path))
s["server_mode"] = "external"
s["server_url"] = endpoint
json.dump(s, open(path, "w"), indent=4)
print(f"✓ plugin server_mode=external, server_url={endpoint}")
PY
fi

echo "Done. Relaunch Krita — the AI Image Generation docker will return, connected to ComfyBox."
