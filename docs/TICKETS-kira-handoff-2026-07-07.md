# ComfyBox — bugs from Kira live-run (handoff 2026-07-07)

Found live overnight against a remote daemon (`10.0.100.134:7870`) during Kira's
sequence-arc renders. Code pointers + assessment added from a read of this repo.
All fixes land here (same codebase); none tested yet — daemon is busy with Kira.

---

## P1 — img2img returns HTTP 500 outright (blocks a whole asset class)
**Symptom:** any render with `init_image` + `image_strength` → HTTP 500 in ~6s,
`ZImage.Img2ImgUtilities.Img2ImgError error 0`. Plain txt2img with identical
params works. Blocks all five sequence arcs.
**Repro:** `POST http://10.0.100.134:7870 …/generate` with `init_image` +
`image_strength: 0.55` on any existing image.
**Code:** route detects img2img when `payload.imagePath` set → `makeImg2ImgRequest`
→ `pipeline.generateImg2Img` (`WarmServer.swift:3055-3059`). Error enum:
`ImageToImagePipeline.swift:134 enum Img2ImgError` and `Img2ImgUtilities.Img2ImgError`
(thrown at `:232 sourceImageNotFound`, `:237 sourceImageLoadFailed`, plus
`maskGenerationFailed`).
**Assessment (unverified):** "error 0" = the enum's first case. If that's
`sourceImageNotFound`, this is the **same path-not-on-server class as P2** —
`init_image` is sent as a *client-side* path the remote daemon can't read (desktop
sends `initImagePath`, EngineService), so it 500s regardless of image validity.
If it's a later case, it's a genuine pipeline bug.
**Next:** reproduce with a **server-local** init_image path to split "path/transfer"
from "pipeline". If transfer: add a **bytes-upload for init_image** (see feature
below). If pipeline: instrument `generateImg2Img` to log which case fires.
**Severity: P1.**

---

## P2 — video jobs never transfer files (either direction)
**Symptom (input):** `image_path` must already exist on the daemon's filesystem; a
non-local path → `400 image_path not found or not a regular file`. No upload/fetch.
**Symptom (output):** completed jobs report a Mac-side `output_path`
(`~/Pictures/ComfyBox/video/…`) and never SCP it back, though a code contract
comment implies the bridge used to.
**Code:** `WarmServer.swift:845 /v1/video/generate` — validates the source image is
a server-local regular file with PNG/JPEG magic (`ReplicateVideoProxy.validateSourceImage`)
before base64-uploading to Replicate. There is **no client→server image upload** and
**no server→client output download**.
**Assessment:** the HTTP video route exists but assumes co-located files. Kira's
scp workaround is a band-aid.
**Fix:** add a bytes-upload API for the init image (reuse the bridge's
`/api/etn/image/<id>` cache pattern) and return output **bytes** (or a fetchable
URL), or formally document the "caller provides/collects server-local paths"
contract and drop the stale comment.
**Severity: P2.**

---

## P3 — WarmServer instability under sustained load
**Symptom:** after a ~46-render marathon, renders began failing instantly at
23:00:10; both MCP bridge processes died at 23:02:32 (`comfybox exit 255`) before
auto-reconnect at 23:02:53. Crash after ~75 min continuous generation.
**Assessment:** likely memory pressure (MLX weights + accumulating buffers / the
tile-leak warnings seen earlier). Hard to repro without a load harness.
**Next:** pull Mac-side logs from that window (`/tmp/comfybox-serve-err.log`,
Console crash reports, `log show --predicate 'process == "ComfyBox"'` around
23:00–23:03) to confirm OOM vs. an unhandled throw. Consider a periodic
weight/cache reset or a memory-watermark guard in the process loop.
**Severity: P2/P3 (stability).**

---

## P4 — MCP API contract skew
**Symptom:** video MCP tools return their payload as **stringified JSON** inside the
MCP content rather than structured fields; job status reports `succeeded` where
consumers expected `completed`. Kira's side is now tolerant of both, but the other
~30 tools carry the same risk.
**Fix:** pin the MCP tool response contract — structured content (not
double-encoded JSON strings) and a single canonical status vocabulary
(`completed`), across all tools. Add a contract test.
**Severity: P3.**

---

## P5 — video model routing ignores the requested model (cost + zero-cloud posture)
**Symptom:** a submit with `model: "ltx"` still ran `wan-video/wan-2.2-i2v-a14b` on
the Replicate backend. Either the param is ignored or there's a silent
local→cloud fallback.
**Code:** `ReplicateVideoProxy.swift:66 i2vModel = "wan-video/wan-2.2-i2v-a14b"`
(hardcoded; the proxy always uses it). `WarmServer.swift:845` **silently falls back
to Replicate** when local LTX-2 isn't configured (`localVideoResponseIfConfigured`
returns nil → `replicateVideoProxy`).
**Assessment:** confirmed root cause — the Replicate proxy ignores the requested
model, and the route silently uses paid cloud when local isn't configured. A
request meant for local `ltx` can render on **paid Replicate** — a cost and
**zero-cloud-posture** violation for non-neutral content.
**Fix:** honor the `model` param; make cloud fallback **explicit/opt-in** (error, or
require `backend: "replicate"`) rather than silent; log the chosen backend+model.
**Severity: P1 for posture, P2 functionally.**

---

## Feature — HTTP video endpoints usable by a remote HTTP-only client
**Ask:** video is effectively MCP-only for Kira; her daemon (HTTP-only, `mcp: null`)
can't render video. An HTTP video route resolves the pending "MCP-for-Kira"
decision without widening her isolation boundary.
**Assessment:** `/v1/video/generate` **already exists** over HTTP (`WarmServer.swift:845`)
— the real gaps are (a) it needs **local LTX-2 configured** on Kira's daemon (else
503 or silent Replicate per P5), and (b) it assumes **co-located files** (P2). So
"make HTTP video work for Kira" = P2 (bytes upload/download) + P5 (local routing
default, no silent cloud) + confirm LTX-2 weights on her daemon.
**Severity: feature (unblocks Kira video, no cloud).**
