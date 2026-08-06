# Reference requests

Known-good and diagnostic request bodies for `POST /v1/video/generate`.
Kept as files so A/B work is reproducible instead of retyped inline.

    curl -s -X POST http://127.0.0.1:7870/v1/video/generate \
      -H 'Content-Type: application/json' \
      -d @qa/reference-requests/audio-reference-library-4s.json

- `audio-reference-library-12s.json` — the clip Todd judged perfect
  (2026-08-05). The baseline any audio change must not regress.
- `audio-reference-library-4s.json` — same scene, one beat, ~5 min.
  Iterate here (skill iron rule 7).

Editing rules: change ONE field per experiment, keep the seed, and give
the output a self-describing name (`EXP_<lever>_<value>.mp4`). Strip the
`_comment` key if a client rejects unknown fields (ComfyBox ignores it).
