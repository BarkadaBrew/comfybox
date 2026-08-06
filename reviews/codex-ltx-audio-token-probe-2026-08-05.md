# LTX-2 audio prompt token probe — 2026-08-05

This P0 probe uses a neutral adult pottery-studio fixture. It does not load, tokenize, or reproduce any current Kira prompt content.

## Setup

- Production path: `LTX2GemmaTokenizer.load(...)` and `untruncatedTokenIds(prompt:)`
- Model directory: `~/.cache/huggingface/hub/models/unsloth/gemma-3-12b-it`
- Tokenizer library: `swift-transformers` 1.1.6
- Limit: 128 tokens
- Marker index convention: number of tokenizer IDs in the prefix before `Audio:`; this includes the tokenizer's BOS token.
- Both variants contain exactly the same visual and audio text. Only section order changes.

The fixture was run as a temporary `ZImageTests` probe linked against the production implementation; the temporary probe was removed after its assertions and captured output passed.

Audio section:

> Audio: quiet room tone, soft rain against the windows, the pottery wheel humming, and the adult artist says "the glaze is ready".

The visual section describes a clearly adult ceramic artist in a public studio. It is 161 tokens without the leading `Visual:` marker, including one BOS token. The isolated audio section is 29 tokens, also including BOS.

## Raw prompt results

| Shape | Total tokens | Audio marker index | Quoted line survives first 128 |
|---|---:|---:|---|
| Audio inside budget, then visual | 191 | 1 | yes |
| Visual first, audio beyond budget | 191 | 164 | no |

Decoded first 128 tokens, audio inside budget:

> \<bos\>Audio: quiet room tone, soft rain against the windows, the pottery wheel humming, and the adult artist says "the glaze is ready". Visual: An adult ceramic artist with warm brown skin, shoulder-length black hair, and calm observant eyes wears a practical indigo work shirt, rolled sleeves, a canvas apron, dark trousers, and clean studio shoes. Her appearance remains consistent: natural features, realistic proportions, relaxed posture, steady hands, and a thoughtful expression. She is clearly an adult professional working in a public community arts studio. Documentary realism, accurate anatomy, natural skin texture, subtle fabric detail, neutral white balance, soft even

Decoded first 128 tokens, audio beyond budget:

> \<bos\>Visual: An adult ceramic artist with warm brown skin, shoulder-length black hair, and calm observant eyes wears a practical indigo work shirt, rolled sleeves, a canvas apron, dark trousers, and clean studio shoes. Her appearance remains consistent: natural features, realistic proportions, relaxed posture, steady hands, and a thoughtful expression. She is clearly an adult professional working in a public community arts studio. Documentary realism, accurate anatomy, natural skin texture, subtle fabric detail, neutral white balance, soft even lighting, and no text on screen. She sits at a wooden wheel shaping a small clay mug while afternoon window light crosses a public studio.

## Guard result

The beyond-budget variant normalizes byte-for-byte to the inside-budget variant:

- Effective prompt hash (SHA-256, first 12 hex): `5d8eeefaa2bf`
- Audio marker index: 1
- Quoted line survives: yes
- Duplicate leading `Visual:` marker: no

This reproduces the truncation mechanism with production tokenization and verifies that section reordering removes order as a variable while retaining the 128-token model limit.

## Live follow-up — 2026-08-06

The deployed guard produced byte-identical decoded A/V streams for a matched
4-second pair where the same audio clause was originally inside vs beyond the
128-token boundary. This verifies prompt-order normalization in the live
production path; it does not establish subjective audio quality.

The first 6-second English probe was not a valid quality discriminator. It put
28 spoken words plus a pause into six seconds (about 280 words per minute), and
Todd judged the resulting voice illegible and undesirable. The prompt itself
was overconstrained even though telemetry proved the quoted text survived.

A corrected probe follows the LTX-2.3 guidance distilled in
`.claude/skills/kira-ltx-tuning/references/ltx-2.3-prompting.md`: one adult
speaker, one ten-word quoted English sentence, an explicit natural voice and
conversational volume, and otherwise silence. It keeps the same seed, request
shape, production configuration, and six-second duration.

- Output: `~/Pictures/ComfyBox/EXP_audio_english_short_line_seed424242_6s.mp4`
- Prompt telemetry: 114 tokens, audio marker index 1, quoted line present and
  surviving, effective hash `b57d491605f9`, no reordering
- Container: H.264 1024x640 + AAC 48 kHz stereo, both 6.041667 seconds
- File SHA-256: `48edbb9d45a359b893443d56234b81d69883f41e0f9f3d90455df0c374ba9c2e`
- Status: render and plumbing verified; subjective intelligibility and voice
  quality are pending Todd's listening verdict
