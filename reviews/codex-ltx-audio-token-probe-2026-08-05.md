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
