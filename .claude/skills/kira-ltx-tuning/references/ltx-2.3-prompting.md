# LTX-2.3 prompting reference

Use this reference when authoring or evaluating LTX-2.3 T2V/I2V prompts,
especially prompts with generated speech or ambient sound.

## Source priority

1. [Lightricks Prompting Guide](https://docs.ltx.io/open-source-model/usage-guides/prompting-guide)
2. [Official LTX-2 repository prompting notes](https://github.com/Lightricks/LTX-2#%EF%B8%8F-prompting-for-ltx-2)
3. [LTX-2.3 practical prompting guide](https://ltxworkflow.com/resources/tutorials/ltx-23-prompting-guide)

The third source is an unaffiliated practical condensation of the official
guidance. Use it for examples and LTX-2.3-specific heuristics; prefer current
Lightricks documentation if the sources diverge.

## Core form

- Write one flowing paragraph in present tense, normally 4–8 descriptive
  sentences and under 200 words.
- Give the shot one dominant event. Layer detail only after the simple version
  works.
- Keep the order legible: subject -> action -> camera -> mood/lighting.
- Prefer literal named motion (`turns`, `walks`, `pours`, `tracks right`) over
  labels such as `dynamic`, `cinematic`, or `epic` without an action.
- Block spatial relationships explicitly: left/right, foreground/background,
  facing direction, subject distance, and what the camera reveals.
- Describe observable physical cues instead of internal emotional states.
- Match detail to shot scale. Close-ups benefit from material, hair, skin,
  fabric, and edge detail; wide shots need environmental structure.
- Treat portrait framing as native composition, not cropped landscape.
- Avoid overloaded scenes, conflicting lighting, chaotic physics, and requests
  for readable text or logos.

## T2V and I2V

For T2V, establish the subject, appearance, environment, dominant action,
camera intent, lighting, and sound. The prompt must construct the whole shot.

For I2V, let the source image establish static appearance. Use clear verbs to
say who moves, what moves, how it moves, and what the camera does. Avoid vague
instructions such as `the scene comes alive` and avoid re-describing the still
so heavily that the requested motion is buried.

## Camera language

- `follows` / `tracks`: subject moves through space
- `pans across`: camera scans a mostly static scene
- `circles around`: reveal the subject from multiple sides
- `tilts upward` / `pushes in`: emphasis or reveal
- `over-the-shoulder`: dialogue or interaction
- `wide establishing shot`: setting context
- `static frame`: stillness or tension
- `handheld movement`: documentary intimacy or chaos

Name pacing deliberately when needed: slow motion, time-lapse, rapid cuts,
lingering shot, continuous shot, freeze-frame, or fade.

## Audio and dialogue

- Describe wanted sound explicitly: speech, ambience, music, tone, intensity,
  and relative volume. Do not rely on `natural audio`.
- Put exact dialogue in quotation marks.
- Specify language and, when relevant, accent.
- Specify a concrete delivery and volume: calm adult voice, conversational
  volume, whisper, mutter, shout, resonant announcer, robotic monotone, etc.
- Budget dialogue to clip duration. A quality probe uses one speaker and one
  short utterance that can be spoken naturally within the clip. Too many words
  can cause skipped, rushed, or unintelligible speech; too few can sound slow.
- For an intelligibility probe, remove competing sound by stating one simple
  positive condition such as `the room remains otherwise silent`. Do not pack
  the positive prompt with a long taxonomy of forbidden noise.
- For an ambience probe, name one intended ambient source and its intensity.
  Do not combine it with the speech-quality discriminator.
- In this engine, start audio-enabled prompts with an explicit `Audio:` clause
  so the 128-token prompt guard can prove the audio marker and quoted line
  survive truncation. Follow with `Visual:` and keep the entire prompt a single
  paragraph.

### Speech-quality probe template

```text
Audio: A single adult woman speaks one short sentence in clear English, in a
calm natural voice at conversational volume and an unhurried pace. She says,
"The blue glaze is dry, so the lesson can begin." The room remains otherwise
silent. Visual: An adult ceramic artist stands beside a finished clay mug in a
quiet community studio. She faces the camera, speaks the line, and makes one
small natural hand gesture. A static medium close-up keeps her face in focus
against softly blurred shelves. Soft even daylight and neutral color create
understated documentary realism.
```

This is a diagnostic template, not a universal style. Keep its seed, request
shape, and engine configuration fixed when comparing another audio lever.

## Evaluation discipline

- Separate prompt compliance from audio fidelity. Prompt logs can prove that
  conditioning survived; only listening can establish intelligibility and
  desirability.
- Change one semantic variable per matched-seed comparison.
- Do not use an ambience-rich prompt to decide whether unexplained background
  noise is a codec/model defect.
- Do not infer a pipeline defect from an utterance whose word count cannot fit
  the requested duration at a natural pace.
