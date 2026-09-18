# Lip-sync measurement

## Use `lip-aperture.swift`, not pixel difference

    swift scripts/analysis/lip-aperture.swift clip.mp4   # one aperture per frame

It reports MOUTH APERTURE — the inner-lip separation from Vision's face
landmarks, normalised by face height. A head that turns, nods, rises or moves
toward camera does not change it. Only an opening mouth does.

## Why this file exists

A pixel-difference metric (mean absolute frame-to-frame change inside a mouth
box) was used first, and it produced two published conclusions that were both
wrong:

- that shortening audio-driven chunks improved lip sync (it did not; the
  "improvement" was the one-time transient of the subject raising her head
  from the opening keyframe as the voice starts, which is a fixed ~40 frames
  and therefore a larger FRACTION of a shorter chunk);
- that a carry-over frame catches the mouth mid-word (it does not; the frame
  was inspected and the mouth is closed).

Both survived because the metric could not tell a moving head from a moving
mouth, and because nobody looked at the frames until late.

## Rules learned the hard way

1. **Verify the region on a real frame before trusting a number.** The first
   mouth box was over the subject's CHEST and scored a confident +0.144.
2. **Onsets masquerade as sync.** Any metric that includes the first ~40
   frames of a chunk is measuring the model settling into the shot.
3. **Always score a control.** An unconditioned clip against the same voice is
   the noise floor. On aperture that floor is r ~ 0.3 — higher than any
   conditioned clip has scored, which is why "there is a correlation" is not
   the same as "there is lip sync".
4. **Report sample counts.** Most of these clips yield ~20 loud frames. Two
   numbers 0.1 apart on n=20 are the same number.
