// LTX2BeatSchedule.swift — Temporal beat scheduling (comfybox#310)
//
// Todd's motion library carries orchestrated multi-beat templates
// (approach -> rhythm -> hands -> tempo) that today get flattened into one
// action string before the model ever sees them — the orchestration is
// prose, not time. This file reimplements the WhatDreamsCost PromptRelay
// technique in MLX: an additive Gaussian penalty on text->video (and
// text->audio) cross-attention that makes each beat's own tokens attend
// mainly to its own frame window, so a single non-chunked render can carry
// true multi-beat motion.
//
// Reference: prompt_relay.py (build_temporal_cost / build_temporal_cost_scaled
// / build_segments / map_token_indices).
//
// Design: `BeatSegment` (wire shape, fractions of clip duration) ->
// `LTX2BeatScheduleLocator.locate` (per-beat token range in the composed,
// left-padded prompt, fail-open per beat) -> `LTX2ResolvedBeat` ->
// `LTX2BeatScheduleBuilder.build{Video,Audio}Bias` (additive bias tensor,
// rebuilt fresh from each render stage's own frame geometry — never reused
// across base/refine, since refine's latent resolution differs). Absent
// `beat_schedule` (empty resolved list) yields `nil` from both builders —
// byte-identical to today's no-bias code path.

import Foundation
import MLX

// MARK: - Wire shape

/// One beat of a multi-beat motion template: a text span plus its fraction
/// of the clip's TOTAL duration. `startFrac`/`endFrac` are resolved to
/// frames per render stage (base vs. refine can have different frame counts
/// / resolutions) and to seconds for the audio cross-attention bias — never
/// stored as absolute frames, so the same `BeatSegment` is valid at any
/// stage's geometry.
public struct BeatSegment: Codable, Sendable, Equatable {
  public var text: String
  public var startFrac: Float
  public var endFrac: Float
  /// Per-beat strength multiplier on the penalty. Nil defaults to 1.0.
  public var strength: Float?

  public init(text: String, startFrac: Float, endFrac: Float, strength: Float? = nil) {
    self.text = text
    self.startFrac = startFrac
    self.endFrac = endFrac
    self.strength = strength
  }
}

// MARK: - Located beat (post token-range locate)

/// A beat whose text span has been located in the composed, left-padded
/// prompt token sequence. `tokenStart..<tokenEnd` indexes the PADDED axis
/// (i.e. the same axis the text encoder's `[B, S]` input occupies), so it
/// can be used directly as a column range into a `(1, 1, videoTokens, S)`
/// bias tensor. Beats that couldn't be located are dropped before this
/// point (fail-open — see `LTX2BeatScheduleLocator`), so every
/// `LTX2ResolvedBeat` is safe to bias.
public struct LTX2ResolvedBeat: Equatable {
  public let tokenStart: Int
  public let tokenEnd: Int
  public let startFrac: Float
  public let endFrac: Float
  public let strength: Float

  public init(tokenStart: Int, tokenEnd: Int, startFrac: Float, endFrac: Float, strength: Float) {
    self.tokenStart = tokenStart
    self.tokenEnd = tokenEnd
    self.startFrac = startFrac
    self.endFrac = endFrac
    self.strength = strength
  }
}

// MARK: - Token-range locate

/// Locates each beat's text as a contiguous token run in the full composed
/// prompt, accounting for the Gemma tokenizer's LEFT padding
/// (`LTX2GemmaTokenizer.encode`: pad tokens go at the START, so a
/// column index in the un-padded sequence needs a constant offset added).
///
/// Beats are matched LEFT TO RIGHT, each search starting where the previous
/// beat's match ended — this keeps ordering monotonic and prevents a short
/// beat text from re-matching an earlier occurrence.
///
/// A beat that can't be located (the tokenizer merged its boundary with
/// adjacent text into a different token — the one fragile part of this
/// design per the FDD) is DROPPED, never fails the render: the caller logs
/// once and the render proceeds with that beat contributing zero bias
/// (identical to not having scheduled it at all).
///
/// comfybox#335: a beat's STANDALONE tokenization differs from its
/// in-context run in two ways, either of which used to defeat the
/// subsequence search (in production, both did — 100% of locates failed):
///   1. BOS — `tokenize` is the production tokenizer's raw encode, which
///      prepends special tokens to standalone text (Gemma: `<bos>` id 2).
///      Those specials appear once at the START of the full prompt, never
///      at a beat's mid-prompt position. `tokenize("")` yields exactly that
///      standalone prefix (Gemma: `[2]`), so it is stripped from every
///      candidate before searching.
///   2. Leading space — SentencePiece-style pretokens keep their leading
///      space, so mid-prompt "… counter. She walks…" contains `▁She`
///      (2625) while standalone "She walks…" starts with bare `She`
///      (5778). Each beat is therefore tokenized BOTH bare and with a
///      leading space, and the earliest match of either form wins. The
///      bare form still covers prompt-start and post-newline positions
///      (after `\n` (107) the next word appears in its bare form).
/// All tokens the search runs against come from the SAME
/// `untruncatedTokenIds` view of the SAME post-reorder `effectivePrompt`
/// the text encoder receives, so a string-level match is exactly a
/// token-level match once these two standalone artifacts are removed.
public enum LTX2BeatScheduleLocator {
  public static func locate(
    beats: [BeatSegment],
    fullPromptTokenIds: [Int],
    maxLength: Int,
    onDrop: ((BeatSegment, String) -> Void)? = nil,
    tokenize: (String) -> [Int]
  ) -> [LTX2ResolvedBeat] {
    let truncatedLen = min(fullPromptTokenIds.count, maxLength)
    let padOffset = max(0, maxLength - fullPromptTokenIds.count)
    // Standalone-encode special-token prefix (Gemma: [2] — BOS). Computed
    // once; empty for tokenizers that add nothing (e.g. the test fakes).
    let standalonePrefix = tokenize("")
    var searchFrom = 0
    var resolved: [LTX2ResolvedBeat] = []

    for beat in beats {
      // Frac hygiene (adversarial review F10): the server-side sanitizer
      // isn't merged yet and the flag defaults on, so the engine enforces
      // its own invariants — clamp fracs into [0, 1] and drop degenerate
      // (endFrac <= startFrac after clamping, or non-finite) spans. Same
      // fail-open contract as an unlocatable beat: drop + log, never fail.
      guard beat.startFrac.isFinite, beat.endFrac.isFinite else {
        onDrop?(beat, "non-finite startFrac/endFrac")
        continue
      }
      let startFrac = min(max(beat.startFrac, 0), 1)
      let endFrac = min(max(beat.endFrac, 0), 1)
      guard endFrac > startFrac else {
        onDrop?(beat, "degenerate span after clamping fracs to [0,1] (endFrac <= startFrac)")
        continue
      }
      // comfybox#335: try the bare and leading-space tokenizations, both
      // with the standalone special-token prefix stripped. Earliest match
      // wins so the left-to-right monotonic contract is preserved.
      var candidates: [[Int]] = []
      for form in [beat.text, " " + beat.text] {
        let ids = strippingStandalonePrefix(tokenize(form), prefix: standalonePrefix)
        if !ids.isEmpty, !candidates.contains(ids) { candidates.append(ids) }
      }
      guard !candidates.isEmpty else {
        onDrop?(beat, "standalone tokenization produced no tokens")
        continue
      }
      var match: (start: Int, count: Int)?
      for ids in candidates {
        if let s = findSubsequence(ids, in: fullPromptTokenIds, from: searchFrom),
           match == nil || s < match!.start {
          match = (s, ids.count)
        }
      }
      guard let (localStart, matchCount) = match else {
        onDrop?(beat, "could not locate beat text as a contiguous run in the composed prompt tokens")
        continue
      }
      let localEnd = localStart + matchCount
      searchFrom = localEnd
      guard localStart < truncatedLen else {
        onDrop?(beat, "beat fell entirely outside the tokenizer's \(maxLength)-token window")
        continue
      }
      let clampedEnd = min(localEnd, truncatedLen)
      resolved.append(LTX2ResolvedBeat(
        tokenStart: padOffset + localStart,
        tokenEnd: padOffset + clampedEnd,
        startFrac: startFrac,
        endFrac: endFrac,
        strength: beat.strength ?? 1.0))
    }
    return resolved
  }

  /// Removes the tokenizer's standalone-encode special-token prefix
  /// (Gemma: BOS `[2]`) from a candidate beat tokenization. A candidate
  /// that is NOTHING BUT the prefix (an effectively empty beat text)
  /// strips to `[]`, which the caller drops as token-less.
  private static func strippingStandalonePrefix(_ ids: [Int], prefix: [Int]) -> [Int] {
    guard !prefix.isEmpty, ids.count >= prefix.count,
          Array(ids.prefix(prefix.count)) == prefix else { return ids }
    return Array(ids.dropFirst(prefix.count))
  }

  private static func findSubsequence(_ needle: [Int], in haystack: [Int], from: Int) -> Int? {
    guard !needle.isEmpty, from >= 0, needle.count <= haystack.count - from else { return nil }
    var i = from
    while i + needle.count <= haystack.count {
      if Array(haystack[i..<(i + needle.count)]) == needle { return i }
      i += 1
    }
    return nil
  }
}

// MARK: - Bias builder

/// Builds the additive Gaussian-penalty cross-attention bias from resolved
/// beats plus one render stage's own frame geometry. Reimplements
/// WhatDreamsCost PromptRelay's math verbatim:
/// `cost = strength · relu(|pos − midpoint| − window)² / (2σ²)`, negated
/// (steers attention AWAY from out-of-window positions, added directly to
/// attention logits). Tokens outside every beat's range — camera/base
/// description, audio-ambience text, padding — get exactly zero bias, so
/// they keep attending everywhere, unchanged.
public enum LTX2BeatScheduleBuilder {
  /// Paper-derived constant, independent of segment length: σ = 1/ln(1/ε),
  /// ε = 1e-3 ⇒ σ ≈ 0.1448.
  public static let sigma: Float = 1.0 / Foundation.log(1.0 / 1e-3)

  /// One beat's frame-space midpoint/window for the CURRENT stage's frame
  /// count (never cached across stages — refine's frame count/resolution
  /// can differ from the base pass, so this is recomputed fresh every call).
  private static func frameGeometry(_ beat: LTX2ResolvedBeat, frames: Int) -> (midpoint: Float, window: Float) {
    let midpoint = (beat.startFrac + beat.endFrac) * 0.5 * Float(frames)
    let span = max(beat.endFrac - beat.startFrac, 0) * Float(frames)
    let window = max(span * 0.5 - 2, 0)
    return (midpoint, window)
  }

  /// `(1, 1, videoTokens, textLen)` additive bias for one stage's frame
  /// geometry, `videoTokens = frames * tokensPerFrame` in the frame-major
  /// token order `LTX2PatchEmbed`/`LTX2VideoGenerator` use. `nil` when
  /// there's nothing to bias (no resolved beats, or every beat's penalty is
  /// zero everywhere) — byte-identical to today's no-bias path.
  public static func buildVideoBias(
    resolved: [LTX2ResolvedBeat],
    frames: Int,
    tokensPerFrame: Int,
    textLen: Int
  ) -> MLXArray? {
    guard !resolved.isEmpty, frames > 0, tokensPerFrame > 0, textLen > 0 else { return nil }
    let videoTokens = frames * tokensPerFrame
    var cost = [Float](repeating: 0, count: videoTokens * textLen)
    var anyNonZero = false

    for beat in resolved {
      guard beat.tokenEnd > beat.tokenStart, beat.tokenStart >= 0, beat.tokenEnd <= textLen else { continue }
      let (midpoint, window) = frameGeometry(beat, frames: frames)
      for q in 0..<videoTokens {
        let frame = Float(q / tokensPerFrame)
        let over = max(abs(frame - midpoint) - window, 0)
        guard over > 0 else { continue }
        let penalty = -(beat.strength * over * over / (2 * sigma * sigma))
        anyNonZero = true
        let rowBase = q * textLen
        for k in beat.tokenStart..<beat.tokenEnd {
          cost[rowBase + k] = penalty
        }
      }
    }
    guard anyNonZero else { return nil }
    return MLXArray(cost, [1, 1, videoTokens, textLen])
  }

  /// `(1, 1, audioTokens, textLen)` additive bias for the audio cross
  /// attention (`audio_attn2`), mapping tokens by their REAL per-token
  /// seconds (`LTX2AudioPatchifier`'s `[start, end]` timings) rather than
  /// PromptRelay's integer-frame approximation — pass the token MIDPOINT
  /// seconds already computed from those timings. `totalFrames`/`fps`
  /// convert the beat's frame-space midpoint/window (same geometry the
  /// video bias uses) into seconds on the same real-time axis.
  public static func buildAudioBias(
    resolved: [LTX2ResolvedBeat],
    totalFrames: Int,
    fps: Float,
    audioTokenMidSeconds: [Float],
    textLen: Int
  ) -> MLXArray? {
    guard !resolved.isEmpty, totalFrames > 0, fps > 0, !audioTokenMidSeconds.isEmpty, textLen > 0 else { return nil }
    let audioTokens = audioTokenMidSeconds.count
    var cost = [Float](repeating: 0, count: audioTokens * textLen)
    var anyNonZero = false

    for beat in resolved {
      guard beat.tokenEnd > beat.tokenStart, beat.tokenStart >= 0, beat.tokenEnd <= textLen else { continue }
      let (midpointFrame, windowFrame) = frameGeometry(beat, frames: totalFrames)
      let midpointSec = midpointFrame / fps
      let windowSec = windowFrame / fps
      for q in 0..<audioTokens {
        let over = max(abs(audioTokenMidSeconds[q] - midpointSec) - windowSec, 0)
        guard over > 0 else { continue }
        let penalty = -(beat.strength * over * over / (2 * sigma * sigma))
        anyNonZero = true
        let rowBase = q * textLen
        for k in beat.tokenStart..<beat.tokenEnd {
          cost[rowBase + k] = penalty
        }
      }
    }
    guard anyNonZero else { return nil }
    return MLXArray(cost, [1, 1, audioTokens, textLen])
  }

  /// Sums two optional additive attention biases (both are logit-space
  /// terms, so this composes exactly): nil+nil = nil, either alone passes
  /// through unchanged, both present sum elementwise (broadcasting).
  public static func combine(_ a: MLXArray?, _ b: MLXArray?) -> MLXArray? {
    switch (a, b) {
    case (nil, nil): return nil
    case (let x?, nil): return x
    case (nil, let y?): return y
    case (let x?, let y?): return x + y
    }
  }
}
