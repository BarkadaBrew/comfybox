// LTX2AudioPromptGuard.swift — keep audio conditioning inside Gemma's token window

import CryptoKit
import Foundation

/// The final, token-aware prompt facts emitted for every LTX-2 render.
///
/// `preTruncationTokenCount`, `audioMarkerTokenIndex`, and the quote fields all
/// describe `effectivePrompt` after any guard rewrite and immediately before
/// the tokenizer applies its fixed sequence-length cap.
public struct LTX2AudioPromptGuardResult: Equatable, Sendable {
    public let effectivePrompt: String
    public let preTruncationTokenCount: Int
    public let audioMarkerTokenIndex: Int?
    public let quotedLinePresent: Bool
    public let quotedLineSurvived: Bool
    public let effectivePromptHash: String
    public let reordered: Bool
}

/// Protects prompt-conditioned audio from tail truncation without changing the
/// model's trained 128-token sequence length.
///
/// Existing callers author a trailing `audio:` clause. If that complete clause
/// would cross the cap, the guard moves it to the front and inserts a stable
/// `Visual:` boundary. A guarded prompt therefore has the idempotent form
/// `Audio: … Visual: …`; future passes can identify the whole audio section
/// rather than guessing where its prose ends.
public enum LTX2AudioPromptGuard {
    private struct Inspection {
        let tokenCount: Int
        let audioMarkerTokenIndex: Int?
        let audioSectionSurvived: Bool?
        let quotedLinePresent: Bool
        let quotedLineSurvived: Bool
        let audioSectionRange: Range<String.Index>?
        let visualMarkerRange: Range<String.Index>?
    }

    public static func prepare(
        prompt: String,
        audio: Bool,
        maxLength: Int,
        tokenize: (String) -> [Int]
    ) throws -> LTX2AudioPromptGuardResult {
        precondition(maxLength > 0, "LTX-2 prompt max length must be positive")

        let initial = inspect(prompt: prompt, maxLength: maxLength, tokenize: tokenize)
        var effectivePrompt = prompt
        var reordered = false

        if audio, let audioRange = initial.audioSectionRange {
            // Check the section in isolation before moving it. If the audio
            // direction itself is too long, no ordering can make it survive.
            let audioSection = String(prompt[audioRange]).trimmingCharacters(in: .whitespacesAndNewlines)
            if tokenize(audioSection).count > maxLength {
                throw LTX2VideoError.audioUnsupported(
                    "the audio section exceeds the \(maxLength)-token prompt window; shorten the sound/dialogue direction")
            }

            if initial.audioSectionSurvived != true ||
                (initial.quotedLinePresent && !initial.quotedLineSurvived) {
                effectivePrompt = reorder(
                    prompt: prompt,
                    audioSectionRange: audioRange,
                    visualMarkerRange: initial.visualMarkerRange)
                reordered = true
            }
        } else if audio, initial.quotedLinePresent, !initial.quotedLineSurvived {
            // There is dialogue but no structural marker telling us what prose
            // can safely move with it. Never silently drop the line or guess.
            throw LTX2VideoError.audioUnsupported(
                "quoted dialogue falls beyond the \(maxLength)-token prompt window; label its section with 'audio:' so it can be reordered safely")
        }

        let final = inspect(prompt: effectivePrompt, maxLength: maxLength, tokenize: tokenize)
        if audio, let survived = final.audioSectionSurvived, !survived {
            throw LTX2VideoError.audioUnsupported(
                "the audio section still falls beyond the \(maxLength)-token prompt window after reordering")
        }
        if audio, final.quotedLinePresent, !final.quotedLineSurvived {
            throw LTX2VideoError.audioUnsupported(
                "quoted dialogue still falls beyond the \(maxLength)-token prompt window after reordering")
        }

        return LTX2AudioPromptGuardResult(
            effectivePrompt: effectivePrompt,
            preTruncationTokenCount: final.tokenCount,
            audioMarkerTokenIndex: final.audioMarkerTokenIndex,
            quotedLinePresent: final.quotedLinePresent,
            quotedLineSurvived: final.quotedLineSurvived,
            effectivePromptHash: hash(effectivePrompt),
            reordered: reordered)
    }

    private static func inspect(
        prompt: String,
        maxLength: Int,
        tokenize: (String) -> [Int]
    ) -> Inspection {
        let audioMarker = prompt.range(
            of: #"\baudio\s*:"#,
            options: [.regularExpression, .caseInsensitive])

        let visualMarker: Range<String.Index>? = audioMarker.flatMap { marker in
            prompt.range(
                of: #"\bvisual\s*:"#,
                options: [.regularExpression, .caseInsensitive],
                range: marker.upperBound..<prompt.endIndex)
        }

        let audioSectionRange: Range<String.Index>? = audioMarker.map { marker in
            marker.lowerBound..<(visualMarker?.lowerBound ?? prompt.endIndex)
        }
        let quoteSearchRange = audioSectionRange ?? (prompt.startIndex..<prompt.endIndex)
        let quoteEnd = completeQuotedLineEnd(in: prompt, range: quoteSearchRange)

        let markerTokenIndex = audioMarker.map {
            tokenize(String(prompt[..<$0.lowerBound])).count
        }
        let sectionSurvived = audioSectionRange.map {
            tokenize(String(prompt[..<$0.upperBound])).count <= maxLength
        }
        let quoteSurvived = quoteEnd.map {
            tokenize(String(prompt[...$0])).count <= maxLength
        } ?? false

        return Inspection(
            tokenCount: tokenize(prompt).count,
            audioMarkerTokenIndex: markerTokenIndex,
            audioSectionSurvived: sectionSurvived,
            quotedLinePresent: quoteEnd != nil,
            quotedLineSurvived: quoteSurvived,
            audioSectionRange: audioSectionRange,
            visualMarkerRange: visualMarker)
    }

    private static func completeQuotedLineEnd(
        in prompt: String,
        range: Range<String.Index>
    ) -> String.Index? {
        let straight = closingQuoteEnd(in: prompt, range: range, open: "\"", close: "\"")
        let curly = closingQuoteEnd(in: prompt, range: range, open: "“", close: "”")
        return [straight, curly].compactMap { $0 }.min()
    }

    private static func closingQuoteEnd(
        in prompt: String,
        range: Range<String.Index>,
        open: Character,
        close: Character
    ) -> String.Index? {
        guard let opening = prompt[range].firstIndex(of: open) else { return nil }
        let contentStart = prompt.index(after: opening)
        guard contentStart < range.upperBound,
              let closing = prompt[contentStart..<range.upperBound].firstIndex(of: close),
              closing > contentStart else { return nil }
        return closing
    }

    private static func reorder(
        prompt: String,
        audioSectionRange: Range<String.Index>,
        visualMarkerRange: Range<String.Index>?
    ) -> String {
        let audioSection = String(prompt[audioSectionRange])
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let before = String(prompt[..<audioSectionRange.lowerBound])
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let afterStart = visualMarkerRange?.upperBound ?? audioSectionRange.upperBound
        let after = String(prompt[afterStart...])
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let visual = removingLeadingVisualMarker(
            from: [before, after].filter { !$0.isEmpty }.joined(separator: " ")
        )
        return visual.isEmpty ? audioSection : audioSection + " Visual: " + visual
    }

    private static func removingLeadingVisualMarker(from text: String) -> String {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let marker = trimmed.range(
            of: #"^visual\s*:"#,
            options: [.regularExpression, .caseInsensitive]
        ) else {
            return trimmed
        }

        return String(trimmed[marker.upperBound...])
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func hash(_ prompt: String) -> String {
        let digest = SHA256.hash(data: Data(prompt.utf8))
        return String(digest.map { String(format: "%02x", $0) }.joined().prefix(12))
    }
}
