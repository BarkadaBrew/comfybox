// SpeechLength.swift — Director tab speech-length readout (WP3).
//
// LTX-2 speaks the quoted dialogue in a prompt. A rough "does the dialogue
// fit the clip?" check: count the words inside double quotes and assume
// 2.5 words per second of speech.

import Foundation

enum SpeechLength {

    /// Words spoken per second of generated speech.
    static let wordsPerSecond = 2.5

    /// Quote characters that open or close a spoken span: straight `"` and
    /// curly `“` `”`. Any of them toggles in/out of a span, so balanced
    /// straight or curly quoting both work; an unbalanced trailing quote
    /// counts to the end of the text.
    private static let quoteCharacters: Set<Character> = ["\"", "\u{201C}", "\u{201D}"]

    static func quotedWordCount(_ text: String) -> Int {
        var inQuote = false
        var inWord = false
        var count = 0
        for ch in text {
            if quoteCharacters.contains(ch) {
                inQuote.toggle()
                inWord = false
                continue
            }
            guard inQuote else { continue }
            if ch.isWhitespace {
                inWord = false
            } else if !inWord {
                inWord = true
                count += 1
            }
        }
        return count
    }

    static func seconds(_ text: String) -> Double {
        Double(quotedWordCount(text)) / wordsPerSecond
    }
}
