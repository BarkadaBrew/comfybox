// SpeechLengthTests.swift — WP3: quoted-dialogue speech-length estimate.

import Foundation
import Testing
@testable import ComfyBoxDesktop

@Suite("SpeechLength")
struct SpeechLengthTests {

    @Test("counts words inside straight and curly double quotes")
    func countsStraightAndCurlyQuotes() {
        #expect(SpeechLength.quotedWordCount(#"She says "hello there" and smiles"#) == 2)
        #expect(SpeechLength.quotedWordCount("He whispers “come with me now” softly") == 4)
        #expect(SpeechLength.quotedWordCount(#""one" then "two three""#) == 3)
    }

    @Test("an unbalanced trailing quote counts to the end")
    func unbalancedQuoteCountsToEnd() {
        #expect(SpeechLength.quotedWordCount(#"She says "wait for me please"#) == 4)
    }

    @Test("seconds is words / 2.5")
    func secondsIsWordsOver2_5() {
        #expect(abs(SpeechLength.seconds(#""a b c d e""#) - 2.0) < 1e-9)
    }

    @Test("no quotes is zero")
    func noQuotesIsZero() {
        #expect(SpeechLength.quotedWordCount("a quiet beach at dusk") == 0)
        #expect(SpeechLength.seconds("") == 0)
    }
}
