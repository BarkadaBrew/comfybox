// AgeFloor.swift — hard age floor at the render boundary (2026-07-28).
//
// Prompt text reaching the generators can be authored by local models (the
// Dan's-PE optimizer for video, arbitrary API clients for images), and one
// such expansion drifted to "a petite 17 year old" in a rendered clip. Canon
// (and policy) is absolute: subjects are adults. This floor rewrites any
// sub-18 age descriptor to the canonical "18-year-old" and scrubs categorical
// minor terms on EVERY prompt right before conditioning — whatever the source.
// Mirrors coffeeshop-server's src/image/age-guard.ts (daemon-side twin).

import Foundation

public enum AgeFloor {
  private static let wordAges =
    "one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen"

  /// (pattern, replacement) pairs, applied in order. NSRegularExpression with
  /// case-insensitive matching; template `$0`-free literal replacements.
  private static let rules: [(NSRegularExpression, String)] = {
    func rx(_ pattern: String) -> NSRegularExpression {
      // Patterns are compile-time constants; a failure is a programmer error.
      // swiftlint:disable:next force_try
      try! NSRegularExpression(pattern: pattern, options: [.caseInsensitive])
    }
    return [
      (rx(#"\b(?:[1-9]|1[0-7]|"# + wordAges + #")[\s-]*(?:year|yr)s?[\s-]*old\b"#), "18-year-old"),
      (rx(#"\bage[d]?[\s:]+(?:[1-9]|1[0-7])\b"#), "age 18"),
      (rx(#"\b(?:teenager|teenaged|teenage|teen|preteen|pre-teen|underage|under-age|jailbait|loli|lolita|schoolgirl|school\s+girl|minor girl|young girl)\b"#), "18-year-old woman"),
      (rx(#"\bbarely[\s-]*legal\b"#), "adult"),
    ]
  }()

  /// Enforce the floor. Unchanged text passes through untouched.
  public static func enforce(_ prompt: String) -> String {
    var out = prompt
    for (pattern, replacement) in rules {
      let range = NSRange(out.startIndex..., in: out)
      out = pattern.stringByReplacingMatches(in: out, options: [], range: range, withTemplate: replacement)
    }
    return out
  }

  /// True when `enforce` would change the prompt (for logging at call sites).
  public static func violates(_ prompt: String) -> Bool {
    rules.contains { pattern, _ in
      pattern.firstMatch(in: prompt, options: [], range: NSRange(prompt.startIndex..., in: prompt)) != nil
    }
  }
}
