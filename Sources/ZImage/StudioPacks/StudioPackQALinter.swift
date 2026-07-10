// StudioPackQALinter.swift — Pack QA and style lint (FR-8 / #201).
//
// Runs a pack's declared QA rules against a composed prompt (pre-generation)
// and the resolved output (post-generation). Rule IDs are matched against a
// small built-in table of checks; an unrecognized rule id (a future pack's
// custom rule with no checker yet) is skipped rather than erroring — this
// is intentionally not a generic rule-expression engine, just enough to
// execute the Life Design pack's five declared checks.

import Foundation

/// The outcome of one QA rule check.
public struct StudioPackQAResult: Sendable, Equatable, Identifiable {
  public var id: String
  public var description: String
  public var passed: Bool
  public var required: Bool

  /// True only when a required rule failed — the one case that should
  /// actually block generation rather than just warn.
  public var blocks: Bool { required && !passed }
}

public enum StudioPackQALinter {
  /// Pre-generation prompt lint: checks the composed prompt/negative prompt
  /// text against the pack's rules. Rules with no built-in checker for their
  /// id are skipped, not reported as failures.
  public static func lintPrompt(
    rules: [StudioPackQARule], prompt: String, negativePrompt: String?
  ) -> [StudioPackQAResult] {
    let lowerPrompt = prompt.lowercased()
    let lowerNegative = (negativePrompt ?? "").lowercased()

    return rules.compactMap { rule -> StudioPackQAResult? in
      let passed: Bool
      switch rule.id {
      case "faceless":
        passed = lowerPrompt.contains("faceless") || lowerPrompt.contains("no facial features")
          || lowerPrompt.contains("no face")
      case "flat-vector-terms":
        passed = lowerPrompt.contains("flat") || lowerPrompt.contains("vector")
      case "no-photorealism":
        passed = lowerNegative.contains("photorealistic") || lowerNegative.contains("gradient")
      default:
        return nil
      }
      return StudioPackQAResult(id: rule.id, description: rule.description, passed: passed, required: rule.required)
    }
  }

  /// Post-generation metadata/export lint: checks whether the actually
  /// resolved output satisfies rules about SVG export and pack provenance.
  /// A rule that doesn't apply to this pack's configuration (e.g. SVG export
  /// wasn't requested at all) is skipped, not reported as failed.
  public static func lintOutput(
    rules: [StudioPackQARule], packId: String, svgWanted: Bool, svgExported: Bool
  ) -> [StudioPackQAResult] {
    rules.compactMap { rule -> StudioPackQAResult? in
      let passed: Bool
      switch rule.id {
      case "svg-export-requested":
        guard svgWanted else { return nil }
        passed = svgExported
      case "has-pack-metadata":
        passed = !packId.isEmpty
      default:
        return nil
      }
      return StudioPackQAResult(id: rule.id, description: rule.description, passed: passed, required: rule.required)
    }
  }
}
