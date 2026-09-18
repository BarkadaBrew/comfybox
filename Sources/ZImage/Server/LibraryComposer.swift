// LibraryComposer.swift — assembling a prompt from library material, with
// placement (PRD-creative-library, release one follow-up).
//
// `LibraryStore.fill` answers "substitute these slot values". That is right
// when the template declares every marker you want to use, and useless when it
// does not: a wardrobe item has nowhere to go in a template with no {OUTFIT},
// and turning a component off leaves its marker sitting in the prompt.
//
// Adapted from ComfyUI-SickOllie's placement state machine (surveyed
// 2026-09-17, `studio_prompt_core.py` — behaviour only, no code): each
// component says HOW it wants to be placed, and the composer reports what it
// actually did, so a picker can show "active / appended / waiting / no marker"
// per slot instead of leaving the user to diff two prompts.
//
//   smart   — substitute into the marker when there is one, else append.
//   token   — substitute ONLY; report `missingMarker` and change nothing.
//   append  — always append, marker or not.
//   prepend — always prepend (what a look's prefix wants).
//   off     — remove the marker and tidy the punctuation it leaves behind.
//
// Pure. The store owns material; this owns assembly.

import Foundation

public enum LibraryPlacement: String, Codable, Sendable, CaseIterable {
  case smart
  case token
  case append
  case prepend
  case off
}

/// One piece of material, and where the caller wants it.
public struct LibraryComponentUse: Sendable, Equatable {
  /// The marker this targets, without braces: `OUTFIT`.
  public var slot: String
  public var value: String
  public var placement: LibraryPlacement
  /// The item this came from, so a render can record what made it.
  public var itemId: String?

  public init(
    slot: String, value: String, placement: LibraryPlacement = .smart, itemId: String? = nil
  ) {
    self.slot = slot
    self.value = value
    self.placement = placement
    self.itemId = itemId
  }
}

/// What the composer did with one component — the thing a UI shows.
public struct LibraryPlacementResult: Codable, Sendable, Equatable {
  public enum Action: String, Codable, Sendable {
    /// Replaced the template's marker.
    case substituted
    case appended
    case prepended
    /// `off`: the marker was removed.
    case removed
    /// `token`: the template has no such marker, so nothing was placed.
    case missingMarker
    /// Nothing to place (blank value).
    case skipped
  }

  public var slot: String
  public var action: Action
  public var itemId: String?
  public var used: Bool { action == .substituted || action == .appended || action == .prepended }

  enum CodingKeys: String, CodingKey {
    case slot, action
    case itemId = "item_id"
  }
}

public struct LibraryComposition: Codable, Sendable, Equatable {
  public var prompt: String
  public var placements: [LibraryPlacementResult]
  /// Markers still in the prompt: unfilled slots, deliberately visible.
  public var unfilled: [String]
  /// Ids of every item that actually landed, for provenance.
  public var usedItemIds: [String]

  enum CodingKeys: String, CodingKey {
    case prompt, placements, unfilled
    case usedItemIds = "used_item_ids"
  }
}

public enum LibraryComposer {

  /// Compose a prompt from a template body and a set of components.
  public static func compose(
    template body: String, components: [LibraryComponentUse]
  ) -> LibraryComposition {
    var prompt = body
    var placements: [LibraryPlacementResult] = []
    var used: [String] = []

    for component in components {
      let value = component.value.trimmingCharacters(in: .whitespacesAndNewlines)
      let marker = "{\(component.slot)}"
      let hasMarker = prompt.contains(marker)

      if component.placement == .off {
        if hasMarker {
          prompt = removeMarker(marker, from: prompt)
          placements.append(.init(slot: component.slot, action: .removed, itemId: component.itemId))
        } else {
          placements.append(.init(slot: component.slot, action: .skipped, itemId: component.itemId))
        }
        continue
      }

      guard !value.isEmpty else {
        placements.append(.init(slot: component.slot, action: .skipped, itemId: component.itemId))
        continue
      }

      switch component.placement {
      case .token where !hasMarker:
        placements.append(
          .init(slot: component.slot, action: .missingMarker, itemId: component.itemId))
      case .token, .smart where hasMarker:
        prompt = prompt.replacingOccurrences(of: marker, with: value)
        placements.append(
          .init(slot: component.slot, action: .substituted, itemId: component.itemId))
        component.itemId.map { used.append($0) }
      case .prepend:
        prompt = join(value, prompt)
        placements.append(.init(slot: component.slot, action: .prepended, itemId: component.itemId))
        component.itemId.map { used.append($0) }
      default:
        // `append`, and `smart` with no marker.
        prompt = join(prompt, value)
        placements.append(.init(slot: component.slot, action: .appended, itemId: component.itemId))
        component.itemId.map { used.append($0) }
      }
    }

    return LibraryComposition(
      prompt: tidy(prompt),
      placements: placements,
      unfilled: LibraryStore.slotMarkers(in: prompt).sorted(),
      usedItemIds: used)
  }

  /// Convenience: a template item plus components, honouring slot defaults for
  /// anything the caller did not supply.
  public static func compose(
    template: LibraryEntry, components: [LibraryComponentUse]
  ) -> LibraryComposition {
    let supplied = Set(components.map(\.slot))
    let defaults = (template.slots ?? []).compactMap { slot -> LibraryComponentUse? in
      guard !supplied.contains(slot.id), let value = slot.defaultValue, !value.isEmpty else {
        return nil
      }
      return LibraryComponentUse(slot: slot.id, value: value, placement: .token)
    }
    return compose(template: template.value, components: components + defaults)
  }

  // MARK: - Text tidying

  /// Turning a component off removes its WHOLE CLAUSE, not just the marker.
  ///
  /// "a woman wearing {OUTFIT}, in {SCENE}, shot on {CAMERA}" with SCENE off
  /// must read "a woman wearing {OUTFIT}, shot on {CAMERA}" — deleting only the
  /// marker leaves "in", a dangling preposition the prompt then has to carry.
  /// A marker that is the entire prompt falls back to plain removal.
  static func removeMarker(_ marker: String, from text: String) -> String {
    let clauses = text.components(separatedBy: ",")
    if clauses.count > 1 {
      let kept = clauses.filter { !$0.contains(marker) }
      if !kept.isEmpty, kept.count < clauses.count {
        return tidy(kept.joined(separator: ","))
      }
    }
    return tidy(text.replacingOccurrences(of: marker, with: ""))
  }

  /// Collapse the punctuation damage that removal and appending cause.
  static func tidy(_ text: String) -> String {
    var out = text
    let fixes: [(String, String)] = [
      (" ,", ","), (",,", ","), (", ,", ","), ("  ", " "), (" .", "."), ("..", "."),
      (",.", "."), ("( ", "("), (" )", ")"),
    ]
    var passes = 0
    var changed = true
    while changed, passes < 8 {
      changed = false
      for (bad, good) in fixes where out.contains(bad) {
        out = out.replacingOccurrences(of: bad, with: good)
        changed = true
      }
      passes += 1
    }
    return out.trimmingCharacters(in: .whitespacesAndNewlines)
      .trimmingCharacters(in: CharacterSet(charactersIn: ","))
      .trimmingCharacters(in: .whitespaces)
  }

  /// Join two prompt fragments with one separator, never two.
  static func join(_ lhs: String, _ rhs: String) -> String {
    let left = lhs.trimmingCharacters(in: .whitespacesAndNewlines)
    let right = rhs.trimmingCharacters(in: .whitespacesAndNewlines)
    if left.isEmpty { return right }
    if right.isEmpty { return left }
    if left.hasSuffix(",") || left.hasSuffix(".") { return left + " " + right }
    return left + ", " + right
  }
}
