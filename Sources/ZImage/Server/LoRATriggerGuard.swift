// LoRATriggerGuard.swift — deterministic trigger-token guarantee.
//
// Todd 2026-08-11: "are you sure it won't get written out?" — it could be.
// Trigger tokens are injected daemon-side BEFORE the sealed prompt rewrite
// (Dan's-PE), and an LLM rewriter can drop unfamiliar tokens (observed with
// the identity weaver). This guard runs AFTER everything, engine-side, where
// the final prompt meets the applied LoRA list: any applied LoRA's known
// trigger that is absent gets prefixed back. Nothing downstream can remove it.

import Foundation

public enum LoRATriggerGuard {
  /// Prefix any missing trigger (first trigger per LoRA) onto the prompt.
  /// Case-insensitive containment check; preserves the given trigger order;
  /// returns the prompt unchanged when everything is already present.
  public static func ensure(prompt: String, triggers: [String]) -> String {
    let lowered = prompt.lowercased()
    let missing = triggers.filter { !$0.isEmpty && !lowered.contains($0.lowercased()) }
    guard !missing.isEmpty else { return prompt }
    return missing.joined(separator: ", ") + ", " + prompt
  }
}
