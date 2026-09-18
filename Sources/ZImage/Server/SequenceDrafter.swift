// SequenceDrafter.swift — a brief plus a preset becomes a timeline
// (FDD-ltx-director-tab §4.9.4, WP15).
//
// The division of labour is the whole design: THE ENGINE OWNS EVERY NUMBER,
// the model owns only prose. Frame math, snapping, ids, seeds, segment
// boundaries and keyframe slots come from the preset (§4.9.3); the model
// writes the global prompt and one line per segment, and nothing else. A
// drafter that could choose frame counts would produce timelines the compiler
// then refuses, and the failure would look like the model's fault.
//
// Flow: build the skeleton from the preset -> ask the model for prose ->
// assemble -> validate -> on failure, ONE repair attempt carrying the
// validator's own issues -> a second failure returns the draft WITH its issues
// rather than a silent fallback.

import Foundation

public struct SequenceDraftRequest: Sendable {
  public var presetId: String
  public var brief: String
  /// Optional keyframe images the caller already has, in frame order.
  public var keyframePaths: [String]
  /// An imported or driving voice track (§4.7).
  public var audioPath: String?

  public init(
    presetId: String, brief: String, keyframePaths: [String] = [], audioPath: String? = nil
  ) {
    self.presetId = presetId
    self.brief = brief
    self.keyframePaths = keyframePaths
    self.audioPath = audioPath
  }
}

/// What the model is allowed to return: prose, nothing else.
public struct SequenceDraftProse: Codable, Sendable, Equatable {
  public var globalPrompt: String
  public var segments: [String]
  /// One line per keyframe slot the caller did not fill, describing the still.
  public var keyframeDescriptions: [String]?

  public init(globalPrompt: String, segments: [String], keyframeDescriptions: [String]? = nil) {
    self.globalPrompt = globalPrompt
    self.segments = segments
    self.keyframeDescriptions = keyframeDescriptions
  }

  enum CodingKeys: String, CodingKey {
    case globalPrompt = "global_prompt"
    case segments
    case keyframeDescriptions = "keyframe_descriptions"
  }
}

public struct SequenceDraft: Codable, Sendable, Equatable {
  public var timeline: DirectorTimeline
  public var presetId: String
  public var templateVersion: String
  public var model: String?
  /// Validator issues that survived the repair attempt. Empty means clean.
  public var issues: [DirectorIssue]
  /// Keyframe slots with no image yet, and what the drafter says should be there.
  public var keyframeDescriptions: [String]
  public var estimatedGpuMinutes: Int

  enum CodingKeys: String, CodingKey {
    case timeline, issues, model
    case presetId = "preset_id"
    case templateVersion = "template_version"
    case keyframeDescriptions = "keyframe_descriptions"
    case estimatedGpuMinutes = "estimated_gpu_minutes"
  }
}

public enum SequenceDraftError: Error, LocalizedError, Equatable {
  case unknownPreset(String)
  case noAuthor
  case authorFailed(String)
  case emptyBrief

  public var errorDescription: String? {
    switch self {
    case .unknownPreset(let id): return "sequence preset '\(id)' not found"
    case .noAuthor:
      return "no assistant provider configured — set providers.assistant in ~/.comfybox/config.json"
    case .authorFailed(let why): return "the drafter did not return a usable timeline: \(why)"
    case .emptyBrief: return "'brief' is required"
    }
  }
}

public enum SequenceDrafter {

  /// Versioned so a Sequence records which authoring rules produced it, and so
  /// the `comfybox-director` skill can be generated from the same text.
  public static let templateVersion = "director-author/1"

  // MARK: - The authoring template

  /// The rules. Deliberately about CRAFT, not about JSON shape — the shape is
  /// enforced by decoding, and telling a model twice makes it obey neither.
  public static func systemPrompt(preset: SequencePreset, segmentCount: Int, keyframeSlots: Int)
    -> String
  {
    var lines: [String] = [
      "You write prompts for LTX-2 video. Return JSON only.",
      "",
      "SHAPE (fixed — do not negotiate):",
      "- global_prompt: one paragraph describing the subject, setting, camera and light.",
      "- segments: EXACTLY \(segmentCount) line(s), in order, one per timed beat.",
    ]
    if keyframeSlots > 0 {
      lines.append(
        "- keyframe_descriptions: EXACTLY \(keyframeSlots) line(s) describing the still image at each keyframe.")
    }
    lines.append(contentsOf: [
      "",
      "CRAFT:",
      "- Present tense, describing what the camera sees.",
      "- The global prompt sets the subject and the room; a segment names only what CHANGES.",
      "- One action per segment. An action needs time: a gesture reads, a costume change does not.",
      "- Name sound inside the action (\"the mug touches the counter with a clear clink\"), not as a list.",
      "- Quote spoken words: she says, \"…\". Only quoted words are spoken by the model.",
      "- No camera moves unless the brief asks; a locked camera is steadier.",
      "- No meta language: no \"prompt\", \"scene number\", \"AI\", \"render\", \"4k\", \"masterpiece\".",
      "- No numbers for timing. The engine owns every frame count.",
    ])
    if let notes = preset.authorNotes, !notes.isEmpty {
      lines.append(contentsOf: ["", "HOUSE STYLE: \(notes)"])
    }
    if preset.audio == "driven" {
      lines.append(
        "AUDIO-DRIVEN: the voice already exists. Describe her speaking naturally; do not invent new dialogue.")
    }
    lines.append(contentsOf: [
      "",
      "Return: {\"global_prompt\": \"…\", \"segments\": [\"…\"]"
        + (keyframeSlots > 0 ? ", \"keyframe_descriptions\": [\"…\"]}" : "}"),
    ])
    return lines.joined(separator: "\n")
  }

  public static func userPrompt(brief: String, preset: SequencePreset, segmentSeconds: [Double])
    -> String
  {
    var text = "BRIEF: \(brief.trimmingCharacters(in: .whitespacesAndNewlines))\n\n"
    text += "The clip is \(String(format: "%.1f", preset.snappedSeconds))s"
    if segmentSeconds.count > 1 {
      let spans = segmentSeconds.enumerated()
        .map { "\($0.offset + 1): \(String(format: "%.1f", $0.element))s" }
        .joined(separator: ", ")
      text += ", in \(segmentSeconds.count) beats (\(spans))."
    } else {
      text += ", one continuous beat."
    }
    return text
  }

  // MARK: - Assembly

  /// Build the timeline from the preset and the model's prose. Pure: every
  /// number here comes from the preset, never from the model.
  public static func assemble(
    prose: SequenceDraftProse,
    preset: SequencePreset,
    request: SequenceDraftRequest,
    idPrefix: String = "d"
  ) -> DirectorTimeline {
    let spans = preset.segmentSpans()
    let keyframeFrames = preset.keyframeFrames()

    var segments: [DirectorTimeline.PromptSegment] = []
    for (index, span) in spans.enumerated() {
      let text = index < prose.segments.count
        ? prose.segments[index]
        : (prose.segments.last ?? prose.globalPrompt)
      segments.append(.init(
        id: "\(idPrefix)s\(index + 1)", startFrame: span.start, lengthFrames: span.length,
        prompt: text.trimmingCharacters(in: .whitespacesAndNewlines)))
    }

    var keyframes: [DirectorTimeline.Keyframe] = []
    for (index, frame) in keyframeFrames.enumerated() {
      guard index < request.keyframePaths.count else { continue }
      keyframes.append(.init(
        id: "\(idPrefix)k\(index + 1)", imagePath: request.keyframePaths[index], frame: frame,
        strength: 1.0))
    }

    var audioClips: [DirectorTimeline.AudioClip] = []
    if let audioPath = request.audioPath, !audioPath.isEmpty {
      audioClips.append(.init(
        id: "\(idPrefix)a1", audioPath: audioPath, startFrame: 0,
        lengthFrames: preset.snappedFrames, trimStartFrames: 0, gain: 1.0))
    }

    var settings = DirectorTimeline.Settings(
      width: preset.width, height: preset.height, lengthFrames: preset.snappedFrames)
    settings.fps = preset.fps
    settings.seed = preset.seed.map { UInt64(max(0, $0)) }
    settings.steps = preset.steps
    settings.negativePrompt = preset.negativePrompt
    settings.character = preset.character
    settings.preset = preset.videoPresetId

    return DirectorTimeline(
      settings: settings,
      globalPrompt: prose.globalPrompt.trimmingCharacters(in: .whitespacesAndNewlines),
      keyframes: keyframes,
      promptSegments: segments,
      audioClips: audioClips,
      audio: .init(mode: audioClips.isEmpty ? .generated : .imported))
  }

  // MARK: - Asking the model

  /// One OpenAI-style chat call that must return JSON. Kept here rather than
  /// reusing `PromptOptimizer` because that one is shaped for prompt rewriting
  /// (character injection, content modes); this needs a plain structured answer.
  public static func ask(
    endpoint: AIProviderEndpoint, system: String, user: String, timeout: TimeInterval = 120,
    session: URLSession = .shared
  ) async throws -> (prose: SequenceDraftProse, model: String) {
    var base = endpoint.baseUrl
    while base.hasSuffix("/") { base.removeLast() }
    guard let url = URL(string: base + "/chat/completions") else {
      throw SequenceDraftError.authorFailed("bad assistant baseUrl: \(endpoint.baseUrl)")
    }
    var request = URLRequest(url: url)
    request.httpMethod = "POST"
    request.timeoutInterval = timeout
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    if let key = endpoint.apiKey, !key.isEmpty {
      request.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization")
    }
    let body: [String: Any] = [
      "model": endpoint.model,
      "messages": [
        ["role": "system", "content": system],
        ["role": "user", "content": user],
      ],
      "temperature": 0.8,
      // A reasoning model spends its budget thinking; 2000 leaves room for the
      // answer (the lesson VisionChat records at 300/320).
      "max_tokens": 2000,
      "response_format": ["type": "json_object"],
    ]
    request.httpBody = try JSONSerialization.data(withJSONObject: body)

    let (data, response) = try await session.data(for: request)
    let status = (response as? HTTPURLResponse)?.statusCode ?? 0
    guard status == 200 else {
      throw SequenceDraftError.authorFailed("assistant returned HTTP \(status)")
    }
    guard let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
          let choices = object["choices"] as? [[String: Any]],
          let message = choices.first?["message"] as? [String: Any],
          let content = message["content"] as? String,
          !content.trimmingCharacters(in: .whitespaces).isEmpty
    else {
      throw SequenceDraftError.authorFailed("assistant returned no content")
    }
    return (try parse(content), endpoint.model)
  }

  /// Models wrap JSON in prose or fences more often than they do not.
  public static func parse(_ content: String) throws -> SequenceDraftProse {
    let fenced = content
      .replacingOccurrences(of: "```json", with: "```")
      .components(separatedBy: "```")
    let candidates = (fenced.count > 1 ? fenced : [content])
      .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
    for candidate in candidates {
      guard let start = candidate.firstIndex(of: "{"), let end = candidate.lastIndex(of: "}"),
            start < end else { continue }
      let slice = String(candidate[start...end])
      if let prose = try? JSONDecoder().decode(SequenceDraftProse.self, from: Data(slice.utf8)),
         !prose.globalPrompt.trimmingCharacters(in: .whitespaces).isEmpty,
         !prose.segments.isEmpty {
        return prose
      }
    }
    throw SequenceDraftError.authorFailed("could not read a timeline out of the answer")
  }

  /// A repair prompt carries the validator's own words — the model cannot
  /// guess what "keyframe_off_grid" means, but it can rewrite a prompt.
  public static func repairPrompt(issues: [DirectorIssue]) -> String {
    let lines = issues.map { "- \($0.code): \($0.message)" }.joined(separator: "\n")
    return """
      The draft was refused:
      \(lines)

      Rewrite ONLY the prose. Keep the same number of segments in the same order.
      """
  }
}
