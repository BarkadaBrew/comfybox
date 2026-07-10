// StudioPack.swift — Domain-specific creative pack schema (Studio Packs, FR-1)
//
// A StudioPack bundles prompt grammar, negative prompt, model/settings
// defaults, a recommended LoRA stack, SVG export defaults, camera/lighting
// defaults, template categories, and QA rules into one reusable production
// recipe. Packs orchestrate existing ComfyBox primitives (presets, LoRA
// library, SVG export, camera directives) rather than duplicating them.
//
// See docs/prd-comfybox-studio-packs.md and GitHub issue #195.

import Foundation

// MARK: - LoRA Reference

/// A LoRA the pack recommends, resolved against the local LoRA library by id.
/// `optional == true` (the default) means a missing LoRA is a warning, not a
/// hard failure — packs must apply cleanly even before a domain-specific LoRA
/// exists locally.
public struct StudioPackLoRARef: Codable, Sendable, Equatable {
  public var loraId: String
  public var scale: Float
  public var optional: Bool

  public init(loraId: String, scale: Float = 1.0, optional: Bool = true) {
    self.loraId = loraId
    self.scale = scale
    self.optional = optional
  }
}

// MARK: - SVG Defaults

public struct StudioPackSVGDefaults: Codable, Sendable, Equatable {
  /// Whether vector-first mode (SVG export alongside PNG) is on by default.
  public var enabled: Bool
  /// One of the CLI's `--svg-preset` values: default, logo, detailed, simplified, bw.
  public var preset: String?

  public init(enabled: Bool = false, preset: String? = nil) {
    self.enabled = enabled
    self.preset = preset
  }
}

// MARK: - QA Rule

/// A declarative quality-check rule. FR-1 only carries this as data; rule
/// *execution* (prompt/metadata lint) is FR-8 (#201).
public struct StudioPackQARule: Codable, Sendable, Equatable, Identifiable {
  public var id: String
  public var description: String
  /// Required rules should block/hard-fail once FR-8 lands; everything else
  /// is a warning-only lint per the PRD.
  public var required: Bool

  public init(id: String, description: String, required: Bool = false) {
    self.id = id
    self.description = description
    self.required = required
  }
}

// MARK: - Template Slot

/// A single fillable slot in a pack template, e.g. `{clinician_role}`.
public struct StudioPackTemplateSlot: Codable, Sendable, Equatable, Identifiable {
  /// Matches the `{id}` placeholder in the owning template's `template` string.
  public var id: String
  public var label: String
  public var placeholder: String
  public var defaultValue: String
  /// Suggested values for a picker; empty means freeform text entry.
  public var options: [String]

  public init(
    id: String, label: String, placeholder: String = "", defaultValue: String = "",
    options: [String] = []
  ) {
    self.id = id
    self.label = label
    self.placeholder = placeholder
    self.defaultValue = defaultValue
    self.options = options
  }
}

// MARK: - Template

/// A slot-based prompt template (FR-2 / #198). `template` contains `{slotId}`
/// placeholders matching entries in `slots`.
public struct StudioPackTemplate: Codable, Sendable, Equatable, Identifiable {
  public var id: String
  public var name: String
  public var category: String
  public var template: String
  public var slots: [StudioPackTemplateSlot]

  enum CodingKeys: String, CodingKey {
    case id, name, category, template, slots
  }

  public init(
    id: String, name: String, category: String, template: String,
    slots: [StudioPackTemplateSlot] = []
  ) {
    self.id = id
    self.name = name
    self.category = category
    self.template = template
    self.slots = slots
  }

  /// Render the template by substituting `{slotId}` with the provided value,
  /// falling back to the slot's default when a value is missing or blank.
  /// Unrecognized placeholders (no matching slot) are left as-is rather than
  /// silently dropped, so a malformed template is visibly wrong, not blank.
  public func render(slotValues: [String: String]) -> String {
    var result = template
    for slot in slots {
      let provided = slotValues[slot.id]?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
      let value = provided.isEmpty ? slot.defaultValue : provided
      result = result.replacingOccurrences(of: "{\(slot.id)}", with: value)
    }
    return result
  }
}

// MARK: - Studio Pack

/// A single Studio Pack: a named, versioned production recipe.
public struct StudioPack: Codable, Sendable, Identifiable, Equatable {
  public var id: String
  public var name: String
  public var description: String
  public var domain: String
  public var version: Int

  // Prompt defaults
  public var promptPrefix: String?
  public var promptSuffix: String?
  public var negativePrompt: String?

  // Model/settings defaults
  public var model: String?
  public var steps: Int?
  public var guidance: Float?
  public var scheduler: String?
  public var width: Int?
  public var height: Int?

  // Recommended LoRA stack
  public var loraStack: [StudioPackLoRARef]

  // SVG/export defaults
  public var svgDefaults: StudioPackSVGDefaults?

  // Camera/lighting defaults — stored as raw identifiers (CameraDirective /
  // LightingDirective live in the Desktop target; ZImage is shared with the
  // server, so this model can't depend on them directly). The Desktop apply
  // path maps these strings onto its own enum cases.
  public var cameraAngle: String?
  public var cameraOrientation: String?
  public var lightingStyle: String?

  // Template categories a pack advertises (used by control template
  // library grouping — FR-5 / #200).
  public var templateCategories: [String]

  // Slot-based prompt templates (FR-2 / #198).
  public var templates: [StudioPackTemplate]

  // QA rules (execution is FR-8 / #201)
  public var qaRules: [StudioPackQARule]

  // Freeform tags for future API/MCP surfacing (FR-9 / #203).
  public var mcpTags: [String]

  enum CodingKeys: String, CodingKey {
    case id, name, description, domain, version
    case promptPrefix = "prompt_prefix"
    case promptSuffix = "prompt_suffix"
    case negativePrompt = "negative_prompt"
    case model, steps, guidance, scheduler, width, height
    case loraStack = "lora_stack"
    case svgDefaults = "svg_defaults"
    case cameraAngle = "camera_angle"
    case cameraOrientation = "camera_orientation"
    case lightingStyle = "lighting_style"
    case templateCategories = "template_categories"
    case templates
    case qaRules = "qa_rules"
    case mcpTags = "mcp_tags"
  }

  public init(
    id: String, name: String, description: String, domain: String, version: Int = 1,
    promptPrefix: String? = nil, promptSuffix: String? = nil, negativePrompt: String? = nil,
    model: String? = nil, steps: Int? = nil, guidance: Float? = nil, scheduler: String? = nil,
    width: Int? = nil, height: Int? = nil,
    loraStack: [StudioPackLoRARef] = [],
    svgDefaults: StudioPackSVGDefaults? = nil,
    cameraAngle: String? = nil, cameraOrientation: String? = nil, lightingStyle: String? = nil,
    templateCategories: [String] = [],
    templates: [StudioPackTemplate] = [],
    qaRules: [StudioPackQARule] = [],
    mcpTags: [String] = []
  ) {
    self.id = id
    self.name = name
    self.description = description
    self.domain = domain
    self.version = version
    self.promptPrefix = promptPrefix
    self.promptSuffix = promptSuffix
    self.negativePrompt = negativePrompt
    self.model = model
    self.steps = steps
    self.guidance = guidance
    self.scheduler = scheduler
    self.width = width
    self.height = height
    self.loraStack = loraStack
    self.svgDefaults = svgDefaults
    self.cameraAngle = cameraAngle
    self.cameraOrientation = cameraOrientation
    self.lightingStyle = lightingStyle
    self.templateCategories = templateCategories
    self.templates = templates
    self.qaRules = qaRules
    self.mcpTags = mcpTags
  }
}

// MARK: - Applying a pack to a prompt

extension StudioPack {
  /// Compose a full prompt from the pack's prefix/suffix and a user's
  /// subject text. Pure function — does not touch any saved preset or
  /// generation state; callers decide what to do with the result.
  public func composePrompt(subject: String) -> String {
    let trimmedSubject = subject.trimmingCharacters(in: .whitespacesAndNewlines)
    var parts: [String] = []
    if let prefix = promptPrefix, !prefix.isEmpty { parts.append(prefix) }
    if !trimmedSubject.isEmpty { parts.append(trimmedSubject) }
    if let suffix = promptSuffix, !suffix.isEmpty { parts.append(suffix) }
    return parts.joined(separator: ", ")
  }
}
