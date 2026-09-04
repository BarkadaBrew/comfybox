// ControlDescriptor.swift — Phase 4 discovery types (FDD-ui-api-parity §3.4, D4;
// comfybox#300).
//
// A ``ControlDescriptor`` is one row of the compile-time ``ControlRegistry``: a
// stable dotted id, human strings, a machine-executable read/write ``ActionRef``,
// and the MCP tool that fronts it. The one rule that stops the registry becoming
// a third truth (§3.4): it declares where a value LIVES and never caches a copy —
// values are resolved per-request by dereferencing `read.pointer` against
// `ServerConfigStore` / `ContentModeStore` / the live control state.

import Foundation

// MARK: - JSONValue

/// Minimal Codable JSON value — carries descriptor `defaultValue`s and resolved
/// control values without inventing a per-control schema. Mirrors the JSON data
/// model exactly (null/bool/int/double/string/array/object).
public enum JSONValue: Codable, Equatable, Sendable {
  case null
  case bool(Bool)
  case int(Int)
  case double(Double)
  case string(String)
  case array([JSONValue])
  case object([String: JSONValue])

  /// Convert a `JSONSerialization` tree (`Any`) into a ``JSONValue``.
  /// Returns nil for values outside the JSON data model.
  public init?(any value: Any) {
    switch value {
    case is NSNull:
      self = .null
    case let number as NSNumber:
      // NSNumber carries bools, ints and doubles; disambiguate bools by objCType.
      if CFGetTypeID(number) == CFBooleanGetTypeID() {
        self = .bool(number.boolValue)
      } else if let int = value as? Int, number.doubleValue == Double(int),
                floor(number.doubleValue) == number.doubleValue,
                !String(cString: number.objCType).contains("d"),
                !String(cString: number.objCType).contains("f") {
        self = .int(int)
      } else {
        self = .double(number.doubleValue)
      }
    case let bool as Bool:
      self = .bool(bool)
    case let int as Int:
      self = .int(int)
    case let double as Double:
      self = .double(double)
    case let string as String:
      self = .string(string)
    case let array as [Any]:
      var items: [JSONValue] = []
      for element in array {
        guard let converted = JSONValue(any: element) else { return nil }
        items.append(converted)
      }
      self = .array(items)
    case let dict as [String: Any]:
      var object: [String: JSONValue] = [:]
      for (key, element) in dict {
        guard let converted = JSONValue(any: element) else { return nil }
        object[key] = converted
      }
      self = .object(object)
    default:
      return nil
    }
  }

  public init(from decoder: Decoder) throws {
    let container = try decoder.singleValueContainer()
    if container.decodeNil() {
      self = .null
    } else if let bool = try? container.decode(Bool.self) {
      self = .bool(bool)
    } else if let int = try? container.decode(Int.self) {
      self = .int(int)
    } else if let double = try? container.decode(Double.self) {
      self = .double(double)
    } else if let string = try? container.decode(String.self) {
      self = .string(string)
    } else if let array = try? container.decode([JSONValue].self) {
      self = .array(array)
    } else if let object = try? container.decode([String: JSONValue].self) {
      self = .object(object)
    } else {
      throw DecodingError.typeMismatch(
        JSONValue.self,
        DecodingError.Context(codingPath: decoder.codingPath, debugDescription: "Unsupported JSON value"))
    }
  }

  public func encode(to encoder: Encoder) throws {
    var container = encoder.singleValueContainer()
    switch self {
    case .null: try container.encodeNil()
    case .bool(let bool): try container.encode(bool)
    case .int(let int): try container.encode(int)
    case .double(let double): try container.encode(double)
    case .string(let string): try container.encode(string)
    case .array(let array): try container.encode(array)
    case .object(let object): try container.encode(object)
    }
  }
}

// MARK: - JSON Pointer (RFC 6901)

/// Minimal RFC 6901 dereference over `JSONSerialization` trees — the resolution
/// primitive behind `GET /v1/controls` values (§3.4: per-request, never cached).
public enum JSONPointer {
  /// Dereference `pointer` (e.g. `/renderDefaults/byFamily/krea2/steps`) in
  /// `object`. Returns nil when any segment is absent — for an optional config
  /// key that means "no value set", not an error. `""` returns the whole doc.
  public static func dereference(_ pointer: String, in object: Any) -> Any? {
    guard !pointer.isEmpty else { return object }
    guard pointer.hasPrefix("/") else { return nil }
    var current: Any = object
    for rawSegment in pointer.dropFirst().components(separatedBy: "/") {
      let segment = rawSegment
        .replacingOccurrences(of: "~1", with: "/")
        .replacingOccurrences(of: "~0", with: "~")
      if let dict = current as? [String: Any] {
        guard let next = dict[segment] else { return nil }
        current = next
      } else if let array = current as? [Any] {
        guard let index = Int(segment), array.indices.contains(index) else { return nil }
        current = array[index]
      } else {
        return nil
      }
    }
    return current
  }
}

// MARK: - Descriptor enums

/// What kind of surface a control belongs to (FDD §3.4).
public enum ControlScope: String, Codable, Sendable {
  case engine
  case queue
  case creative
  case provider
  case model
  case kira
}

/// The control's value type. `.action` controls have no value — POSTing the
/// write route IS the control (e.g. `queue.pause`).
public enum ControlType: String, Codable, Sendable {
  case int
  case double
  case bool
  case string
  case `enum`
  case object
  case action
}

/// Which host serves the control (federated, FDD §3.2). Every descriptor in
/// this repo's registry is `.comfybox`; `.kiraDaemon` descriptors arrive with
/// Phase 2's `docs/kira-control-api.md` federation and are covered by a
/// coffeeshop-server contract test, not this repo's parity test (§3.5 step 4).
public enum ControlHost: String, Codable, Sendable {
  case comfybox
  case kiraDaemon
}

// MARK: - ActionRef

/// A machine-executable reference to the HTTP action that reads or writes a
/// control: `{host, method, path, pointer}` (§3.4). For config-backed controls
/// the write is `{PATCH, /v1/config, pointer: /renderDefaults/...}` — one route,
/// no URL-per-control drift engine (§3.3).
public struct ActionRef: Codable, Sendable, Equatable {
  public let host: ControlHost
  public let method: String
  public let path: String
  /// RFC 6901 JSON Pointer into the read/write document; nil when the action's
  /// body/response is not document-shaped (e.g. queue actions).
  public let pointer: String?

  public init(host: ControlHost = .comfybox, method: String, path: String, pointer: String? = nil) {
    self.host = host
    self.method = method
    self.path = path
    self.pointer = pointer
  }
}

// MARK: - ControlDescriptor

/// One discoverable control (FDD §3.4). Encodes with `range` as `{min, max}`
/// for a stable, self-describing wire shape.
public struct ControlDescriptor: Codable, Sendable {
  /// Stable dotted handle, e.g. `render.defaults.krea2.steps`.
  public let id: String
  public let title: String
  public let summary: String
  public let scope: ControlScope
  public let type: ControlType
  public let range: ClosedRange<Double>?
  /// Allowed values for `.enum` controls.
  public let allowed: [String]?
  public let unit: String?
  /// The engine/built-in default the control falls back to when unset — a
  /// DECLARED constant, not a resolved value (values are resolved per-request).
  public let defaultValue: JSONValue?
  public let read: ActionRef?
  public let write: ActionRef?
  /// MCP tool that fronts the write (must resolve in `MCPToolRegistry` — §3.5
  /// step 4). Nil for controls with no agent tool (e.g. content-mode edits,
  /// which are exempted with a reason in `ParityExemptions`).
  public let mcpTool: String?
  public let host: ControlHost
  public let mutatesEngine: Bool
  public let requiresRestart: Bool
  /// The headless-parity phase that made this control API-reachable
  /// (`phase0` control plane, `phase3` config/content modes).
  public let since: String

  public init(
    id: String,
    title: String,
    summary: String,
    scope: ControlScope,
    type: ControlType,
    range: ClosedRange<Double>? = nil,
    allowed: [String]? = nil,
    unit: String? = nil,
    defaultValue: JSONValue? = nil,
    read: ActionRef? = nil,
    write: ActionRef? = nil,
    mcpTool: String? = nil,
    host: ControlHost = .comfybox,
    mutatesEngine: Bool = false,
    requiresRestart: Bool = false,
    since: String
  ) {
    self.id = id
    self.title = title
    self.summary = summary
    self.scope = scope
    self.type = type
    self.range = range
    self.allowed = allowed
    self.unit = unit
    self.defaultValue = defaultValue
    self.read = read
    self.write = write
    self.mcpTool = mcpTool
    self.host = host
    self.mutatesEngine = mutatesEngine
    self.requiresRestart = requiresRestart
    self.since = since
  }

  private enum CodingKeys: String, CodingKey {
    case id, title, summary, scope, type, range, allowed, unit, defaultValue
    case read, write, mcpTool, host, mutatesEngine, requiresRestart, since
  }

  private struct RangeDTO: Codable {
    let min: Double
    let max: Double
  }

  public init(from decoder: Decoder) throws {
    let c = try decoder.container(keyedBy: CodingKeys.self)
    id = try c.decode(String.self, forKey: .id)
    title = try c.decode(String.self, forKey: .title)
    summary = try c.decode(String.self, forKey: .summary)
    scope = try c.decode(ControlScope.self, forKey: .scope)
    type = try c.decode(ControlType.self, forKey: .type)
    if let dto = try c.decodeIfPresent(RangeDTO.self, forKey: .range) {
      range = dto.min...dto.max
    } else {
      range = nil
    }
    allowed = try c.decodeIfPresent([String].self, forKey: .allowed)
    unit = try c.decodeIfPresent(String.self, forKey: .unit)
    defaultValue = try c.decodeIfPresent(JSONValue.self, forKey: .defaultValue)
    read = try c.decodeIfPresent(ActionRef.self, forKey: .read)
    write = try c.decodeIfPresent(ActionRef.self, forKey: .write)
    mcpTool = try c.decodeIfPresent(String.self, forKey: .mcpTool)
    host = try c.decode(ControlHost.self, forKey: .host)
    mutatesEngine = try c.decode(Bool.self, forKey: .mutatesEngine)
    requiresRestart = try c.decode(Bool.self, forKey: .requiresRestart)
    since = try c.decode(String.self, forKey: .since)
  }

  public func encode(to encoder: Encoder) throws {
    var c = encoder.container(keyedBy: CodingKeys.self)
    try c.encode(id, forKey: .id)
    try c.encode(title, forKey: .title)
    try c.encode(summary, forKey: .summary)
    try c.encode(scope, forKey: .scope)
    try c.encode(type, forKey: .type)
    if let range {
      try c.encode(RangeDTO(min: range.lowerBound, max: range.upperBound), forKey: .range)
    }
    try c.encodeIfPresent(allowed, forKey: .allowed)
    try c.encodeIfPresent(unit, forKey: .unit)
    try c.encodeIfPresent(defaultValue, forKey: .defaultValue)
    try c.encodeIfPresent(read, forKey: .read)
    try c.encodeIfPresent(write, forKey: .write)
    try c.encodeIfPresent(mcpTool, forKey: .mcpTool)
    try c.encode(host, forKey: .host)
    try c.encode(mutatesEngine, forKey: .mutatesEngine)
    try c.encode(requiresRestart, forKey: .requiresRestart)
    try c.encode(since, forKey: .since)
  }
}
