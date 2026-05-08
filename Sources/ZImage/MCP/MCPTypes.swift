// MCPTypes.swift — JSON-RPC 2.0 protocol types for MCP server
//
// Implements the Model Context Protocol (MCP) 2024-11-05 wire format.
// All types are Codable for direct JSON serialization over stdio.

import Foundation

// MARK: - JSON-RPC 2.0 Request

/// Incoming JSON-RPC 2.0 message (request or notification).
/// Notifications have no `id` field.
public struct MCPRequest: Codable, Sendable {
  public let jsonrpc: String
  public let id: MCPRequestId?
  public let method: String
  public let params: MCPParams?

  public init(jsonrpc: String = "2.0", id: MCPRequestId? = nil, method: String, params: MCPParams? = nil) {
    self.jsonrpc = jsonrpc
    self.id = id
    self.method = method
    self.params = params
  }
}

/// Request ID — can be a string or integer per JSON-RPC 2.0 spec.
public enum MCPRequestId: Codable, Sendable, Equatable {
  case string(String)
  case integer(Int)

  public init(from decoder: Decoder) throws {
    let container = try decoder.singleValueContainer()
    if let intVal = try? container.decode(Int.self) {
      self = .integer(intVal)
      return
    }
    if let strVal = try? container.decode(String.self) {
      self = .string(strVal)
      return
    }
    throw DecodingError.typeMismatch(
      MCPRequestId.self,
      DecodingError.Context(codingPath: decoder.codingPath, debugDescription: "Expected String or Int for JSON-RPC id")
    )
  }

  public func encode(to encoder: Encoder) throws {
    var container = encoder.singleValueContainer()
    switch self {
    case .string(let s): try container.encode(s)
    case .integer(let i): try container.encode(i)
    }
  }
}

/// Params object — decoded lazily as raw JSON to avoid schema coupling.
public struct MCPParams: Codable, Sendable {
  public let raw: [String: AnyCodable]

  public init(_ dict: [String: AnyCodable] = [:]) {
    self.raw = dict
  }

  public init(from decoder: Decoder) throws {
    let container = try decoder.singleValueContainer()
    self.raw = try container.decode([String: AnyCodable].self)
  }

  public func encode(to encoder: Encoder) throws {
    var container = encoder.singleValueContainer()
    try container.encode(raw)
  }

  /// Get a string value from params.
  public func string(_ key: String) -> String? {
    raw[key]?.stringValue
  }

  /// Get an integer value from params.
  public func integer(_ key: String) -> Int? {
    raw[key]?.intValue
  }

  /// Get a boolean value from params.
  public func bool(_ key: String) -> Bool? {
    raw[key]?.boolValue
  }

  /// Get a double value from params.
  public func number(_ key: String) -> Double? {
    raw[key]?.doubleValue
  }

  /// Get an array value from params.
  public func array(_ key: String) -> [AnyCodable]? {
    raw[key]?.arrayValue
  }

  /// Get a nested dictionary from params.
  public func dict(_ key: String) -> [String: AnyCodable]? {
    raw[key]?.dictValue
  }
}

// MARK: - JSON-RPC 2.0 Response

/// Outgoing JSON-RPC 2.0 response.
public struct MCPResponse: Codable, Sendable {
  public let jsonrpc: String
  public let id: MCPRequestId?
  public let result: AnyCodable?
  public let error: MCPError?

  public init(id: MCPRequestId?, result: AnyCodable) {
    self.jsonrpc = "2.0"
    self.id = id
    self.result = result
    self.error = nil
  }

  public init(id: MCPRequestId?, error: MCPError) {
    self.jsonrpc = "2.0"
    self.id = id
    self.result = nil
    self.error = error
  }
}

// MARK: - JSON-RPC Error

/// JSON-RPC 2.0 error object.
public struct MCPError: Codable, Sendable {
  public let code: Int
  public let message: String
  public let data: AnyCodable?

  public init(code: Int, message: String, data: AnyCodable? = nil) {
    self.code = code
    self.message = message
    self.data = data
  }

  // Standard JSON-RPC error codes
  public static let parseError = MCPError(code: -32700, message: "Parse error")
  public static let invalidRequest = MCPError(code: -32600, message: "Invalid Request")
  public static let methodNotFound = MCPError(code: -32601, message: "Method not found")
  public static let invalidParams = MCPError(code: -32602, message: "Invalid params")
  public static let internalError = MCPError(code: -32603, message: "Internal error")

  /// Application-level error for tool execution failures.
  public static func toolError(_ message: String) -> MCPError {
    MCPError(code: -32000, message: message)
  }
}

// MARK: - MCP Tool Definition

/// Tool definition for the MCP tools/list response.
public struct MCPToolDefinition: Sendable {
  public let name: String
  public let description: String
  public let inputSchema: [String: Any]

  public init(name: String, description: String, inputSchema: [String: Any]) {
    self.name = name
    self.description = description
    self.inputSchema = inputSchema
  }

  /// Serialize inputSchema to JSON-compatible dictionary (for tools/list response).
  public func inputSchemaJSON() -> Any {
    return inputSchema
  }
}

// MARK: - MCP Content Block

/// A content block in an MCP tool result.
public struct MCPContentBlock: Sendable {
  public let type: String
  public let text: String?

  public init(text: String) {
    self.type = "text"
    self.text = text
  }
}

// MARK: - MCP Tool Result

/// Result of a tools/call invocation.
public struct MCPToolResult: Sendable {
  public let content: [MCPContentBlock]
  public let isError: Bool

  public init(text: String) {
    self.content = [MCPContentBlock(text: text)]
    self.isError = false
  }

  public init(error: String) {
    self.content = [MCPContentBlock(text: error)]
    self.isError = true
  }

  /// Encode as a JSON-serializable dictionary for the MCP response.
  func toResponseDict() -> [String: Any] {
    var result: [String: Any] = [
      "content": content.map { block -> [String: Any] in
        var dict: [String: Any] = ["type": block.type]
        if let text = block.text { dict["text"] = text }
        return dict
      }
    ]
    if isError {
      result["isError"] = true
    }
    return result
  }
}

// MARK: - AnyCodable (type-erased JSON wrapper)

/// Type-erased Codable wrapper for arbitrary JSON values.
/// Supports null, bool, int, double, string, array, and dictionary.
public struct AnyCodable: Codable, Sendable {
  public let value: Any

  public init(_ value: Any) {
    self.value = AnyCodable.sanitize(value)
  }

  // MARK: Accessors

  public var stringValue: String? { value as? String }
  public var intValue: Int? { value as? Int }
  public var doubleValue: Double? {
    if let d = value as? Double { return d }
    if let i = value as? Int { return Double(i) }
    return nil
  }
  public var boolValue: Bool? { value as? Bool }
  public var arrayValue: [AnyCodable]? { value as? [AnyCodable] }
  public var dictValue: [String: AnyCodable]? { value as? [String: AnyCodable] }
  public var isNil: Bool { value is NSNull }

  /// Recursively unwrap AnyCodable wrappers back to plain [String: Any] / [Any].
  /// Used when embedding an AnyCodable tree inside another [String: Any] dict.
  public var rawValue: Any {
    switch value {
    case let dict as [String: AnyCodable]:
      return dict.mapValues { $0.rawValue }
    case let arr as [AnyCodable]:
      return arr.map { $0.rawValue }
    default:
      return value
    }
  }

  // MARK: Codable

  public init(from decoder: Decoder) throws {
    let container = try decoder.singleValueContainer()
    if container.decodeNil() {
      value = NSNull()
    } else if let b = try? container.decode(Bool.self) {
      value = b
    } else if let i = try? container.decode(Int.self) {
      value = i
    } else if let d = try? container.decode(Double.self) {
      value = d
    } else if let s = try? container.decode(String.self) {
      value = s
    } else if let arr = try? container.decode([AnyCodable].self) {
      value = arr
    } else if let dict = try? container.decode([String: AnyCodable].self) {
      value = dict
    } else {
      throw DecodingError.typeMismatch(
        AnyCodable.self,
        DecodingError.Context(codingPath: decoder.codingPath, debugDescription: "Unsupported JSON value type")
      )
    }
  }

  public func encode(to encoder: Encoder) throws {
    var container = encoder.singleValueContainer()
    switch value {
    case is NSNull:
      try container.encodeNil()
    case let b as Bool:
      try container.encode(b)
    case let i as Int:
      try container.encode(i)
    case let d as Double:
      try container.encode(d)
    case let s as String:
      try container.encode(s)
    case let arr as [AnyCodable]:
      try container.encode(arr)
    case let dict as [String: AnyCodable]:
      try container.encode(dict)
    default:
      if let dict = value as? [String: Any] {
        try container.encode(dict.mapValues { AnyCodable($0) })
      } else if let arr = value as? [Any] {
        try container.encode(arr.map { AnyCodable($0) })
      } else {
        try container.encode(String(describing: value))
      }
    }
  }

  /// Sanitize a value from [String: Any] into properly nested AnyCodable types.
  private static func sanitize(_ value: Any) -> Any {
    switch value {
    case let dict as [String: Any]:
      return dict.mapValues { AnyCodable($0) } as [String: AnyCodable]
    case let arr as [Any]:
      return arr.map { AnyCodable($0) } as [AnyCodable]
    default:
      return value
    }
  }
}

// MARK: - MCP Method Constants

/// Standard MCP method names.
public enum MCPMethod {
  public static let initialize = "initialize"
  public static let notificationsInitialized = "notifications/initialized"
  public static let toolsList = "tools/list"
  public static let toolsCall = "tools/call"
  public static let ping = "ping"
}
