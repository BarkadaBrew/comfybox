// ComfyBridgeOrderedJSON.swift — Order-preserving JSON serialization for /object_info
//
// ComfyUI frontend maps widgets_values positionally to input names based on
// the key order in the /object_info JSON response. Swift dictionaries ([String: Any])
// do not preserve insertion order, so JSONSerialization scrambles the keys.
//
// This module provides:
// - OrderedDict: a key-value container that preserves insertion order
// - orderedJSONData(_:): a recursive JSON serializer that respects OrderedDict ordering

import Foundation

/// A dictionary-like container that preserves insertion order.
/// When serialized by `orderedJSONData`, keys appear in the order they were inserted.
final class OrderedDict {
  var entries: [(String, Any)]

  init(_ entries: [(String, Any)]) {
    self.entries = entries
  }

  subscript(key: String) -> Any? {
    get { entries.first { $0.0 == key }?.1 }
    set {
      if let newValue {
        if let idx = entries.firstIndex(where: { $0.0 == key }) {
          entries[idx] = (key, newValue)
        } else {
          entries.append((key, newValue))
        }
      } else {
        entries.removeAll { $0.0 == key }
      }
    }
  }

  var keys: [String] { entries.map(\.0) }
  var count: Int { entries.count }
}

/// Serialize any value to a JSON string, preserving key order for OrderedDict instances.
/// Regular [String: Any] dictionaries are serialized with keys in alphabetical order.
func orderedJSONData(_ value: Any) -> Data? {
  var output = ""
  serializeValue(value, to: &output)
  return output.data(using: .utf8)
}

/// Serialize any value to a JSON string fragment.
private func serializeValue(_ value: Any, to output: inout String) {
  switch value {
  case let ordered as OrderedDict:
    output += "{"
    for (i, (key, val)) in ordered.entries.enumerated() {
      if i > 0 { output += "," }
      output += "\""
      escapeJSONString(key, to: &output)
      output += "\":"
      serializeValue(val, to: &output)
    }
    output += "}"

  case let dict as [String: Any]:
    output += "{"
    let keys = dict.keys.sorted()
    for (i, key) in keys.enumerated() {
      if i > 0 { output += "," }
      output += "\""
      escapeJSONString(key, to: &output)
      output += "\":"
      serializeValue(dict[key]!, to: &output)
    }
    output += "}"

  case let array as [Any]:
    output += "["
    for (i, val) in array.enumerated() {
      if i > 0 { output += "," }
      serializeValue(val, to: &output)
    }
    output += "]"

  case let string as String:
    output += "\""
    escapeJSONString(string, to: &output)
    output += "\""

  case let bool as Bool:
    output += bool ? "true" : "false"

  case let int as Int:
    output += "\(int)"

  case let float as Float:
    if float == Float(Int(float)) {
      output += "\(Int(float))"
    } else {
      output += "\(float)"
    }

  case let double as Double:
    if double == Double(Int(double)) {
      output += "\(Int(double))"
    } else {
      output += "\(double)"
    }

  case is NSNull:
    output += "null"

  default:
    // Fallback: try JSONSerialization for unknown types
    if let data = try? JSONSerialization.data(withJSONObject: value),
       let str = String(data: data, encoding: .utf8) {
      output += str
    } else {
      output += "null"
    }
  }
}

/// Escape a string for JSON output, appending to the output buffer.
private func escapeJSONString(_ string: String, to output: inout String) {
  for char in string {
    switch char {
    case "\"": output += "\\\""
    case "\\": output += "\\\\"
    case "\n": output += "\\n"
    case "\r": output += "\\r"
    case "\t": output += "\\t"
    default:
      if char.asciiValue.map({ $0 < 0x20 }) ?? false {
        output += String(format: "\\u%04x", char.asciiValue!)
      } else {
        output.append(char)
      }
    }
  }
}
