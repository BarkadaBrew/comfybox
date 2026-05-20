// ComfyBridgeMultipart.swift — Minimal multipart/form-data parser

import Foundation

enum ComfyBridgeMultipart {
  struct ParseError: Error, CustomStringConvertible {
    let description: String
    init(_ description: String) {
      self.description = description
    }
  }

  static func parse(body: Data, contentType: String?) throws -> [String: Data] {
    guard let boundary = boundary(from: contentType), !boundary.isEmpty else {
      throw ParseError("Missing multipart boundary")
    }

    let delimiter = Data("--\(boundary)".utf8)
    guard let firstDelimiter = body.range(of: delimiter) else {
      throw ParseError("Multipart boundary not found")
    }

    var fields: [String: Data] = [:]
    var cursor = firstDelimiter.upperBound

    while cursor < body.count {
      if hasBytes([45, 45], in: body, at: cursor) {
        break
      }
      if hasBytes([13, 10], in: body, at: cursor) {
        cursor += 2
      }

      guard let nextDelimiter = body.range(of: delimiter, options: [], in: cursor..<body.count) else {
        throw ParseError("Unterminated multipart section")
      }

      var partEnd = nextDelimiter.lowerBound
      if partEnd >= 2 && body[partEnd - 2] == 13 && body[partEnd - 1] == 10 {
        partEnd -= 2
      }

      if let field = parsePart(body.subdata(in: cursor..<partEnd)) {
        fields[field.name] = field.data
      }

      cursor = nextDelimiter.upperBound
    }

    return fields
  }

  private static func parsePart(_ part: Data) -> (name: String, data: Data)? {
    let separator = Data("\r\n\r\n".utf8)
    guard let separatorRange = part.range(of: separator),
          let headers = String(data: part.subdata(in: 0..<separatorRange.lowerBound), encoding: .utf8) else {
      return nil
    }

    let disposition = headers
      .components(separatedBy: "\r\n")
      .first { $0.lowercased().hasPrefix("content-disposition:") } ?? ""
    guard let name = dispositionName(from: disposition) else { return nil }

    return (name, part.subdata(in: separatorRange.upperBound..<part.count))
  }

  private static func boundary(from contentType: String?) -> String? {
    guard let contentType else { return nil }
    for segment in contentType.split(separator: ";") {
      let trimmed = segment.trimmingCharacters(in: .whitespacesAndNewlines)
      guard trimmed.lowercased().hasPrefix("boundary=") else { continue }
      var value = String(trimmed.dropFirst("boundary=".count))
      if value.hasPrefix("\""), value.hasSuffix("\""), value.count >= 2 {
        value.removeFirst()
        value.removeLast()
      }
      return value
    }
    return nil
  }

  private static func dispositionName(from header: String) -> String? {
    for segment in header.split(separator: ";") {
      let trimmed = segment.trimmingCharacters(in: .whitespacesAndNewlines)
      guard trimmed.lowercased().hasPrefix("name=") else { continue }
      var value = String(trimmed.dropFirst("name=".count))
      if value.hasPrefix("\""), value.hasSuffix("\""), value.count >= 2 {
        value.removeFirst()
        value.removeLast()
      }
      return value
    }
    return nil
  }

  private static func hasBytes(_ bytes: [UInt8], in data: Data, at index: Int) -> Bool {
    guard index >= 0, index + bytes.count <= data.count else { return false }
    for (offset, byte) in bytes.enumerated() where data[index + offset] != byte {
      return false
    }
    return true
  }
}
