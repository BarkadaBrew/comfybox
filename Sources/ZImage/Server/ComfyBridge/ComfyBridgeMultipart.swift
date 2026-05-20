// ComfyBridgeMultipart.swift — Minimal multipart/form-data parser for image uploads

import Foundation

enum ComfyBridgeMultipart {
  struct Form {
    let fields: [String: String]
    let files: [File]

    func field(named name: String) -> String? {
      fields[name]
    }

    func file(named name: String) -> File? {
      files.first { $0.name == name }
    }
  }

  struct File {
    let name: String
    let filename: String?
    let contentType: String?
    let data: Data
  }

  struct ParseError: Error, CustomStringConvertible {
    let description: String
    init(_ description: String) {
      self.description = description
    }
  }

  static func parse(body: Data, contentType: String?) throws -> Form {
    guard let boundary = boundary(from: contentType), !boundary.isEmpty else {
      throw ParseError("Missing multipart boundary")
    }

    let delimiter = Data("--\(boundary)".utf8)
    guard let firstDelimiter = body.range(of: delimiter) else {
      throw ParseError("Multipart boundary not found")
    }

    var cursor = firstDelimiter.upperBound
    var fields: [String: String] = [:]
    var files: [File] = []

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

      if partEnd > cursor {
        let part = body.subdata(in: cursor..<partEnd)
        if let parsed = try parsePart(part) {
          switch parsed {
          case .field(let name, let value):
            fields[name] = value
          case .file(let file):
            files.append(file)
          }
        }
      }

      cursor = nextDelimiter.upperBound
    }

    return Form(fields: fields, files: files)
  }

  private enum Part {
    case field(name: String, value: String)
    case file(File)
  }

  private static func parsePart(_ data: Data) throws -> Part? {
    let headerSeparator = Data("\r\n\r\n".utf8)
    guard let separator = data.range(of: headerSeparator) else {
      return nil
    }

    let headerData = data.subdata(in: 0..<separator.lowerBound)
    guard let headerString = String(data: headerData, encoding: .utf8) else {
      throw ParseError("Invalid multipart headers")
    }

    var headers: [String: String] = [:]
    for line in headerString.components(separatedBy: "\r\n") where !line.isEmpty {
      guard let colon = line.firstIndex(of: ":") else { continue }
      let key = line[..<colon].trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
      let value = line[line.index(after: colon)...].trimmingCharacters(in: .whitespacesAndNewlines)
      headers[key] = value
    }

    guard let disposition = headers["content-disposition"] else {
      return nil
    }
    let parameters = dispositionParameters(from: disposition)
    guard let name = parameters["name"] else {
      return nil
    }

    let payload = data.subdata(in: separator.upperBound..<data.count)
    if let filename = parameters["filename"] {
      return .file(File(
        name: name,
        filename: filename,
        contentType: headers["content-type"],
        data: payload
      ))
    }

    let value = String(data: payload, encoding: .utf8) ?? ""
    return .field(name: name, value: value)
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

  private static func dispositionParameters(from header: String) -> [String: String] {
    var result: [String: String] = [:]
    for segment in header.split(separator: ";") {
      let trimmed = segment.trimmingCharacters(in: .whitespacesAndNewlines)
      guard let equals = trimmed.firstIndex(of: "=") else { continue }
      let key = trimmed[..<equals].trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
      var value = trimmed[trimmed.index(after: equals)...].trimmingCharacters(in: .whitespacesAndNewlines)
      if value.hasPrefix("\""), value.hasSuffix("\""), value.count >= 2 {
        value.removeFirst()
        value.removeLast()
      }
      result[key] = String(value)
    }
    return result
  }

  private static func hasBytes(_ bytes: [UInt8], in data: Data, at index: Int) -> Bool {
    guard index >= 0, index + bytes.count <= data.count else {
      return false
    }
    for (offset, byte) in bytes.enumerated() where data[index + offset] != byte {
      return false
    }
    return true
  }
}
