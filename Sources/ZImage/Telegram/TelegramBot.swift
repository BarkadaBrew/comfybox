// TelegramBot.swift — Zero-dependency Telegram Bot API client
//
// Long-polling bot using URLSession. No external dependencies.
// Handles getUpdates, sendMessage, sendPhoto, sendDocument.

import Foundation
import Logging

// MARK: - Types

public struct TelegramUpdate: Sendable {
  public let updateId: Int
  public let message: TelegramMessage?
  public let callbackQuery: TelegramCallbackQuery?
}

public final class TelegramMessage: @unchecked Sendable {
  public let messageId: Int
  public let chatId: Int
  public let userId: Int
  public let firstName: String
  public let text: String?
  public let caption: String?
  public let photo: [TelegramPhotoSize]?
  public let replyToMessage: TelegramMessage?
  public let date: Int

  public init(
    messageId: Int, chatId: Int, userId: Int, firstName: String,
    text: String?, caption: String?, photo: [TelegramPhotoSize]?,
    replyToMessage: TelegramMessage?, date: Int
  ) {
    self.messageId = messageId
    self.chatId = chatId
    self.userId = userId
    self.firstName = firstName
    self.text = text
    self.caption = caption
    self.photo = photo
    self.replyToMessage = replyToMessage
    self.date = date
  }
}

public struct TelegramCallbackQuery: Sendable {
  public let id: String
  public let userId: Int
  public let firstName: String
  public let messageId: Int?
  public let chatId: Int?
  public let data: String?
}

public struct TelegramPhotoSize: Sendable {
  public let fileId: String
  public let width: Int
  public let height: Int
}

public struct InlineKeyboard: Sendable {
  public let rows: [[InlineButton]]

  public func toJSON() -> [[[String: Any]]] {
    return rows.map { row in
      row.map { button -> [String: Any] in
        var dict: [String: Any] = ["text": button.text]
        dict["callback_data"] = button.callbackData
        return dict
      }
    }
  }
}

public struct InlineButton: Sendable {
  public let text: String
  public let callbackData: String
}

public struct SendResult: Sendable {
  public let ok: Bool
  public let messageId: Int?
}

// MARK: - TelegramBot

public final class TelegramBot: @unchecked Sendable {
  public struct Configuration: Sendable {
    public let botToken: String
    public let allowedUserIds: Set<Int>
    public let pollTimeoutSeconds: Int
    public let retryDelaySeconds: Int

    public init(
      botToken: String,
      allowedUserIds: Set<Int>,
      pollTimeoutSeconds: Int = 30,
      retryDelaySeconds: Int = 5
    ) {
      self.botToken = botToken
      self.allowedUserIds = allowedUserIds
      self.pollTimeoutSeconds = pollTimeoutSeconds
      self.retryDelaySeconds = retryDelaySeconds
    }
  }

  private let config: Configuration
  private let baseURL: String
  private let session: URLSession
  private let logger: Logger
  private var offset: Int = 0
  private var running = false

  public init(configuration: Configuration, logger: Logger = Logger(label: "comfybox.telegram")) {
    self.config = configuration
    self.baseURL = "https://api.telegram.org/bot\(configuration.botToken)"
    self.logger = logger

    let sessionConfig = URLSessionConfiguration.ephemeral
    // Long poll timeout + buffer for network overhead
    sessionConfig.timeoutIntervalForRequest = TimeInterval(configuration.pollTimeoutSeconds + 10)
    sessionConfig.timeoutIntervalForResource = TimeInterval(configuration.pollTimeoutSeconds + 30)
    self.session = URLSession(configuration: sessionConfig)
  }

  // MARK: - Polling

  /// Start long polling. Runs until stop() is called.
  public func startPolling(handler: @escaping (TelegramUpdate) async -> Void) async {
    running = true
    logger.info("Telegram bot: polling started (allowed users: \(config.allowedUserIds))")

    while running {
      do {
        let updates = try await getUpdates()
        for update in updates {
          offset = update.updateId + 1

          // Auth check
          let userId = update.message?.userId ?? update.callbackQuery?.userId
          guard let uid = userId, config.allowedUserIds.contains(uid) else {
            if let uid = userId {
              logger.warning("Telegram bot: rejected update from unauthorized user \(uid)")
            }
            continue
          }

          await handler(update)
        }
      } catch {
        if !running { break }
        logger.error("Telegram bot: poll error — \(error.localizedDescription). Retrying in \(config.retryDelaySeconds)s.")
        try? await Task.sleep(nanoseconds: UInt64(config.retryDelaySeconds) * 1_000_000_000)
      }
    }

    logger.info("Telegram bot: polling stopped")
  }

  /// Stop polling gracefully.
  public func stop() {
    running = false
  }

  // MARK: - Outbound API

  /// Send a text message.
  public func sendMessage(chatId: Int, text: String, replyTo: Int? = nil) async throws -> SendResult {
    var body: [String: Any] = [
      "chat_id": chatId,
      "text": text,
      "parse_mode": "HTML"
    ]
    if let replyTo = replyTo {
      body["reply_to_message_id"] = replyTo
    }

    return try await callMethod("sendMessage", body: body)
  }

  /// Send a photo via multipart/form-data upload.
  public func sendPhoto(
    chatId: Int,
    imageData: Data,
    filename: String,
    caption: String? = nil,
    replyMarkup: InlineKeyboard? = nil
  ) async throws -> SendResult {
    let boundary = "ComfyBox-\(UUID().uuidString)"
    var body = Data()

    // chat_id field
    appendMultipartField(&body, boundary: boundary, name: "chat_id", value: "\(chatId)")

    // caption field
    if let caption = caption {
      appendMultipartField(&body, boundary: boundary, name: "caption", value: caption)
      appendMultipartField(&body, boundary: boundary, name: "parse_mode", value: "HTML")
    }

    // reply_markup field
    if let markup = replyMarkup {
      let markupJSON: [String: Any] = ["inline_keyboard": markup.toJSON()]
      if let markupData = try? JSONSerialization.data(withJSONObject: markupJSON),
         let markupString = String(data: markupData, encoding: .utf8) {
        appendMultipartField(&body, boundary: boundary, name: "reply_markup", value: markupString)
      }
    }

    // photo file
    appendMultipartFile(&body, boundary: boundary, name: "photo", filename: filename, mimeType: "image/png", data: imageData)

    // Close boundary
    body.append("--\(boundary)--\r\n".data(using: .utf8)!)

    guard let url = URL(string: "\(baseURL)/sendPhoto") else {
      return SendResult(ok: false, messageId: nil)
    }

    var request = URLRequest(url: url)
    request.httpMethod = "POST"
    request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
    request.httpBody = body

    let (data, _) = try await session.data(for: request)
    return parseResult(data)
  }

  /// Send a document via multipart/form-data upload.
  public func sendDocument(
    chatId: Int,
    fileData: Data,
    filename: String,
    mimeType: String,
    caption: String? = nil,
    replyMarkup: InlineKeyboard? = nil
  ) async throws -> SendResult {
    let boundary = "ComfyBox-\(UUID().uuidString)"
    var body = Data()

    appendMultipartField(&body, boundary: boundary, name: "chat_id", value: "\(chatId)")
    if let caption = caption {
      appendMultipartField(&body, boundary: boundary, name: "caption", value: caption)
      appendMultipartField(&body, boundary: boundary, name: "parse_mode", value: "HTML")
    }
    if let markup = replyMarkup {
      let markupJSON: [String: Any] = ["inline_keyboard": markup.toJSON()]
      if let markupData = try? JSONSerialization.data(withJSONObject: markupJSON),
         let markupString = String(data: markupData, encoding: .utf8) {
        appendMultipartField(&body, boundary: boundary, name: "reply_markup", value: markupString)
      }
    }

    appendMultipartFile(&body, boundary: boundary, name: "document", filename: filename, mimeType: mimeType, data: fileData)
    body.append("--\(boundary)--\r\n".data(using: .utf8)!)

    guard let url = URL(string: "\(baseURL)/sendDocument") else {
      return SendResult(ok: false, messageId: nil)
    }

    var request = URLRequest(url: url)
    request.httpMethod = "POST"
    request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
    request.httpBody = body

    let (data, _) = try await session.data(for: request)
    return parseResult(data)
  }

  /// Answer a callback query (acknowledge inline button press).
  public func answerCallbackQuery(id: String, text: String? = nil) async throws {
    var body: [String: Any] = ["callback_query_id": id]
    if let text = text { body["text"] = text }
    let _ = try await callMethod("answerCallbackQuery", body: body)
  }

  /// Edit reply markup on an existing message.
  public func editMessageReplyMarkup(chatId: Int, messageId: Int, markup: InlineKeyboard? = nil) async throws {
    var body: [String: Any] = [
      "chat_id": chatId,
      "message_id": messageId
    ]
    if let markup = markup {
      body["reply_markup"] = ["inline_keyboard": markup.toJSON()]
    }
    let _ = try await callMethod("editMessageReplyMarkup", body: body)
  }

  // MARK: - Private

  private func getUpdates() async throws -> [TelegramUpdate] {
    let body: [String: Any] = [
      "offset": offset,
      "timeout": config.pollTimeoutSeconds,
      "allowed_updates": ["message", "callback_query"]
    ]

    guard let url = URL(string: "\(baseURL)/getUpdates") else { return [] }
    var request = URLRequest(url: url)
    request.httpMethod = "POST"
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    request.httpBody = try JSONSerialization.data(withJSONObject: body)

    let (data, _) = try await session.data(for: request)

    guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
          let ok = json["ok"] as? Bool, ok,
          let result = json["result"] as? [[String: Any]] else {
      return []
    }

    return result.compactMap { parseUpdate($0) }
  }

  private func callMethod(_ method: String, body: [String: Any]) async throws -> SendResult {
    guard let url = URL(string: "\(baseURL)/\(method)") else {
      return SendResult(ok: false, messageId: nil)
    }

    var request = URLRequest(url: url)
    request.httpMethod = "POST"
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    request.httpBody = try JSONSerialization.data(withJSONObject: body)

    let (data, _) = try await session.data(for: request)
    return parseResult(data)
  }

  private func parseResult(_ data: Data) -> SendResult {
    guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
      return SendResult(ok: false, messageId: nil)
    }
    let ok = json["ok"] as? Bool ?? false
    let messageId = (json["result"] as? [String: Any])?["message_id"] as? Int
    return SendResult(ok: ok, messageId: messageId)
  }

  private func parseUpdate(_ json: [String: Any]) -> TelegramUpdate? {
    guard let updateId = json["update_id"] as? Int else { return nil }

    var message: TelegramMessage? = nil
    if let msgJson = json["message"] as? [String: Any] {
      message = parseMessage(msgJson)
    }

    var callbackQuery: TelegramCallbackQuery? = nil
    if let cbJson = json["callback_query"] as? [String: Any] {
      callbackQuery = parseCallbackQuery(cbJson)
    }

    return TelegramUpdate(updateId: updateId, message: message, callbackQuery: callbackQuery)
  }

  private func parseMessage(_ json: [String: Any]) -> TelegramMessage? {
    guard let messageId = json["message_id"] as? Int,
          let chat = json["chat"] as? [String: Any],
          let chatId = chat["id"] as? Int else { return nil }

    let from = json["from"] as? [String: Any]
    let userId = from?["id"] as? Int ?? 0
    let firstName = from?["first_name"] as? String ?? ""

    var photos: [TelegramPhotoSize]? = nil
    if let photoArray = json["photo"] as? [[String: Any]] {
      photos = photoArray.compactMap { p in
        guard let fileId = p["file_id"] as? String,
              let width = p["width"] as? Int,
              let height = p["height"] as? Int else { return nil }
        return TelegramPhotoSize(fileId: fileId, width: width, height: height)
      }
    }

    var replyTo: TelegramMessage? = nil
    if let replyJson = json["reply_to_message"] as? [String: Any] {
      replyTo = parseMessage(replyJson)
    }

    return TelegramMessage(
      messageId: messageId,
      chatId: chatId,
      userId: userId,
      firstName: firstName,
      text: json["text"] as? String,
      caption: json["caption"] as? String,
      photo: photos,
      replyToMessage: replyTo,
      date: json["date"] as? Int ?? 0
    )
  }

  private func parseCallbackQuery(_ json: [String: Any]) -> TelegramCallbackQuery? {
    guard let id = json["id"] as? String else { return nil }

    let from = json["from"] as? [String: Any]
    let userId = from?["id"] as? Int ?? 0
    let firstName = from?["first_name"] as? String ?? ""

    let msg = json["message"] as? [String: Any]
    let messageId = msg?["message_id"] as? Int
    let chat = msg?["chat"] as? [String: Any]
    let chatId = chat?["id"] as? Int

    return TelegramCallbackQuery(
      id: id,
      userId: userId,
      firstName: firstName,
      messageId: messageId,
      chatId: chatId,
      data: json["data"] as? String
    )
  }

  // MARK: - Multipart Helpers

  private func appendMultipartField(_ body: inout Data, boundary: String, name: String, value: String) {
    body.append("--\(boundary)\r\n".data(using: .utf8)!)
    body.append("Content-Disposition: form-data; name=\"\(name)\"\r\n\r\n".data(using: .utf8)!)
    body.append("\(value)\r\n".data(using: .utf8)!)
  }

  private func appendMultipartFile(_ body: inout Data, boundary: String, name: String, filename: String, mimeType: String, data: Data) {
    body.append("--\(boundary)\r\n".data(using: .utf8)!)
    body.append("Content-Disposition: form-data; name=\"\(name)\"; filename=\"\(filename)\"\r\n".data(using: .utf8)!)
    body.append("Content-Type: \(mimeType)\r\n\r\n".data(using: .utf8)!)
    body.append(data)
    body.append("\r\n".data(using: .utf8)!)
  }
}
