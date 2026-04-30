// ComfyBridgeWebSocket.swift — WebSocket connection manager for ComfyUI bridge
//
// Implements RFC 6455 WebSocket framing over NWConnection. Manages per-client
// connections and provides methods to send ComfyUI protocol messages.
//
// Phase 1: Accept connections, send status on connect, handle ping/pong.
// Phase 2: Will add progress events, preview images, and execution messages.

import Foundation
import Logging
import Network
import CryptoKit

/// Manages WebSocket connections for the ComfyUI bridge protocol.
///
/// Each Krita plugin instance connects with a unique `clientId`. The manager
/// keeps connections alive and routes execution events to the correct client.
final class ComfyWebSocketManager {
  private let logger: Logger
  private let lock = NSLock()
  private var connections: [String: WebSocketConnection] = [:]

  init(logger: Logger) {
    self.logger = logger
  }

  /// Register a new WebSocket connection after the HTTP 101 upgrade.
  /// Sends the initial `{"type": "status"}` message and begins listening for frames.
  func registerConnection(clientId: String, connection: NWConnection, queue: DispatchQueue) {
    let ws = WebSocketConnection(
      clientId: clientId,
      connection: connection,
      queue: queue,
      logger: logger,
      onClose: { [weak self] id in
        self?.removeConnection(id: id)
      }
    )

    lock.lock()
    // Close any existing connection with the same clientId.
    if let existing = connections[clientId] {
      logger.info("ComfyWS: replacing existing connection for clientId=\(clientId)")
      existing.close()
    }
    connections[clientId] = ws
    lock.unlock()

    ws.start()
  }

  /// Send a text message to a specific client.
  func send(to clientId: String, text: String) {
    lock.lock()
    let ws = connections[clientId]
    lock.unlock()

    ws?.sendText(text)
  }

  /// Send a text message to all connected clients.
  func broadcast(text: String) {
    lock.lock()
    let all = Array(connections.values)
    lock.unlock()

    for ws in all {
      ws.sendText(text)
    }
  }

  /// Number of active connections.
  var connectionCount: Int {
    lock.lock()
    defer { lock.unlock() }
    return connections.count
  }

  private func removeConnection(id: String) {
    lock.lock()
    connections.removeValue(forKey: id)
    lock.unlock()
    logger.info("ComfyWS: connection closed for clientId=\(id)")
  }

  // MARK: - WebSocket Accept Key

  /// Compute the Sec-WebSocket-Accept value per RFC 6455 Section 4.2.2.
  static func computeAcceptKey(from clientKey: String) -> String {
    let magic = "258EAFA5-E914-47DA-95CA-5AB5DC11AD35"
    let combined = clientKey + magic
    let data = Data(combined.utf8)
    let digest = Insecure.SHA1.hash(data: data)
    return Data(digest).base64EncodedString()
  }
}

// MARK: - WebSocket Connection

/// A single WebSocket connection handling RFC 6455 framing.
private final class WebSocketConnection {
  private let clientId: String
  private let connection: NWConnection
  private let queue: DispatchQueue
  private let logger: Logger
  private let onClose: (String) -> Void

  private var isActive = true
  /// Self-retention to keep the connection alive while frames are being processed.
  private var retainSelf: WebSocketConnection?

  init(
    clientId: String,
    connection: NWConnection,
    queue: DispatchQueue,
    logger: Logger,
    onClose: @escaping (String) -> Void
  ) {
    self.clientId = clientId
    self.connection = connection
    self.queue = queue
    self.logger = logger
    self.onClose = onClose
  }

  func start() {
    retainSelf = self
    // Send the initial status message that triggers Krita's "connected" event.
    let statusMsg = #"{"type":"status"}"#
    sendText(statusMsg)
    receiveFrame()
  }

  func close() {
    guard isActive else { return }
    isActive = false
    connection.cancel()
    retainSelf = nil
  }

  // MARK: - Send

  func sendText(_ text: String) {
    guard isActive else { return }
    let payload = Data(text.utf8)
    let frame = encodeFrame(opcode: .text, payload: payload)
    connection.send(content: frame, completion: .contentProcessed { [weak self] error in
      if let error {
        self?.logger.warning("ComfyWS: send error for \(self?.clientId ?? "?"): \(error)")
        self?.close()
        self?.onClose(self?.clientId ?? "")
      }
    })
  }

  func sendBinary(_ data: Data) {
    guard isActive else { return }
    let frame = encodeFrame(opcode: .binary, payload: data)
    connection.send(content: frame, completion: .contentProcessed { [weak self] error in
      if let error {
        self?.logger.warning("ComfyWS: binary send error for \(self?.clientId ?? "?"): \(error)")
        self?.close()
        self?.onClose(self?.clientId ?? "")
      }
    })
  }

  // MARK: - Receive

  private func receiveFrame() {
    guard isActive else { return }

    // Read at least 2 bytes (minimum WebSocket frame header).
    connection.receive(minimumIncompleteLength: 2, maximumLength: 65_536) { [weak self] data, _, isComplete, error in
      guard let self, self.isActive else { return }

      if isComplete || error != nil {
        self.close()
        self.onClose(self.clientId)
        return
      }

      guard let data, data.count >= 2 else {
        self.receiveFrame()
        return
      }

      self.handleReceivedData(data)
      self.receiveFrame()
    }
  }

  private func handleReceivedData(_ data: Data) {
    let byte0 = data[data.startIndex]
    let opcode = WebSocketOpcode(rawValue: byte0 & 0x0F) ?? .text

    switch opcode {
    case .ping:
      // Respond with pong, echoing the payload.
      let maskBit = data[data.startIndex + 1]
      let payloadLength = Int(maskBit & 0x7F)
      let maskStart = data.startIndex + 2
      let dataStart = maskStart + (maskBit & 0x80 != 0 ? 4 : 0)

      if dataStart + payloadLength <= data.endIndex {
        var payload = Data(data[dataStart..<(dataStart + payloadLength)])
        // Unmask if masked (client frames are always masked per RFC 6455).
        if maskBit & 0x80 != 0, maskStart + 4 <= data.endIndex {
          let mask = Array(data[maskStart..<(maskStart + 4)])
          for i in 0..<payload.count {
            payload[payload.startIndex + i] ^= mask[i % 4]
          }
        }
        sendPong(payload: payload)
      } else {
        sendPong(payload: Data())
      }

    case .close:
      // Send close frame back and tear down.
      let closeFrame = encodeFrame(opcode: .close, payload: Data())
      connection.send(content: closeFrame, completion: .contentProcessed { [weak self] _ in
        self?.close()
        if let clientId = self?.clientId {
          self?.onClose(clientId)
        }
      })

    case .text, .binary:
      // Phase 1: log and ignore incoming data frames.
      // Phase 2 may need to handle client commands.
      break

    case .pong:
      // Received pong — connection is alive, nothing to do.
      break
    }
  }

  private func sendPong(payload: Data) {
    let frame = encodeFrame(opcode: .pong, payload: payload)
    connection.send(content: frame, completion: .contentProcessed { _ in })
  }

  // MARK: - Frame Encoding (Server → Client)

  /// Encode a WebSocket frame. Server frames are never masked (RFC 6455 Section 5.1).
  private func encodeFrame(opcode: WebSocketOpcode, payload: Data) -> Data {
    var frame = Data()

    // FIN bit + opcode.
    frame.append(0x80 | opcode.rawValue)

    // Payload length (no mask bit for server frames).
    let length = payload.count
    if length < 126 {
      frame.append(UInt8(length))
    } else if length <= 65535 {
      frame.append(126)
      frame.append(UInt8((length >> 8) & 0xFF))
      frame.append(UInt8(length & 0xFF))
    } else {
      frame.append(127)
      for i in (0..<8).reversed() {
        frame.append(UInt8((length >> (8 * i)) & 0xFF))
      }
    }

    frame.append(payload)
    return frame
  }
}

// MARK: - WebSocket Opcodes

private enum WebSocketOpcode: UInt8 {
  case text = 0x1
  case binary = 0x2
  case close = 0x8
  case ping = 0x9
  case pong = 0xA
}
