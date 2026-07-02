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

  /// Send a binary message to a specific client.
  func sendBinary(to clientId: String, data: Data) {
    lock.lock()
    let ws = connections[clientId]
    lock.unlock()

    ws?.sendBinary(data)
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
    let magic = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
    let combined = clientKey + magic
    let data = Data(combined.utf8)
    let digest = Insecure.SHA1.hash(data: data)
    return Data(digest).base64EncodedString()
  }
}

// MARK: - WebSocket Connection

/// A single WebSocket connection handling RFC 6455 framing with proper
/// buffering, extended payload lengths, and continuation frame support.
private final class WebSocketConnection {
  private let clientId: String
  private let connection: NWConnection
  private let queue: DispatchQueue
  private let logger: Logger
  private let onClose: (String) -> Void

  private var isActive = true
  /// Self-retention to keep the connection alive while frames are being processed.
  private var retainSelf: WebSocketConnection?

  /// Receive buffer for assembling frames from partial TCP reads.
  private var receiveBuffer = Data()
  /// Buffer for reassembling fragmented messages (continuation frames).
  private var fragmentBuffer = Data()
  /// Opcode of the first frame in a fragmented sequence.
  private var fragmentOpcode: WebSocketOpcode?

  /// Maximum single message size (16 MB) to prevent memory exhaustion.
  private static let maxMessageSize = 16 * 1024 * 1024

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
    let statusMsg = jsonString([
      "type": "status",
      "data": [
        "sid": clientId,
        "status": [
          "exec_info": [
            "queue_remaining": 0
          ]
        ]
      ] as [String: Any]
    ])
    sendText(statusMsg)
    receiveLoop()
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

  private func receiveLoop() {
    guard isActive else { return }

    connection.receive(minimumIncompleteLength: 1, maximumLength: 65_536) { [weak self] data, _, isComplete, error in
      guard let self, self.isActive else { return }

      if let data, !data.isEmpty {
        self.receiveBuffer.append(data)
      }

      if isComplete || error != nil {
        self.close()
        self.onClose(self.clientId)
        return
      }

      // Parse as many complete frames as possible from the buffer.
      self.drainFrames()
      self.receiveLoop()
    }
  }

  /// Parse and handle all complete frames currently in the receive buffer.
  private func drainFrames() {
    while isActive {
      guard let frame = parseFrame() else { break }
      handleFrame(frame)
    }
  }

  /// Attempt to parse one WebSocket frame from the receive buffer.
  /// Returns nil if the buffer does not contain a complete frame.
  /// Consumes the frame bytes from the buffer on success.
  private func parseFrame() -> ParsedFrame? {
    let buf = receiveBuffer
    guard buf.count >= 2 else { return nil }

    let byte0 = buf[buf.startIndex]
    let byte1 = buf[buf.startIndex + 1]

    let fin = (byte0 & 0x80) != 0
    let opcode = WebSocketOpcode(rawValue: byte0 & 0x0F) ?? .text
    let masked = (byte1 & 0x80) != 0
    let lengthField = byte1 & 0x7F

    var headerSize = 2
    var payloadLength: Int

    if lengthField < 126 {
      payloadLength = Int(lengthField)
    } else if lengthField == 126 {
      // 16-bit extended payload length.
      guard buf.count >= 4 else { return nil }
      payloadLength = Int(buf[buf.startIndex + 2]) << 8 | Int(buf[buf.startIndex + 3])
      headerSize = 4
    } else {
      // 64-bit extended payload length.
      guard buf.count >= 10 else { return nil }
      var length64: UInt64 = 0
      for i in 0..<8 {
        length64 = (length64 << 8) | UInt64(buf[buf.startIndex + 2 + i])
      }
      headerSize = 10

      // Sanity check — reject frames larger than our limit. Compare as UInt64
      // before converting to Int: a length with the top bit set would otherwise
      // overflow to a negative Int, bypass this check, and crash on slicing.
      guard length64 <= UInt64(Self.maxMessageSize) else {
        logger.warning("ComfyWS: frame too large (\(length64) bytes), closing \(clientId)")
        closeWithProtocolError()
        return nil
      }
      payloadLength = Int(length64)
    }

    let maskSize = masked ? 4 : 0
    let totalFrameSize = headerSize + maskSize + payloadLength

    guard buf.count >= totalFrameSize else { return nil }

    // Extract and unmask payload.
    let maskStart = buf.startIndex + headerSize
    let dataStart = maskStart + maskSize

    var payload = Data(buf[dataStart..<(dataStart + payloadLength)])
    if masked {
      let mask = Array(buf[maskStart..<(maskStart + 4)])
      for i in 0..<payload.count {
        payload[payload.startIndex + i] ^= mask[i % 4]
      }
    }

    // Consume the frame from the buffer.
    receiveBuffer.removeSubrange(receiveBuffer.startIndex..<(receiveBuffer.startIndex + totalFrameSize))

    return ParsedFrame(fin: fin, opcode: opcode, payload: payload)
  }

  /// Handle a fully parsed WebSocket frame, including fragmentation reassembly.
  private func handleFrame(_ frame: ParsedFrame) {
    switch frame.opcode {
    case .continuation:
      // Continuation frame — append to fragment buffer.
      guard fragmentOpcode != nil else {
        logger.warning("ComfyWS: unexpected continuation frame for \(clientId)")
        closeWithProtocolError()
        return
      }
      fragmentBuffer.append(frame.payload)
      if fragmentBuffer.count > Self.maxMessageSize {
        logger.warning("ComfyWS: fragmented message too large for \(clientId)")
        closeWithProtocolError()
        return
      }
      if frame.fin {
        // Reassembly complete.
        let completePayload = fragmentBuffer
        let opcode = fragmentOpcode!
        fragmentBuffer.removeAll()
        fragmentOpcode = nil
        handleMessage(opcode: opcode, payload: completePayload)
      }

    case .text, .binary:
      if frame.fin {
        // Single-frame message.
        handleMessage(opcode: frame.opcode, payload: frame.payload)
      } else {
        // First frame of a fragmented message.
        fragmentOpcode = frame.opcode
        fragmentBuffer = frame.payload
      }

    case .ping:
      // Control frames may appear between fragmented data frames.
      sendPong(payload: frame.payload)

    case .pong:
      // Received pong — connection is alive, nothing to do.
      break

    case .close:
      // Send close frame back and tear down.
      let closeFrame = encodeFrame(opcode: .close, payload: Data())
      connection.send(content: closeFrame, completion: .contentProcessed { [weak self] _ in
        self?.close()
        if let clientId = self?.clientId {
          self?.onClose(clientId)
        }
      })
    }
  }

  /// Handle a complete (possibly reassembled) WebSocket message.
  private func handleMessage(opcode: WebSocketOpcode, payload: Data) {
    // Phase 1: log and ignore incoming data frames.
    // Phase 2 may need to handle client commands.
    _ = opcode
    _ = payload
  }

  private func closeWithProtocolError() {
    // Send close frame with 1002 (protocol error) status code.
    var payload = Data()
    payload.append(UInt8(1002 >> 8))
    payload.append(UInt8(1002 & 0xFF))
    let frame = encodeFrame(opcode: .close, payload: payload)
    connection.send(content: frame, completion: .contentProcessed { [weak self] _ in
      self?.close()
      if let clientId = self?.clientId {
        self?.onClose(clientId)
      }
    })
  }

  private func sendPong(payload: Data) {
    let frame = encodeFrame(opcode: .pong, payload: payload)
    connection.send(content: frame, completion: .contentProcessed { _ in })
  }

  private func jsonString(_ dict: [String: Any]) -> String {
    guard let data = try? JSONSerialization.data(withJSONObject: dict),
          let str = String(data: data, encoding: .utf8) else {
      return "{}"
    }
    return str
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

// MARK: - Parsed Frame

private struct ParsedFrame {
  let fin: Bool
  let opcode: WebSocketOpcode
  let payload: Data
}

// MARK: - WebSocket Opcodes

private enum WebSocketOpcode: UInt8 {
  case continuation = 0x0
  case text = 0x1
  case binary = 0x2
  case close = 0x8
  case ping = 0x9
  case pong = 0xA
}
