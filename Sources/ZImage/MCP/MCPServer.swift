// MCPServer.swift — Main MCP server orchestrator
//
// Combines transport (stdio), registry (tool catalog), and executor (HTTP proxy).
// Handles the full MCP protocol flow:
//   initialize -> initialized notification -> tools/list -> tools/call
//
// Reads JSON-RPC from stdin, writes responses to stdout, logs to stderr.
// Blocks on stdin until it closes or the process is interrupted.

import Foundation

/// MCP server that bridges stdio JSON-RPC 2.0 to WarmServer HTTP API.
public final class MCPServer {
  private let client: WarmServerClient
  private let executor: MCPToolExecutor

  /// Whether the server has been initialized (client sent initialize request).
  private var initialized = false

  /// Server version reported in initialize response.
  public static let version = "1.0.0"

  public init(host: String = "127.0.0.1", port: UInt16 = 7870) {
    self.client = WarmServerClient(host: host, port: port)
    self.executor = MCPToolExecutor(client: client)
  }

  /// Run the MCP server — blocks on stdin until it closes.
  public func run() {
    startParentDeathWatchdog()
    log("Starting — bridging to WarmServer at \(client.host):\(client.port)")

    let decoder = JSONDecoder()

    while let line = readLine(strippingNewline: true) {
      guard !line.isEmpty else { continue }

      guard let data = line.data(using: .utf8) else {
        log("Invalid UTF-8 input")
        continue
      }

      // Parse JSON-RPC request
      let request: MCPRequest
      do {
        request = try decoder.decode(MCPRequest.self, from: data)
      } catch {
        log("Invalid JSON: \(line.prefix(200)) — \(error)")
        // Per JSON-RPC spec, respond with parse error
        sendError(id: nil, error: .parseError)
        continue
      }

      if let id = request.id {
        // Request — dispatch and respond synchronously.
        // We use a semaphore to bridge async tool execution into the
        // synchronous readline loop. This is intentional: MCP over stdio
        // is inherently serial (one request at a time).
        let semaphore = DispatchSemaphore(value: 0)
        var response: MCPResponse?
        Task {
          response = await handleRequest(id: id, method: request.method, params: request.params)
          semaphore.signal()
        }
        semaphore.wait()
        if let resp = response {
          send(resp)
        }
      } else {
        // Notification — no response needed
        handleNotification(method: request.method)
      }
    }

    log("stdin closed — shutting down")
  }

  // MARK: - Request Handling

  private func handleRequest(id: MCPRequestId, method: String, params: MCPParams?) async -> MCPResponse {
    switch method {
    case MCPMethod.initialize:
      return handleInitialize(id: id, params: params)

    case MCPMethod.toolsList:
      return handleToolsList(id: id)

    case MCPMethod.toolsCall:
      return await handleToolsCall(id: id, params: params)

    case MCPMethod.ping:
      return MCPResponse(id: id, result: AnyCodable([:] as [String: Any]))

    default:
      log("Unknown method: \(method)")
      return MCPResponse(id: id, error: .methodNotFound)
    }
  }

  // MARK: - Initialize

  private func handleInitialize(id: MCPRequestId, params: MCPParams?) -> MCPResponse {
    initialized = true

    let clientName = params?.dict("clientInfo")?["name"]?.stringValue ?? "unknown"
    let protocolVersion = params?.string("protocolVersion") ?? "2024-11-05"
    log("Client connected: \(clientName) (protocol \(protocolVersion))")

    let result: [String: Any] = [
      "protocolVersion": "2024-11-05",
      "capabilities": [
        "tools": [
          "listChanged": false,
        ] as [String: Any],
      ] as [String: Any],
      "serverInfo": [
        "name": "comfybox",
        "version": MCPServer.version,
      ] as [String: Any],
    ]

    return MCPResponse(id: id, result: AnyCodable(result))
  }

  // MARK: - Tools List

  private func handleToolsList(id: MCPRequestId) -> MCPResponse {
    // Build tools list using JSONSerialization to ensure clean JSON output.
    // MCPToolDefinition stores inputSchema as [String: Any] which serializes
    // cleanly via JSONSerialization without AnyCodable debug descriptions.
    var toolDicts: [[String: Any]] = []
    for tool in MCPToolRegistry.tools {
      toolDicts.append([
        "name": tool.name,
        "description": tool.description,
        "inputSchema": tool.inputSchema,
      ] as [String: Any])
    }
    let result: [String: Any] = ["tools": toolDicts]
    return MCPResponse(id: id, result: AnyCodable(result))
  }

  // MARK: - Tools Call

  private func handleToolsCall(id: MCPRequestId, params: MCPParams?) async -> MCPResponse {
    guard let toolName = params?.string("name") else {
      return MCPResponse(id: id, error: MCPError.invalidParams)
    }

    // Verify tool exists in the registry
    guard MCPToolRegistry.tool(named: toolName) != nil else {
      return MCPResponse(id: id, error: MCPError(code: -32602, message: "Unknown tool: \(toolName)"))
    }

    // Extract tool arguments from params.arguments
    let arguments: MCPParams?
    if let argsDict = params?.dict("arguments") {
      arguments = MCPParams(argsDict)
    } else {
      arguments = nil
    }

    log("Executing tool: \(toolName)")
    let result = await executor.execute(name: toolName, arguments: arguments)
    log("Tool \(toolName) completed (isError=\(result.isError))")

    return MCPResponse(id: id, result: AnyCodable(result.toResponseDict()))
  }

  // MARK: - Notifications

  private func handleNotification(method: String) {
    switch method {
    case MCPMethod.notificationsInitialized:
      log("Client initialized")
    default:
      log("Notification: \(method)")
    }
  }

  // MARK: - Transport

  /// Send a JSON-RPC response to stdout using JSONSerialization for reliable output.
  private func send(_ response: MCPResponse) {
    // Build the response dict manually to avoid AnyCodable encoding issues.
    var dict: [String: Any] = ["jsonrpc": response.jsonrpc]

    if let id = response.id {
      switch id {
      case .integer(let i): dict["id"] = i
      case .string(let s): dict["id"] = s
      }
    }

    if let result = response.result {
      dict["result"] = result.rawValue
    }

    if let error = response.error {
      var errorDict: [String: Any] = [
        "code": error.code,
        "message": error.message,
      ]
      if let data = error.data {
        errorDict["data"] = data.rawValue
      }
      dict["error"] = errorDict
    }

    guard let data = try? JSONSerialization.data(withJSONObject: dict, options: [.sortedKeys]),
          let line = String(data: data, encoding: .utf8) else {
      log("Failed to encode response")
      return
    }
    // Write to stdout with newline, then flush
    print(line)
    fflush(stdout)
  }

  /// Send an error response.
  private func sendError(id: MCPRequestId?, error: MCPError) {
    let response = MCPResponse(id: id, error: error)
    send(response)
  }

  /// Log a message to stderr (stdout is reserved for JSON-RPC).
  private func log(_ message: String) {
    let logLine = "[mcp-server:comfybox] \(message)\n"
    FileHandle.standardError.write(Data(logLine.utf8))
  }
  // MARK: - Parent Death Watchdog

  /// Start a background watchdog that exits the process when the parent dies.
  ///
  /// When the parent process (SSH session, Claude.app, etc.) exits, macOS
  /// reparents this process to launchd (pid 1). Without this watchdog, the
  /// MCP server keeps running indefinitely as an orphan — blocked on stdin
  /// with no client to serve — leaking memory and file descriptors.
  ///
  /// Polls `getppid()` every 5 seconds. When the parent PID changes (parent
  /// died → reparented to pid 1), the process exits cleanly.
  private func startParentDeathWatchdog() {
    let parentPid = getppid()
    guard parentPid > 1 else {
      // Already orphaned (parent is launchd) — no client to serve. Exit
      // immediately rather than blocking on stdin with no one reading.
      log("Parent is launchd (pid 1) at startup — no parent, exiting")
      _exit(0)
    }

    DispatchQueue.global(qos: .utility).async {
      while true {
        sleep(5)
        if getppid() != parentPid {
          // Use _exit (not exit) to avoid atexit/stdio flush racing with
          // the main thread's readLine(). Use raw write(2) for the same
          // reason — FileHandle.standardError is not thread-safe with the
          // main thread's log() calls.
          let msg = "[mcp-server:comfybox] Parent process exited — shutting down\n"
          msg.withCString { ptr in
            _ = write(STDERR_FILENO, ptr, msg.utf8.count)
          }
          _exit(0)
        }
      }
    }
  }

}
