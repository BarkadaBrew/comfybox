// HTTPKit.swift — the smallest HTTP/1.1 server that serves this catalog.
//
// Deliberately hand-rolled rather than reusing the engine's WarmServer: this
// library must not depend on ZImage/MLX, so the gallery process stays small,
// starts instantly, and can be rebuilt without touching the engine binary.

import Foundation
import Network

public enum HTTPKit {

    /// Hard cap on a single request. The routes take short query strings and, at
    /// most, a small JSON body; anything larger is a client that has lost its
    /// place, and buffering it would be a way to grow the process without bound.
    public static let maxRequestBytes = 1 << 20

    /// A connection that never finishes its request must not hold a buffer (and
    /// a file descriptor) forever. Loopback peers are fast; 15s is generous.
    static let requestTimeout: TimeInterval = 15

    public struct Request: Sendable {
        public let method: String
        public let target: String
        /// Header names are lowercased on parse.
        public let headers: [String: String]
        public let body: Data

        public init(method: String, target: String, headers: [String: String], body: Data) {
            self.method = method; self.target = target
            self.headers = headers.reduce(into: [:]) { $0[$1.key.lowercased()] = $1.value }
            self.body = body
        }

        public var path: String { target.split(separator: "?", maxSplits: 1).first.map(String.init) ?? target }
        public var query: [String: String] { HTTPKit.queryParameters(of: target) }
    }

    public struct Response: Sendable {
        public let status: Int
        public let contentType: String
        public let body: Data

        public init(status: Int, contentType: String = "application/json", body: Data) {
            self.status = status; self.contentType = contentType; self.body = body
        }

        /// Serialisation failure becomes a 500, never a 200 with an empty body:
        /// a caller must not read "no results" out of "the server could not say".
        public static func json(_ object: Any, status: Int = 200) -> Response {
            guard JSONSerialization.isValidJSONObject(object),
                  let data = try? JSONSerialization.data(withJSONObject: object) else {
                return Response(status: 500, body: Data(#"{"error":"unserializable response"}"#.utf8))
            }
            return Response(status: status, body: data)
        }

        public static func error(_ status: Int, _ message: String) -> Response {
            json(["error": message], status: status)
        }

        var wireData: Data {
            var head = "HTTP/1.1 \(status) \(Response.reason(status))\r\n"
            head += "Content-Type: \(contentType)\r\n"
            head += "Content-Length: \(body.count)\r\n"
            head += "Connection: close\r\n\r\n"
            return Data(head.utf8) + body
        }

        static func reason(_ status: Int) -> String {
            switch status {
            case 200: return "OK"
            case 400: return "Bad Request"
            case 403: return "Forbidden"
            case 404: return "Not Found"
            case 413: return "Payload Too Large"
            case 500: return "Internal Server Error"
            default: return "Error"
            }
        }
    }

    /// Parse `?a=1&b=two%20words`. A key with no `=` yields "".
    public static func queryParameters(of target: String) -> [String: String] {
        guard let qIndex = target.firstIndex(of: "?") else { return [:] }
        let raw = String(target[target.index(after: qIndex)...])
        var out: [String: String] = [:]
        for pair in raw.split(separator: "&") {
            let parts = pair.split(separator: "=", maxSplits: 1, omittingEmptySubsequences: false)
            let key = String(parts[0]).removingPercentEncoding ?? String(parts[0])
            let value = parts.count > 1
                ? (String(parts[1]).replacingOccurrences(of: "+", with: " ").removingPercentEncoding ?? "")
                : ""
            out[key] = value
        }
        return out
    }

    /// Parse a request out of a buffer that may still be arriving.
    ///
    /// Returns nil until the message is COMPLETE — headers terminated by a blank
    /// line and a body at least as long as any declared Content-Length. A single
    /// `NWConnection.receive` is a read, not a message boundary: treating one as
    /// the other drops requests that happen to straddle two TCP segments, which
    /// is exactly the kind of failure that only shows up under load.
    public static func parseComplete(_ data: Data) -> Request? {
        guard let text = String(data: data, encoding: .utf8),
              let headerEnd = text.range(of: "\r\n\r\n") else { return nil }
        let headLines = text[..<headerEnd.lowerBound].components(separatedBy: "\r\n")
        guard let requestLine = headLines.first else { return nil }
        let parts = requestLine.split(separator: " ")
        guard parts.count >= 2 else { return nil }

        var headers: [String: String] = [:]
        for line in headLines.dropFirst() {
            guard let colon = line.firstIndex(of: ":") else { continue }
            let name = String(line[..<colon]).trimmingCharacters(in: .whitespaces).lowercased()
            let value = String(line[line.index(after: colon)...]).trimmingCharacters(in: .whitespaces)
            headers[name] = value
        }

        var body = Data(text[headerEnd.upperBound...].utf8)
        // `declared >= 0`: `Content-Length: -1` is a client that is lying, and
        // `prefix(-1)` is a precondition failure — a one-header way to kill the
        // process. A negative length declares nothing, so nothing is enforced.
        if let declared = headers["content-length"].flatMap(Int.init), declared >= 0 {
            if body.count < declared { return nil }   // still arriving
            // …and TRUNCATE an overshoot. Content-Length is the message
            // boundary, not a minimum: anything past it belongs to whatever the
            // peer sent next, and while no route read a body it did not matter.
            // Now that POST /v1/catalog/file parses this as JSON, handing the
            // parser those trailing bytes turns a perfectly valid request into a
            // 400 that the caller cannot see the cause of.
            // Re-based, not merely sliced: a `Data` slice keeps the parent's
            // indices, and a consumer that reaches for `body[0]` traps.
            if body.count > declared { body = Data(body.prefix(declared)) }
        }
        return Request(method: String(parts[0]), target: String(parts[1]),
                       headers: headers, body: body)
    }

    public enum ServerError: Error, LocalizedError {
        case invalidPort(UInt16)
        case alreadyStarted
        case bindFailed(String)
        case notReadyInTime(TimeInterval)

        public var errorDescription: String? {
            switch self {
            case let .invalidPort(p): return "invalid port \(p)"
            case .alreadyStarted: return "server is already started"
            case let .bindFailed(m): return "listener failed: \(m)"
            case let .notReadyInTime(t): return "listener not ready within \(t)s"
            }
        }
    }

    /// A continuation must be resumed exactly once, and `stateUpdateHandler`
    /// plus the timeout can both fire.
    final class Once: @unchecked Sendable {
        private let lock = NSLock()
        private var done = false
        func run(_ body: () -> Void) {
            lock.lock()
            if done { lock.unlock(); return }
            done = true
            lock.unlock()
            body()
        }
    }

    /// Bound to loopback only. The gallery holds raw prompt text under the
    /// provenance contract; it is not a LAN service.
    public final class Server: @unchecked Sendable {
        private let requestedPort: UInt16
        private let handler: @Sendable (Request) async -> Response

        /// `listener` and `started` are touched from the caller's thread, the
        /// listener's queue and `boundPort` readers, so they are guarded rather
        /// than merely `@unchecked`.
        private let lock = NSLock()
        private var listener: NWListener?
        private var started = false

        public init(port: UInt16, handler: @escaping @Sendable (Request) async -> Response) {
            self.requestedPort = port
            self.handler = handler
        }

        /// Bind and WAIT for the listener to reach `.ready`, surfacing `.failed`
        /// to the caller.
        ///
        /// `NWListener.start(queue:)` is non-throwing and reports its outcome
        /// only through `stateUpdateHandler`. Without this, an `EADDRINUSE` —
        /// the exact case an overlapping launchd restart produces — left the
        /// listener `.failed` with nobody watching, while the caller went on to
        /// log "listening" and block forever. A service that cannot bind must
        /// fail loudly enough for launchd to recycle it, not sit there healthy
        /// and deaf.
        ///
        /// Pass port 0 to let the kernel choose; read `boundPort` afterwards.
        public func start(timeout: TimeInterval = 5) async throws {
            lock.lock()
            if started { lock.unlock(); throw ServerError.alreadyStarted }
            started = true
            lock.unlock()

            do {
                try await bind(timeout: timeout)
            } catch {
                lock.lock(); started = false; let l = listener; listener = nil; lock.unlock()
                l?.cancel()
                throw error
            }
        }

        private func bind(timeout: TimeInterval) async throws {
            guard let nwPort = NWEndpoint.Port(rawValue: requestedPort) else {
                throw ServerError.invalidPort(requestedPort)
            }
            let params = NWParameters.tcp
            // The single line that makes this not a LAN service.
            params.requiredLocalEndpoint = NWEndpoint.hostPort(host: "127.0.0.1", port: nwPort)
            // A restart within TIME_WAIT must not fail to bind; launchd restarts
            // this process and a two-minute dead window is not acceptable.
            params.allowLocalEndpointReuse = true
            let l = try NWListener(using: params)

            lock.lock(); listener = l; lock.unlock()

            let once = Once()
            try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
                l.stateUpdateHandler = { state in
                    switch state {
                    case .ready:
                        once.run { cont.resume() }
                    case let .failed(error):
                        once.run { cont.resume(throwing: ServerError.bindFailed("\(error)")) }
                    case let .waiting(error):
                        // For a loopback listener `.waiting` means the address is
                        // unavailable — in practice the port is taken. It will
                        // retry forever, which is exactly the silent hang.
                        once.run { cont.resume(throwing: ServerError.bindFailed("\(error)")) }
                    case .cancelled:
                        once.run { cont.resume(throwing: ServerError.bindFailed("cancelled")) }
                    default:
                        break
                    }
                }
                // STRONG capture, deliberately. NWListener is retained by the
                // network framework once started, so with `[weak self]` a caller
                // who lets the Server go out of scope — which the CLI entry point
                // did — is left with a socket that still ACCEPTS and binds, and
                // silently discards every request. A listening port that answers
                // nothing is the worst failure available here, because every
                // health check passes. A started server owns itself until
                // `stop()`; the cycle is the point.
                l.newConnectionHandler = { conn in self.accept(conn) }
                l.start(queue: .global(qos: .userInitiated))

                DispatchQueue.global(qos: .utility).asyncAfter(deadline: .now() + timeout) {
                    once.run { cont.resume(throwing: ServerError.notReadyInTime(timeout)) }
                }
            }
        }

        /// Cancel FIRST, then replace the handler with one that closes anything
        /// still arriving. Assigning nil to `newConnectionHandler` on a live
        /// listener has undocumented semantics; `{ $0.cancel() }` breaks the
        /// self-reference just as well and is defined behaviour either way.
        public func stop() {
            lock.lock()
            let l = listener
            listener = nil
            started = false
            lock.unlock()

            l?.cancel()
            l?.newConnectionHandler = { $0.cancel() }
            l?.stateUpdateHandler = nil
        }

        /// The port actually bound. Valid once `start()` has returned; nil before.
        public var boundPort: UInt16? {
            lock.lock(); defer { lock.unlock() }
            return listener?.port?.rawValue
        }

        private func accept(_ conn: NWConnection) {
            conn.start(queue: .global(qos: .userInitiated))
            DispatchQueue.global(qos: .utility)
                .asyncAfter(deadline: .now() + HTTPKit.requestTimeout) { conn.cancel() }
            read(conn, buffer: Data())
        }

        private func read(_ conn: NWConnection, buffer: Data) {
            conn.receive(minimumIncompleteLength: 1, maximumLength: 1 << 16) {
                [weak self] chunk, _, isComplete, error in
                guard let self, error == nil else { conn.cancel(); return }
                var buffer = buffer
                if let chunk { buffer.append(chunk) }

                if buffer.count > HTTPKit.maxRequestBytes {
                    conn.send(content: Response.error(413, "request too large").wireData,
                              completion: .contentProcessed { _ in conn.cancel() })
                    return
                }
                if let req = HTTPKit.parseComplete(buffer) {
                    Task {
                        let res = await self.handler(req)
                        conn.send(content: res.wireData,
                                  completion: .contentProcessed { _ in conn.cancel() })
                    }
                    return
                }
                guard !isComplete else { conn.cancel(); return }   // peer stopped mid-request
                self.read(conn, buffer: buffer)
            }
        }
    }
}
