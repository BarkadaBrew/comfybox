// MCPPortProbe.swift — "is anything listening on this port" for comfybox#153
//
// A raw, best-effort TCP connect check. No HTTP, no health semantics — the
// startup decision (MCPBridgeStartupPolicy) does not care WHAT is
// listening, only THAT something is: the bridge never starts a server, so
// an occupied port is always a reason to connect, healthy or not — the
// caller decides separately whether to report it as healthy by hitting
// /health after connecting.

import Foundation
#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

public enum MCPPortProbe {
  /// Returns true if a TCP connect to host:port succeeds within `timeoutMs`.
  /// Non-blocking connect + poll() so an unreachable/firewalled target can't
  /// hang the bridge's startup.
  public static func isOccupied(host: String = "127.0.0.1", port: UInt16, timeoutMs: Int32 = 250) -> Bool {
    let fd = socket(AF_INET, SOCK_STREAM, 0)
    guard fd >= 0 else { return false }
    defer { close(fd) }

    let flags = fcntl(fd, F_GETFL, 0)
    _ = fcntl(fd, F_SETFL, flags | O_NONBLOCK)

    var addr = sockaddr_in()
    addr.sin_family = sa_family_t(AF_INET)
    addr.sin_port = port.bigEndian
    guard inet_pton(AF_INET, host, &addr.sin_addr) == 1 else { return false }

    let connectResult = withUnsafePointer(to: &addr) { ptr -> Int32 in
      ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockaddrPtr in
        connect(fd, sockaddrPtr, socklen_t(MemoryLayout<sockaddr_in>.size))
      }
    }

    if connectResult == 0 {
      return true
    }
    guard errno == EINPROGRESS else {
      return false
    }

    var pfd = pollfd(fd: fd, events: Int16(POLLOUT), revents: 0)
    let pollResult = poll(&pfd, 1, timeoutMs)
    guard pollResult > 0, Int32(pfd.revents) & POLLOUT != 0 else {
      return false
    }

    var socketError: Int32 = 0
    var length = socklen_t(MemoryLayout<Int32>.size)
    guard getsockopt(fd, SOL_SOCKET, SO_ERROR, &socketError, &length) == 0 else {
      return false
    }
    return socketError == 0
  }
}
