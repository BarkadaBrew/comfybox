import XCTest

@testable import ZImage
import Darwin

/// comfybox#153 — "is anything listening" probe. Per the burndown rules, no
/// test may bind :7870; every case here binds an ephemeral port (bind 0) and
/// lets the kernel assign one.
final class MCPPortProbeTests: XCTestCase {

  /// Binds and listens on an ephemeral loopback port. Returns the fd (caller
  /// must close it) and the assigned port.
  private func bindEphemeralListener() throws -> (fd: Int32, port: UInt16) {
    let fd = socket(AF_INET, SOCK_STREAM, 0)
    guard fd >= 0 else { throw XCTSkip("could not create a test socket") }

    var reuse: Int32 = 1
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuse, socklen_t(MemoryLayout<Int32>.size))

    var addr = sockaddr_in()
    addr.sin_family = sa_family_t(AF_INET)
    addr.sin_port = 0  // ask the kernel for a free ephemeral port
    addr.sin_addr.s_addr = inet_addr("127.0.0.1")

    let bindResult = withUnsafePointer(to: &addr) { ptr -> Int32 in
      ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockaddrPtr in
        Darwin.bind(fd, sockaddrPtr, socklen_t(MemoryLayout<sockaddr_in>.size))
      }
    }
    guard bindResult == 0 else {
      close(fd)
      throw XCTSkip("could not bind an ephemeral test port")
    }
    guard listen(fd, 1) == 0 else {
      close(fd)
      throw XCTSkip("could not listen on the ephemeral test port")
    }

    var assigned = sockaddr_in()
    var length = socklen_t(MemoryLayout<sockaddr_in>.size)
    _ = withUnsafeMutablePointer(to: &assigned) { ptr -> Int32 in
      ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockaddrPtr in
        getsockname(fd, sockaddrPtr, &length)
      }
    }
    return (fd, assigned.sin_port.byteSwapped)
  }

  func testOccupiedPortIsDetected() throws {
    let (fd, port) = try bindEphemeralListener()
    defer { close(fd) }

    XCTAssertTrue(MCPPortProbe.isOccupied(host: "127.0.0.1", port: port, timeoutMs: 500))
  }

  func testFreePortAfterListenerClosesIsNotOccupied() throws {
    let (fd, port) = try bindEphemeralListener()
    close(fd)  // release the port before probing

    XCTAssertFalse(MCPPortProbe.isOccupied(host: "127.0.0.1", port: port, timeoutMs: 500))
  }
}
