import XCTest

@testable import ZImage

/// FDD-glimmer-gpu-slot §3.1: leased top-priority inference slots.
final class InferenceSlotTableTests: XCTestCase {

  /// Mutable injected clock.
  final class Clock: @unchecked Sendable {
    private let lock = NSLock()
    private var t = Date(timeIntervalSince1970: 1_000_000)
    func now() -> Date { lock.lock(); defer { lock.unlock() }; return t }
    func advance(_ s: TimeInterval) { lock.lock(); t = t.addingTimeInterval(s); lock.unlock() }
  }

  func testAcquireGetRelease() {
    let table = InferenceSlotTable()
    XCTAssertFalse(table.isHeld())
    let slot = table.acquire(holder: "kira:chat", ttl: nil)
    XCTAssertEqual(slot.holder, "kira:chat")
    XCTAssertTrue(table.isHeld())
    XCTAssertEqual(table.get(id: slot.id)?.id, slot.id)
    XCTAssertEqual(table.activeSlots().count, 1)
    XCTAssertTrue(table.release(id: slot.id))
    XCTAssertFalse(table.release(id: slot.id), "second release is a no-op")
    XCTAssertFalse(table.isHeld())
    XCTAssertNil(table.get(id: slot.id))
  }

  func testTTLIsClamped() {
    let clock = Clock()
    let table = InferenceSlotTable(now: clock.now)
    let dflt = table.acquire(holder: "a", ttl: nil)
    XCTAssertEqual(dflt.expiresAt.timeIntervalSince(clock.now()), InferenceSlotTable.defaultTTL, accuracy: 0.001)
    let big = table.acquire(holder: "b", ttl: 5000)
    XCTAssertEqual(big.expiresAt.timeIntervalSince(clock.now()), InferenceSlotTable.maxTTL, accuracy: 0.001)
    let tiny = table.acquire(holder: "c", ttl: 0)
    XCTAssertEqual(tiny.expiresAt.timeIntervalSince(clock.now()), 1, accuracy: 0.001)
  }

  func testExpiryDropsTheSlot() {
    let clock = Clock()
    let table = InferenceSlotTable(now: clock.now)
    let slot = table.acquire(holder: "vision", ttl: 10)
    clock.advance(9)
    XCTAssertTrue(table.isHeld())
    clock.advance(2)
    XCTAssertNil(table.get(id: slot.id), "expired slots are invisible")
    XCTAssertFalse(table.isHeld())
    XCTAssertTrue(table.activeSlots().isEmpty)
  }

  func testRenewExtends() {
    let clock = Clock()
    let table = InferenceSlotTable(now: clock.now)
    let slot = table.acquire(holder: "chat", ttl: 10)
    clock.advance(8)
    let renewed = table.renew(id: slot.id, ttl: 30)
    XCTAssertEqual(renewed?.expiresAt.timeIntervalSince(clock.now()) ?? 0, 30, accuracy: 0.001)
    clock.advance(20)
    XCTAssertTrue(table.isHeld(), "renewal outlived the original ttl")
    XCTAssertNil(table.renew(id: "nope", ttl: 30))
  }

  func testOnEmptyFiresOncePerEmptying() {
    let clock = Clock()
    let table = InferenceSlotTable(now: clock.now)
    let count = LockedCounter()
    table.onEmpty = { count.increment() }
    let a = table.acquire(holder: "a", ttl: 60)
    let b = table.acquire(holder: "b", ttl: 5)
    XCTAssertTrue(table.release(id: a.id))
    XCTAssertEqual(count.value, 0, "still one slot held")
    clock.advance(6)
    table.sweep()
    XCTAssertEqual(count.value, 1, "expiry of the last slot empties the table")
    table.sweep()
    XCTAssertEqual(count.value, 1, "already empty: no second fire")
    _ = b
    let c = table.acquire(holder: "c", ttl: 60)
    XCTAssertTrue(table.release(id: c.id))
    XCTAssertEqual(count.value, 2)
  }

  func testWaitUntilFreeReturnsImmediatelyWhenEmpty() async {
    let table = InferenceSlotTable()
    await table.waitUntilFree()
  }

  func testWaitUntilFreeResumesOnLastRelease() async throws {
    let table = InferenceSlotTable()
    let a = table.acquire(holder: "a", ttl: 60)
    let b = table.acquire(holder: "b", ttl: 60)
    let freed = LockedCounter()
    let waiter = Task { await table.waitUntilFree(); freed.increment() }
    try await Task.sleep(nanoseconds: 50_000_000)
    XCTAssertTrue(table.release(id: a.id))
    try await Task.sleep(nanoseconds: 50_000_000)
    XCTAssertEqual(freed.value, 0, "one slot still held")
    XCTAssertTrue(table.release(id: b.id))
    await waiter.value
    XCTAssertEqual(freed.value, 1)
  }

  func testWaitUntilFreeHonoursCancellation() async throws {
    let table = InferenceSlotTable()
    _ = table.acquire(holder: "a", ttl: 60)
    let waiter = Task { await table.waitUntilFree() }
    try await Task.sleep(nanoseconds: 50_000_000)
    waiter.cancel()
    await waiter.value
    XCTAssertTrue(table.isHeld(), "cancelling a waiter does not release anyone's slot")
  }
}

final class LockedCounter: @unchecked Sendable {
  private let lock = NSLock()
  private var n = 0
  func increment() { lock.lock(); n += 1; lock.unlock() }
  var value: Int { lock.lock(); defer { lock.unlock() }; return n }
}
