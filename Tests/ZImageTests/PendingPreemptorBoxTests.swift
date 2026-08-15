import XCTest
@testable import ZImage

/// #1479 (final review, finding I1): the pending-preemptor slot is claimed by
/// two racing parties — the coordinator, which claims whatever is parked, and
/// a per-episode timeout watchdog, which must only ever claim the entry ITS
/// OWN episode armed.
///
/// The bug these tests pin: episode A's watchdog is still asleep when A's
/// episode completes and a LATER image job (B) arms the slot and raises its
/// own preemption signal. With an unqualified claim, A's stale watchdog woke
/// up, took B's entry, cleared B's raise and B's in-flight flag, and ran B
/// unpreempted — a silent degradation with B's continuation still outstanding.
/// The episode token makes that claim fail instead.
///
/// Exercised through `TokenedSlot<Int>` rather than the live
/// `PendingPreemptorBox` (`= TokenedSlot<PendingPreemptor>`) so the claim
/// semantics can be tested without a real `GeneratePayload` + continuation
/// pair; it is the same generic code path either way.
final class PendingPreemptorBoxTests: XCTestCase {
  func testUnqualifiedClaimIsExactlyOnce() {
    let box = TokenedSlot<Int>()
    box.set(7)
    XCTAssertEqual(box.claim(), 7)
    XCTAssertNil(box.claim())
  }

  func testTokenedClaimSucceedsForItsOwnEntry() {
    let box = TokenedSlot<Int>()
    let token = box.set(7)
    XCTAssertEqual(box.claim(matching: token), 7)
    XCTAssertNil(box.claim(matching: token), "a tokened claim is still exactly-once")
  }

  /// The finding itself: A's watchdog must not be able to claim B's entry.
  func testStaleWatchdogTokenCannotClaimALaterEpisodesEntry() {
    let box = TokenedSlot<Int>()
    let tokenA = box.set(1)      // episode A arms
    XCTAssertEqual(box.claim(), 1)  // A's coordinator claims it — episode A over
    let tokenB = box.set(2)      // episode B arms
    XCTAssertNotEqual(tokenA, tokenB)

    // A's watchdog finally wakes up. It must NOT get B's entry.
    XCTAssertNil(box.claim(matching: tokenA))
    // ...and B's entry is untouched, still claimable by B's own parties.
    XCTAssertEqual(box.claim(matching: tokenB), 2)
  }

  /// Same hijack, without A's entry ever having been claimed: B simply
  /// overwrites the slot (the coordinator never observed A's yield).
  func testTokenedClaimFailsAfterTheSlotIsRearmed() {
    let box = TokenedSlot<Int>()
    let tokenA = box.set(1)
    let tokenB = box.set(2)
    XCTAssertNil(box.claim(matching: tokenA))
    XCTAssertEqual(box.claim(matching: tokenB), 2)
  }

  func testTokenedClaimFailsOnAnEmptySlot() {
    let box = TokenedSlot<Int>()
    XCTAssertNil(box.claim(matching: 1))
    XCTAssertNil(box.claim())
  }

  /// Tokens never repeat within a process, so a token can only ever identify
  /// the one entry it was minted for.
  func testTokensAreUniquePerSet() {
    let box = TokenedSlot<Int>()
    var seen = Set<UInt64>()
    for i in 0..<100 {
      let t = box.set(i)
      XCTAssertTrue(seen.insert(t).inserted, "token \(t) was reused")
    }
  }

  /// The coordinator's unqualified claim always wins over a still-sleeping
  /// watchdog for the SAME episode — exactly-once across both entry points.
  func testCoordinatorClaimBeatsItsOwnWatchdog() {
    let box = TokenedSlot<Int>()
    let token = box.set(9)
    XCTAssertEqual(box.claim(), 9)
    XCTAssertNil(box.claim(matching: token))
  }

  func testConcurrentClaimsYieldExactlyOneWinner() {
    let box = TokenedSlot<Int>()
    let token = box.set(42)
    let winners = NSMutableArray()
    let lock = NSLock()
    DispatchQueue.concurrentPerform(iterations: 32) { i in
      let claimed = i.isMultiple(of: 2) ? box.claim() : box.claim(matching: token)
      if let claimed {
        lock.lock(); winners.add(claimed); lock.unlock()
      }
    }
    XCTAssertEqual(winners.count, 1)
  }
}
