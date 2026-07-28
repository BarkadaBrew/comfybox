// AgeFloorTests.swift — the render-boundary age floor (2026-07-28).
import XCTest
@testable import ZImage

final class AgeFloorTests: XCTestCase {
  func testRewritesSubEighteenAges() {
    XCTAssertEqual(AgeFloor.enforce("a petite 17 year old filipina, hips rolling"),
                   "a petite 18-year-old filipina, hips rolling")
    XCTAssertEqual(AgeFloor.enforce("a 16-year-old by the window"), "a 18-year-old by the window")
    XCTAssertEqual(AgeFloor.enforce("seventeen-year-old dancer"), "18-year-old dancer")
    XCTAssertEqual(AgeFloor.enforce("aged 15, smiling"), "age 18, smiling")
  }

  func testScrubsCategoricalMinorTerms() {
    XCTAssertEqual(AgeFloor.enforce("a teen in the kitchen"), "a 18-year-old woman in the kitchen")
    XCTAssertEqual(AgeFloor.enforce("barely legal look"), "adult look")
    XCTAssertFalse(AgeFloor.enforce("underage preteen loli").lowercased().contains("loli"))
  }

  func testLeavesAdultTextUntouched() {
    for ok in [
      "an 18-year-old Filipina, hips rolling",
      "a 25 year old woman reading",
      "still life of pears in a ceramic bowl, deep chiaroscuro",
      "a 60-year-old man with a salt-and-pepper beard",
      "teal linen tablecloth",
    ] {
      XCTAssertEqual(AgeFloor.enforce(ok), ok)
      XCTAssertFalse(AgeFloor.violates(ok), ok)
    }
  }

  func testViolatesFlagsExactly() {
    XCTAssertTrue(AgeFloor.violates("a petite 17 year old"))
    XCTAssertFalse(AgeFloor.violates("a petite 18 year old"))
  }
}
