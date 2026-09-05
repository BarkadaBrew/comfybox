// QueryDecodingVectorsTests.swift — comfybox#387.
//
// The catalog's own query decoder (`HTTPKit.queryParameters`) used to treat a
// literal `+` as a space (`application/x-www-form-urlencoded` behavior) while
// the engine's `HTTPRequest.queryParameters` (comfybox#380/#381, pinned in
// `QueryParametersPercentDecodingTests.swift` in this same directory) decodes
// per RFC 3986, where `+` stays literal. Ruling on #387: one convention for
// both services — RFC 3986, matching the engine.
//
// TWIN FILE — keep byte-for-byte identical (except this header comment and
// the target-specific decode call at the bottom):
//   Tests/ComfyBoxCatalogTests/QueryDecodingVectorsTests.swift
//
// `ZImageTests` and `ComfyBoxCatalogTests` share no common module — `ZImage`
// and `ComfyBoxCatalog` do not depend on each other — so this vector table
// cannot be a single `import`ed source file. It is duplicated verbatim
// instead. `testVectorTableHashMatchesItsTwin` below is the drift guard: it
// hashes this file's table with a hard-coded literal that is copied
// identically into the twin. Change the vectors in ONE of these files
// without updating the other (table AND hash literal, in both places) and
// that file's hash test fails.

import XCTest
@testable import ZImage

struct QueryDecodingVector {
    let name: String
    let query: String
    let key: String
    let expected: String
}

/// One vector per case in the #387 ruling: space, `#`, `%25`, non-ASCII, `+`
/// (both percent-encoded and literal), malformed escape, and key decoding.
let queryDecodingVectors: [QueryDecodingVector] = [
    .init(name: "space", query: "model=a%20b", key: "model", expected: "a b"),
    .init(name: "hash", query: "model=track%231.safetensors", key: "model", expected: "track#1.safetensors"),
    .init(name: "percent25", query: "model=100%25off.safetensors", key: "model", expected: "100%off.safetensors"),
    .init(name: "nonASCII", query: "model=caf%C3%A9.safetensors", key: "model", expected: "café.safetensors"),
    .init(name: "plusEncoded", query: "model=z-image%2Bextra.safetensors", key: "model",
          expected: "z-image+extra.safetensors"),
    .init(name: "plusLiteral", query: "model=a+b", key: "model", expected: "a+b"),
    .init(name: "keyDecoding", query: "mod%65l=krea2", key: "model", expected: "krea2"),
    .init(name: "malformedTrailingPercent", query: "model=abc%", key: "model", expected: "abc%"),
    .init(name: "malformedNonHex", query: "model=50%zz", key: "model", expected: "50%zz"),
]

/// Deterministic FNV-1a 64-bit over the table's canonical text form. NOT
/// Swift's `Hashable`/`hashValue` — those are seeded per process (hash
/// randomization) and would differ between the two test binaries even for
/// byte-identical content, which would defeat the whole point of this check.
func hashQueryDecodingVectors(_ vectors: [QueryDecodingVector]) -> UInt64 {
    var hash: UInt64 = 0xcbf29ce484222325
    let prime: UInt64 = 0x100000001b3
    for v in vectors {
        for field in [v.name, v.query, v.key, v.expected] {
            for byte in field.utf8 {
                hash ^= UInt64(byte)
                hash = hash &* prime
            }
            hash ^= 0xFF
            hash = hash &* prime
        }
    }
    return hash
}

final class QueryDecodingVectorsTests: XCTestCase {

    /// The drift guard. This literal must be identical to the one in the twin
    /// file named above. Recompute (see the twin's comment) and update BOTH
    /// literals together whenever the table changes.
    func testVectorTableHashMatchesItsTwin() {
        XCTAssertEqual(
            hashQueryDecodingVectors(queryDecodingVectors), 0x9895a01037f4c6ec,
            "queryDecodingVectors diverged from its twin in "
            + "Tests/ComfyBoxCatalogTests/QueryDecodingVectorsTests.swift — "
            + "update both tables and both hash literals together")
    }

    /// The engine side of the shared contract: `HTTPRequest.queryParameters`
    /// must decode every vector exactly like the catalog's
    /// `HTTPKit.queryParameters` (pinned on its own copy of this table).
    func testHTTPRequestDecodesEveryVector() {
        for v in queryDecodingVectors {
            let request = HTTPRequest(method: "GET", path: "/x", queryString: v.query, headers: [:], body: Data())
            XCTAssertEqual(request.queryParameters[v.key], v.expected, v.name)
        }
    }
}
