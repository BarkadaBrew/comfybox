import XCTest

@testable import ZImage

/// WP-E4 (FDD-krea2-raw-recipe §3.4, AC-17) — no phantom names anywhere in
/// `/object_info`. Every `sampler_name` and every `scheduler` option list
/// resolves without throwing, `uni_pc` / `dpmpp_2m_sde` are gone, and the
/// union of the advertised scheduler lists equals `SigmaScheduleKind.allCases`
/// ∪ the declared aliases — so a new enum case (bong_tangent, WP-E11) is
/// advertised the commit it resolves, and never before.
final class ComfyBridgeObjectInfoSamplerTests: XCTestCase {

  private struct OptionLists {
    var samplers: [String: [String]] = [:]   // node → list
    var schedulers: [String: [String]] = [:]
  }

  /// Walk every node's required + optional inputs for `sampler_name` /
  /// `scheduler` option lists.
  private func collect() -> OptionLists {
    let info = ComfyBridgeObjectInfo.build()
    var lists = OptionLists()
    for (node, definition) in info {
      guard let def = definition as? [String: Any], let input = def["input"] as? [String: Any] else { continue }
      for section in ["required", "optional"] {
        guard let raw = input[section] else { continue }
        let entries: [(String, Any)]
        if let dict = raw as? [String: Any] {
          entries = dict.map { ($0.key, $0.value) }
        } else if let ordered = raw as? OrderedDict {
          entries = ordered.entries
        } else {
          continue
        }
        for (key, value) in entries {
          guard let arr = value as? [Any], let options = arr.first as? [String] else { continue }
          if key == "sampler_name" { lists.samplers[node] = options }
          if key == "scheduler" { lists.schedulers[node] = options }
        }
      }
    }
    return lists
  }

  func testSamplerListsAreFoundAndResolve() throws {
    let lists = collect()
    XCTAssertEqual(Set(lists.samplers.keys), ["KSamplerSelect", "KSampler", "KSamplerAdvanced"])
    for (node, options) in lists.samplers {
      XCTAssertFalse(options.isEmpty, node)
      XCTAssertFalse(options.contains("uni_pc"), "\(node) still advertises uni_pc")
      XCTAssertFalse(options.contains("dpmpp_2m_sde"), "\(node) still advertises dpmpp_2m_sde")
      for name in options {
        XCTAssertNotNil(try RecipeNameResolver.resolveSchedulerKind(name), "\(node) advertises '\(name)' which does not resolve")
      }
      // Every SchedulerKind is reachable from the advertised list, so a new
      // sampler cannot land unadvertised.
      let reachable = Set(try options.compactMap { try RecipeNameResolver.resolveSchedulerKind($0) })
      XCTAssertEqual(reachable, Set(SchedulerKind.allCases), node)
    }
  }

  func testSchedulerListsAreFoundAndResolve() throws {
    let lists = collect()
    XCTAssertEqual(Set(lists.schedulers.keys), ["BasicScheduler", "KSampler", "KSamplerAdvanced"])
    var union = Set<String>()
    for (node, options) in lists.schedulers {
      for name in options {
        XCTAssertNotNil(try RecipeNameResolver.resolveSigmaScheduleKind(name), "\(node) advertises '\(name)' which does not resolve")
      }
      union.formUnion(options)
    }
    let expected = Set(SigmaScheduleKind.allCases.map(\.rawValue)).union(RecipeNameResolver.sigmaScheduleAliases.keys)
    XCTAssertEqual(union, expected)
    // The Krita defaults stay advertised.
    for kept in ["normal", "simple", "sgm_uniform", "ddim_uniform", "karras", "exponential", "beta", "beta57"] {
      XCTAssertTrue(union.contains(kept), "'\(kept)' must stay advertised")
    }
  }

  func testNoPhantomNamesInSerializedObjectInfo() throws {
    let data = try XCTUnwrap(orderedJSONData(ComfyBridgeObjectInfo.build()))
    let text = String(decoding: data, as: UTF8.self)
    XCTAssertFalse(text.contains("\"uni_pc\""))
    XCTAssertFalse(text.contains("\"dpmpp_2m_sde\""))
  }
}
