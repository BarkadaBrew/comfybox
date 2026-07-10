// ModelReferenceResolverTests.swift — Gallery → Generate model matching
// (coffeeshop-server#1180)

import Testing
@testable import ComfyBoxDesktop

@Suite("ModelReferenceResolver")
struct ModelReferenceResolverTests {
  @Test("matches the currently active model by basename, ignoring path/extension/case")
  func matchesActiveByBasename() {
    let result = ModelReferenceResolver.resolve(
      "cyberrealisticZImage_v50",
      currentModel: "/Users/todd/Models-working/cyberrealistic-z-image/cyberrealisticZImage_v50.safetensors",
      availableModels: []
    )
    #expect(result == .alreadyActive)
  }

  @Test("matches the currently active model when it's already a bare spec like krea2")
  func matchesActiveBareSpec() {
    let result = ModelReferenceResolver.resolve(
      "krea-2-turbo", currentModel: "krea-2-turbo", availableModels: []
    )
    #expect(result == .alreadyActive)
  }

  @Test("resolves against the catalog by id when not currently active")
  func resolvesAgainstCatalogById() {
    let result = ModelReferenceResolver.resolve(
      "krea2-turbo-q8", currentModel: "some-other-model",
      availableModels: [(id: "krea2-turbo-q8", displayName: "Krea-2-Turbo")]
    )
    #expect(result == .resolved("krea2-turbo-q8"))
  }

  @Test("resolves against the catalog by display name, ignoring hyphens/case")
  func resolvesAgainstCatalogByDisplayName() {
    let result = ModelReferenceResolver.resolve(
      "krea-2-turbo", currentModel: "some-other-model",
      availableModels: [(id: "krea2-turbo-q8", displayName: "Krea-2-Turbo")]
    )
    #expect(result == .resolved("krea2-turbo-q8"))
  }

  @Test("returns unresolved for an empty or unknown reference rather than guessing")
  func unresolvedForUnknownOrEmpty() {
    #expect(ModelReferenceResolver.resolve("", currentModel: "x", availableModels: []) == .unresolved)
    #expect(
      ModelReferenceResolver.resolve(
        "totally-unknown-model", currentModel: "x",
        availableModels: [(id: "krea2-turbo-q8", displayName: "Krea-2-Turbo")]
      ) == .unresolved
    )
  }

  @Test("nil currentModel doesn't crash and falls through to catalog matching")
  func nilCurrentModelFallsThroughToCatalog() {
    let result = ModelReferenceResolver.resolve(
      "krea2-turbo-q8", currentModel: nil,
      availableModels: [(id: "krea2-turbo-q8", displayName: "Krea-2-Turbo")]
    )
    #expect(result == .resolved("krea2-turbo-q8"))
  }
}
