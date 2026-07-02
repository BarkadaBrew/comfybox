// ProviderFormTests.swift — AI-provider Settings form ↔ registry mapping.

import Testing
import Foundation
import ZImage
@testable import ComfyBoxDesktop

@Suite("ProviderForm")
struct ProviderFormTests {

    @Test("empty form maps to no endpoint")
    func emptyIsNil() {
        #expect(EndpointForm().toEndpoint() == nil)
    }

    @Test("base URL without model is not a configured endpoint")
    func requiresBothUrlAndModel() {
        var f = EndpointForm()
        f.baseUrl = "http://localhost:1234/v1"
        #expect(f.toEndpoint() == nil)
    }

    @Test("populated form maps to an endpoint; empty api key stays nil")
    func populatedMapsThrough() {
        var f = EndpointForm()
        f.baseUrl = "http://localhost:1234/v1"
        f.model = "dans-pe-v1.3.0-24b-heresy@8bit"
        let e = f.toEndpoint()
        #expect(e?.baseUrl == "http://localhost:1234/v1")
        #expect(e?.model == "dans-pe-v1.3.0-24b-heresy@8bit")
        #expect(e?.apiKey == nil)

        f.apiKey = "sk-123"
        #expect(f.toEndpoint()?.apiKey == "sk-123")
    }

    @Test("whitespace is trimmed and blanks ignored")
    func trimsWhitespace() {
        var f = EndpointForm()
        f.baseUrl = "  http://h/v1  "
        f.model = " m "
        f.apiKey = "   "
        let e = f.toEndpoint()
        #expect(e?.baseUrl == "http://h/v1")
        #expect(e?.model == "m")
        #expect(e?.apiKey == nil)
    }

    @Test("registry round-trips through the form bundle")
    func registryRoundTrip() {
        let registry = AIProviderRegistry(
            promptOptimization: AIProviderEndpoint(baseUrl: "http://localhost:1234/v1", model: "dans-pe-v1.3.0-24b-heresy@8bit"),
            vision: AIProviderEndpoint(baseUrl: "http://localhost:1235/v1", model: "moondream", apiKey: "k")
        )
        let bundle = ProviderFormBundle(registry)
        #expect(bundle.prompt.model == "dans-pe-v1.3.0-24b-heresy@8bit")
        #expect(bundle.vision.apiKey == "k")
        #expect(bundle.captioning.baseUrl.isEmpty)

        let back = bundle.toRegistry()
        #expect(back == registry)
        #expect(back.captioning == nil)
    }
}
