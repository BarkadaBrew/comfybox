// VisionService.swift — Local vision captioning & tagging
//
// Uses the configured vision/captioning provider (a local vision model served
// by LM Studio, OpenAI-compatible /chat/completions with image content) to
// caption an image and propose tags. Request building and response parsing are
// pure so they're unit-tested without a server.

import Foundation
import AppKit
import ZImage

@Observable
@MainActor
public final class VisionService {
    private let engine: EngineService
    /// FDD-glimmer-gpu-slot: host:port list whose calls hold an inference slot.
    private let glimmerEndpoints: [String]
    public init(
        engine: EngineService,
        glimmerEndpoints: [String] = InferenceSlotClient.glimmerEndpoints(environment: ProcessInfo.processInfo.environment)
    ) {
        self.engine = engine
        self.glimmerEndpoints = glimmerEndpoints
    }

    /// The caption request's own timeout, and its overall deadline (slot wait +
    /// request): a batch caption waits up to 8 min per image for the GPU.
    nonisolated static let captionRequestSec: TimeInterval = 120
    nonisolated static let captionDeadlineSec: TimeInterval = 600

    public struct Description: Sendable, Equatable {
        public var caption: String
        public var tags: [String]
    }

    public enum VisionError: Error, LocalizedError {
        case noProvider, badURL, requestFailed(Int), emptyReply, imageUnreadable
        public var errorDescription: String? {
            switch self {
            case .noProvider: return "No vision provider configured (Settings → AI Providers → Vision/Captioning)."
            case .badURL: return "Invalid provider URL."
            case .requestFailed(let c): return "Vision request failed (HTTP \(c))."
            case .emptyReply: return "Vision model returned nothing."
            case .imageUnreadable: return "Couldn't read the image."
            }
        }
    }

    // MARK: - Pure helpers (tested)

    static let instruction = """
    Describe this image for a digital-asset library. Respond with ONLY a JSON object:
    {"caption": "one concise sentence", "tags": ["5-12 lowercase keyword tags"]}
    No prose, no code fences.
    """

    /// Shared with the engine's MCP repair path (ZImage.VisionChat): 1200 tokens +
    /// reasoning_budget 256, so a reasoning vision model (Glimmer) answers instead
    /// of spending a 320-token budget thinking.
    public nonisolated static func requestBody(model: String, base64PNG: String) -> [String: Any] {
        VisionChat.body(model: model, prompt: instruction, base64PNG: base64PNG)
    }

    /// Parse a chat-completions reply into a Description (tolerant of fences/prose).
    public nonisolated static func parseDescription(from reply: String) -> Description? {
        guard let json = extractJSONObject(from: reply),
              let data = json.data(using: .utf8),
              let obj = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
        else {
            // Fallback: treat the whole reply as a caption.
            let trimmed = reply.trimmingCharacters(in: .whitespacesAndNewlines)
            return trimmed.isEmpty ? nil : Description(caption: trimmed, tags: [])
        }
        let caption = (obj["caption"] as? String)?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let rawTags = (obj["tags"] as? [String]) ?? []
        let tags = rawTags.map { $0.lowercased().trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
        return Description(caption: caption, tags: tags)
    }

    /// First balanced {...} block in the text.
    nonisolated static func extractJSONObject(from text: String) -> String? {
        guard let start = text.firstIndex(of: "{") else { return nil }
        var depth = 0
        var idx = start
        while idx < text.endIndex {
            let ch = text[idx]
            if ch == "{" { depth += 1 }
            else if ch == "}" { depth -= 1; if depth == 0 { return String(text[start...idx]) } }
            idx = text.index(after: idx)
        }
        return nil
    }

    // MARK: - Describe

    /// - Parameter onWaitStatus: "Waiting for the GPU (~Ns)" while a Glimmer
    ///   slot is pending (called off the main actor).
    public func describe(imagePath: String, onWaitStatus: (@Sendable (String) -> Void)? = nil) async throws -> Description {
        let config = try await engine.fetchServerConfig()
        guard let provider = config.providers.captioning ?? config.providers.vision else {
            throw VisionError.noProvider
        }
        var base = provider.baseUrl
        if base.hasSuffix("/") { base.removeLast() }
        guard let url = URL(string: base + "/chat/completions") else { throw VisionError.badURL }

        // Downscale to keep the request light.
        guard let b64 = await Self.base64DownscaledPNG(path: imagePath, maxDimension: 768) else {
            throw VisionError.imageUnreadable
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.timeoutInterval = Self.captionRequestSec
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if let key = provider.apiKey, !key.isEmpty {
            request.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization")
        }
        request.httpBody = try JSONSerialization.data(withJSONObject: Self.requestBody(model: provider.model, base64PNG: b64))

        let (data, response) = try await withGlimmerSlot(baseURL: base, onWaitStatus: onWaitStatus) { [request] in
            try await URLSession.shared.data(for: request)
        }
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            throw VisionError.requestFailed((response as? HTTPURLResponse)?.statusCode ?? -1)
        }
        guard case .text(let content) = VisionChat.parseReply(data),
              let desc = Self.parseDescription(from: content)
        else { throw VisionError.emptyReply }
        return desc
    }

    /// FDD-glimmer-gpu-slot: on Glimmer, hold a top-priority inference slot for
    /// the request (an in-flight video parks); elsewhere run it as is.
    private func withGlimmerSlot<T: Sendable>(
        baseURL: String, onWaitStatus: (@Sendable (String) -> Void)?, _ body: () async throws -> T
    ) async throws -> T {
        guard let transport = engine.transport else { return try await body() }
        return try await InferenceSlotClient(transport: transport, endpoints: glimmerEndpoints).withSlot(
            holder: "comfybox-desktop:caption", baseURL: baseURL,
            deadline: Date().addingTimeInterval(Self.captionDeadlineSec), reserveSec: Self.captionRequestSec,
            onWait: onWaitStatus.map { report in { @Sendable wait in report(wait.statusText) } },
            body)
    }

    /// Load, downscale, and PNG-base64 an image off the main actor.
    nonisolated static func base64DownscaledPNG(path: String, maxDimension: CGFloat) async -> String? {
        await Task.detached {
            guard let img = NSImage(contentsOfFile: path),
                  let tiff = img.tiffRepresentation,
                  let rep = NSBitmapImageRep(data: tiff) else { return nil }
            let w = CGFloat(rep.pixelsWide), h = CGFloat(rep.pixelsHigh)
            let scale = min(1, maxDimension / max(w, h))
            let tw = Int(w * scale), th = Int(h * scale)
            guard let resized = NSBitmapImageRep(
                bitmapDataPlanes: nil, pixelsWide: tw, pixelsHigh: th,
                bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
                colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0) else { return nil }
            NSGraphicsContext.saveGraphicsState()
            NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: resized)
            rep.draw(in: NSRect(x: 0, y: 0, width: tw, height: th))
            NSGraphicsContext.restoreGraphicsState()
            return resized.representation(using: .png, properties: [:])?.base64EncodedString()
        }.value
    }
}
