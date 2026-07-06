// CloudImageProvider.swift — Replicate & Fal image-generation backends
//
// Lets Generate render via a cloud API instead of the local MLX server. Each
// provider builds a model-appropriate request, submits it, waits for the
// result, and downloads the image into the output directory (from where the
// DAM ingests it). Body-building and output-parsing are pure static functions
// so they're unit-tested without network access.

import Foundation

/// Which backend Generate should use.
public enum CloudProvider: String, CaseIterable, Identifiable, Sendable {
    case local = "Local"
    case replicate = "Replicate"
    case fal = "Fal"
    public var id: String { rawValue }

    /// A sensible default model id for the cloud providers.
    public var defaultModel: String {
        switch self {
        case .local: return ""
        case .replicate: return "black-forest-labs/flux-schnell"
        case .fal: return "fal-ai/flux/schnell"
        }
    }

    /// Whether the provider needs an API key configured.
    public var needsKey: Bool { self != .local }
}

public struct CloudImageParams: Sendable {
    public var prompt: String
    public var negativePrompt: String?
    public var width: Int
    public var height: Int
    public var steps: Int
    public var seed: UInt64   // 0 = random
    public var initImagePath: String?   // for provider-side img2img (data URI)

    public init(prompt: String, negativePrompt: String? = nil, width: Int, height: Int,
                steps: Int, seed: UInt64 = 0, initImagePath: String? = nil) {
        self.prompt = prompt; self.negativePrompt = negativePrompt
        self.width = width; self.height = height
        self.steps = steps; self.seed = seed; self.initImagePath = initImagePath
    }
}

public enum CloudImageError: Error, LocalizedError {
    case missingKey(String)
    case badResponse(Int, String)
    case noOutput
    case timedOut
    case badURL

    public var errorDescription: String? {
        switch self {
        case .missingKey(let p): return "\(p) API key not set — add it in Settings → AI Providers."
        case .badResponse(let code, let msg): return "Cloud provider error (\(code)): \(msg)"
        case .noOutput: return "Cloud provider returned no image."
        case .timedOut: return "Cloud generation timed out."
        case .badURL: return "Invalid provider URL."
        }
    }
}

public struct CloudImageClient: Sendable {
    public let provider: CloudProvider
    public let model: String
    public let apiKey: String

    public init(provider: CloudProvider, model: String, apiKey: String) {
        self.provider = provider
        self.model = model.isEmpty ? provider.defaultModel : model
        self.apiKey = apiKey
    }

    // MARK: - Pure request/response helpers (tested)

    /// Map a pixel size to the nearest standard aspect-ratio string that the
    /// Flux family on Replicate/Fal accepts.
    public static func aspectRatio(width: Int, height: Int) -> String {
        guard width > 0, height > 0 else { return "1:1" }
        let ratio = Double(width) / Double(height)
        let table: [(String, Double)] = [
            ("1:1", 1.0), ("16:9", 16.0/9), ("9:16", 9.0/16),
            ("3:2", 3.0/2), ("2:3", 2.0/3), ("4:3", 4.0/3), ("3:4", 3.0/4),
            ("21:9", 21.0/9),
        ]
        return table.min(by: { abs($0.1 - ratio) < abs($1.1 - ratio) })?.0 ?? "1:1"
    }

    /// Replicate prediction `input` body for the Flux family.
    public static func replicateInput(_ p: CloudImageParams) -> [String: Any] {
        var input: [String: Any] = [
            "prompt": p.prompt,
            "aspect_ratio": aspectRatio(width: p.width, height: p.height),
            "num_inference_steps": max(1, min(p.steps, 4)),   // schnell caps at 4
            "output_format": "png",
            "num_outputs": 1,
        ]
        if p.seed > 0 { input["seed"] = Int(truncatingIfNeeded: p.seed) }
        return input
    }

    /// Parse a Replicate prediction response → (status, id, outputURLs).
    public static func parseReplicate(_ data: Data) -> (status: String, id: String?, urls: [String]) {
        guard let obj = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else {
            return ("unknown", nil, [])
        }
        let status = obj["status"] as? String ?? "unknown"
        let id = obj["id"] as? String
        var urls: [String] = []
        if let s = obj["output"] as? String { urls = [s] }
        else if let arr = obj["output"] as? [String] { urls = arr }
        return (status, id, urls)
    }

    /// Fal request body for the Flux family.
    public static func falInput(_ p: CloudImageParams) -> [String: Any] {
        var input: [String: Any] = [
            "prompt": p.prompt,
            "image_size": ["width": p.width, "height": p.height],
            "num_inference_steps": max(1, p.steps),
            "num_images": 1,
        ]
        if p.seed > 0 { input["seed"] = Int(truncatingIfNeeded: p.seed) }
        return input
    }

    /// Parse a Fal response → image URLs.
    public static func parseFal(_ data: Data) -> [String] {
        guard let obj = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else { return [] }
        if let images = obj["images"] as? [[String: Any]] {
            return images.compactMap { $0["url"] as? String }
        }
        if let image = obj["image"] as? [String: Any], let url = image["url"] as? String {
            return [url]
        }
        return []
    }

    // MARK: - Generate

    /// Generate an image via the cloud provider and download it to `directory`.
    /// Returns the local file URL.
    public func generate(_ params: CloudImageParams, downloadTo directory: URL) async throws -> URL {
        guard !apiKey.isEmpty else { throw CloudImageError.missingKey(provider.rawValue) }
        let remoteURL: String
        switch provider {
        case .replicate: remoteURL = try await runReplicate(params)
        case .fal: remoteURL = try await runFal(params)
        case .local: throw CloudImageError.badURL
        }
        return try await downloadImage(from: remoteURL, to: directory)
    }

    // MARK: - Replicate (create prediction, wait, poll if needed)

    private func runReplicate(_ params: CloudImageParams) async throws -> String {
        guard let url = URL(string: "https://api.replicate.com/v1/models/\(model)/predictions") else {
            throw CloudImageError.badURL
        }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("wait", forHTTPHeaderField: "Prefer")   // block up to 60s
        request.httpBody = try JSONSerialization.data(withJSONObject: ["input": Self.replicateInput(params)])

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse, (200...299).contains(http.statusCode) else {
            let code = (response as? HTTPURLResponse)?.statusCode ?? -1
            throw CloudImageError.badResponse(code, String(data: data, encoding: .utf8) ?? "")
        }

        var parsed = Self.parseReplicate(data)
        if let first = parsed.urls.first, parsed.status == "succeeded" { return first }

        // `Prefer: wait` may still return processing for slow models — poll.
        guard let id = parsed.id else { throw CloudImageError.noOutput }
        for _ in 0..<120 {   // ~2 min at 1s
            try await Task.sleep(nanoseconds: 1_000_000_000)
            parsed = try await pollReplicate(id: id)
            if parsed.status == "succeeded", let first = parsed.urls.first { return first }
            if parsed.status == "failed" || parsed.status == "canceled" {
                throw CloudImageError.badResponse(200, "prediction \(parsed.status)")
            }
        }
        throw CloudImageError.timedOut
    }

    private func pollReplicate(id: String) async throws -> (status: String, id: String?, urls: [String]) {
        guard let url = URL(string: "https://api.replicate.com/v1/predictions/\(id)") else {
            throw CloudImageError.badURL
        }
        var request = URLRequest(url: url)
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        let (data, _) = try await URLSession.shared.data(for: request)
        return Self.parseReplicate(data)
    }

    // MARK: - Fal (synchronous)

    private func runFal(_ params: CloudImageParams) async throws -> String {
        guard let url = URL(string: "https://fal.run/\(model)") else { throw CloudImageError.badURL }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("Key \(apiKey)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = 180
        request.httpBody = try JSONSerialization.data(withJSONObject: Self.falInput(params))

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse, (200...299).contains(http.statusCode) else {
            let code = (response as? HTTPURLResponse)?.statusCode ?? -1
            throw CloudImageError.badResponse(code, String(data: data, encoding: .utf8) ?? "")
        }
        guard let first = Self.parseFal(data).first else { throw CloudImageError.noOutput }
        return first
    }

    // MARK: - Download

    private func downloadImage(from urlString: String, to directory: URL) async throws -> URL {
        guard let url = URL(string: urlString) else { throw CloudImageError.badURL }
        let (data, response) = try await URLSession.shared.data(from: url)
        guard let http = response as? HTTPURLResponse, (200...299).contains(http.statusCode) else {
            throw CloudImageError.badResponse((response as? HTTPURLResponse)?.statusCode ?? -1, "download failed")
        }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let ext = url.pathExtension.isEmpty ? "png" : url.pathExtension
        let name = "\(provider.rawValue.lowercased())-\(Int(Date().timeIntervalSince1970)).\(ext)"
        let dest = directory.appendingPathComponent(name)
        try data.write(to: dest)
        return dest
    }
}
