// ImageRecipe.swift — reconstruct a Generate-tab recipe from an image's
// embedded metadata (EXIF:UserComment JSON), for "Send to Generate".
import Foundation
import ImageIO

public struct ImageRecipe {
    public let preset: GenerationPreset
    public let contentMode: ContentMode?

    /// Build a recipe from a params dict (the embedded UserComment JSON, or a
    /// dict derived from a DAMAsset). Returns nil when there's nothing usable.
    public static func from(params: [String: Any]) -> ImageRecipe? {
        let prompt = params["prompt"] as? String ?? ""
        guard !prompt.isEmpty || params["seed"] != nil || params["model"] != nil else { return nil }

        let loras: [PresetLoRA] = (params["loras"] as? [[String: Any]] ?? []).compactMap { l in
            guard let name = l["name"] as? String, !name.isEmpty else { return nil }
            let scale = ((l["scale"] as? NSNumber)?.floatValue) ?? 1.0
            let filename = name.hasSuffix(".safetensors") ? name : name + ".safetensors"
            return PresetLoRA(id: name, filename: filename, scale: scale)
        }

        let preset = GenerationPreset(
            id: "from-image",
            name: "From image",
            promptTemplate: prompt,
            negativePrompt: params["negative_prompt"] as? String,
            modelId: params["model"] as? String,
            loras: loras,
            steps: (params["steps"] as? NSNumber)?.intValue ?? 9,
            guidance: (params["guidance"] as? NSNumber)?.floatValue ?? 3.5,
            width: (params["width"] as? NSNumber)?.intValue ?? 1024,
            height: (params["height"] as? NSNumber)?.intValue ?? 1024,
            seed: (params["seed"] as? NSNumber)?.uint64Value
        )
        let mode = (params["content_mode"] as? String).flatMap(ContentMode.init(rawValue:))
        return ImageRecipe(preset: preset, contentMode: mode)
    }

    /// Read an image file's embedded UserComment JSON; fall back to the
    /// DAMAsset's structured fields for images without embedded params.
    public static func read(fromImageAt path: String, fallback: DAMAsset?) -> ImageRecipe? {
        var params: [String: Any] = [:]
        if let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
           let props = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [CFString: Any],
           let exif = props[kCGImagePropertyExifDictionary] as? [CFString: Any],
           let uc = exif[kCGImagePropertyExifUserComment] as? String,
           let d = uc.data(using: .utf8),
           let j = try? JSONSerialization.jsonObject(with: d) as? [String: Any] {
            params = j
        }
        if params.isEmpty, let a = fallback {
            var fromAsset: [String: Any] = [:]
            fromAsset["prompt"] = a.prompt
            fromAsset["negative_prompt"] = a.negativePrompt
            fromAsset["seed"] = a.seed
            fromAsset["steps"] = a.steps
            fromAsset["guidance"] = a.guidance
            fromAsset["width"] = a.width
            fromAsset["height"] = a.height
            fromAsset["content_mode"] = a.contentMode
            params = fromAsset
        }
        return from(params: params)
    }
}
