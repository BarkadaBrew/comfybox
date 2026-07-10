// LoRACompatibility.swift — Match LoRAs to the model they're designed for
//
// A LoRA trained for one base family (Z-Image, Flux, Krea 2, Qwen, SDXL…) won't
// work on another. This normalizes the many string spellings (CivitAI base
// models, mflux variants, server families) into a canonical family token so the
// UI can show whether a LoRA fits the active model.

import Foundation

public enum LoRACompatibility {
    /// Canonical family for a model/LoRA string, or "" when unknown.
    public static func family(from raw: String) -> String {
        let s = raw.lowercased()
        // Order matters: check more specific tokens first.
        if s.contains("krea 2") || s.contains("krea2") || s.contains("krea-2") { return "krea2" }
        if s.contains("flux2") || s.contains("flux.2") || s.contains("klein") { return "flux2" }
        if s.contains("zeta") || s.contains("z-image") || s.contains("zimage") || s == "z_image" { return "z-image" }
        if s.contains("qwen") { return "qwen" }
        if s.contains("wan") { return "wan" }
        if s.contains("ltx") { return "ltx" }
        if s.contains("hidream") { return "hidream" }
        if s.contains("sd 3.5") || s.contains("sd3.5") || s.contains("sd35") { return "sd35" }
        if s.contains("sd 1.5") || s.contains("sd1.5") || s.contains("sd15") { return "sd15" }
        // SDXL family (incl. Pony / Illustrious / NoobAI derivatives).
        if s.contains("sdxl") || s.contains("pony") || s.contains("illustrious") || s.contains("noobai") { return "sdxl" }
        // Flux family (incl. dev/schnell/krea-dev — Krea 1 is Flux-based).
        if s.contains("flux") || s.contains("krea-dev") || s.contains("dev-krea")
            || s == "dev" || s == "schnell" || s.contains("krea dev") { return "flux" }
        return ""
    }

    public enum Status: Equatable {
        case compatible
        case incompatible(loraFamily: String, modelFamily: String)
        case unknown
    }

    /// Whether `loraCompatibility` fits `modelIdentifier` (name or family).
    public static func status(loraCompatibility: String, modelIdentifier: String?) -> Status {
        let loraFam = family(from: loraCompatibility)
        guard let model = modelIdentifier, !model.isEmpty else { return .unknown }
        let modelFam = family(from: model)
        if loraFam.isEmpty || modelFam.isEmpty { return .unknown }
        return loraFam == modelFam ? .compatible : .incompatible(loraFamily: loraFam, modelFamily: modelFam)
    }

    /// Short human label for a family token.
    public static func label(for family: String) -> String {
        switch family {
        case "z-image": return "Z-Image"
        case "flux": return "Flux"
        case "krea2": return "Krea 2"
        case "flux2": return "Flux.2"
        case "qwen": return "Qwen"
        case "sdxl": return "SDXL"
        case "sd15": return "SD 1.5"
        case "sd35": return "SD 3.5"
        case "wan": return "Wan"
        case "ltx": return "LTX"
        case "hidream": return "HiDream"
        case "": return "?"
        default: return family
        }
    }
}
