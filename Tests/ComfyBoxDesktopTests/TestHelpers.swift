// TestHelpers.swift — Shared test utilities and data factories

import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
@testable import ComfyBoxDesktop

// MARK: - Sample Data Factories

enum TestData {
    static func makeAsset(
        id: String = "test-asset-1",
        filename: String = "test-image.png",
        prompt: String? = "a beautiful landscape",
        seed: Int? = 42,
        steps: Int? = 9,
        guidance: Double? = 3.5,
        modelFamily: String? = "flux",
        rating: Int = 0,
        favorite: Bool = false,
        contentMode: String? = nil,
        characterName: String? = nil,
        width: Int? = 1024,
        height: Int? = 1024,
        source: String? = nil
    ) -> DAMAsset {
        DAMAsset(
            id: id,
            kind: "image",
            filename: filename,
            absolutePath: "/tmp/test-images/\(filename)",
            fileSize: 1_234_567,
            sha256: nil,
            width: width,
            height: height,
            createdAt: Date(timeIntervalSince1970: 1_700_000_000),
            modifiedAt: Date(timeIntervalSince1970: 1_700_000_000),
            ingestedAt: Date(timeIntervalSince1970: 1_700_000_000),
            orphaned: false,
            prompt: prompt,
            negativePrompt: nil,
            seed: seed,
            steps: steps,
            guidance: guidance,
            modelFamily: modelFamily,
            rating: rating,
            favorite: favorite,
            contentMode: contentMode,
            characterName: characterName,
            source: source
        )
    }

    static func makePreset(
        id: String = "preset-1",
        name: String = "Test Preset",
        promptTemplate: String = "a test prompt",
        steps: Int = 9,
        guidance: Float = 3.5,
        width: Int = 1024,
        height: Int = 1024
    ) -> GenerationPreset {
        GenerationPreset(
            id: id,
            name: name,
            promptTemplate: promptTemplate,
            modelId: "test-model",
            loras: [],
            steps: steps,
            guidance: guidance,
            width: width,
            height: height,
            sampler: nil,
            createdAt: Date(timeIntervalSince1970: 1_700_000_000),
            modifiedAt: Date(timeIntervalSince1970: 1_700_000_000)
        )
    }

    static func makeCharacter(
        id: String = "char-1",
        name: String = "Test Character",
        description: String = "A test character",
        promptSnippet: String = "test character prompt"
    ) -> CharacterEntry {
        CharacterEntry(
            id: id,
            name: name,
            description: description,
            defaultLoras: ["test-lora"],
            promptSnippet: promptSnippet,
            tags: ["test", "sample"]
        )
    }

    static func makeLoRAInfo(
        id: String = "lora-1",
        filename: String = "test-lora.safetensors",
        isActive: Bool = false,
        quarantined: Bool = false
    ) -> LoRAInfo {
        LoRAInfo(
            id: id,
            filename: filename,
            modelCompatibility: "flux",
            format: "safetensors",
            rank: 16,
            sizeBytes: 50_000_000,
            quarantined: quarantined,
            tags: ["test"],
            category: "style",
            triggerwords: ["testtrigger"],
            recommendedScale: 0.8,
            isActive: isActive
        )
    }

    static func makeModelInfo(
        id: String = "model-1",
        family: String = "flux",
        displayName: String = "Test Model"
    ) -> ModelInfo {
        ModelInfo(
            id: id,
            family: family,
            variant: "dev",
            quantization: "q8",
            displayName: displayName,
            description: "A test model",
            parametersBillions: 12.0,
            defaultSteps: 9,
            defaultGuidance: 3.5,
            supportsGuidance: true,
            supportsLoRA: true,
            defaultResolution: "1024x1024",
            estimatedVRAM_GB: 24.0,
            huggingFaceId: "test-org/test-model"
        )
    }

    static func makePoolModel(
        id: String = "pool-1",
        model: String = "test-org/test-model",
        active: Bool = true
    ) -> PoolModelInfo {
        PoolModelInfo(
            id: id,
            model: model,
            family: "flux",
            vramMB: 24576,
            active: active,
            lastUsed: "2025-01-01T00:00:00Z"
        )
    }

    static func makeQueueInfo(
        isRendering: Bool = false,
        pendingCount: Int = 0,
        renderCount: Int = 42,
        currentJobId: String? = nil,
        progressPercent: Double? = nil
    ) -> QueueInfo {
        QueueInfo(
            isRendering: isRendering,
            pendingCount: pendingCount,
            renderCount: renderCount,
            uptimeSeconds: 3600,
            lastRenderDurationMs: 15000,
            lastError: nil,
            memoryUsageMB: 8192,
            currentJobId: currentJobId,
            progressPercent: progressPercent
        )
    }

    static func makeLoRASelection(
        id: String = "lora-1",
        filename: String = "test-lora.safetensors",
        scale: Float = 1.0
    ) -> LoRASelection {
        LoRASelection(id: id, filename: filename, scale: scale)
    }

    /// Writes a real, decodable PNG to `path` — needed wherever a test
    /// depends on `AssetIngestor.generateThumbnail`, which no-ops on fake
    /// (non-image) bytes because `CGImageSourceCreateWithURL` can't decode
    /// them.
    @discardableResult
    static func writeRealPNG(at path: String, width: Int = 32, height: Int = 32) -> Bool {
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: 0,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return false }
        context.setFillColor(CGColor(red: 0.3, green: 0.6, blue: 0.9, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        guard let cgImage = context.makeImage() else { return false }
        guard let destination = CGImageDestinationCreateWithURL(
            URL(fileURLWithPath: path) as CFURL, UTType.png.identifier as CFString, 1, nil
        ) else { return false }
        CGImageDestinationAddImage(destination, cgImage, nil)
        return CGImageDestinationFinalize(destination)
    }
}
