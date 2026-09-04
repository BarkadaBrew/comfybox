// SubjectMasker.swift — Vision foreground mask for background removal
//
// Wraps VNGenerateForegroundInstanceMaskRequest (macOS 14). Returns a
// source-resolution single-channel CIImage, white = subject. Cached per
// source path for the session so toggling Remove Background is free.

import Foundation
import CoreImage
import CoreGraphics
import Vision

public enum SubjectMaskError: Error, Equatable {
    case noSubject
    case visionFailed(String)
}

public actor SubjectMasker {
    private var cache: [String: CIImage] = [:]

    public init() {}

    public func mask(for source: CGImage, cacheKey: String) async throws -> CIImage {
        if let hit = cache[cacheKey] { return hit }
        let request = VNGenerateForegroundInstanceMaskRequest()
        let handler = VNImageRequestHandler(cgImage: source, options: [:])
        do { try handler.perform([request]) } catch { throw SubjectMaskError.visionFailed(error.localizedDescription) }
        guard let result = request.results?.first, !result.allInstances.isEmpty else { throw SubjectMaskError.noSubject }
        let buffer: CVPixelBuffer
        do {
            buffer = try result.generateScaledMaskForImage(forInstances: result.allInstances, from: handler)
        } catch { throw SubjectMaskError.visionFailed(error.localizedDescription) }
        var image = CIImage(cvPixelBuffer: buffer)
        // Vision returns a float mask sized to the source; force exact extent.
        let target = CGRect(x: 0, y: 0, width: source.width, height: source.height)
        if image.extent.size != target.size {
            image = image.transformed(by: CGAffineTransform(scaleX: target.width / image.extent.width,
                                                            y: target.height / image.extent.height))
        }
        image = image.cropped(to: target)
        cache[cacheKey] = image
        return image
    }
}
