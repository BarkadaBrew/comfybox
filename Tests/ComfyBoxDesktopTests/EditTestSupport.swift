// EditTestSupport.swift — synthetic images and pixel readback for editor tests
import Foundation
import CoreGraphics
import ImageIO
@testable import ComfyBoxDesktop

enum EditTestSupport {
    static func rgbaBytes(_ image: CGImage) -> [UInt8] {
        let w = image.width, h = image.height
        var bytes = [UInt8](repeating: 0, count: w * h * 4)
        let cs = CGColorSpace(name: CGColorSpace.sRGB)!
        let ctx = CGContext(data: &bytes, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                            space: cs, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        return bytes
    }

    static func pixel(_ image: CGImage, x: Int, y: Int) -> (r: UInt8, g: UInt8, b: UInt8, a: UInt8) {
        let b = rgbaBytes(image)
        let i = (y * image.width + x) * 4
        return (b[i], b[i + 1], b[i + 2], b[i + 3])
    }

    static func gray(_ image: CGImage, x: Int, y: Int) -> UInt8 { pixel(image, x: x, y: y).r }

    static func horizontalGradient(width: Int, height: Int) -> CGImage {
        var bytes = [UInt8](repeating: 255, count: width * height * 4)
        for y in 0..<height {
            for x in 0..<width {
                let v = UInt8((x * 255) / max(width - 1, 1))
                let i = (y * width + x) * 4
                bytes[i] = v; bytes[i + 1] = v; bytes[i + 2] = v; bytes[i + 3] = 255
            }
        }
        return make(bytes, width: width, height: height)
    }

    static func solid(r: UInt8, g: UInt8, b: UInt8, width: Int, height: Int) -> CGImage {
        var bytes = [UInt8](repeating: 255, count: width * height * 4)
        for p in 0..<(width * height) {
            bytes[p * 4] = r; bytes[p * 4 + 1] = g; bytes[p * 4 + 2] = b; bytes[p * 4 + 3] = 255
        }
        return make(bytes, width: width, height: height)
    }

    /// Straight (non-premultiplied) alpha: `r/g/b` are the true channel values regardless of `a`,
    /// so a partially transparent solid still reads back with its literal color under straight alpha.
    /// CGBitmapContext only supports premultiplied/none alpha layouts, so this builds the CGImage
    /// directly from a CGDataProvider instead of drawing through a context.
    static func solidRGBA(r: UInt8, g: UInt8, b: UInt8, a: UInt8, width: Int, height: Int) -> CGImage {
        var bytes = [UInt8](repeating: 0, count: width * height * 4)
        for p in 0..<(width * height) {
            bytes[p * 4] = r; bytes[p * 4 + 1] = g; bytes[p * 4 + 2] = b; bytes[p * 4 + 3] = a
        }
        let cs = CGColorSpace(name: CGColorSpace.sRGB)!
        let provider = CGDataProvider(data: Data(bytes) as CFData)!
        return CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width * 4,
                       space: cs, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
                       provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
    }

    private static func make(_ bytes: [UInt8], width: Int, height: Int) -> CGImage {
        var copy = bytes
        let cs = CGColorSpace(name: CGColorSpace.sRGB)!
        let ctx = CGContext(data: &copy, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4,
                            space: cs, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        return ctx.makeImage()!
    }
}
