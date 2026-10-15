// ImageIO.swift — reference-image preprocessing for the editing path.
//
// Mirrors `load_image_native`: RGB (RGBA over white) → smart_resize to a
// factor-32 grid within [min, max] pixels → ImageNet normalize → flatten 16px
// patches in (c, p, q) order.
//
// ⚠ Documented deviation: the resample kernel. PIL uses BICUBIC here (the
// reference's `Image.resize` default); we use CoreGraphics high-quality
// interpolation. Parity tests bypass this by injecting the oracle's
// preprocessed pixel_values; the production path accepts the kernel delta
// (input preprocessing, upstream of all weights). If it ever matters,
// qwen25vl-mlx-swift ships parity-locked HF-style resampling to borrow.

import Accelerate
import CoreGraphics
import Foundation
import ImageIO
import MLX
import UniformTypeIdentifiers

public enum SenseNovaImageIO {

    public static let imagenetMean: [Float] = [0.485, 0.456, 0.406]
    public static let imagenetStd: [Float] = [0.229, 0.224, 0.225]

    /// Exact port of `smart_resize` (Qwen2.5-VL rounding rules).
    public static func smartResize(
        height: Int, width: Int, factor: Int = 32,
        minPixels: Int = 65536, maxPixels: Int = 4_194_304
    ) -> (height: Int, width: Int) {
        func round_(_ n: Double) -> Int { Int((n / Double(factor)).rounded()) * factor }
        func ceil_(_ n: Double) -> Int { Int((n / Double(factor)).rounded(.up)) * factor }
        func floor_(_ n: Double) -> Int { Int((n / Double(factor)).rounded(.down)) * factor }

        var hBar = max(factor, round_(Double(height)))
        var wBar = max(factor, round_(Double(width)))
        if hBar * wBar > maxPixels {
            let beta = (Double(height * width) / Double(maxPixels)).squareRoot()
            hBar = max(factor, floor_(Double(height) / beta))
            wBar = max(factor, floor_(Double(width) / beta))
        } else if hBar * wBar < minPixels {
            let beta = (Double(minPixels) / Double(height * width)).squareRoot()
            hBar = ceil_(Double(height) * beta)
            wBar = ceil_(Double(width) * beta)
        }
        return (hBar, wBar)
    }

    /// Load an image file → `EditImage` (und-stream reference input).
    public static func loadEditImage(
        url: URL, patchSize: Int = 16,
        minPixels: Int = 512 * 512, maxPixels: Int = 2048 * 2048
    ) throws -> EditImage {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let cg = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else { throw SenseNovaError.badConfig("cannot read image at \(url.path)") }
        return try editImage(from: cg, patchSize: patchSize, minPixels: minPixels, maxPixels: maxPixels)
    }

    /// Encoded image bytes (PNG/JPEG) → `EditImage`.
    public static func loadEditImage(
        data: Data, patchSize: Int = 16,
        minPixels: Int = 512 * 512, maxPixels: Int = 2048 * 2048
    ) throws -> EditImage {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
              let cg = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else { throw SenseNovaError.badConfig("cannot decode image data") }
        return try editImage(from: cg, patchSize: patchSize, minPixels: minPixels, maxPixels: maxPixels)
    }

    static func editImage(
        from cg: CGImage, patchSize: Int, minPixels: Int, maxPixels: Int
    ) throws -> EditImage {

        let (h, w) = smartResize(
            height: cg.height, width: cg.width, factor: 2 * patchSize,
            minPixels: minPixels, maxPixels: maxPixels)

        // draw RGBA over white → RGBA8 buffer at target size
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
        guard let ctx = CGContext(
            data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
            space: colorSpace, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { throw SenseNovaError.badConfig("cannot create bitmap context") }
        ctx.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: w, height: h))
        ctx.interpolationQuality = .high
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
        guard let data = ctx.data else { throw SenseNovaError.badConfig("no bitmap data") }

        // RGBA8 → fp32 CHW normalized
        let pixelCount = w * h
        let bytes = data.bindMemory(to: UInt8.self, capacity: pixelCount * 4)
        var chw = [Float](repeating: 0, count: 3 * pixelCount)
        for c in 0 ..< 3 {
            let mean = imagenetMean[c]
            let std = imagenetStd[c]
            for i in 0 ..< pixelCount {
                chw[c * pixelCount + i] = (Float(bytes[i * 4 + c]) / 255.0 - mean) / std
            }
        }

        let image = MLXArray(chw, [1, 3, h, w])
        // (c, p, q)-ordered 16px patch flatten == patchifyChannelFirst
        let (gh, gw) = (h / patchSize, w / patchSize)
        let flat = image.reshaped([1, 3, gh, patchSize, gw, patchSize])
            .transposed(0, 2, 4, 1, 3, 5)
            .reshaped([gh * gw, 3 * patchSize * patchSize])
        return EditImage(pixelValues: flat, gridH: gh, gridW: gw)
    }

    // MARK: - generation-space images (the init-image / SR-probe path)

    /// Decode an image file into the generation space: (1, 3, H, W) fp32 in
    /// [-1, 1] (`x·0.5 + 0.5` → RGB), RGBA drawn over white at its native size
    /// (a 1:1 draw, no resampling). `scale > 1` Lanczos-upsamples by that
    /// integer factor with vImage — centre-aligned, unlike a scaled CGContext
    /// draw, which shifts content by up to 0.26 px (AB-L-0239).
    public static func loadGenImage(url: URL, scale: Int = 1) throws -> MLXArray {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let cg = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else { throw SenseNovaError.badConfig("cannot read image at \(url.path)") }
        let (w0, h0) = (cg.width, cg.height)
        var rgba = try rgba8OverWhite(cg)
        var (w, h) = (w0, h0)
        if scale > 1 {
            (w, h) = (w0 * scale, h0 * scale)
            var dst = [UInt8](repeating: 0, count: w * h * 4)
            let err: vImage_Error = rgba.withUnsafeMutableBytes { sp in
                dst.withUnsafeMutableBytes { dp in
                    var src = vImage_Buffer(
                        data: sp.baseAddress, height: vImagePixelCount(h0),
                        width: vImagePixelCount(w0), rowBytes: w0 * 4)
                    var out = vImage_Buffer(
                        data: dp.baseAddress, height: vImagePixelCount(h),
                        width: vImagePixelCount(w), rowBytes: w * 4)
                    return vImageScale_ARGB8888(&src, &out, nil, vImage_Flags(kvImageHighQualityResampling))
                }
            }
            guard err == kvImageNoError else {
                throw SenseNovaError.badConfig("vImageScale failed: \(err)")
            }
            rgba = dst
        }
        let n = w * h
        var chw = [Float](repeating: 0, count: 3 * n)
        for c in 0 ..< 3 {
            for i in 0 ..< n { chw[c * n + i] = Float(rgba[i * 4 + c]) / 127.5 - 1 }
        }
        return MLXArray(chw, [1, 3, h, w])
    }

    /// Write a generation-space image (1, 3, H, W) in [-1, 1] as an 8-bit RGB
    /// PNG (rounded, clamped).
    public static func savePNG(_ image: MLXArray, to url: URL) throws {
        precondition(image.ndim == 4 && image.dim(0) == 1 && image.dim(1) == 3, "expected (1, 3, H, W)")
        let (h, w) = (image.dim(2), image.dim(3))
        let n = w * h
        let px = clip((image.asType(.float32) * 0.5 + 0.5) * 255 + 0.5, min: 0, max: 255)
        let chw = px.reshaped([3 * n]).asArray(Float.self)
        var rgba = [UInt8](repeating: 255, count: n * 4)
        for i in 0 ..< n {
            rgba[i * 4] = UInt8(chw[i])
            rgba[i * 4 + 1] = UInt8(chw[n + i])
            rgba[i * 4 + 2] = UInt8(chw[2 * n + i])
        }
        guard let provider = CGDataProvider(data: Data(rgba) as CFData),
              let cg = CGImage(
                width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: w * 4,
                space: CGColorSpace(name: CGColorSpace.sRGB)!,
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent),
              let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)
        else { throw SenseNovaError.badConfig("cannot create PNG at \(url.path)") }
        CGImageDestinationAddImage(dest, cg, nil)
        guard CGImageDestinationFinalize(dest) else {
            throw SenseNovaError.badConfig("cannot write PNG at \(url.path)")
        }
    }

    /// RGBA8 bytes of `cg` drawn over white at its own size, row 0 = top.
    static func rgba8OverWhite(_ cg: CGImage) throws -> [UInt8] {
        let (w, h) = (cg.width, cg.height)
        guard let ctx = CGContext(
            data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
            space: CGColorSpace(name: CGColorSpace.sRGB)!,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { throw SenseNovaError.badConfig("cannot create bitmap context") }
        ctx.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: w, height: h))
        ctx.interpolationQuality = .none
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
        guard let data = ctx.data else { throw SenseNovaError.badConfig("no bitmap data") }
        return Array(UnsafeBufferPointer(start: data.bindMemory(to: UInt8.self, capacity: w * h * 4), count: w * h * 4))
    }
}
