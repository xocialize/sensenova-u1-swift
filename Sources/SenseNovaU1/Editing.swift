// Editing.swift — mirrors `it2i_generate` (image-conditioned generation /
// editing) and its input plumbing (`_build_it2i_inputs`, `get_thw_indexes`).
//
// Reference-image tokens ride the UND stream: ImageNet-normalized pixels →
// und vision patchify → spliced into the prompt embedding at the contiguous
// `<IMG_CONTEXT>` runs. All patches of one image share ONE temporal index;
// h/w carry token-grid coordinates. CFG default for editing: cfg 4.0,
// img_cfg 1.0 → two branches, v = v_ic + cfg·(v_c − v_ic).
// ⚠ cfg-interval semantics here are EXCLUSIVE-with-lo==0-fallback — different
// from t2i's inclusive check (trap ledger #13); both copied verbatim.

import Foundation
import MLX
import MLXNN

/// A preprocessed und-stream reference image (ImageNet-normalized, flattened
/// 16px patches in (c, p, q) order — `load_image_native` output layout).
public struct EditImage {
    public let pixelValues: MLXArray  // (gridH·gridW, 3·16·16)
    public let gridH: Int             // 16px-patch grid
    public let gridW: Int

    public init(pixelValues: MLXArray, gridH: Int, gridW: Int) {
        precondition(pixelValues.dim(0) == gridH * gridW)
        self.pixelValues = pixelValues
        self.gridH = gridH
        self.gridW = gridW
    }

    /// tokens after the 2×2 dense merge
    public var tokenCount: Int { (gridH / 2) * (gridW / 2) }
    public var tokenGrid: (h: Int, w: Int) { (gridH / 2, gridW / 2) }
}

public enum SpecialToken {
    public static let imgStart: Int32 = 151670
    public static let imgEnd: Int32 = 151671
    public static let imgContext: Int32 = 151669
}

extension Conversation {
    /// `it2i_generate` prompt assembly: pad missing `<image>` placeholders,
    /// then expand each to `<img><IMG_CONTEXT>·n</img>` in order.
    public static func expandImagePlaceholders(
        prompt: String, imageTokenCounts: [Int]
    ) throws -> String {
        var p = prompt
        let placeholders = p.components(separatedBy: "<image>").count - 1
        // Reference asserts `len(images) >= prompt.count('<image>')`; without it a
        // prompt containing a literal "<image>" leaves an unexpanded placeholder
        // that silently tokenizes as text (no IMG_CONTEXT run, so the splice
        // preconditions still pass).
        guard placeholders <= imageTokenCounts.count else {
            throw SenseNovaError.badConfig(
                "prompt has \(placeholders) <image> placeholders but only "
                    + "\(imageTokenCounts.count) image(s) were supplied")
        }
        if imageTokenCounts.count > placeholders {
            let missing = imageTokenCounts.count - placeholders
            if placeholders == 0 && imageTokenCounts.count > 1 {
                p = (1 ... imageTokenCounts.count).map { "Image-\($0):<image>\n" }.joined() + prompt
            } else {
                p = String(repeating: "<image>\n", count: missing) + p
            }
        }
        for n in imageTokenCounts {
            let block = imgStartToken + String(repeating: imgContextToken, count: n) + imgEndToken
            if let r = p.range(of: "<image>") { p.replaceSubrange(r, with: block) }
        }
        return p
    }

    /// cond branch: gen system message + prompt-with-images + closed think + <img>
    public static func editCondPrompt(_ prompt: String, imageTokenCounts: [Int]) throws -> String {
        let expanded = try expandImagePlaceholders(prompt: prompt, imageTokenCounts: imageTokenCounts)
        return buildPrompt(
            userMessage: expanded, systemMessage: systemMessageForGen,
            appendText: "<think>\n\n</think>\n\n" + imgStartToken)
    }

    /// img-cond branch: images only, default (empty→omitted) system, no think
    public static func editImgCondPrompt(imageTokenCounts: [Int]) throws -> String {
        let expanded = try expandImagePlaceholders(
            prompt: String(repeating: "<image>", count: imageTokenCounts.count),
            imageTokenCounts: imageTokenCounts)
        return buildPrompt(userMessage: expanded, systemMessage: "", appendText: imgStartToken)
    }
}

extension NEOChatModel {

    /// `get_thw_indexes`: t advances on every non-IMG_CONTEXT token AND on the
    /// token following `<img>`; all context tokens of one image share one t and
    /// carry row-major (h, w) token-grid coordinates.
    public func getTHWIndexes(ids: [Int32], images: [EditImage]) -> THWIndexes {
        var t = [Int32](); t.reserveCapacity(ids.count)
        var h = [Int32](repeating: 0, count: ids.count)
        var w = [Int32](repeating: 0, count: ids.count)
        var run: Int32 = -1
        var prevWasImgStart = false
        var imageIdx = 0
        var withinImage = 0
        for (i, id) in ids.enumerated() {
            let isContext = id == SpecialToken.imgContext
            run += (prevWasImgStart ? 1 : 0) + (isContext ? 0 : 1)
            t.append(run)
            if isContext {
                let grid = images[imageIdx].tokenGrid
                h[i] = Int32(withinImage / grid.w)
                w[i] = Int32(withinImage % grid.w)
                withinImage += 1
                if withinImage == images[imageIdx].tokenCount {
                    imageIdx += 1
                    withinImage = 0
                }
            }
            prevWasImgStart = (id == SpecialToken.imgStart)
        }
        return THWIndexes(t: MLXArray(t), h: MLXArray(h), w: MLXArray(w))
    }

    /// `_build_it2i_inputs`: token embeds with und-vision embeds spliced into
    /// the contiguous `<IMG_CONTEXT>` runs (splice by concat — runs are
    /// contiguous by construction).
    public func buildIT2IInputs(
        ids: [Int32], images: [EditImage]
    ) -> (embeds: MLXArray, indexes: THWIndexes, mask: MLXArray, maxT: Int32) {
        let s = ids.count
        let tokenEmbeds = languageModel.model.embedTokens(MLXArray(ids).reshaped([1, s]))

        var segments: [MLXArray] = []
        var cursor = 0
        var imageIdx = 0
        var i = 0
        while i < s {
            if ids[i] == SpecialToken.imgContext {
                let start = i
                while i < s, ids[i] == SpecialToken.imgContext { i += 1 }
                let runLength = i - start
                precondition(imageIdx < images.count, "more IMG_CONTEXT runs than images")
                let image = images[imageIdx]
                precondition(
                    runLength == image.tokenCount,
                    "IMG_CONTEXT run \(runLength) != image tokens \(image.tokenCount)")
                if start > cursor {
                    segments.append(tokenEmbeds[0..., cursor ..< start, 0...])
                }
                // Match the conv weight dtype: ImageIO produces fp32 while the
                // model (and the parity fixtures) are bf16 — without this the
                // shipped path runs the patchify at a different precision than
                // the gated one.
                let pixels = image.pixelValues.asType(visionModel.embeddings.patchEmbedding.weight.dtype)
                let vit = visionModel(pixels, gridH: image.gridH, gridW: image.gridW)
                segments.append(vit.reshaped([1, image.tokenCount, -1]).asType(tokenEmbeds.dtype))
                cursor = i
                imageIdx += 1
            } else {
                i += 1
            }
        }
        precondition(imageIdx == images.count, "unused images: \(images.count - imageIdx)")
        if cursor < s { segments.append(tokenEmbeds[0..., cursor ..< s, 0...]) }
        let embeds = segments.count == 1 ? segments[0] : concatenated(segments, axis: 1)

        let indexes = getTHWIndexes(ids: ids, images: images)
        let mask = createBlockCausalMask(tIndexes: indexes.t)
        let maxT = indexes.t.max().item(Int32.self)
        return (embeds, indexes, mask, maxT)
    }

    func prefillEmbeds(_ embeds: MLXArray, indexes: THWIndexes, mask: MLXArray) -> [KVPair] {
        let (_, kv) = languageModel.model(
            embeds: embeds, stream: .und, indexes: indexes, mask: mask, collectKV: true)
        eval(kv.map(\.keys) + kv.map(\.values))
        return kv
    }

    /// Mirrors `it2i_generate` (non-think, batch 1).
    /// - cond: prompt + reference images; imgCond: images only; uncond: empty.
    ///   Branch need is derived from (cfgScale, imgCfgScale) exactly as the
    ///   reference does; pass only the id sets the branches require.
    public func it2iGenerate(
        condIds: [Int32],
        imgCondIds: [Int32]?,
        uncondIds: [Int32]?,
        images: [EditImage],
        width: Int,
        height: Int,
        params: T2IParams = T2IParams(),
        imgCfgScale: Float = 1.0,
        injectedNoise: MLXArray? = nil,
        initImage: MLXArray? = nil,
        onStep: ((Int, Int) -> Void)? = nil
    ) throws -> MLXArray {
        let px = config.pixelsPerToken
        let (tokenH, tokenW) = (height / px, width / px)
        let (gridH, gridW) = (height / config.patchSize, width / config.patchSize)
        let l = tokenH * tokenW

        let needsCFG = !(params.cfgScale == 1 && imgCfgScale == 1)
        let needsImgCond = needsCFG && (imgCfgScale == 1 || params.cfgScale != imgCfgScale)
        let needsUncond = needsCFG && imgCfgScale != 1

        // -- prefill branches --
        let (condEmbeds, condIdx, condMask, condMaxT) = buildIT2IInputs(ids: condIds, images: images)
        let prefixCond = prefillEmbeds(condEmbeds, indexes: condIdx, mask: condMask)

        var prefixImgCond: [KVPair] = []
        var imgCondMaxT: Int32 = 0
        if needsImgCond {
            guard let imgCondIds else {
                throw SenseNovaError.badConfig(
                    "imgCondIds required for cfgScale=\(params.cfgScale), imgCfgScale=\(imgCfgScale)")
            }
            let (e, idx, m, maxT) = buildIT2IInputs(ids: imgCondIds, images: images)
            prefixImgCond = prefillEmbeds(e, indexes: idx, mask: m)
            imgCondMaxT = maxT
        }
        var prefixUncond: [KVPair] = []
        var uncondMaxT: Int32 = 0
        if needsUncond {
            guard let uncondIds else {
                throw SenseNovaError.badConfig(
                    "uncondIds required for imgCfgScale=\(imgCfgScale)")
            }
            prefixUncond = prefillText(uncondIds)
            uncondMaxT = Int32(uncondIds.count - 1)
        }

        func imageIndexes(afterT maxT: Int32) -> THWIndexes {
            let t = MLXArray([Int32](repeating: maxT + 1, count: l))
            let idx = MLXArray(Int32(0) ..< Int32(l))
            return THWIndexes(t: t, h: idx.floorDivide(MLXArray(Int32(tokenW))), w: idx % Int32(tokenW))
        }
        let idxCond = imageIndexes(afterT: condMaxT)
        let idxImgCond = imageIndexes(afterT: imgCondMaxT)
        let idxUncond = imageIndexes(afterT: uncondMaxT)

        // -- init noise (and, for an init-image run, the partial-t start) --
        let sigma = noiseScale(tokenH: tokenH, tokenW: tokenW)
        let noise: MLXArray
        if let injected = injectedNoise {
            noise = injected
        } else {
            MLXRandom.seed(params.seed)
            noise = (MLXRandom.normal([1, 3, height, width]) * sigma).asType(.bfloat16)
        }
        let startT: Float = initImage == nil ? 0 : params.startT
        var image = Self.initialState(
            noise: noise, initImage: initImage, startT: startT, mix: params.initNoiseMix,
            height: height, width: width)

        let timesteps = shiftedTimesteps(
            numSteps: params.numSteps, shift: params.timestepShift,
            enable: params.enableTimestepShift, startT: startT)

        // loop-invariant noise-scale embedding, at the reference's L-row width
        // (see t2iDenoise for why a 1-row broadcast is not equivalent)
        let noiseEmb: MLXArray? =
            config.addNoiseScaleEmbedding
            ? fmModules.noiseScaleEmbedder(
                MLXArray([Float](repeating: sigma / config.noiseScaleMaxValue, count: l)))
                .reshaped([1, l, -1])
            : nil

        for step in 0 ..< params.numSteps {
            try Task.checkCancellation()  // CAN cadence: per denoise step
            let t = timesteps[step]
            let tNext = timesteps[step + 1]
            // it2i interval semantics: exclusive bounds, or lo == 0 (verbatim)
            let useCFG = (t > params.cfgInterval.0 && t < params.cfgInterval.1) || params.cfgInterval.0 == 0

            var z = patchify(image, patchSize: px)
            let imageInput = patchifyChannelFirst(image, patchSize: config.patchSize)
                .reshaped([-1, 3 * config.patchSize * config.patchSize])
            var imageEmbeds = fmModules.visionGen(imageInput, gridH: gridH, gridW: gridW)
                .reshaped([1, l, -1])

            let tExpanded = MLXArray([Float](repeating: t, count: l))
            var timeEmb = fmModules.timestepEmbedder(tExpanded).reshaped([1, l, -1])
            if let noiseEmb { timeEmb = timeEmb + noiseEmb }
            imageEmbeds = imageEmbeds + timeEmb.asType(imageEmbeds.dtype)

            func branch(_ idx: THWIndexes, _ prefix: [KVPair]) -> MLXArray {
                predictV(
                    imageEmbeds: imageEmbeds, indexes: idx, prefixKV: prefix,
                    z: z, t: t, tokenH: tokenH, tokenW: tokenW, tEps: params.tEps)
            }

            let vCond = branch(idxCond, prefixCond)
            var v = vCond
            if useCFG, needsCFG {
                if params.cfgScale == 1 && imgCfgScale == 1 {
                    v = vCond
                } else if imgCfgScale == 1 {
                    let vImgCond = branch(idxImgCond, prefixImgCond)
                    v = vImgCond + params.cfgScale * (vCond - vImgCond)
                } else if params.cfgScale == imgCfgScale {
                    let vUncond = branch(idxUncond, prefixUncond)
                    v = vUncond + params.cfgScale * (vCond - vUncond)
                } else {
                    let vImgCond = branch(idxImgCond, prefixImgCond)
                    let vUncond = branch(idxUncond, prefixUncond)
                    v = vUncond
                        + params.cfgScale * (vCond - vImgCond)
                        + imgCfgScale * (vImgCond - vUncond)
                }
                if params.cfgScale > 1 || imgCfgScale > 1 {
                    switch params.cfgNorm {
                    case .global, .channel:
                        v = rescaleToCondNorm(v, vCond, mode: params.cfgNorm)
                    case .none, .cfgZeroStar:
                        break  // cfg_zero_star is a t2i-only option upstream
                    }
                }
            }

            z = z + (tNext - t) * v
            image = unpatchify(z, patchSize: px, height: height, width: width)
            eval(image)
            Memory.clearCache()
            onStep?(step + 1, params.numSteps)
        }
        return image
    }

    /// `cfg_norm` global/channel rescale (shared shape with t2i's inline code).
    func rescaleToCondNorm(_ v: MLXArray, _ vCond: MLXArray, mode: CFGNorm) -> MLXArray {
        let axes: [Int] = mode == .global ? [1, 2] : [2]
        let nc = MLX.sqrt((vCond.asType(.float32) * vCond.asType(.float32)).sum(axes: axes, keepDims: true))
        let nv = MLX.sqrt((v.asType(.float32) * v.asType(.float32)).sum(axes: axes, keepDims: true))
        let scale = clip(nc / (nv + 1e-8), min: 0, max: 1.0).asType(v.dtype)
        return v * scale
    }
}
