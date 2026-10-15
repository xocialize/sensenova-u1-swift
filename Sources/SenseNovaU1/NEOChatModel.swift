// NEOChatModel.swift — mirrors modeling_neo_chat.py (t2i path first; it2i/edit,
// think mode, and VQA chat follow in later milestones — see PORTING-SPEC §10).
//
// Pixel-space rectified flow: state z lives as 32px patches (channel-LAST,
// `nhwpqc`); every step the current image is ALSO re-patchified at 16px
// (channel-FIRST, `nhwcpq`) to feed the gen-stream vision patchify. The head
// predicts x̂ (clean image), velocity is derived: v = (x̂ − z)/max(1−t, t_eps).
// Defaults follow the reference CLI, not config.json: shift 3.0, t_eps 0.02,
// 50 steps, cfg 4.0 (trap ledger #4/#5).

import Foundation
import MLX
import MLXNN
import MLXRandom

public enum CFGNorm: String, Sendable {
    case none, global, channel, cfgZeroStar
}

public struct T2IParams: Sendable {
    public var numSteps = 50
    public var cfgScale: Float = 4.0
    public var cfgNorm: CFGNorm = .none
    public var timestepShift: Float = 3.0
    public var enableTimestepShift = true
    public var cfgInterval: (Float, Float) = (0.0, 1.0)
    public var tEps: Float = 0.02
    public var seed: UInt64 = 42
    /// Model t the loop starts at when an init image is given (0 = pure noise,
    /// the default and the only meaningful value without one; 1 = clean image).
    /// FluxSR's T_L = 0.35 noise fraction is `startT = 0.65` here.
    public var startT: Float = 0
    /// With an init image: mix noise in at `startT` (SDEdit, on-distribution
    /// for an untrained prior) or feed the clean image as the state (the
    /// FluxSR / HYPIR one-step entry a trained restorer expects).
    public var initNoiseMix = true
    public init() {}
}

public final class NEOChatModel: Module {
    public let config: NEOChatConfig

    @ModuleInfo(key: "vision_model") var visionModel: NEOVisionModel
    @ModuleInfo(key: "language_model") var languageModel: Qwen3MoTForCausalLM
    @ModuleInfo(key: "fm_modules") var fmModules: FMModules

    public init(_ config: NEOChatConfig) {
        self.config = config
        self._visionModel.wrappedValue = NEOVisionModel(config.vision)
        self._languageModel.wrappedValue = Qwen3MoTForCausalLM(config.llm)
        self._fmModules.wrappedValue = FMModules(config)
        super.init()
    }

    // MARK: - patchify / unpatchify (pure reshapes; images are NCHW like the reference)

    /// (B, 3, H, W) → (B, L, p²·3), per-patch layout (p, q, c) — `nhwpqc`.
    public func patchify(_ images: MLXArray, patchSize p: Int) -> MLXArray {
        let (b, h, w) = (images.dim(0), images.dim(2), images.dim(3))
        let (gh, gw) = (h / p, w / p)
        return images.reshaped([b, 3, gh, p, gw, p])
            .transposed(0, 2, 4, 3, 5, 1)
            .reshaped([b, gh * gw, p * p * 3])
    }

    /// (B, 3, H, W) → (B, gh·gw, 3·p²), per-patch layout (c, p, q) — `nhwcpq`.
    public func patchifyChannelFirst(_ images: MLXArray, patchSize p: Int) -> MLXArray {
        let (b, h, w) = (images.dim(0), images.dim(2), images.dim(3))
        let (gh, gw) = (h / p, w / p)
        return images.reshaped([b, 3, gh, p, gw, p])
            .transposed(0, 2, 4, 1, 3, 5)
            .reshaped([b, gh * gw, 3 * p * p])
    }

    /// inverse of `patchify`: (B, L, p²·3) → (B, 3, H, W).
    public func unpatchify(_ x: MLXArray, patchSize p: Int, height: Int, width: Int) -> MLXArray {
        let b = x.dim(0)
        let (gh, gw) = (height / p, width / p)
        return x.reshaped([b, gh, gw, p, p, 3])
            .transposed(0, 5, 1, 3, 2, 4)
            .reshaped([b, 3, gh * p, gw * p])
    }

    // MARK: - schedule

    /// `_apply_time_schedule` with the runtime-forced "standard" branch:
    /// σ = 1−t; σ′ = s·σ / (1 + (s−1)·σ); t′ = 1−σ′.
    ///
    /// `startT` (model t, 0 = pure noise, 1 = clean image) begins the schedule
    /// part-way for an init-image run: the linear grid runs from the t whose
    /// shifted value is `startT` up to 1, so the step spacing is the same shape
    /// a full schedule has over that interval. `startT == 0` is the full
    /// schedule, bit-identical to the pre-`startT` form (0 + 1·x == x).
    public func shiftedTimesteps(
        numSteps: Int, shift: Float, enable: Bool, startT: Float = 0
    ) -> [Float] {
        Self.timestepSchedule(numSteps: numSteps, shift: shift, enable: enable, startT: startT)
    }

    public static func timestepSchedule(
        numSteps: Int, shift: Float, enable: Bool, startT: Float = 0
    ) -> [Float] {
        let t0: Float = startT == 0 ? 0 : unshiftedT(startT, shift: shift, enable: enable)
        let linear = (0 ... numSteps).map { t0 + (1 - t0) * (Float($0) / Float(numSteps)) }
        guard enable else { return linear }
        return linear.map { t in
            let sigma = 1 - t
            let shifted = shift * sigma / (1 + (shift - 1) * sigma)
            return 1 - shifted
        }
    }

    /// Inverse of the shift: the linear t whose shifted value is `t`.
    /// σ′ = s·σ / (1 + (s−1)·σ)  ⇔  σ = σ′ / (s − (s−1)·σ′).
    public static func unshiftedT(_ t: Float, shift: Float, enable: Bool) -> Float {
        guard enable else { return t }
        let shiftedSigma = 1 - t
        let sigma = shiftedSigma / (shift - (shift - 1) * shiftedSigma)
        return 1 - sigma
    }

    /// The state the denoise loop starts from. With no `initImage` it is the
    /// (σ-scaled) noise. With one, at `startT`: the SDEdit mix
    /// t·x + (1−t)·noise (`initNoiseMix`), or the image itself (the FluxSR /
    /// HYPIR entry, which feeds a clean x as if it sat at t).
    public static func initialState(
        noise: MLXArray, initImage: MLXArray?, startT: Float, mix: Bool,
        height: Int, width: Int
    ) -> MLXArray {
        guard let x = initImage else { return noise }
        precondition(
            x.shape == [1, 3, height, width],
            "initImage shape \(x.shape) != [1, 3, \(height), \(width)]")
        guard mix else { return x.asType(.bfloat16) }
        let mixed = startT * x.asType(.float32) + (1 - startT) * noise.asType(.float32)
        return mixed.asType(.bfloat16)
    }

    /// resolution-scaled initial-noise σ: min(√(L/base), max) — merge² cancels
    /// against the 16px grid exactly as in the reference.
    public func noiseScale(tokenH: Int, tokenW: Int) -> Float {
        let l = Float(tokenH * tokenW)
        let raw = (l / config.noiseScaleBaseImageSeqLen).squareRoot() * config.noiseScale
        return min(raw, config.noiseScaleMaxValue)
    }

    // MARK: - prefill

    /// und-stream prefill of a text prompt: returns per-layer KV.
    /// Text tokens: t = position, h = w = 0; block-causal mask (pure causal here).
    public func prefillText(_ ids: [Int32]) -> [KVPair] {
        let s = ids.count
        let t = MLXArray(ids.indices.map { Int32($0) })
        let zerosIdx = MLXArray([Int32](repeating: 0, count: s))
        let indexes = THWIndexes(t: t, h: zerosIdx, w: zerosIdx)
        let mask = createBlockCausalMask(tIndexes: t)
        let embeds = languageModel.model.embedTokens(MLXArray(ids).reshaped([1, s]))
        let (_, kv) = languageModel.model(
            embeds: embeds, stream: .und, indexes: indexes, mask: mask, collectKV: true)
        eval(kv.map(\.keys) + kv.map(\.values))
        return kv
    }

    // MARK: - denoise core (mirrors `_t2i_predict_v`)

    func predictV(
        imageEmbeds: MLXArray,      // (B, L, hidden) — gen-vision embeds + time emb
        indexes: THWIndexes,
        prefixKV: [KVPair],
        z: MLXArray,                // (B, L, 3072)
        t: Float,
        tokenH: Int,
        tokenW: Int,
        tEps: Float
    ) -> MLXArray {
        let (hidden, _) = languageModel.model(
            embeds: imageEmbeds, stream: .gen, indexes: indexes, mask: nil,
            prefixKV: prefixKV, collectKV: false)
        let b = hidden.dim(0)
        let px = config.pixelsPerToken  // 32

        // token grid is NHWC natively for our ConvDecoder
        let grid = hidden.reshaped([b, tokenH, tokenW, hidden.dim(2)])
        let img = fmModules.fmHead(grid)  // (B, H, W, 3) NHWC
        // repack to z layout `b (h w) (p q c)`
        let xhat = img.reshaped([b, tokenH, px, tokenW, px, 3])
            .transposed(0, 1, 3, 2, 4, 5)
            .reshaped([b, tokenH * tokenW, px * px * 3])

        return (xhat - z) / max(1 - t, tEps)
    }

    // MARK: - CFG combine (mirrors t2i_generate exactly, incl. cfg_zero_star)

    func combineCFG(
        _ vCond: MLXArray, _ vUncond: MLXArray,
        cfgScale: Float, norm: CFGNorm, stepIndex: Int
    ) -> MLXArray {
        switch norm {
        case .cfgZeroStar:
            let b = vCond.dim(0)
            let pos = vCond.reshaped([b, -1]).asType(.float32)
            let neg = vUncond.reshaped([b, -1]).asType(.float32)
            let dot = (pos * neg).sum(axis: 1, keepDims: true)
            let sq = (neg * neg).sum(axis: 1, keepDims: true) + 1e-8
            var alpha = (dot / sq).asType(vCond.dtype)
            alpha = alpha.reshaped([b] + Array(repeating: 1, count: vCond.ndim - 1))
            if stepIndex <= 0 {
                return vCond * 0
            }
            return vUncond * alpha + cfgScale * (vCond - vUncond * alpha)
        case .none, .global, .channel:
            let v = vUncond + cfgScale * (vCond - vUncond)
            guard norm == .global || norm == .channel else { return v }
            return rescaleToCondNorm(v, vCond, mode: norm)
        }
    }

    // MARK: - T2I

    /// Mirrors `t2i_generate` (non-think, batch 1). Token ids come from the
    /// Conversation builder (or parity fixtures). `injectedNoise` (1,3,H,W),
    /// already σ-scaled, replaces the RNG for parity runs.
    /// Returns the final image in generation-normalized space (1, 3, H, W);
    /// denormalize with `x·0.5 + 0.5`.
    public func t2iGenerate(
        condIds: [Int32],
        uncondIds: [Int32]?,
        width: Int,
        height: Int,
        params: T2IParams = T2IParams(),
        injectedNoise: MLXArray? = nil,
        initImage: MLXArray? = nil,
        onStep: ((Int, Int) -> Void)? = nil
    ) throws -> MLXArray {
        let needsCFG = params.cfgScale > 1 && uncondIds != nil
        let prefixCond = prefillText(condIds)
        let prefixUncond = needsCFG ? prefillText(uncondIds!) : []
        return try t2iDenoise(
            prefixCond: prefixCond, condImageT: Int32(condIds.count),
            prefixUncond: prefixUncond,
            uncondImageT: needsCFG ? Int32(uncondIds!.count) : 0,
            needsCFG: needsCFG, width: width, height: height, params: params,
            injectedNoise: injectedNoise, initImage: initImage, onStep: onStep)
    }

    /// The shared t2i denoise loop (non-think and think paths).
    /// `condImageT` / `uncondImageT`: the constant temporal index the image
    /// tokens carry in each branch (reference: `text_len`, or `maxT + 1` after
    /// a think phase).
    func t2iDenoise(
        prefixCond: [KVPair],
        condImageT: Int32,
        prefixUncond: [KVPair],
        uncondImageT: Int32,
        needsCFG: Bool,
        width: Int,
        height: Int,
        params: T2IParams,
        injectedNoise: MLXArray?,
        initImage: MLXArray? = nil,
        onStep: ((Int, Int) -> Void)?
    ) throws -> MLXArray {
        let px = config.pixelsPerToken           // 32
        let (tokenH, tokenW) = (height / px, width / px)
        let (gridH, gridW) = (height / config.patchSize, width / config.patchSize)
        let l = tokenH * tokenW

        // -- image-token indexes: t = const per branch, h/w = grid coords --
        func imageIndexes(t tVal: Int32) -> THWIndexes {
            let t = MLXArray([Int32](repeating: tVal, count: l))
            let idx = MLXArray(Int32(0) ..< Int32(l))
            let h = idx.floorDivide(MLXArray(Int32(tokenW)))
            let w = idx % Int32(tokenW)
            return THWIndexes(t: t, h: h, w: w)
        }
        let idxCond = imageIndexes(t: condImageT)
        let idxUncond = needsCFG ? imageIndexes(t: uncondImageT) : idxCond

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

        // sigma is loop-invariant, so the noise-scale embedding is too — compute
        // it once at the SAME L-row width the reference uses. (Computing it on a
        // single row and broadcasting is NOT equivalent: MLX dispatches a
        // different matmul kernel at M=1 and the rounding difference amplifies
        // through the 42-layer stack — measured e2e cos 0.986 → 0.969.)
        let noiseEmb: MLXArray? =
            config.addNoiseScaleEmbedding
            ? fmModules.noiseScaleEmbedder(
                MLXArray([Float](repeating: sigma / config.noiseScaleMaxValue, count: l)))
                .reshaped([1, l, -1])
            : nil

        // -- denoise loop --
        for step in 0 ..< params.numSteps {
            try Task.checkCancellation()  // CAN cadence: per denoise step
            let t = timesteps[step]
            let tNext = timesteps[step + 1]

            var z = patchify(image, patchSize: px)
            let imageInput = patchifyChannelFirst(image, patchSize: config.patchSize)
                .reshaped([-1, 3 * config.patchSize * config.patchSize])  // reference `.view(B·grid, -1)`
            var imageEmbeds = fmModules.visionGen(imageInput, gridH: gridH, gridW: gridW)
                .reshaped([1, l, -1])

            // time (+ noise-scale) conditioning, added to every image token
            let tExpanded = MLXArray([Float](repeating: t, count: l))
            var timeEmb = fmModules.timestepEmbedder(tExpanded).reshaped([1, l, -1])
            if let noiseEmb { timeEmb = timeEmb + noiseEmb }
            imageEmbeds = imageEmbeds + timeEmb.asType(imageEmbeds.dtype)

            let vCond = predictV(
                imageEmbeds: imageEmbeds, indexes: idxCond, prefixKV: prefixCond,
                z: z, t: t, tokenH: tokenH, tokenW: tokenW, tEps: params.tEps)

            var v = vCond
            // t2i interval semantics: inclusive bounds (differs from interleave!)
            if needsCFG, t >= params.cfgInterval.0, t <= params.cfgInterval.1 {
                let vUncond = predictV(
                    imageEmbeds: imageEmbeds, indexes: idxUncond, prefixKV: prefixUncond,
                    z: z, t: t, tokenH: tokenH, tokenW: tokenW, tEps: params.tEps)
                v = combineCFG(vCond, vUncond, cfgScale: params.cfgScale,
                               norm: params.cfgNorm, stepIndex: step)
            }

            z = z + (tNext - t) * v
            image = unpatchify(z, patchSize: px, height: height, width: width)
            eval(image)
            Memory.clearCache()  // long-denoise buffer-cache ratchet discipline
            onStep?(step + 1, params.numSteps)
        }
        return image
    }
}
