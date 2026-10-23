// MLXEngine package over the SenseNovaU1 core — one resident unified model
// serving `textToImage`, `imageEdit` AND `imageAnalysis` (VQA) on every tier.
//
// SenseNova-U1.5-8B-MoT (Apache-2.0): NEO-unify Mixture-of-Transformers,
// pixel-space rectified flow, no VAE. The Swift core is parity-locked against
// the reference PyTorch implementation (components <1e-4; e2e per-pass cos
// 0.999+; edit e2e 34.6 dB) — this wrapper is a thin conformance layer.
//
// Weights are fetched from OUR mlx-community artifacts (fetch address), while
// `Provenance.sourceRepo` names the upstream ORIGIN — the split-provenance
// rule; `ProvenanceSplitTests` asserts the two never swap.

import CoreGraphics
import Foundation
import ImageIO
import MLX
import MLXToolKit
import SenseNovaU1
import UniformTypeIdentifiers

/// The published artifact tiers (mlx-community). `8step` tiers carry the
/// official 8-step distillation LoRA pre-merged and run cfg-free — fast T2I
/// only; use `bf16`/`q8` for 50-step quality T2I and editing. `pose` tiers
/// carry the RefControl pose adapter pre-merged and keep every surface.
public enum SenseNovaVariant: String, Codable, Sendable, CaseIterable {
    case bf16 = "bf16"
    case q8 = "8bit"
    case fast8 = "8step-8bit"
    case fast4 = "8step-4bit"

    /// **8-step distill, V2 adapter** (upstream `SenseNova-U1.5-8B-MoT-LoRA-8step-V2`,
    /// 2026-09-24, upstream's recommended 8-step LoRA: better colour balance,
    /// more natural contrast, less oversharpening than V1). Same 882-tensor
    /// adapter layout as V1, merged through the same path; separate repos so the
    /// V1 tiers keep reproducing their published renders at fixed seed (AB-T-0228).
    case fast8V2 = "8step-v2-8bit"
    case fast4V2 = "8step-v2-4bit"

    /// **Pose specialty, bf16.** The RefControl pose adapter (v1 @3000, rank 32
    /// / alpha 32, 294 `*_mot_gen` gen-stream projections) merged into the base
    /// checkpoint at fp32 and cast — the same merge the base tiers get for the
    /// 8-step distill, over a different adapter.
    case poseBf16 = "pose-bf16"

    /// **Pose specialty, int8** — the shipping pose tier. The merged 8-bit
    /// artifact reproduces the bf16 renders on all five valid held-out fixtures
    /// (AB-R-0194), so the 20 GB tier is not a quality compromise here. The
    /// duplicate-figure effect seen at an earlier checkpoint under int8 was
    /// checkpoint-specific, not a property of the tier.
    case pose8 = "pose-8bit"

    public var repo: String { "mlx-community/SenseNova-U1.5-8B-MoT-\(rawValue)" }

    public var quant: Quant {
        switch self {
        case .bf16, .poseBf16: return .bf16
        case .q8, .fast8, .fast8V2, .pose8: return .int8
        case .fast4, .fast4V2: return .int4
        }
    }

    /// LoRA-merged distill tiers run cfg-free at 8 steps (the vendor contract).
    public var isDistilled: Bool {
        self == .fast8 || self == .fast4 || self == .fast8V2 || self == .fast4V2
    }

    /// Pose tiers are a **base-checkpoint merge, not a distill** — the sampler
    /// contract is unchanged (50-step-class, cfg 4), and think mode stays enabled.
    public var isPoseTuned: Bool { self == .pose8 || self == .poseBf16 }

    /// 28 steps on the pose tiers is the A/B's setting, not a guess: the
    /// 28-vs-50 comparison at fixed seed costs 1.84× for ~6 % more
    /// high-frequency texture, and every pose receipt was rendered at 28.
    public var defaultSteps: Int {
        if isDistilled { return 8 }
        return isPoseTuned ? 28 : 50
    }

    public var defaultGuidance: Float { isDistilled ? 1.0 : 4.0 }

    /// The surfaces this tier serves — all three, on every tier. The 8-step
    /// distill adapters touch only the 294 gen-stream (`*_mot_gen`) projections:
    /// against the base artifact at the same quant exactly 882 tensors differ
    /// and the understanding stream is bit-identical, so VQA answers match the
    /// base tier token for token; and the distilled edit at 8 steps / cfg 1
    /// matches the 50-step cfg-4 base edit (28.7 dB) in a tenth of the time
    /// (AB-R-0490). The gates that refused edit and VQA on distill tiers
    /// (2026-08 → 2026-10-11) were never evidence-based. Think mode remains the
    /// one base-tier-only feature (`distilledTierCannotThink`).
    public var capabilities: Set<Capability> { [.textToImage, .imageEdit, .imageAnalysis] }

    /// C6 model-level ranking metadata. Only the pose tiers advertise
    /// `.poseDriven` — the base tiers read a skeleton as something to paste,
    /// not as a pose to apply (the zero-adapter experiment in the eval), so
    /// declaring it fleet-wide for this package would mis-rank them.
    public var specialties: [SpecialtyWeight] {
        isPoseTuned ? [SpecialtyWeight(.poseDriven, strength: 0.8)] : []
    }

    /// The two-slot prompt the pose adapter was trained and A/B-ed with.
    ///
    /// **Slot order is load-bearing**: image 1 is the POSE (an OpenPose-style
    /// skeleton render), image 2 is the IDENTITY/appearance reference. Pass
    /// them to `IEditRequest(images:)` in that order —
    /// `IEditRequest(images: [skeleton, identity], prompt: SenseNovaVariant.posePrompt)`.
    /// Swapping the slots degrades the render (the `pose-slot-swap` gate); a
    /// weak skeleton (no arm joints) also loses to the reference's own pose
    /// prior, which is a fixture-quality failure rather than a model one
    /// (AB-R-0196).
    public static let posePrompt =
        "Image-1: <image>\nImage-2: <image>\n"
        + "apply pose from image 1 with reference from image 2"
}

/// The working set a configuration is declared for: a second footprint over
/// the SAME artifact, charged through `FootprintConfigured`, and the request
/// bounds `run()` enforces so that the declaration stays true.
///
/// The manifest's footprints are quant-keyed and sized for the native
/// 2048²-class envelope, so a 16 GB Mac (≈ 12.7 GB budget) refuses the int4
/// tier's 12.2 + 4.5 GB, although its 768² fill measures 12.51 GB. A window
/// declares only what was measured: `footprint(for:)` is nil for every other
/// variant, and `load()` refuses such a pairing.
public enum SenseNovaWindow: String, Codable, Sendable, CaseIterable {
    /// The native envelope: outputs and references up to the 2048²-class
    /// buckets, multi-reference edits, either guidance axis. The manifest's
    /// quant-keyed footprints apply.
    case native

    /// **The 16 GB Generative Fill window** (AB-D-0141, AB-T-0278), declared
    /// for `8step-v2-4bit`: output ≤ 768² by area, one reference, scaled into
    /// the same area, cfg-free (guidance 1, imageGuidance 1). VQA images are
    /// scaled into it too.
    ///
    /// Measured with a 768² single-reference edit, 8 steps, cfg 1, the same to
    /// the MiB on three machines: floor 11,230 MiB and peak 11,930 MiB on a
    /// real 16 GB M1 Pro (AB-R-0602), 11,235 / 11,931 on the 24 GB M5 Pro
    /// (AB-R-0551) and the M5 Max (AB-R-0537). Declared 11.8 GB resident +
    /// 0.75 GB activation = 12.55 GB, under the 12.7 GB budget. On the 16 GB
    /// Mac the process `phys_footprint` was 12.60 GB, swap grew 1.9 GB on the
    /// first load only, and a fill took 41–43 s warm (5.1–5.4 s/step).
    ///
    /// 1024² peaks 12,203 MiB, 973 MiB over the floor: declared honestly that
    /// is 12.82 GB, over the budget, so the window refuses it rather than
    /// declare it (it did run on the 16 GB Mac, with nothing else resident).
    /// A second reference or a guidance branch adds a sequence per step, so
    /// the window refuses those too.
    case fill768 = "fill-768"

    /// The window's side; its area is the most an output or reference may
    /// cover. nil = native.
    public var side: Int? {
        switch self {
        case .native: return nil
        case .fill768: return 768
        }
    }

    public var maxPixels: Int? { side.map { $0 * $0 } }

    /// The declared footprint for `variant` under this window, or nil: native
    /// (the manifest's quant-keyed footprints apply) or not measured.
    public func footprint(for variant: SenseNovaVariant) -> QuantFootprint? {
        switch (self, variant) {
        case (.fill768, .fast4V2):
            return QuantFootprint(
                quant: .int4, residentBytes: 11_800_000_000,
                peakActivationBytes: 750_000_000)
        default:
            return nil
        }
    }
}

/// Init-time configuration (C9): variant + explicit snapshot override + defaults.
public struct SenseNovaU1Configuration:
    PackageConfiguration, ModelStorable, QuantConfigured, FootprintConfigured, WeightSourcing
{
    public var variant: SenseNovaVariant
    /// The working set this configuration is declared for. `.native` (the
    /// default) charges the manifest's quant-keyed footprint; `.fill768` is
    /// the 16 GB fill tier's declaration over the same artifact.
    public var window: SenseNovaWindow
    /// Explicit artifact root (config.json + model-*.safetensors). Empty =
    /// resolve from the model store, materializing from `variant.repo` first run.
    public var snapshotPath: String
    public var modelsRootDirectory: URL?

    public var quant: Quant { variant.quant }

    /// The window's declared split, over the quant-keyed manifest footprint;
    /// nil (native) falls through to it.
    public var residentBytesHint: UInt64? { window.footprint(for: variant)?.residentBytes }
    public var peakActivationBytesHint: UInt64? {
        window.footprint(for: variant)?.peakActivationBytes
    }

    /// Fresh-machine sources (MAT): one self-contained artifact per variant —
    /// quant-tiered configs exclude every byte their tier doesn't need by
    /// construction (each tier is its own repo).
    ///
    /// ⚠ `revision: "main"` here is OUR artifact repo's main, not the upstream
    /// checkpoint's — this seam never fetches `sensenova/…`. The upstream pin
    /// lives on `Provenance.revision` (see the manifest), and it stays pinned
    /// precisely so an upstream re-shard cannot silently change what these
    /// artifacts were built from.
    public var weightSources: [WeightSource] {
        [WeightSource(role: "model", repo: variant.repo, revision: "main", matching: ["*"])]
    }

    public func missingWeightSources(storeRoot: URL?) -> [WeightSource] {
        if !snapshotPath.isEmpty,
            FileManager.default.fileExists(
                atPath: URL(fileURLWithPath: snapshotPath)
                    .appendingPathComponent("config.json").path)
        {
            return []
        }
        return defaultMissingWeightSources(storeRoot: storeRoot)
    }

    public func resolvedSnapshotDirectory(storeRoot: URL?) -> URL? {
        if !snapshotPath.isEmpty { return URL(fileURLWithPath: snapshotPath) }
        let store = ModelStore(root: storeRoot)
        let fm = FileManager.default
        if let flat = store.directory(for: variant.repo),
            fm.fileExists(atPath: flat.appendingPathComponent("config.json").path)
        {
            return flat
        }
        if let snap = store.snapshotDirectory(for: variant.repo, revision: "main"),
            fm.fileExists(atPath: snap.appendingPathComponent("config.json").path)
        {
            return snap
        }
        return store.directory(for: variant.repo)
    }

    public init(
        variant: SenseNovaVariant = .q8,
        window: SenseNovaWindow = .native,
        snapshotPath: String = "",
        modelsRootDirectory: URL? = nil
    ) {
        self.variant = variant
        self.window = window
        self.snapshotPath = snapshotPath
        self.modelsRootDirectory = modelsRootDirectory
    }

    private enum CodingKeys: String, CodingKey {
        case variant, window, snapshotPath
    }

    /// Configurations encoded before 0.6.0 carry no window: they are native.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        variant = try container.decode(SenseNovaVariant.self, forKey: .variant)
        window = try container.decodeIfPresent(SenseNovaWindow.self, forKey: .window) ?? .native
        snapshotPath = try container.decode(String.self, forKey: .snapshotPath)
        modelsRootDirectory = nil
    }
}

public enum SenseNovaU1PackageError: Error, LocalizedError {
    case unreadableSnapshot(String)
    case imageDecode
    case pngEncode
    @available(*, deprecated, message: "no longer thrown: distill tiers serve imageEdit (AB-R-0490)")
    case distilledTierCannotEdit
    case distilledTierCannotThink
    case negativePromptUnsupportedOnEdit
    case sizeNotTokenAligned(width: Int, height: Int, multiple: Int)
    @available(*, deprecated, message: "no longer thrown: distill tiers serve imageAnalysis (AB-R-0490)")
    case distilledTierCannotAnalyze
    case metaDataOutOfRange(key: String, value: Double, range: ClosedRange<Double>)
    /// A window with no declared footprint for this variant (`load()`).
    case windowNotDeclared(window: SenseNovaWindow, variant: SenseNovaVariant)
    /// A request outside the configuration's window (`run()`).
    case outsideWindow(window: SenseNovaWindow, reason: String)

    public var errorDescription: String? {
        switch self {
        case .unreadableSnapshot(let p): return "SenseNova artifact not readable at \(p)."
        case .imageDecode: return "Could not decode an input image."
        case .pngEncode: return "PNG encoding failed."
        case .distilledTierCannotEdit:
            return "No longer thrown: the 8step distill tiers serve imageEdit (AB-R-0490)."
        case .distilledTierCannotThink:
            return "Think mode needs the base tiers; use the bf16 or 8bit variant."
        case .negativePromptUnsupportedOnEdit:
            return "negativePrompt is not wired on the edit surface (the branch that would "
                + "carry it is disabled at imgCfgScale 1); omit it or use textToImage."
        case .sizeNotTokenAligned(let w, let h, let m):
            return "width/height must be multiples of \(m) (got \(w)x\(h))."
        case .distilledTierCannotAnalyze:
            return "No longer thrown: the 8step distill tiers serve imageAnalysis (AB-R-0490)."
        case .metaDataOutOfRange(let key, let value, let range):
            return "metaData \(key) = \(value) is outside \(range.lowerBound)...\(range.upperBound)."
        case .windowNotDeclared(let window, let variant):
            return "The \(window.rawValue) window declares no footprint for \(variant.rawValue) "
                + "(a window is declared only where it was measured); use the native window."
        case .outsideWindow(let window, let reason):
            return "Outside the \(window.rawValue) window: \(reason)."
        }
    }
}

@InferenceActor
public final class SenseNovaU1Package: ModelPackage {
    public typealias Configuration = SenseNovaU1Configuration

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            license: LicenseDeclaration(weightLicense: .apache2, portCodeLicense: .mit),
            // Upstream origin, PINNED. Hub `main` moved on 2026-08-24 to
            // `19bc874e` ("Convert to BF16 and re-shard": 8 shards / 32.7 GB,
            // against the pinned `07d76f61`'s fp32 13 shards / 47 GB). Same
            // weights at the dtype this port casts to at load, so no behaviour
            // change is expected — but every published artifact was converted
            // from `07d76f61`, and a rebuild from today's `main` starts from a
            // different index and shard layout. The pin is what makes that
            // statement checkable, so it does NOT follow `main`; the note
            // rides the registry row. Nothing the wrapper fetches is affected
            // (see `weightSources` — that address is our own mlx-community
            // artifact repo).
            provenance: Provenance(
                sourceRepo: "sensenova/SenseNova-U1.5-8B-MoT",
                revision: "07d76f61474b9c6e6999e22c559314d3439b8c81", tier: 3),
            requirements: RequirementsManifest(
                // Split footprints, 2048² envelope, M5 Max.
                //
                // int8 is RE-BASELINED from in-app `phys_footprint` (SenseNova
                // Demo, Engine pane): floor 19.97 GB measured post-load with the
                // MLX pool trimmed, worst peak 23.5 GB over a cold+warm pair ⇒
                // activation 3.53 GB. The previous split (19.5 / 5.0) was wrong
                // in BOTH directions at once — under-declaring the persistent
                // floor the governor reserves for the model's lifetime while
                // over-declaring the transient by 1.4×, which needlessly denies
                // co-residency to other models. The totals happened to match,
                // which is exactly why only the split shows the error.
                //
                // 2026-08-26: ALL FOUR tiers now measured headless, one process
                // per number (in-process residue makes back-to-back runs
                // unusable), with the pre-load baseline captured so `resident`
                // is weights and not process overhead — it measured 0.01 GB, so
                // floor IS weights here. bf16 and int4 were under-declared by
                // ~3.3% against the CLI-derived figures and are corrected:
                // bf16 35.17 GB measured (was 34.0), int4 11.87 (was 11.5).
                // int8 measured 19.98 against 20.5 declared — left alone, since
                // over-declaring resident is the safe direction.
                //
                // Activations are LEFT AS DECLARED. Measured T2I peaks at the
                // 2048² envelope are 2.93 / 3.35 / 2.52 GB, comfortably inside
                // 4.0 / 4.2 / 4.5. Editing (2026-10-11, headless CLI, 768² → 2048²
                // single-reference edit, 8 steps, cfg 1, 8step-v2 tiers): peak
                // 22.9 GB int8 / 15.2 GB int4 — 3.97 GB over the post-load
                // active set on both, i.e. the vision-token activation lands
                // inside the declared 4.2 / 4.5 GB and the peak equals the load
                // peak. bf16 edit is unmeasured; it runs on the same headroom.
                //
                // 2026-10-14: the artifact load no longer evaluates the random-init
                // model (WeightLoading.loadArtifact quantizes structurally), so the
                // load peak IS the floor: 11.2 GB int4 / 19.0 GB int8. A 1024²
                // single-reference edit then peaks at 12.2 GB int4 / 19.9 GB int8
                // (768²: 11.9 / 19.7), i.e. the int4 tier's 1024² fill window sits
                // under a 12.7 GB MLX limit (16 GB class) with 0.5 GB to spare.
                // The declared activation stays the 2048² envelope's.
                //
                // 2026-10-22: the 16 GB fill tier is a second declaration over
                // the int4 artifact, `SenseNovaWindow.fill768` (11.8 + 0.75 GB,
                // measured on a real 16 GB Mac, AB-R-0602), charged through the
                // configuration's `FootprintConfigured` hints. These quant-keyed
                // footprints are unchanged; they stay the native envelope's.
                footprints: [
                    QuantFootprint(
                        quant: .bf16, residentBytes: 35_500_000_000,
                        peakActivationBytes: 4_000_000_000),
                    QuantFootprint(
                        quant: .int8, residentBytes: 20_500_000_000,
                        peakActivationBytes: 4_200_000_000),
                    QuantFootprint(
                        quant: .int4, residentBytes: 12_200_000_000,
                        peakActivationBytes: 4_500_000_000),
                ],
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
                // Pro-class since 2026-10-14: the int4 tier measured on a 24 GB
                // M5 Pro (16-core GPU) with the Max's memory profile to the MiB
                // (floor 11,235 MiB; 1024² edit peak 12,203, 2048² 12,569) at
                // 2.1–2.4× the Max's step time, bit-identical across runs, and
                // admitted as declared on that machine's 19.07 GB budget
                // (Docs/receipts/int4-m5pro-24gb). A 16 GB M1 Pro matched it to
                // the MiB as well (AB-R-0602). Base chips are unmeasured, so the
                // floor stays here until a base-chip receipt relaxes it (AB-A-0181).
                // DeviceProfile.current() reports .max on every host today, so
                // this floor only bites once the engine detects real tiers.
                chipFloor: .pro
            ),
            specialties: [],
            surfaces: [
                T2IContract.descriptor(
                    name: "sensenova-u1",
                    summary: "SenseNova-U1.5-8B-MoT unified T2I (pixel-space rectified "
                        + "flow, no VAE): native 2048²-class buckets, text rendering, "
                        + "50-step base or cfg-free 8-step distill tiers. metaData: "
                        + "think=true for the reasoning mode (base tiers).",
                    modes: []
                ),
                IEditContract.descriptor(
                    name: "sensenova-u1-edit",
                    summary: "Identity-preserving instruction editing on the same "
                        + "resident model (image-conditioned generation; 50 steps / "
                        + "cfg 4 on the base and pose tiers, 8 steps cfg-free on the "
                        + "8step distill tiers; metaData imageGuidance (1...4) adds "
                        + "the image-CFG axis). N reference images are conditioned "
                        + "on in the order given.",
                    modes: []
                ),
                ImageAnalysisContract.descriptor(
                    name: "sensenova-u1-vqa",
                    summary: "Visual question answering on the SAME resident model "
                        + "(understanding stream): grounded description, reading, and "
                        + "reasoning about an image. metaData: maxTokens, temperature. "
                        + "Every tier (the distill adapters leave the understanding "
                        + "stream untouched).",
                    modes: []
                ),
            ]
        )
    }

    private let configuration: Configuration
    private var model: NEOChatModel?
    private var tokenizer: SenseNovaTokenizer?

    /// metaData is caller-supplied and may arrive typed (native clients) or as
    /// strings (JSON tool-calling), so every reader accepts both.
    nonisolated static func number(
        _ metaData: MetaData, _ key: String, in range: ClosedRange<Double>
    ) throws -> Double? {
        let raw: Double?
        switch metaData[key] {
        case .double(let d): raw = d
        case .int(let i): raw = Double(i)
        case .string(let s): raw = Double(s)
        default: raw = nil
        }
        guard let value = raw else { return nil }
        guard range.contains(value) else {
            throw SenseNovaU1PackageError.metaDataOutOfRange(key: key, value: value, range: range)
        }
        return value
    }

    nonisolated static func flag(_ metaData: MetaData, _ key: String) -> Bool {
        switch metaData[key] {
        case .bool(let b): return b
        case .string(let s): return (s as NSString).boolValue
        case .int(let i): return i != 0
        default: return false
        }
    }

    /// The window's request bounds: the shape its footprint was measured on
    /// (output area, one reference, no guidance branch). Native admits all.
    nonisolated static func checkWindow(
        _ window: SenseNovaWindow, width: Int, height: Int, references: Int,
        guidance: Float, imageGuidance: Float
    ) throws {
        guard let side = window.side, let maxPixels = window.maxPixels else { return }
        if width * height > maxPixels {
            throw SenseNovaU1PackageError.outsideWindow(
                window: window, reason: "\(width)×\(height) is larger than \(side)² by area")
        }
        if references > 1 {
            throw SenseNovaU1PackageError.outsideWindow(
                window: window, reason: "it takes one reference image, got \(references)")
        }
        if guidance != 1 || imageGuidance != 1 {
            throw SenseNovaU1PackageError.outsideWindow(
                window: window,
                reason: "it runs cfg-free (guidance and imageGuidance 1), "
                    + "got \(guidance) and \(imageGuidance)")
        }
    }

    /// A reference or VQA image, scaled into the window's area when there is one.
    private func loadImage(_ data: Data) throws -> EditImage {
        guard let maxPixels = configuration.window.maxPixels else {
            return try SenseNovaImageIO.loadEditImage(data: data)
        }
        return try SenseNovaImageIO.loadEditImage(data: data, maxPixels: maxPixels)
    }

    public nonisolated init(configuration: Configuration) {
        self.configuration = configuration
    }

    public func load() async throws {
        guard model == nil else { return }
        let window = configuration.window
        if window != .native, window.footprint(for: configuration.variant) == nil {
            throw SenseNovaU1PackageError.windowNotDeclared(
                window: window, variant: configuration.variant)
        }
        // Engine-executed materialization (contract 1.24) fetches the declared
        // artifact before load(); this guard is the offline backstop.
        guard let snapshot = configuration.resolvedSnapshotDirectory(
            storeRoot: configuration.modelsRootDirectory),
            FileManager.default.fileExists(
                atPath: snapshot.appendingPathComponent("config.json").path)
        else {
            throw SenseNovaU1PackageError.unreadableSnapshot(
                configuration.snapshotPath.isEmpty
                    ? configuration.variant.repo : configuration.snapshotPath)
        }
        // Artifacts load with zero conversion transient (mmap + verified
        // update); the HF-checkpoint fallback path sanitizes at load.
        let model = WeightLoading.isArtifact(snapshot)
            ? try WeightLoading.loadArtifact(from: snapshot)
            : try WeightLoading.load(from: snapshot, dtype: .bfloat16)
        self.tokenizer = try await SenseNovaTokenizer.load(from: snapshot)
        self.model = model
    }

    public func unload() async {
        model = nil
        tokenizer = nil
        MLX.Memory.clearCache()  // release the retained MLX pool so eviction frees RSS
    }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        // CAN-1: entry checkpoint FIRST — before notLoaded validation. Mid-run
        // cadence lives in the core (per denoise step / per decoded token).
        try Task.checkCancellation()
        guard let model, let tokenizer else { throw PackageError.notLoaded }

        guard configuration.variant.capabilities.contains(request.capability) else {
            throw PackageError.unsupportedCapability(request.capability)
        }
        switch request.capability {
        case .textToImage:
            guard let t2i = request as? T2IRequest else {
                throw PackageError.unsupportedCapability(request.capability)
            }
            return try await runT2I(t2i, model: model, tokenizer: tokenizer)
        case .imageAnalysis:
            guard let analysis = request as? ImageAnalysisRequest else {
                throw PackageError.unsupportedCapability(request.capability)
            }
            return try await runAnalysis(analysis, model: model, tokenizer: tokenizer)
        case .imageEdit:
            guard let edit = request as? IEditRequest else {
                throw PackageError.unsupportedCapability(request.capability)
            }
            return try await runEdit(edit, model: model, tokenizer: tokenizer)
        default:
            throw PackageError.unsupportedCapability(request.capability)
        }
    }

    /// Contract 1.18: report at the same seams the CAN checkpoints sit on, so a
    /// consumer bound to `engine.runProgress` sees a live step counter instead of
    /// an indeterminate spinner for the length of a 2048² render.
    nonisolated static var denoiseReporter: (Int, Int) -> Void {
        { step, totalSteps in
            RunProgress.report(.denoise, step: step, totalSteps: totalSteps)
        }
    }

    // MARK: - T2I

    private func runT2I(
        _ request: T2IRequest, model: NEOChatModel, tokenizer: SenseNovaTokenizer
    ) async throws -> T2IResponse {
        let variant = configuration.variant
        var params = T2IParams()
        params.numSteps = request.steps ?? configuration.variant.defaultSteps
        params.cfgScale = request.guidanceScale.map(Float.init) ?? configuration.variant.defaultGuidance
        params.seed = request.seed ?? 0
        let width = request.width ?? configuration.window.side ?? 1024
        let height = request.height ?? configuration.window.side ?? 1024

        // token grid is 32 px; a non-multiple silently truncates via integer
        // division and would make the declared Image size a lie.
        let px = model.config.pixelsPerToken
        guard width % px == 0, height % px == 0 else {
            throw SenseNovaU1PackageError.sizeNotTokenAligned(
                width: width, height: height, multiple: px)
        }
        try Self.checkWindow(
            configuration.window, width: width, height: height, references: 0,
            guidance: params.cfgScale, imageGuidance: 1)

        let think: Bool = {
            switch request.metaData["think"] {
            case .bool(let b): return b
            case .string(let raw): return (raw as NSString).boolValue
            default: return false
            }
        }()
        if think, variant.isDistilled {
            throw SenseNovaU1PackageError.distilledTierCannotThink
        }
        let image: MLXArray
        if think {
            let condThinkIds = tokenizer.encode(
                Conversation.buildPrompt(
                    userMessage: request.prompt,
                    systemMessage: Conversation.systemMessageForGen,
                    appendText: "<think>\n"))
            let uncond = params.cfgScale > 1
                ? tokenizer.encode(
                    Conversation.t2iUncondPrompt(negativePrompt: request.negativePrompt ?? ""))
                : nil
            let suffix = tokenizer.encode("\n\n" + Conversation.imgStartToken)
            var thinkTokens = 0
            (image, _) = try model.t2iGenerateThink(
                condThinkIds: condThinkIds, uncondIds: uncond, imgSuffixIds: suffix,
                width: width, height: height, params: params,
                onToken: { _ in
                    thinkTokens += 1
                    RunProgress.report(.generate, step: thinkTokens)
                },
                onStep: Self.denoiseReporter)
        } else {
            let (cond, uncond) = tokenizer.t2iIDs(
                prompt: request.prompt, negativePrompt: request.negativePrompt ?? "")
            image = try model.t2iGenerate(
                condIds: cond, uncondIds: params.cfgScale > 1 ? uncond : nil,
                width: width, height: height, params: params,
                onStep: Self.denoiseReporter)
        }
        try Task.checkCancellation()
        return T2IResponse(image: try Self.pngArtifact(image))
    }

    // MARK: - Edit

    private func runEdit(
        _ request: IEditRequest, model: NEOChatModel, tokenizer: SenseNovaTokenizer
    ) async throws -> IEditResponse {
        guard !request.images.isEmpty else { throw SenseNovaU1PackageError.imageDecode }
        if let negative = request.negativePrompt, !negative.isEmpty {
            throw SenseNovaU1PackageError.negativePromptUnsupportedOnEdit
        }
        let inputs = try request.images.map { try loadImage($0.data) }
        // Output size: explicit wins, else first input's aspect at ~2048² px
        // (reference `_resolve_output_size`); under a window, the first
        // input's own size, which loadImage already scaled into it.
        let px = model.config.pixelsPerToken
        let (width, height): (Int, Int)
        if let w = request.width, let h = request.height {
            guard w % px == 0, h % px == 0 else {
                throw SenseNovaU1PackageError.sizeNotTokenAligned(width: w, height: h, multiple: px)
            }
            (width, height) = (w, h)
        } else if configuration.window != .native {
            (width, height) = (inputs[0].gridW * 16, inputs[0].gridH * 16)
        } else {
            let target = 2048 * 2048
            let (h, w) = SenseNovaImageIO.smartResize(
                height: inputs[0].gridH * 16, width: inputs[0].gridW * 16,
                factor: 32, minPixels: target, maxPixels: target)
            (width, height) = (w, h)
        }

        var params = T2IParams()
        // Tier defaults, as T2I: 8 steps / cfg 1 on the distill tiers (the
        // vendor's 8-step edit recipe), 28 on pose (its A/B setting), 50 / cfg 4
        // on the base tiers.
        params.numSteps = request.steps ?? configuration.variant.defaultSteps
        params.cfgScale = request.guidanceScale.map(Float.init) ?? configuration.variant.defaultGuidance
        params.seed = request.seed ?? 0

        // The image-CFG axis (the reference's `img_cfg_scale`, the Space's
        // "Image Guidance"): at 1.0 the uncond branch is unused; above it the
        // three-branch combine needs an empty-prompt prefix too.
        let imgCfgScale = Float(
            try Self.number(request.metaData, "imageGuidance", in: 1.0...4.0) ?? 1.0)
        try Self.checkWindow(
            configuration.window, width: width, height: height, references: inputs.count,
            guidance: params.cfgScale, imageGuidance: imgCfgScale)

        let counts = inputs.map(\.tokenCount)
        let condIds = try tokenizer.encode(
            Conversation.editCondPrompt(request.prompt, imageTokenCounts: counts))
        let imgCondIds = try tokenizer.encode(
            Conversation.editImgCondPrompt(imageTokenCounts: counts))
        let uncondIds: [Int32]? =
            imgCfgScale > 1
            ? tokenizer.encode(Conversation.t2iUncondPrompt()) : nil

        let image = try model.it2iGenerate(
            condIds: condIds, imgCondIds: imgCondIds, uncondIds: uncondIds,
            images: inputs, width: width, height: height,
            params: params, imgCfgScale: imgCfgScale,
            onStep: Self.denoiseReporter)
        try Task.checkCancellation()
        return IEditResponse(image: try Self.pngArtifact(image))
    }

    // MARK: - VQA

    private func runAnalysis(
        _ request: ImageAnalysisRequest, model: NEOChatModel, tokenizer: SenseNovaTokenizer
    ) async throws -> ImageAnalysisResponse {
        let image = try loadImage(request.image.data)

        var sampling = SamplingParams()
        sampling.maxNewTokens = Int(
            try Self.number(request.metaData, "maxTokens", in: 1...4096) ?? 512)
        sampling.temperature = Float(
            try Self.number(request.metaData, "temperature", in: 0...2) ?? 0)
        sampling.seed = request.metaData["seed"].flatMap {
            if case .int(let i) = $0 { return UInt64(max(0, i)) }
            return nil
        } ?? 0

        let wantsReasoning = Self.flag(request.metaData, "think")
        let userMessage = try Conversation.expandImagePlaceholders(
            prompt: "<image>\n" + request.prompt, imageTokenCounts: [image.tokenCount])
        let ids = tokenizer.encode(
            Conversation.vqaPrompt(userMessage: userMessage, think: wantsReasoning))

        var produced = 0
        let answer = try model.chat(ids: ids, images: [image], params: sampling) { _ in
            produced += 1
            RunProgress.report(.generate, step: produced, totalSteps: sampling.maxNewTokens)
        }
        try Task.checkCancellation()
        // The canonical text is the ANSWER; a think block is never it. Callers
        // that want the deliberation ask for it (metaData includeReasoning).
        let (text, reasoning) = Conversation.splitReasoning(tokenizer.decode(answer))
        if Self.flag(request.metaData, "includeReasoning"), let reasoning {
            return ImageAnalysisResponse(text: "<think>\n\(reasoning)\n</think>\n\n\(text)")
        }
        return ImageAnalysisResponse(text: text)
    }

    // MARK: - output encoding (canonical PNG artifact, C3)

    /// (1, 3, H, W) tensor → the canonical `Image` artifact, with the declared
    /// dimensions taken from the TENSOR (never from the request, which may have
    /// been rounded down by the token grid).
    nonisolated static func pngArtifact(_ tensor: MLXArray) throws -> Image {
        Image(
            format: .png, data: try encodePNG(tensor),
            width: tensor.dim(3), height: tensor.dim(2))
    }

    /// (1, 3, H, W) generation-normalized tensor → PNG.
    nonisolated static func encodePNG(_ tensor: MLXArray) throws -> Data {
        let h = tensor.dim(2)
        let w = tensor.dim(3)
        let rgb01 = clip(tensor.asType(.float32) * 0.5 + 0.5, min: 0, max: 1)
        let hwc = rgb01[0].transposed(1, 2, 0)  // (H, W, 3)
        let bytes = MLX.round(hwc * 255).asType(.uint8).asArray(UInt8.self)

        let cs = CGColorSpace(name: CGColorSpace.sRGB)!
        guard let ctx = CGContext(
            data: nil, width: w, height: h, bitsPerComponent: 8,
            bytesPerRow: w * 4, space: cs,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        else { throw SenseNovaU1PackageError.pngEncode }
        let buf = ctx.data!.bindMemory(to: UInt8.self, capacity: w * h * 4)
        for i in 0 ..< (w * h) {
            buf[i * 4] = bytes[i * 3]
            buf[i * 4 + 1] = bytes[i * 3 + 1]
            buf[i * 4 + 2] = bytes[i * 3 + 2]
            buf[i * 4 + 3] = 255
        }
        guard let image = ctx.makeImage() else { throw SenseNovaU1PackageError.pngEncode }
        let out = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(
            out, UTType.png.identifier as CFString, 1, nil)
        else { throw SenseNovaU1PackageError.pngEncode }
        CGImageDestinationAddImage(dest, image, nil)
        guard CGImageDestinationFinalize(dest) else {
            throw SenseNovaU1PackageError.pngEncode
        }
        return out as Data
    }
}

extension SenseNovaU1Package {
    /// The author one-liner the engine registers.
    public nonisolated static var registration: PackageRegistration {
        .of(SenseNovaU1Package.self)
    }

    /// The **pose specialty** manifest: the same model, the same requirements
    /// and the same two license layers, advertising `.poseDriven` (C6) over a
    /// single surface.
    ///
    /// Why a second manifest rather than a specialty on the shared one: a
    /// `PackageManifest` is static per package TYPE, while pose-ness is a
    /// property of two of the six artifact tiers. Declaring `.poseDriven` on
    /// the shared manifest would tell the Model Manager that the plain bf16
    /// and 8-bit tiers apply a skeleton, which the zero-adapter experiment
    /// says they do not — they paste it into the frame.
    ///
    /// It declares **only `imageEdit`**. The merge is a base-checkpoint merge,
    /// so T2I, think mode and VQA all still run on these weights — but
    /// registering them here would make the pose tier the *default* backer for
    /// those capabilities too (last registration wins routing), and a
    /// pose-conditioned gen stream is not what a plain `textToImage` call
    /// wants. Register the base tiers for those.
    public nonisolated static var poseManifest: PackageManifest {
        let base = manifest
        return PackageManifest(
            license: base.license,
            provenance: base.provenance,
            requirements: base.requirements,
            specialties: [SpecialtyWeight(.poseDriven, strength: 0.8)],
            surfaces: [
                IEditContract.descriptor(
                    name: "sensenova-u1-pose-edit",
                    summary: "Pose transfer as a two-reference edit: image 1 is an "
                        + "OpenPose-style skeleton, image 2 the identity/appearance "
                        + "reference, and the prompt names the slots — "
                        + "\"Image-1: <image>\\nImage-2: <image>\\napply pose from image 1 "
                        + "with reference from image 2\" (SenseNovaVariant.posePrompt). "
                        + "Slot ORDER is load-bearing. Needs a pose tier "
                        + "(pose-8bit / pose-bf16); 28 steps, cfg 4, 768² is the "
                        + "receipted setting.",
                    modes: []
                )
            ]
        )
    }

    /// Registration for the pose specialty. Same package type and the same
    /// resident core — only the advertisement differs.
    ///
    /// The factory refuses a non-pose configuration on purpose: registering
    /// this manifest against, say, the plain 8-bit tier would advertise
    /// `.poseDriven` for weights that carry no pose adapter, which is exactly
    /// the kind of quiet mis-declaration C6 ranking cannot recover from.
    public nonisolated static var poseRegistration: PackageRegistration {
        PackageRegistration(manifest: poseManifest) { configuration in
            guard let typed = configuration as? Configuration else {
                throw PackageError.configurationMismatch(
                    expected: String(describing: Configuration.self),
                    got: String(describing: type(of: configuration)))
            }
            guard typed.variant.isPoseTuned else {
                throw PackageError.configurationMismatch(
                    expected: "SenseNovaU1Configuration with a pose variant "
                        + "(.pose8 / .poseBf16)",
                    got: "variant .\(typed.variant)")
            }
            return SenseNovaU1Package(configuration: typed)
        }
    }
}
