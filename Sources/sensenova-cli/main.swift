// sensenova-cli — manual T2I runs for perf/eyeball work.
// Prompt ids come from .npy fixtures until the tokenizer artifact lands
// (tokenizer.json generation is on the spec's open-items list).
//
// Usage:
//   sensenova-cli --weights DIR --cond-ids cond.npy --uncond-ids uncond.npy \
//     --width 1024 --height 1024 --steps 20 --out out.npy [--seed 42]
//
// `--edit-image` is REPEATABLE — the reference conditions on N images and the
// port's `it2iGenerate(images:)` always took an array; one flag was the only
// thing capping the CLI at a single reference. Multi-reference prompts name
// their slots ("Image-1: <image>\nImage-2: <image>\n..."), which is what the
// pose tier expects (skeleton first, identity second).
//
// `--diff-artifacts A --diff-artifacts B` compares two converted artifacts
// tensor-by-tensor and exits non-zero on any difference — the gate form, since
// file hashes are not reproducible across converts (see the mode below).

import Foundation
import MLX
import SenseNovaU1

func arg(_ name: String) -> String? {
    let a = CommandLine.arguments
    guard let i = a.firstIndex(of: "--\(name)"), i + 1 < a.count else { return nil }
    return a[i + 1]
}

func flag(_ name: String) -> Bool {
    CommandLine.arguments.contains("--\(name)")
}

/// Every value given for a repeatable flag, in command-line order.
func args(_ name: String) -> [String] {
    let a = CommandLine.arguments
    return a.indices.compactMap { i in
        a[i] == "--\(name)" && i + 1 < a.count ? a[i + 1] : nil
    }
}

func loadIDs(_ path: String) throws -> [Int32] {
    try MLX.loadArray(url: URL(fileURLWithPath: path)).asType(.int32).asArray(Int32.self)
}

let weights = URL(fileURLWithPath: arg("weights") ?? "/Volumes/Satechi/Development/mlxengine-image/weights/SenseNova-U1.5-8B-MoT")
// Repeatable: all references are conditioned on, in the order given.
let editImages: [EditImage] = try args("edit-image").map {
    try SenseNovaImageIO.loadEditImage(url: URL(fileURLWithPath: $0))
}
// editing default output: FIRST input's aspect at ~2048² pixels (factor 32)
var width = Int(arg("width") ?? "1024")!
var height = Int(arg("height") ?? "1024")!
if let img = editImages.first, arg("width") == nil, arg("height") == nil {
    let target = Int(arg("target-pixels") ?? "\(2048 * 2048)")!
    let (h, w) = SenseNovaImageIO.smartResize(
        height: img.gridH * 16, width: img.gridW * 16, factor: 32,
        minPixels: target, maxPixels: target)
    width = w
    height = h
    print("[cli] edit output size resolved to \(width)x\(height)")
}
var params = T2IParams()
params.numSteps = Int(arg("steps") ?? "20")!
if let s = arg("seed") { params.seed = UInt64(s)! }
if let c = arg("cfg") { params.cfgScale = Float(c)! }
let outPath = arg("out") ?? "sensenova_out.npy"

// --- artifact diff mode: --diff-artifacts A --diff-artifacts B ---
// Order-independent, byte-exact tensor comparison of two artifact directories.
// `save(arrays:)` hands MLX an UNORDERED map, so two converts of identical
// weights lay their tensors out in a different order inside the shards and get
// different file hashes — `shasum` is therefore not a reproducibility test for
// this format, and the tensor content is. Exits 1 on any difference so it can
// be used as a gate.
let diffPaths = args("diff-artifacts")
if !diffPaths.isEmpty {
    guard diffPaths.count == 2 else {
        print("[diff] need exactly two --diff-artifacts DIR arguments")
        exit(2)
    }
    func artifactTensors(_ dir: URL) throws -> [String: MLXArray] {
        let files = try FileManager.default.contentsOfDirectory(
            at: dir, includingPropertiesForKeys: nil)
            .filter { $0.lastPathComponent.hasPrefix("model-") && $0.pathExtension == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !files.isEmpty else {
            throw SenseNovaError.badWeights("no shards under \(dir.path)")
        }
        var out: [String: MLXArray] = [:]
        for url in files { out.merge(try loadArrays(url: url)) { _, new in new } }
        return out
    }
    let lhs = try artifactTensors(URL(fileURLWithPath: diffPaths[0]))
    let rhs = try artifactTensors(URL(fileURLWithPath: diffPaths[1]))
    var problems: [String] = []
    for key in Set(lhs.keys).subtracting(rhs.keys).sorted() { problems.append("only in A: \(key)") }
    for key in Set(rhs.keys).subtracting(lhs.keys).sorted() { problems.append("only in B: \(key)") }
    var compared = 0
    for key in Set(lhs.keys).intersection(rhs.keys).sorted() {
        let a = lhs[key]!
        let b = rhs[key]!
        if a.shape != b.shape {
            problems.append("\(key): shape \(a.shape) vs \(b.shape)")
        } else if a.dtype != b.dtype {
            problems.append("\(key): dtype \(a.dtype) vs \(b.dtype)")
        } else {
            let ad: Data = a.asData()
            let bd: Data = b.asData()
            if ad != bd {
                switch a.dtype {
                case .float32, .float16, .bfloat16, .float64:
                    let delta = MLX.abs(a.asType(.float32) - b.asType(.float32))
                        .max().item(Float.self)
                    problems.append("\(key): differs (max |Δ| \(delta))")
                default:
                    // packed quantized payloads — a float delta over a uint32
                    // bit pattern is noise, so count bytes instead
                    let n = zip(ad, bd).reduce(0) { $1.0 == $1.1 ? $0 : $0 + 1 }
                    problems.append("\(key): differs (\(n)/\(ad.count) packed bytes)")
                }
            }
        }
        compared += 1
    }
    print("[diff] compared \(compared) tensors")
    if problems.isEmpty {
        print("[diff] IDENTICAL — every tensor matches byte-for-byte")
        exit(0)
    }
    print("[diff] \(problems.count) DIFFERENCE(S):")
    for line in problems.prefix(20) { print("[diff]   \(line)") }
    if problems.count > 20 { print("[diff]   … \(problems.count - 20) more") }
    exit(1)
}

// --- VQA mode: --vqa "question" [--edit-image img] → prints the answer ---
if let question = arg("vqa") {
    let tok = try await SenseNovaTokenizer.load(from: weights)
    var userMessage = question
    var images: [EditImage] = []
    if !editImages.isEmpty {
        userMessage = try Conversation.expandImagePlaceholders(
            prompt: String(repeating: "<image>\n", count: editImages.count) + question,
            imageTokenCounts: editImages.map(\.tokenCount))
        images = editImages
    }
    let ids = tok.encode(
        Conversation.vqaPrompt(userMessage: userMessage, think: flag("think")))
    let t0v = Date()
    // Artifact-aware: artifacts are already in our key layout (and may be
    // quantized), so they must NOT go through sanitize() — re-transposing an
    // NHWC conv weight fails the verified update.
    let model: NEOChatModel
    if WeightLoading.isArtifact(weights) {
        print("[cli] loading OFFLINE ARTIFACT ...")
        model = try WeightLoading.loadArtifact(from: weights)
    } else {
        print("[cli] loading model bf16 ...")
        model = try WeightLoading.load(from: weights, dtype: .bfloat16)
    }
    print("[cli] loaded in \(String(format: "%.1f", Date().timeIntervalSince(t0v)))s")
    var sampling = SamplingParams()
    sampling.maxNewTokens = Int(arg("max-tokens") ?? "512")!
    let t1 = Date()
    var count = 0
    let answer = try model.chat(ids: ids, images: images, params: sampling) { _ in count += 1 }
    let dt = Date().timeIntervalSince(t1)
    print("[cli] answer (\(count) tokens, \(String(format: "%.1f", Double(count) / dt)) tok/s):")
    let (text, reasoning) = Conversation.splitReasoning(tok.decode(answer))
    if let reasoning { print("--- reasoning ---\n\(reasoning)\n--- answer ---") }
    print(text)
    exit(0)
}

let condIds: [Int32]
let uncondIds: [Int32]?
var imgCondIds: [Int32]? = nil
if arg("convert") != nil {
    condIds = []
    uncondIds = nil
} else if let prompt = arg("prompt") {
    let tok = try await SenseNovaTokenizer.load(from: weights)
    if !editImages.isEmpty {
        let counts = editImages.map(\.tokenCount)
        condIds = try tok.encode(Conversation.editCondPrompt(prompt, imageTokenCounts: counts))
        imgCondIds = try tok.encode(Conversation.editImgCondPrompt(imageTokenCounts: counts))
        uncondIds = nil  // editing default img_cfg == 1 needs no uncond branch
        print("[cli] edit prompt tokenized: \(editImages.count) image(s), "
            + "\(condIds.count) cond, \(imgCondIds!.count) img-cond ids")
    } else {
        let pair = tok.t2iIDs(prompt: prompt)
        condIds = pair.cond
        uncondIds = params.cfgScale > 1 ? pair.uncond : nil
        print("[cli] prompt tokenized: \(condIds.count) cond ids\(uncondIds != nil ? ", \(uncondIds!.count) uncond" : " (no CFG)")")
    }
} else {
    condIds = try loadIDs(arg("cond-ids")!)
    uncondIds = arg("uncond-ids").map { try! loadIDs($0) }
}

let loraURL = arg("lora").map { URL(fileURLWithPath: $0) }
let quantBits = arg("quant").flatMap(Int.init)
// --mlx-limit-mb N: MLX's memory limit (MiB), the small-Mac emulation. Not a hard cap: past
// it MLX waits for pending work before allocating, so it reshapes scheduling and can slow a
// run, never fail it (BRIDGE-036). The cache limit is set to the same figure so the pool
// can't inflate past it. --ballast-mb N keeps an N MiB MLX array alive across the run, a
// stand-in for another resident model (the editor's segmenter).
if let mb = arg("mlx-limit-mb").flatMap(Int.init) {
    Memory.memoryLimit = mb << 20
    Memory.cacheLimit = mb << 20
    print("[cli] MLX memory limit \(Memory.memoryLimit >> 20) MB, cache limit \(Memory.cacheLimit >> 20) MB")
}
var ballast: MLXArray? = nil
if let mb = arg("ballast-mb").flatMap(Int.init) {
    ballast = MLXArray.ones([mb << 20 / 4], dtype: .float32)
    eval(ballast!)
    print("[cli] ballast \(mb) MB resident (active \(GPU.activeMemory >> 20) MB)")
}
var t0 = Date()
let model: NEOChatModel
if WeightLoading.isArtifact(weights) {
    print("[cli] loading OFFLINE ARTIFACT (quant/LoRA baked in) ...")
    model = try WeightLoading.loadArtifact(from: weights)
} else {
    print("[cli] loading model bf16\(loraURL != nil ? " + LoRA" : "")\(quantBits.map { " + q\($0)" } ?? "") ...")
    model = try WeightLoading.load(from: weights, dtype: .bfloat16, loraURL: loraURL)
    if let bits = quantBits {
        let tq = Date()
        WeightLoading.quantizeStreams(model, bits: bits)
        MLX.GPU.clearCache()
        print("[cli] quantized streams to q\(bits) in \(String(format: "%.1f", Date().timeIntervalSince(tq)))s")
    }
}

// --- convert mode: save the loaded (merged/quantized) model as an artifact ---
if let outDir = arg("convert") {
    print("[cli] load+prep took \(String(format: "%.1f", Date().timeIntervalSince(t0)))s; saving artifact ...")
    try WeightLoading.saveArtifact(
        model: model, to: URL(fileURLWithPath: outDir), sourceDir: weights,
        quantBits: quantBits,
        loraMergedNote: loraURL.map { $0.deletingPathExtension().lastPathComponent })
    exit(0)
}
print("[cli] loaded in \(String(format: "%.1f", Date().timeIntervalSince(t0)))s")
print("[cli] resident after load: \(GPU.activeMemory / (1 << 20)) MB active, peak \(GPU.peakMemory / (1 << 20)) MB")

// --- SR probe mode (SENSENOVA-SR-EVAL §7 step 1):
//   --init-dir DIR --out-dir DIR [--scale 4] [--start-t 0.7] [--init-no-noise] [--ref-init]
//   [--prompt "..."] [--steps N] [--cfg 1] [--suffix S] [--only-low] [--limit N] [--skip-existing]
//   [--timing f.json]
// Every PNG in --init-dir is Lanczos-upsampled by --scale, entered as the denoise state at
// --start-t (model t: 0 = noise, 1 = image; the SDEdit mix with the seeded noise, or the clean
// image with --init-no-noise), denoised in --steps steps and written to --out-dir as an 8-bit
// PNG. --ref-init also passes the LR through the und stream as an edit reference, with --prompt
// as the instruction (arm b); without it --prompt is a plain T2I prompt (arm a). --suffix S
// renames bench inputs `<name>__<regime>__low.png` to `…__S.png` for `vosrgate score`.
if let initDir = arg("init-dir") {
    guard let outDirPath = arg("out-dir") else { print("[probe] --out-dir required"); exit(2) }
    let outDir = URL(fileURLWithPath: outDirPath)
    try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
    let scale = Int(arg("scale") ?? "4")!
    params.startT = Float(arg("start-t") ?? "0.7")!
    params.initNoiseMix = !flag("init-no-noise")
    params.cfgScale = Float(arg("cfg") ?? "1.0")!
    let refInit = flag("ref-init")
    let suffix = arg("suffix")
    let limit = Int(arg("limit") ?? "0")!
    var files = try FileManager.default.contentsOfDirectory(
        at: URL(fileURLWithPath: initDir), includingPropertiesForKeys: nil)
        .filter { $0.pathExtension.lowercased() == "png" }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }
    if flag("only-low") { files = files.filter { $0.lastPathComponent.hasSuffix("__low.png") } }
    if let listPath = arg("list") {
        let keep = Set(try String(contentsOfFile: listPath, encoding: .utf8)
            .split(separator: "\n").map { String($0).trimmingCharacters(in: .whitespaces) })
        files = files.filter { keep.contains($0.deletingPathExtension().lastPathComponent) }
    }
    if limit > 0 { files = Array(files.prefix(limit)) }
    let tok = try await SenseNovaTokenizer.load(from: weights)
    let prompt = arg("prompt") ?? "A sharp, high-resolution photograph."
    var timing: [[String: Any]] = []
    print("[probe] \(files.count) inputs · x\(scale) · start-t \(params.startT) · noise-mix \(params.initNoiseMix) · steps \(params.numSteps) · cfg \(params.cfgScale) · ref-init \(refInit) · seed \(params.seed)")
    print("[probe] prompt: \(prompt)")
    for (i, url) in files.enumerated() {
        var outName = url.lastPathComponent
        if let suffix, outName.hasSuffix("__low.png") {
            outName = String(outName.dropLast("__low.png".count)) + "__\(suffix).png"
        }
        let outURL = outDir.appendingPathComponent(outName)
        if flag("skip-existing"), FileManager.default.fileExists(atPath: outURL.path) { continue }
        let t1 = Date()
        let x = try SenseNovaImageIO.loadGenImage(url: url, scale: scale)
        let (h, w) = (x.dim(2), x.dim(3))
        guard h % 32 == 0, w % 32 == 0 else {
            print("[probe] skip \(outName): \(w)x\(h) is not a multiple of 32")
            continue
        }
        let img: MLXArray
        if refInit {
            let ref = try SenseNovaImageIO.loadEditImage(url: url)
            let cond = try tok.encode(
                Conversation.editCondPrompt(prompt, imageTokenCounts: [ref.tokenCount]))
            img = try model.it2iGenerate(
                condIds: cond, imgCondIds: nil, uncondIds: nil, images: [ref],
                width: w, height: h, params: params, imgCfgScale: 1.0, initImage: x)
        } else {
            let cond = tok.t2iIDs(prompt: prompt).cond
            img = try model.t2iGenerate(
                condIds: cond, uncondIds: nil, width: w, height: h, params: params, initImage: x)
        }
        eval(img)
        try SenseNovaImageIO.savePNG(img, to: outURL)
        let dt = Date().timeIntervalSince(t1)
        timing.append([
            "file": outName, "seconds": dt, "dit_forwards": params.numSteps,
            "dit_ms_per_call": dt * 1000 / Double(params.numSteps), "width": w, "height": h,
        ])
        print(String(format: "[probe] %d/%d %@ %dx%d %.2fs  active %d MB", i + 1, files.count, outName, w, h, dt, GPU.activeMemory >> 20))
        Memory.clearCache()
    }
    if let tp = arg("timing") {
        try JSONSerialization.data(withJSONObject: timing, options: [.prettyPrinted])
            .write(to: URL(fileURLWithPath: tp))
    }
    print("[probe] done: \(timing.count) written · peak memory \(GPU.peakMemory >> 20) MB")
    exit(0)
}

t0 = Date()
// --- think-mode T2I: AR reasoning block, then denoise ---
if flag("think"), let prompt = arg("prompt"), editImages.isEmpty {
    let tok = try await SenseNovaTokenizer.load(from: weights)
    let condThinkIds = tok.encode(
        Conversation.buildPrompt(
            userMessage: prompt, systemMessage: Conversation.systemMessageForGen,
            appendText: "<think>\n"))
    let uncond = params.cfgScale > 1 ? tok.encode(Conversation.t2iUncondPrompt()) : nil
    let imgSuffix = tok.encode("\n\n" + Conversation.imgStartToken)
    print("[cli] think-mode: reasoning ...")
    let (thinkImage, thinkIds) = try model.t2iGenerateThink(
        condThinkIds: condThinkIds, uncondIds: uncond, imgSuffixIds: imgSuffix,
        width: width, height: height, params: params
    ) { _ in
    } onStep: { step, total in
        if step % 5 == 0 || step == total {
            print("[cli] step \(step)/\(total)")
        }
    }
    print("[cli] --- think ---\n\(tok.decode(thinkIds))\n[cli] --- end think ---")
    eval(thinkImage)
    try MLX.save(array: thinkImage.asType(.float32), url: URL(fileURLWithPath: outPath))
    print("[cli] wrote \(outPath)")
    exit(0)
}
let image: MLXArray
if !editImages.isEmpty {
    image = try model.it2iGenerate(
        condIds: condIds, imgCondIds: imgCondIds, uncondIds: uncondIds,
        images: editImages, width: width, height: height,
        params: params, imgCfgScale: Float(arg("img-cfg") ?? "1.0")!
    ) { step, total in
        if step % 5 == 0 || step == total {
            let dt = Date().timeIntervalSince(t0)
            print("[cli] step \(step)/\(total)  \(String(format: "%.2f", dt / Double(step)))s/step  active \(GPU.activeMemory / (1 << 20)) MB")
        }
    }
} else {
    image = try model.t2iGenerate(
        condIds: condIds, uncondIds: uncondIds, width: width, height: height,
        params: params
    ) { step, total in
        if step % 5 == 0 || step == total {
            let dt = Date().timeIntervalSince(t0)
            print("[cli] step \(step)/\(total)  \(String(format: "%.2f", dt / Double(step)))s/step  active \(GPU.activeMemory / (1 << 20)) MB")
        }
    }
}
eval(image)
let wall = Date().timeIntervalSince(t0)
print("[cli] \(params.numSteps)-step \(width)x\(height): \(String(format: "%.1f", wall))s total, \(String(format: "%.2f", wall / Double(params.numSteps)))s/step")
print("[cli] peak memory: \(GPU.peakMemory / (1 << 20)) MB")

try MLX.save(array: image.asType(.float32), url: URL(fileURLWithPath: outPath))
print("[cli] wrote \(outPath)")
if let ballast { print("[cli] ballast released: \(ballast.size >> 18) MB") }
