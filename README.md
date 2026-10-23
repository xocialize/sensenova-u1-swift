# sensenova-u1-swift

Native Apple Silicon (MLX-Swift) port of **[SenseNova-U1.5-8B-MoT](https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT)** — SenseTime's unified multimodal flagship: text-to-image, instruction image editing, and visual question answering in one NEO-unify **Mixture-of-Transformers** model (8.1B understanding + 8.1B generation streams on a shared 42-layer Qwen3 skeleton, pixel-space rectified flow, **no VAE**).

All three capabilities run locally on a Mac:

- **Text-to-image** — up to the native 2048²-class trained buckets, 50-step base or 8-step distilled tier, optional `<think>` reasoning mode
- **Image editing** — identity-preserving instruction edits (`it2iGenerate`)
- **VQA / chat** — image-grounded question answering on the understanding stream

## Performance (M5 Max, prebuilt artifacts)

| Artifact | Peak memory | 1024² image | 2048² image |
|---|---|---|---|
| [`SenseNova-U1.5-8B-MoT-8step-4bit`](https://huggingface.co/mlx-community/SenseNova-U1.5-8B-MoT-8step-4bit) | **14.8 GB** | **3.2 s** | **15.8 s** |
| [`SenseNova-U1.5-8B-MoT-8step-8bit`](https://huggingface.co/mlx-community/SenseNova-U1.5-8B-MoT-8step-8bit) | 22.4 GB | 3.8 s | 19.5 s |
| [`SenseNova-U1.5-8B-MoT-8step-v2-4bit`](https://huggingface.co/mlx-community/SenseNova-U1.5-8B-MoT-8step-v2-4bit) | 14.7 GB | 3.2 s | — |
| [`SenseNova-U1.5-8B-MoT-8step-v2-8bit`](https://huggingface.co/mlx-community/SenseNova-U1.5-8B-MoT-8step-v2-8bit) | 22.4 GB | 3.3 s | 13.4 s |
| [`SenseNova-U1.5-8B-MoT-8bit`](https://huggingface.co/mlx-community/SenseNova-U1.5-8B-MoT-8bit) | 22.9 GB | ~7.4 s (cfg 4) | — |
| [`SenseNova-U1.5-8B-MoT-bf16`](https://huggingface.co/mlx-community/SenseNova-U1.5-8B-MoT-bf16) | 35.1 GB | ~6.6 s (cfg 4) | ~40 s (cfg 4) |
| [`SenseNova-U1.5-8B-MoT-pose-8bit`](https://huggingface.co/mlx-community/SenseNova-U1.5-8B-MoT-pose-8bit) | 22.5 GB | — | — |
| [`SenseNova-U1.5-8B-MoT-pose-bf16`](https://huggingface.co/mlx-community/SenseNova-U1.5-8B-MoT-pose-bf16) | 35.1 GB | — | — |

`8step` artifacts have the official [8-step distillation LoRA](https://huggingface.co/xocialize/SenseNova-U1.5-8B-MoT-LoRAs) pre-merged and run **cfg-free** (single forward per step) — use them for fast T2I **and fast editing** (8 steps, cfg 1 — the vendor's 8-step edit recipe: a 2048² jacket edit in 15 s against 169 s for the 50-step base tier, 28.7 dB apart, AB-R-0490). VQA runs on every tier — the distill adapters leave the understanding stream bit-identical to the base tier, so the answers match token for token. Use `bf16`/`8bit` for 50-step quality T2I and for think mode, the one base-tier-only feature. Per the fleet quantization doctrine for diffusion paths, **8-bit reproduces the bf16 image (cos 0.998); 4-bit is a declared opt-in tier** that produces a different-but-equally-valid draw (cos ~0.92 at fixed seed).

`8step-v2` artifacts carry upstream's **V2** 8-step adapter (2026-09-24, their recommended one) through the same merge; the V1 tiers stay published so existing seeds keep reproducing. At fixed seed over five prompts V2 lowers mean saturation by 0.07 and RMS contrast by 0.045, cuts the 99th-percentile gradient by a quarter (less oversharpening) and removes the shadow/highlight clipping V1 shows (8.4 % → 0.3 % of pixels below 2 %; the V1 poster blew 64 % of its pixels, V2 none), with composition preserved (cos 0.86–0.99). V2 int8 reproduces its bf16 merge at cos 0.995 / 30.9 dB; V2 int4 is tighter than V1's (cos 0.976 vs ~0.92). Receipt: `Docs/receipts/8step-v2-ab/`.

**16 GB Macs: the fill window.** `SenseNovaU1Configuration(variant: .fast4V2, window: .fill768)` is the 16 GB Generative Fill tier (AB-D-0141): the same `8step-v2-4bit` artifact, declared at 11.8 GB resident + 0.75 GB activation (12.55 GB) instead of the native envelope's 12.2 + 4.5 GB, so MLXEngine admits it on a 16 GB Mac's ≈ 12.7 GB budget. In exchange `run()` holds every request to what that number was measured on: output ≤ 768² by area, one reference scaled into the same area, cfg-free; an edit's default size is its reference's. On a real 16 GB MacBook Pro (M1 Pro) a 768² fill peaked at 11,930 MiB with a 12.60 GB process footprint, swap grew 1.9 GB on the first load only, and a fill took 41–43 s warm (5.1–5.4 s/step; receipt AB-R-0602). Evict every other MLX resident first. Base chips are unmeasured, so `chipFloor` stays `.pro`.

`pose` artifacts are a **base-checkpoint merge, not a distill** — see [Pose transfer](#pose-transfer-the-pose-tiers) below. Every surface stays enabled on them; a 768² two-reference edit runs in **20.2 s at 0.72 s/step**, 2.6 s to load, 19.0 GB resident. Published 2026-09-02 and verified against a fresh download (`Docs/publish/pose-artifacts.md`).

## Quick start (CLI)

```bash
swift build -c release

# fast tier: 1024² in ~3 s
.build/release/sensenova-cli \
  --weights /path/to/SenseNova-U1.5-8B-MoT-8step-4bit \
  --prompt "A cinematic mountain lake at sunrise, realistic photography." \
  --width 1024 --height 1024 --steps 8 --cfg 1.0 --out out.npy

# quality tier + think mode
.build/release/sensenova-cli --weights /path/to/SenseNova-U1.5-8B-MoT-bf16 \
  --think --prompt "Minimalist poster for a jazz concert titled 'BLUE HOUR'" \
  --width 1024 --height 1024 --steps 50 --cfg 4.0 --out poster.npy

# instruction editing
.build/release/sensenova-cli --weights /path/to/SenseNova-U1.5-8B-MoT-bf16 \
  --edit-image photo.png \
  --prompt "Change the jacket to cobalt blue. Preserve the face, pose, and lighting." \
  --steps 50 --cfg 4.0 --out edited.npy

# multi-reference editing — `--edit-image` is repeatable, references are
# conditioned on in the order given, and the FIRST one drives the output size
.build/release/sensenova-cli --weights /path/to/SenseNova-U1.5-8B-MoT-pose-8bit \
  --edit-image skeleton.png --edit-image identity.jpg \
  --prompt $'Image-1: <image>\nImage-2: <image>\napply pose from image 1 with reference from image 2' \
  --width 768 --height 768 --steps 28 --cfg 4.0 --out posed.npy

# VQA
.build/release/sensenova-cli --weights /path/to/SenseNova-U1.5-8B-MoT-bf16 \
  --vqa "Describe this image in one sentence." --edit-image photo.png
```

Outputs are `.npy` tensors in the model's normalized space (`x·0.5 + 0.5` → RGB); pipe through any converter or use the library API. The CLI also loads the original HF checkpoint directly (`--weights <hf-snapshot>` + optional `--quant 8|4` and `--lora <file>`), and `--convert OUTDIR` produces the prebuilt artifacts above.

## Pose transfer (the `pose` tiers)

`SenseNova-U1.5-8B-MoT-pose-{8bit,bf16}` carry the RefControl **pose adapter**
pre-merged: a rank-32 LoRA over the 294 gen-stream (`*_mot_gen`) projections,
trained on pose triples and merged into the base checkpoint with the same
`applyLoRA` path the 8-step distill uses. Because it is a base merge and not a
distillation, editing, think mode and VQA all still work — only the gen stream's
behaviour changes.

Pose transfer is a **two-reference edit**, and slot order is load-bearing:

```swift
let response = try await package.run(
    IEditRequest(
        images: [skeleton, identity],           // 1 = pose, 2 = appearance
        prompt: SenseNovaVariant.posePrompt,    // names both slots
        width: 768, height: 768, steps: 28, seed: 42))
```

`SenseNovaVariant.posePrompt` is the exact string the adapter was trained and
A/B-ed with:

```
Image-1: <image>
Image-2: <image>
apply pose from image 1 with reference from image 2
```

Image 1 is an OpenPose-style skeleton render, image 2 the identity/appearance
frame. Swapping them makes the model reproduce the appearance frame's own pose
and ignore the skeleton (receipt: `Docs/receipts/pose-gates/`). A skeleton
missing arm joints loses to the reference's pose prior the same way — that is a
fixture-quality problem, not a model one.

The tier advertises the `poseDriven` specialty through a **second registration**,
`SenseNovaU1Package.poseRegistration`, which declares only the `imageEdit`
surface so it cannot become the default backer for plain text-to-image. It
refuses a configuration whose variant is not a pose tier.

Building the artifacts from the HF checkpoint:

```bash
sensenova-cli --weights <HF base> --lora pose_lora_003000.safetensors \
  --quant 8 --convert SenseNova-U1.5-8B-MoT-pose-8bit
sensenova-cli --weights <HF base> --lora pose_lora_003000.safetensors \
  --convert SenseNova-U1.5-8B-MoT-pose-bf16
```

`--diff-artifacts A --diff-artifacts B` compares two artifacts tensor-by-tensor
and exits non-zero on any difference. Use it, not `shasum`: `saveArtifact` hands
MLX an unordered map, so two converts of identical weights lay their tensors out
in a different order inside the shards and get different file hashes while every
tensor matches.

## Library

```swift
import SenseNovaU1

let model = try WeightLoading.loadArtifact(from: artifactDir)   // or .load(from:) for HF checkpoints
let tok = try await SenseNovaTokenizer.load(from: artifactDir)
let (cond, uncond) = tok.t2iIDs(prompt: "A tabby cat on a red velvet chair")
var params = T2IParams()                    // 50 steps, cfg 4.0, shift 3.0 (reference defaults)
let image = model.t2iGenerate(condIds: cond, uncondIds: uncond,
                              width: 1024, height: 1024, params: params)
```

`it2iGenerate` (editing), `chat` (VQA), and `t2iGenerateThink` (reasoning mode) are the other entry points.

## Parity

The port is gated against the reference PyTorch implementation ([OpenSenseNova/SenseNova-U1](https://github.com/OpenSenseNova/SenseNova-U1) @ `a62fd54`, weights `07d76f6`):

- component parity (fp32, CPU stream): all ops < 1e-4, decoder layers at relative 1e-5 across both streams
- e2e bf16 vs the torch oracle: per-pass velocity cosine 0.999+; editing e2e at cos 0.9990 / 34.6 dB
- THW positional indexing exactly integer-equal; tokenizer ids exactly equal
- production-grid renders eyeball-gated at 1024² and 2048²

Oracle fixtures and capture harnesses live in the companion `sensenova-u1-oracle` workspace (spec: `PORTING-SPEC.md`, including a 21-item trap ledger — dual RoPE conventions, the `[::2]` frequency-range trick, fp32 gen-stream activations to 4.3e5, and more).

**NAX split-K GEMM (fixed):** mlx-swift ≤ 0.31.6 JIT builds mis-instantiate the NAX split-K bf16 GEMM ([mlx#3797](https://github.com/ml-explore/mlx/issues/3797)), which used to force a ≤896-row chunk on the FFN down-projection. mlx-swift 0.32.3 (the manifest floor) carries the fix, the chunk is gone, and `NAXProbeTests` guards the shape on every test pass.

## GPU numerics: the pixel head's convs (2026-09-24)

mlx's Metal `conv2d` takes a Winograd F(6×6,3×3) path when the conv is 3×3, stride 1, dilation 1,
groups 1, C % 32 == 0, O % 32 == 0, C + O ≥ 256 and N·H·W ≥ 4096. On M5 that path is lossy, and
badly so in bf16: about 5.8e-2 relL2 per conv, 35× the bf16 rounding floor.

Both `fm_head` (ConvDecoder) convs fall inside it at production sizes, in bf16:

- conv1: 1024→1024 at 2× the token grid (64² for a 1024² image)
- conv2: 256→192 at 4× the token grid (128²)

They run on every velocity prediction, twice per step with CFG. The component fixture uses a 4×6
token grid, which sits below the window, and the GPU e2e parity runs at 256², so neither could see
this.

Both convs are now `WinogradFreeConv2d`. **Default `.conv3d`** (`fmHead.convRoute`, type
`SenseNovaConvRoute`).

Head only, at the 1024² token grid in bf16, against fp32 on the same inputs and weights:

| Route | relL2 | max error | Time per call |
|---|---|---|---|
| `.winograd` (old default) | 7.5e-2 | 6.3 | 4.0 ms |
| `.fp32Winograd` | 8.6e-3 | 0.70 | 5.9 ms |
| `.conv3d` | 4.1e-3 | 0.17 | 7.0 ms |

End to end at 1024², 20 steps, seed 42, with the oracle's T2I prompt, comparing `.winograd` against
`.conv3d`:

- The images differ at 39.3 dB (8-bit). 73% of pixels change, 17.5% by more than 4 levels, and
  the largest difference is 117 levels.
- The difference map is blocky, following the head's tiling.
- At 3× the raw render shows chroma speckle on fine texture (fur, the nose bridge) that the routed
  render does not have.
- Time is 21.4 s vs 20.6 s: no measurable cost.

Controls and tests:

- Environment override: `SENSENOVA_CONV_ROUTE=winograd|conv3d|fp32Winograd`.
- `SENSENOVA_PARITY=1 swift test -c release -Xswiftc -enable-testing --filter FMHeadGPULaneTests`
  compares the head against fp32.

## License & provenance

- Port code: **MIT** (this repository)
- Model weights: **Apache-2.0** — SenseNova-U1.5-8B-MoT by [SenseTime / SenseNova](https://huggingface.co/sensenova) ([paper](https://arxiv.org/abs/2605.12500), [reference implementation](https://github.com/OpenSenseNova/SenseNova-U1)). The prebuilt artifacts above are format conversions (bf16 cast / group-64 quantization / LoRA merge) of the upstream checkpoint, redistributed under the same license with credit; original license text and provenance are stated on each artifact card.
- **Upstream revision, pinned.** Every artifact here was converted from `07d76f61` (fp32 gen stream, 13 shards, 47 GB). Hub `main` moved on 2026-08-24 to `19bc874e`, "Convert to BF16 and re-shard" (8 shards, 32.7 GB) — the same weights at the dtype this port casts to at load, so no behaviour change is expected, but a different index and shard layout to rebuild from. `Provenance.revision` stays pinned so that statement is checkable; nothing the wrapper *fetches* is affected, because `weightSources` addresses our own mlx-community artifact repos rather than the upstream checkpoint.
