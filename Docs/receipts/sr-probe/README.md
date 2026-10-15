# SenseNova-U1.5 zero-training SR probe — SENSENOVA-SR-EVAL §7 step 1 (2026-10-15)

_Mac Studio M5 Max 128 GB · `sensenova-u1-swift` at this commit (release build) · artifact
`SenseNova-U1.5-8B-MoT-8step-v2-8bit`, rebuilt here from the pinned base `07d76f61` + the vendor's
`SenseNova-U1.5-8B-MoT-LoRA-8step-V2.safetensors` with `--quant 8 --convert` (19.9 GB, 2292 tensors) · no training,
no artifact published._

**Question.** Before any LoRA spend: does the U1.5 prior, entered part-way with a bicubic-upsampled LR as the
state, hold text without hallucinating — and what does it do on the fleet's restoration gates? The memo's go/no-go:
arm (b) must beat the un-restored input on ScreenSR ×1 word-F1 (where VOSR2 loses) and match VOSR2 on SA-Text lv3
CORRUPTED.

## What was added to the port

- `T2IParams.startT` (model t: 0 = noise, 1 = image) and `T2IParams.initNoiseMix`; `t2iGenerate` / `it2iGenerate`
  take `initImage: MLXArray?` (1, 3, H, W in [-1, 1]). The schedule is `NEOChatModel.timestepSchedule(…, startT:)`:
  the linear grid runs from the t whose shifted value is `startT` up to 1 (`unshiftedT` is the inverse of the
  shift), so a partial run keeps the full schedule's spacing over its interval. `startT == 0` is bit-identical to
  the previous schedule (`PartialScheduleTests` pins it, plus the inverse round-trip and the partial endpoints).
- The state at `startT`: SDEdit mix `t·x + (1−t)·σ·ε` (default; on-distribution for an untrained prior) or the
  clean image as the state (`initNoiseMix = false`, the FluxSR / HYPIR entry a *trained* restorer expects).
- `SenseNovaImageIO.loadGenImage(url:scale:)` (PNG → generation space, vImage Lanczos ×scale, centre-aligned per
  AB-L-0239) and `savePNG`.
- `sensenova-cli --init-dir … --out-dir … --scale --start-t --steps --cfg [--ref-init] [--init-no-noise] [--suffix]
  [--list] [--only-low] [--skip-existing] [--timing]`: one model load per directory.

## Arms

| arm | state entry | prompt / prefix |
|---|---|---|
| **a** | bicubic-×scale LR mixed with seeded noise at `start-t` | T2I prefix, "A sharp, high-resolution image." |
| **b** | same | the LR also through the und stream as an edit reference; instruction "Upscale this image. Preserve every word and letter exactly as written." |
| a-nonoise | the clean bicubic LR as the state at t 0.7 (FluxSR-literal) | as a |

`start-t` ∈ {0.7, 0.5} (noise fraction 0.3 / 0.5 of the resolution-scaled σ: σ = 8 at 2048², 2 at 512²),
steps ∈ {1, 2}, cfg 1 (the 8-step tier is cfg-free), seed 42. Method names: `U1-<arm>-t<start-t>-s<steps>`.

## Inputs and scorers (bench-identical to VOSR-EVAL §7 / AB-R-0448)

- **Bench**: the 27 `<name>__<regime>__low.png` 512² crops (9 images × bicubic / lanczos / jpeg) → 2048², scored by
  `vosrgate score` (SSIMULACRA2 vs the lossless reference, gradient, FidelityGuard downsample consistency, octave
  SSIMs) and `nrscore` (SigLIP2 NR). The classical arms and reference are symlinked from the Iris run, and
  `analyze_probe.py` checks they reproduce the 2026-09-26 rows before merging VOSR2 / SeedVR2 / RealPLKSR /
  Real-ESRGAN / Iris-3B from their receipts.
- **SA-Text lv3 and lv1**: the 200-image subset every learned arm ran on (128² → 512²), Vision OCR (`vosrgate ocr`),
  word-F1 vs the dataset transcriptions and the N7 CORRUPTED / FIXED accounting against the un-restored bicubic.
- **ScreenSR**: the four fixture pages (001, 002, 010, 095; 512²) at ×1, ×2, ×4; Vision word-F1 vs the HR page's
  OCR (the §11 scorer — it reproduces §11's input rows 0.937 / 0.116 for 010 / 095 to the digit). Pages 001 and
  002 carry no Latin tokens for Vision and score as n/a. Page 090 is not in the fixture set.
- **Grid**: gradient-spectrum energy at the 32-px (U1 token) and 16-px harmonics vs neighbouring bins, jpeg regime.

`run_probe.sh` is the chain; `logs/chain.log` the transcript; `analysis.txt` the tables; `logs/timing_*.json` the
per-image wall times; `eyeball/` the smoke-run crops (Bayer_photo jpeg: Lanczos / arm a / arm b / reference).

## Results (13 cells; `analysis.txt` has every table, `logs/chain*.log` the transcripts)

Input identity vs the 2026-09-26 run: 0 classical-arm / reference mismatches, so the merged rows are comparable.

### Verdict against the pre-registered go/no-go: **NO-GO** for the zero-training prior

| criterion | required | measured (best cell) | |
|---|---|---|---|
| ScreenSR ×1, arm (b) beats the un-restored input (page 010 0.937) | > 0.937 | 0.892 (b, t 0.9, 1 step), 22 words corrupted; VOSR2 ×1 0.904 / 19 | **fail** |
| SA-Text lv3 CORRUPTED matches VOSR2 (0 corrupted, 55 fixed, F1 0.242) | ≈ | 5–8 corrupted, 8–21 fixed, F1 0.105–0.146 (input 0.106) | **fail** — no lift |

### G1 / G3 — the prior restores nothing on its own

SSIMULACRA2 vs the lossless 2048² reference, jpeg regime (the regime that decides G1); FidelityGuard worst tile.

| arm | jpeg/photo FR | jpeg/graphic FR | worst-tile DC photo | base-band SSIM photo |
|---|---:|---:|---:|---:|
| Lanczos floor | −26.53 | −5.04 | 0.995 | 0.993 |
| RealPLKSR | −19.33 | 10.02 | 0.956 | 0.994 |
| **VOSR2** | **−19.17** | **12.28** | 0.927 | 0.994 |
| Iris-3B (AB-R-0448) | −27.94 | 2.12 | 0.861 | 0.988 |
| U1-b t 0.9, 1 step | −29.97 | 0.31 | 0.824 | 0.977 |
| U1-a t 0.9, 1 step | −30.89 | −1.01 | 0.817 | 0.975 |
| U1-b t 0.7, 1 step | −43.29 | −14.04 | 0.733 | 0.962 |
| U1-a t 0.7, 1 step | −51.99 | −29.11 | 0.674 | 0.941 |
| U1-b t 0.5, 1 step | −50.35 | −19.89 | 0.630 | 0.944 |
| U1-a t 0.5, 1 step | −62.77 | −49.24 | 0.491 | 0.878 |
| U1-a t 0.7, no noise (FluxSR-literal) | −60.87 | −63.65 | 0.129 | 0.802 |

- Every cell fails G1; the best (t 0.9) sits at the Lanczos floor on graphics and 3–4 points under it on photos,
  11 points under VOSR2. The trend is monotone: the less noise mixed in, the closer to the input; there is no entry
  point at which the untrained prior restores. Two steps never beat one at the same t.
- Arm (b) (LR as an und-stream reference) is better than (a) at every t on every gate: +9 FR on photos at t 0.7, +15
  on graphics; the reference holds colour and layout (base-band 0.962 vs 0.941). That is the edit training working.
- The **FluxSR-literal entry** (clean LR fed as the state at t 0.7, the way a *trained* one-step restorer is used)
  is the worst cell on every fidelity axis: the model, seeing a clean image at a noisy t, emits a structured
  "velocity" that paints diagonal cyan striping over flat regions (`eyeball/PubSec_graphic__jpeg__…nonoise__crop.png`,
  gradient 5.6 vs reference 1.6, worst tile 0.13). Text pages survive it only because the letters dominate.
  So the FTD entry is meaningless before training — the probe cannot test it, only the SDEdit entry.
- SigLIP2 NR sits at 0.87–0.92 for every cell (VOSR2 0.913, Iris 0.922): the NR oracle does not see any of this,
  as AB-L-0221 said.

### Text

**SA-Text** (200-image subset, 128² → 512², word-F1 vs the dataset transcriptions; N7 accounting vs the un-restored
bicubic):

| arm | lv3 F1 | lv3 CORRUPTED / FIXED | lv1 F1 | lv1 CORRUPTED / FIXED |
|---|---:|---:|---:|---:|
| un-restored | 0.106 | — | 0.231 | — |
| **VOSR2** | **0.242** | **0 / 55** | **0.360** | **3 / 57** |
| Iris-3B | 0.185 | 5 / 35 | — | — |
| U1-b t 0.5, 1 step | 0.146 | 6 / 21 | 0.290 | 10 / 32 |
| U1-a t 0.7, 1 step | 0.123 | 6 / 15 | 0.281 | — |
| U1-b t 0.7, 1 step | 0.115 | 5 / 8 | 0.267 | 10 / 22 |
| U1-b t 0.9, 1 step | 0.105 | 8 / 9 | 0.257 | — |
| U1-a t 0.7, no noise | 0.135 | 6 / 20 | 0.271 | 10 / 22 |

No cell lifts illegible text the way VOSR2 does (its 55 fixed words against our best 21), and every cell corrupts
5–11 input-correct words where VOSR2 corrupts 0 (lv3) / 3 (lv1).

**ScreenSR** (word-F1 vs the HR page's OCR; pages 010 dense / 095 sparse; CORRUPTED / FIXED over both):

| arm | ×1 010 / 095 | ×2 010 / 095 | ×4 010 / 095 | corrupted / fixed at ×2 |
|---|---|---|---|---|
| un-restored input | 0.937 / 0.116 | — | — | — |
| VOSR2 (§10–§11.2) | 0.904 / 0.444 (19 / 57) | 0.952 / 0.635 | 0.920 / 0.625 | — |
| U1-b t 0.9, 1 step | 0.892 / 0.043 (22 / 4) | **0.937 / 0.631** | 0.931 / 0.628 | **8 / 99** |
| U1-a t 0.9, 1 step | 0.870 / 0.109 (24 / 5) | 0.924 / 0.588 | 0.931 / 0.650 | 10 / 89 |
| U1-b t 0.7, 1 step | 0.731 / 0.211 (61 / 18) | 0.877 / 0.482 | 0.849 / 0.522 | 25 / 75 |
| U1-a t 0.7, 1 step | 0.616 / 0.000 (98 / 2) | 0.844 / 0.463 | 0.851 / 0.510 | 30 / 67 |
| U1-a t 0.5, 1 step | 0.244 / 0.026 (196 / 2) | 0.387 / 0.172 | 0.455 / 0.204 | 155 / 22 |

- **At ×1 every cell rewrites legible text** (`eyeball/screensr010__x1__…crop.png`: "nothing" → "nuthing",
  "Mummie" → "Murnmie" at t 0.7), the AB-L-0172 mechanism — the memo's hypothesis that the und-stream reading of the
  LR would hold the letters is **not supported** at ×1: (b) at t 0.9 corrupts 22 words, VOSR2 19, input 0.
- **At ×2 and ×4 with the light entry, the prior is text-benign**: (b) t 0.9 reads the dense page at 0.937 (input
  0.937, VOSR2 ×2 0.952) and the sparse page at 0.631 (VOSR2 ×2 0.635), 8 corrupted words. Level with VOSR2 on
  these two pages, not better, and bought with photo fidelity at the Lanczos floor (G1 above): at t 0.9 the model
  mostly passes the bicubic input through.
- Heavier entries trade text for nothing: t 0.5 corrupts 155–196 words.

### The no-noise entry at lighter t (second extra batch)

A clean state at t 0.9 is a 10 % velocity step from the bicubic input, and it is the one cell that clears the
Lanczos floor on SSIMULACRA2 — while failing G3 harder than any SDEdit cell:

| arm | jpeg/photo FR | jpeg/graphic FR | worst-tile DC photo | base-band photo | SA-Text lv3 corr / fixed | ScreenSR ×1 010 (corr / fixed) | ×2 010 / 095 |
|---|---:|---:|---:|---:|---|---|---|
| Lanczos floor | −26.53 | −5.04 | 0.995 | 0.993 | — | 0.937 (0 / 0) | — |
| VOSR2 | −19.17 | 12.28 | 0.927 | 0.994 | 0 / 55 | 0.904 (19 / 57) | 0.952 / 0.635 |
| U1-b t 0.9, no noise | −25.85 | 7.92 | **0.433** | 0.933 | 7 / 13 | **0.928 (12 / 5)** | 0.918 / 0.582 |
| U1-a t 0.9, no noise | −25.58 | 7.54 | 0.452 | 0.936 | 2 / 11 | 0.920 (17 / 6) | 0.918 / 0.592 |
| U1-b t 0.9, SDEdit | −29.97 | 0.31 | 0.824 | 0.977 | 8 / 9 | 0.892 (22 / 4) | 0.937 / 0.631 |

- On graphics the 10 % step is a mild learned sharpening (+8 over bicubic, RealPLKSR 10, VOSR2 12) with no grid
  (32-px ratio 1.06, reference 1.06); on photos it is 1 point over the floor and 6 under VOSR2.
- The worst-tile fidelity of 0.43–0.45 is the striping again, at lower amplitude: on flat tiles the "velocity" of
  a clean image is structured noise, so FidelityGuard flags invented content where the SDEdit entry at the same t
  is at 0.82. Base-band 0.93 is colour drift (VOSR2 0.99).
- ScreenSR ×1: the gentlest arm measured — 12 words corrupted (VOSR2 19), but 5 fixed (VOSR2 57) and still below
  the un-restored 0.937. It fails the ×1 criterion by less than the SDEdit cells and fixes almost nothing.
- Verdict unchanged: no zero-training cell passes G1 (RealPLKSR −19.3 is the bar) or G3.

### Grid, cost

- 32-px grid ratio (jpeg, mean over 9 images; 1.0 = none): every SDEdit cell 1.10–1.27, no-noise t 0.9 1.06,
  no-noise t 0.7 1.90 (the striping), reference 1.06, bicubic 1.06, Lanczos 1.74, Iris-3B 1.27. No token-grid
  artifact from ConvDecoder at any entry; ADL has nothing to fix here. 16-px ratios in `analysis.txt`.
- One 2048² forward: 1.95 s (a) / 2.06 s (b, incl. the 512² reference prefill); 512²: 0.26 s; peak 20.2 GB. The
  bench cell (27 images) runs in 55 s against VOSR2's 35 s *per image* (PyTorch MPS, 25 DiT calls) — the speed and
  the untiled 2048² pass are real, and are the only probe results that favour U1.

### What this settles for the memo

1. The zero-training route is closed: no entry point restores, and the text behaviour the memo bet on (holding
   legible text at ×1 through the reference) does not exist in the untrained prior.
2. The one positive: at light entries the prior does not *invent* text at ≥ ×2 (8 corrupted words on the ScreenSR
   pages, VOSR2-level reads), and the reference path measurably steadies colour and layout. A LoRA pilot would start
   from a text-benign, fidelity-poor prior and have to learn restoration entirely from the FTD / MSE objective —
   which is what FluxSR's recipe is for, but the probe gives no evidence it would then *beat* VOSR2 on text, only
   that it would not start behind on corruption.
3. The FluxSR-literal entry cannot be evaluated without training; the probe's SDEdit entry is the only
   zero-training proxy, and it is pass-through at light t and hallucination at heavy t.
