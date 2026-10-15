#!/bin/zsh
# SenseNova-U1.5 zero-training SR probe (SENSENOVA-SR-EVAL §7 step 1) — the sweep, then the scores.
# Sequential: the GPU is shared by the model, SSIMULACRA2 scoring and the SigLIP2 NR scorer.
#
# Cells: arm a (plain prompt, LR as the state) and arm b (LR also an und-stream reference, instruction prompt)
#        × start-t {0.7, 0.5} (model t; noise fraction 0.3 / 0.5) × steps {1, 2}, SDEdit noise mix;
#        plus one FluxSR-literal entry (clean LR as the state at t 0.7, no noise, arm a, 1 step).
# Inputs: bench 27 lows (512² → 2048²), SA-Text lv3 + lv1 on the 200-image subset every learned arm ran on
#        (128² → 512²), ScreenSR pages 001/002/010/095 at ×1 / ×2 / ×4 (512² → 512² / 1024² / 2048²).
set -u
cd "$(dirname "$0")"
CLI=../../../.build/release/sensenova-cli
W=/Volumes/Satechi/Development/mlxengine-image/weights/SenseNova-U1.5-8B-MoT-8step-v2-8bit
V=/Volumes/Satechi/Development/mlxengine-forge/Tools/vosrgate/.build/release/vosrgate
NR=/Volumes/Satechi/Development/mlxengine-forge/Tools/nrscore/.build/out/Products/Release/nrscore
PA="A sharp, high-resolution image."
PB="Upscale this image. Preserve every word and letter exactly as written."
stage() { echo "=== $1  $(date +%T)  load $(uptime | sed 's/.*averages: //')"; }

# cell <arm> <start-t> <steps> [extra flags]
cell() {
  local arm=$1 t=$2 s=$3; shift 3
  local extra=("$@")
  local M="U1-${arm}-t${t}-s${s}"
  [[ "${extra[*]}" == *init-no-noise* ]] && M="${M}-nonoise"
  local flags=(--weights $W --start-t $t --steps $s --cfg 1 --seed 42 "${extra[@]}")
  if [[ $arm == b ]]; then flags+=(--ref-init --prompt "$PB"); else flags+=(--prompt "$PA"); fi
  stage "$M bench (27 × 512² → 2048²)"
  $CLI "${flags[@]}" --init-dir bench/dump --only-low --scale 4 --out-dir bench/dump --suffix $M \
      --skip-existing --timing logs/timing_${M}_bench.json 2>&1 | grep -E "^\[probe\] (done|[0-9]+/[0-9]+ .*(27|1)/27)|error|Error"
  for lv in lq3 lq1; do
    stage "$M SA-Text $lv (200 × 128² → 512²)"
    $CLI "${flags[@]}" --init-dir satext/$lv --list satext/subset200.txt --scale 4 --out-dir satext/out/$M/$lv \
        --skip-existing --timing logs/timing_${M}_${lv}.json 2>&1 | grep -E "^\[probe\] done|error|Error"
    $V ocr satext/out/$M/$lv ocr/satext_${M}_${lv}.json 2>&1 | tail -1
  done
  for sc in 1 2 4; do
    stage "$M ScreenSR ×$sc"
    $CLI "${flags[@]}" --init-dir screensr/lr --scale $sc --out-dir screensr/out/$M/x$sc \
        --skip-existing --timing logs/timing_${M}_screensr_x${sc}.json 2>&1 | grep -E "^\[probe\] done|error|Error"
    $V ocr screensr/out/$M/x$sc ocr/screensr_${M}_x${sc}.json 2>&1 | tail -1
  done
}

mkdir -p logs ocr satext/out screensr/out
cell a 0.7 1
cell b 0.7 1
cell a 0.5 1
cell b 0.5 1
cell a 0.7 2
cell b 0.7 2
cell a 0.5 2
cell b 0.5 2
cell a 0.7 1 --init-no-noise

stage "score bench (SSIMULACRA2 / gradient / FidelityGuard / octaves)"
$V score bench/dump bench/score.csv 2>&1 | tail -2
stage "nrscore bench"
$NR bench/nr.json bench/dump 2>&1 | tail -2
stage "analyze"
python3 -I analyze_probe.py > analysis.txt 2>&1; tail -5 analysis.txt
stage "CHAIN DONE"
