#!/bin/zsh
# Extra cells after the main chain (run_probe.sh): a lighter entry (start-t 0.9 = 10 % noise) for both arms, since
# the 30 % entry rewrites legible ×1 text; and the first cell's SA-Text stages, which the main chain missed (its
# satext/lq* links were directory symlinks the CLI could not list; now real directories). Then re-score.
# Usage: ./run_probe_extra.sh ["a 0.9 1" "b 0.9 1" "a 0.7 1"]
set -u
cd "$(dirname "$0")"
CLI=../../../.build/release/sensenova-cli
W=/Volumes/Satechi/Development/mlxengine-image/weights/SenseNova-U1.5-8B-MoT-8step-v2-8bit
V=/Volumes/Satechi/Development/mlxengine-forge/Tools/vosrgate/.build/release/vosrgate
NR=/Volumes/Satechi/Development/mlxengine-forge/Tools/nrscore/.build/out/Products/Release/nrscore
PA="A sharp, high-resolution image."
PB="Upscale this image. Preserve every word and letter exactly as written."
stage() { echo "=== $1  $(date +%T)  load $(uptime | sed 's/.*averages: //')"; }

cell() {
  local arm=$1 t=$2 s=$3; shift 3
  local extra=("$@")
  local M="U1-${arm}-t${t}-s${s}"
  [[ "${extra[*]}" == *init-no-noise* ]] && M="${M}-nonoise"
  local flags=(--weights $W --start-t $t --steps $s --cfg 1 --seed 42 "${extra[@]}")
  if [[ $arm == b ]]; then flags+=(--ref-init --prompt "$PB"); else flags+=(--prompt "$PA"); fi
  stage "$M bench (27 × 512² → 2048²)"
  $CLI "${flags[@]}" --init-dir bench/dump --only-low --scale 4 --out-dir bench/dump --suffix $M \
      --skip-existing --timing logs/timing_${M}_bench.json 2>&1 | grep -E "^\[probe\] done|error|Error"
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

if (( $# == 0 )); then set -- "a 0.9 1" "b 0.9 1" "a 0.7 1"; fi
for spec in "$@"; do cell ${=spec}; done

stage "score bench (SSIMULACRA2 / gradient / FidelityGuard / octaves)"
$V score bench/dump bench/score.csv 2>&1 | tail -2
stage "nrscore bench"
$NR bench/nr.json bench/dump 2>&1 | tail -2
stage "analyze"
python3 -I analyze_probe.py > analysis.txt 2>&1; tail -5 analysis.txt
stage "EXTRA DONE"
