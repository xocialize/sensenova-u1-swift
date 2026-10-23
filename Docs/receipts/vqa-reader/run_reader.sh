#!/bin/zsh
# U1.5 VQA as a text reader, on the SR probe's inputs: SA-Text lv3 / lv1 / hq (200-image subset) and the four
# ScreenSR pages (lr and hr). Output in the `vosrgate ocr` JSON shape so the probe's scorers apply unchanged.
set -u
cd "$(dirname "$0")"
CLI=../../../.build/release/sensenova-cli
W=/Volumes/Satechi/Development/mlxengine-image/weights/SenseNova-U1.5-8B-MoT-8step-v2-8bit
SP=$(cd ../sr-probe && pwd)
stage() { echo "=== $1  $(date +%T)"; }
run() {  # run <name> <dir> <max-tokens>
  stage "$1 ($(ls $2 | wc -l | tr -d ' ') images, max-tokens $3)"
  $CLI --weights $W --vqa-dir $2 --out reads/u1_$1.json --timing logs/timing_$1.json --max-tokens $3 2>&1 \
    | grep -E "^\[vqa\] (wrote|[0-9]+/[0-9]+ .*(50|100|150|200)/)|error|Error" | tee logs/$1.log
}
run satext_lq3 $SP/satext/lq3 256
run satext_lq1 $SP/satext/lq1 256
run satext_hq $(pwd)/hq 256
run screensr_lr $SP/screensr/lr 1024
run screensr_hr $SP/screensr/hr 1024
stage "analyze"
python3 -I analyze_reader.py > analysis.txt 2>&1; tail -3 analysis.txt
stage "READER DONE"
