#!/bin/bash
cd /workspace/TAU/experiments/interplay
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUN1_LOG="/workspace/TAU/experiments/interplay/results/logs/crosstask_ov_FULL_20260610_071354.log"
echo "[pipeline] waiting for run1 to finish: $RUN1_LOG"
until grep -qE "\[done\]|EXIT_" "$RUN1_LOG" 2>/dev/null; do sleep 60; done
echo "[pipeline] run1 finished; starting run2 (mechanism decomposition)"
LOG2="results/logs/mechanism_decomp_$(date +%Y%m%d_%H%M%S).log"
echo "$LOG2" > /tmp/run2_log.txt
python mechanism_decomposition.py --cuda 0 --families arithmetic --n-prompts 15 > "$LOG2" 2>&1
echo "EXIT_$? PIPELINE_DONE" >> "$LOG2"
echo "[pipeline] done"
