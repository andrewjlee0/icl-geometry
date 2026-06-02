#!/usr/bin/env bash
# Sequential runner — ONE job at a time on ONE GPU. No concurrency, nothing to
# OOM, nothing to collide. Slower but it WILL finish. Survives disconnect (nohup).
# Skips jobs whose .pkl already exists, so it's safe to re-run after any stop.
set -u
cd /workspace/TAU/experiments/pairing
RES=/workspace/TAU/results
mkdir -p "$RES/logs"
MAIN="$RES/logs/run_main.log"
echo "START $(date)" | tee "$MAIN"

run() {
  local tag="$1"; shift
  if [ -f "$RES/${tag}.pkl" ]; then
    echo "SKIP  $tag (already done)" | tee -a "$MAIN"; return
  fi
  echo ">>>>> RUN  $tag   $(date +%H:%M:%S)" | tee -a "$MAIN"
  CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python -u "$@" --cuda 0 \
    > "$RES/logs/job_${tag}.log" 2>&1
  if [ -f "$RES/${tag}.pkl" ]; then
    echo "      OK   $tag" | tee -a "$MAIN"
  else
    echo "      FAIL $tag  (see $RES/logs/job_${tag}.log)" | tee -a "$MAIN"
  fi
}

run 01_corruption_icl_hendel                                01_corruption_icl_accuracy.py --dataset hendel
run 01_corruption_icl_nonce_arithmetic                      01_corruption_icl_accuracy.py --dataset nonce+arithmetic
run 02_corruption_tvpatch_hendel                            02_corruption_tv_patching.py --dataset hendel --save-activations
run 02_corruption_tvpatch_nonce_arithmetic                  02_corruption_tv_patching.py --dataset nonce+arithmetic --save-activations
run 03_ablation_icl_hendel_pairing_pooled                   03_ablation_icl_accuracy.py --dataset hendel --head-set pairing --scope pooled
run 03_ablation_icl_hendel_pairing_task                     03_ablation_icl_accuracy.py --dataset hendel --head-set pairing --scope task
run 03_ablation_icl_hendel_aggregation_pooled               03_ablation_icl_accuracy.py --dataset hendel --head-set aggregation --scope pooled
run 03_ablation_icl_hendel_aggregation_task                 03_ablation_icl_accuracy.py --dataset hendel --head-set aggregation --scope task
run 03_ablation_icl_nonce_arithmetic_pairing_pooled         03_ablation_icl_accuracy.py --dataset nonce+arithmetic --head-set pairing --scope pooled
run 03_ablation_icl_nonce_arithmetic_pairing_nonce          03_ablation_icl_accuracy.py --dataset nonce+arithmetic --head-set pairing --scope nonce
run 03_ablation_icl_nonce_arithmetic_pairing_arithmetic     03_ablation_icl_accuracy.py --dataset nonce+arithmetic --head-set pairing --scope arithmetic
run 03_ablation_icl_nonce_arithmetic_pairing_task           03_ablation_icl_accuracy.py --dataset nonce+arithmetic --head-set pairing --scope task
run 03_ablation_icl_nonce_arithmetic_aggregation_pooled     03_ablation_icl_accuracy.py --dataset nonce+arithmetic --head-set aggregation --scope pooled
run 03_ablation_icl_nonce_arithmetic_aggregation_nonce      03_ablation_icl_accuracy.py --dataset nonce+arithmetic --head-set aggregation --scope nonce
run 03_ablation_icl_nonce_arithmetic_aggregation_arithmetic 03_ablation_icl_accuracy.py --dataset nonce+arithmetic --head-set aggregation --scope arithmetic
run 03_ablation_icl_nonce_arithmetic_aggregation_task       03_ablation_icl_accuracy.py --dataset nonce+arithmetic --head-set aggregation --scope task
run 04_ablation_tvpatch_hendel_pairing_pooled               04_ablation_tv_patching.py --dataset hendel --head-set pairing --scope pooled --save-activations
run 04_ablation_tvpatch_hendel_pairing_task                 04_ablation_tv_patching.py --dataset hendel --head-set pairing --scope task --save-activations
run 04_ablation_tvpatch_hendel_aggregation_pooled           04_ablation_tv_patching.py --dataset hendel --head-set aggregation --scope pooled --save-activations
run 04_ablation_tvpatch_hendel_aggregation_task             04_ablation_tv_patching.py --dataset hendel --head-set aggregation --scope task --save-activations
run 04_ablation_tvpatch_nonce_arithmetic_pairing_pooled     04_ablation_tv_patching.py --dataset nonce+arithmetic --head-set pairing --scope pooled --save-activations
run 04_ablation_tvpatch_nonce_arithmetic_pairing_nonce      04_ablation_tv_patching.py --dataset nonce+arithmetic --head-set pairing --scope nonce --save-activations
run 04_ablation_tvpatch_nonce_arithmetic_pairing_arithmetic 04_ablation_tv_patching.py --dataset nonce+arithmetic --head-set pairing --scope arithmetic --save-activations
run 04_ablation_tvpatch_nonce_arithmetic_pairing_task       04_ablation_tv_patching.py --dataset nonce+arithmetic --head-set pairing --scope task --save-activations
run 04_ablation_tvpatch_nonce_arithmetic_aggregation_pooled     04_ablation_tv_patching.py --dataset nonce+arithmetic --head-set aggregation --scope pooled --save-activations
run 04_ablation_tvpatch_nonce_arithmetic_aggregation_nonce      04_ablation_tv_patching.py --dataset nonce+arithmetic --head-set aggregation --scope nonce --save-activations
run 04_ablation_tvpatch_nonce_arithmetic_aggregation_arithmetic 04_ablation_tv_patching.py --dataset nonce+arithmetic --head-set aggregation --scope arithmetic --save-activations
run 04_ablation_tvpatch_nonce_arithmetic_aggregation_task       04_ablation_tv_patching.py --dataset nonce+arithmetic --head-set aggregation --scope task --save-activations
run 05_attention_knockout_hendel_pairing_pooled             05_attention_knockout.py --dataset hendel --head-set pairing --scope pooled
run 05_attention_knockout_hendel_aggregation_pooled         05_attention_knockout.py --dataset hendel --head-set aggregation --scope pooled
run 05_attention_knockout_nonce_arithmetic_pairing_pooled   05_attention_knockout.py --dataset nonce+arithmetic --head-set pairing --scope pooled
run 05_attention_knockout_nonce_arithmetic_aggregation_pooled   05_attention_knockout.py --dataset nonce+arithmetic --head-set aggregation --scope pooled

N=$(ls "$RES"/*.pkl 2>/dev/null | grep -v head_sets | wc -l)
echo "DONE $(date)  --  $N / 32 results present" | tee -a "$MAIN"
