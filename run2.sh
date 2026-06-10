#!/usr/bin/env bash
# Regenerate head-set caches (with the fixed random control) and re-run ONLY the
# head-dependent jobs (03, 04, 05). Leaves 01/02 corruption results untouched.
# Sequential, one GPU, survives disconnect. Skips jobs whose .pkl already exists,
# so safe to re-run if interrupted.
set -u
cd /workspace/TAU/experiments/pairing
RES=/workspace/TAU/results
mkdir -p "$RES/logs"
MAIN="$RES/logs/rerun_heads_main.log"
echo "START $(date)" | tee "$MAIN"

# --- 1. delete old head caches (they have the contaminated random control) ---
echo "removing old head-set caches..." | tee -a "$MAIN"
rm -f "$RES"/head_sets_hendel_pct10.pkl "$RES"/head_sets_nonce_arithmetic_pct10.pkl

# --- 2. rebuild head sets (uses fixed matched_random_heads) ---
for ds in hendel nonce+arithmetic; do
  echo ">>> SCORE $ds  $(date +%H:%M:%S)" | tee -a "$MAIN"
  CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python -u score_heads.py --dataset "$ds" --cuda 0 \
    > "$RES/logs/score_${ds//+/_}.log" 2>&1
  dsl="${ds//+/_}"
  if [ -f "$RES/head_sets_${dsl}_pct10.pkl" ]; then echo "    OK score $ds" | tee -a "$MAIN"
  else echo "    !!! FAILED score $ds" | tee -a "$MAIN"; fi
done

# --- 3. re-run ONLY head-dependent jobs (03, 04, 05). Force overwrite. ---
run() {
  local tag="$1"; shift
  echo ">>> RUN  $tag   $(date +%H:%M:%S)" | tee -a "$MAIN"
  CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python -u "$@" --cuda 0 \
    > "$RES/logs/job_${tag}.log" 2>&1
  if [ -f "$RES/${tag}.pkl" ]; then echo "    OK   $tag" | tee -a "$MAIN"
  else echo "    FAIL $tag (see $RES/logs/job_${tag}.log)" | tee -a "$MAIN"; fi
}

# delete old head-dependent results so they're regenerated with the fixed control
rm -f "$RES"/03_ablation_icl_*.pkl "$RES"/04_ablation_tvpatch_*.pkl "$RES"/05_attention_knockout_*.pkl

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
echo "DONE $(date)  --  $N results present (incl. untouched 01/02)" | tee -a "$MAIN"
