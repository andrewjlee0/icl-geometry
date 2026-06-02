#!/usr/bin/env bash
# run_all.sh — full pairing-mechanism pipeline across all GPUs.
# Stages: 0) build nonce+arithmetic dataset  1) score heads (all scopes)
#         2) fan out corruption/ablation/knockout across GPUs.
# Stages 0-1 are serial prerequisites (GPU 0); stage 2 saturates all GPUs.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
while [ "$REPO_ROOT" != "/" ] && ! { [ -d "$REPO_ROOT/utils" ] && [ -d "$REPO_ROOT/configs" ]; }; do
    REPO_ROOT="$(dirname "$REPO_ROOT")"
done
EXP_DIR="$REPO_ROOT/experiments/pairing"
LOG_DIR="$REPO_ROOT/results/logs"
QUEUE_FILE="$REPO_ROOT/results/.job_queue"
LOCK_FILE="$REPO_ROOT/results/.queue.lock"
mkdir -p "$LOG_DIR"

SESSION="pairing"
NP_FLAG=""; [ -n "${N_PROMPTS:-}" ] && NP_FLAG="--n-prompts ${N_PROMPTS}"
DATASETS=("hendel" "nonce+arithmetic")
HEAD_SETS=("pairing" "aggregation")
ABLATION_SCOPES="${ABLATION_SCOPES:-pooled category task}"

if [ -n "${N_GPUS:-}" ]; then NGPU="$N_GPUS"
elif command -v nvidia-smi >/dev/null 2>&1; then NGPU="$(nvidia-smi -L | wc -l)"
else NGPU=1; fi
[ "$NGPU" -lt 1 ] && NGPU=1

job_tag() { echo "$1" | sed -E 's/\.py//; s/--//g; s/[^a-zA-Z0-9]+/_/g; s/_+/_/g; s/^_|_$//g'; }
categories_for() { case "$1" in nonce+arithmetic) echo "nonce arithmetic";; *) echo "";; esac; }

build_jobs() {
    for ds in "${DATASETS[@]}"; do
        echo "01_corruption_icl_accuracy.py --dataset ${ds}"
        echo "02_corruption_tv_patching.py --dataset ${ds} --save-activations"
    done
    for ds in "${DATASETS[@]}"; do
        for sc in $ABLATION_SCOPES; do
            if [ "$sc" = "category" ]; then
                cats="$(categories_for "$ds")"; [ -z "$cats" ] && continue
                for c in $cats; do for hs in "${HEAD_SETS[@]}"; do
                    echo "03_ablation_icl_accuracy.py --dataset ${ds} --head-set ${hs} --scope ${c}"
                    echo "04_ablation_tv_patching.py --dataset ${ds} --head-set ${hs} --scope ${c} --save-activations"
                done; done
            else
                for hs in "${HEAD_SETS[@]}"; do
                    echo "03_ablation_icl_accuracy.py --dataset ${ds} --head-set ${hs} --scope ${sc}"
                    echo "04_ablation_tv_patching.py --dataset ${ds} --head-set ${hs} --scope ${sc} --save-activations"
                done
            fi
        done
    done
    for ds in "${DATASETS[@]}"; do for hs in "${HEAD_SETS[@]}"; do
        echo "05_attention_knockout.py --dataset ${ds} --head-set ${hs} --scope pooled"
    done; done
}

prereqs() {
    if [ -z "${SKIP_DATASET:-}" ]; then
        if [ ! -f "$REPO_ROOT/data/nonce_arithmetic_splits.pkl" ] && \
           [ ! -f "$REPO_ROOT/configs/nonce_arithmetic_splits.pkl" ]; then
            echo "[stage 0] building nonce+arithmetic dataset..."
            ( cd "$REPO_ROOT" && CUDA_VISIBLE_DEVICES=0 python data/make_nonce_arithmetic_splits.py \
                  --cuda 0 > "$LOG_DIR/00_make_nonce_arithmetic.log" 2>&1 ) \
                  || { echo "[stage 0] FAILED"; return 1; }
        else echo "[stage 0] nonce+arithmetic splits exist, skipping."; fi
    fi
    echo "[stage 1] scoring heads at all scopes..."
    for ds in "${DATASETS[@]}"; do
        dsl="$(echo "$ds" | tr '+' '_')"
        cache="$REPO_ROOT/results/head_sets_${dsl}_pct10.pkl"
        if [ -f "$cache" ]; then echo "[stage 1] $ds exists, skipping"; continue; fi
        echo "[stage 1] scoring $ds ..."
        ( cd "$EXP_DIR" && CUDA_VISIBLE_DEVICES=0 python score_heads.py \
              --dataset "$ds" --cuda 0 $NP_FLAG > "$LOG_DIR/01_score_${dsl}.log" 2>&1 ) \
              || { echo "[stage 1] $ds FAILED"; return 1; }
    done
    echo "[prereqs] done."
}

gpu_worker() {
    local gpu="$1"
    while true; do
        local job
        job="$(flock "$LOCK_FILE" bash -c '
            q="'"$QUEUE_FILE"'"; [ -s "$q" ] || exit 1
            head -n1 "$q"; tail -n +2 "$q" > "$q.tmp" && mv "$q.tmp" "$q"')" || break
        [ -z "$job" ] && break
        local tag; tag="$(job_tag "$job")"; local log="$LOG_DIR/${tag}.gpu${gpu}.log"
        echo "[gpu $gpu] START  $job"
        ( cd "$EXP_DIR" && CUDA_VISIBLE_DEVICES="$gpu" python $job --cuda "$gpu" $NP_FLAG > "$log" 2>&1 )
        [ $? -eq 0 ] && echo "[gpu $gpu] DONE   $job" || echo "[gpu $gpu] FAIL  $job ($log)"
    done
    echo "[gpu $gpu] done."
}

run_pipeline() {
    prereqs || { echo "prerequisites failed; aborting."; return 1; }
    build_jobs > "$QUEUE_FILE"; : > "$LOCK_FILE"
    local total; total="$(wc -l < "$QUEUE_FILE")"
    echo "=== stage 2: $total jobs across $NGPU GPU(s) | scopes: $ABLATION_SCOPES ==="
    local pids=()
    for g in $(seq 0 $((NGPU - 1))); do gpu_worker "$g" & pids+=($!); done
    wait "${pids[@]}"
    echo "=== ALL DONE. results/ ==="
}

export REPO_ROOT EXP_DIR LOG_DIR QUEUE_FILE LOCK_FILE NGPU NP_FLAG SESSION

if [ -n "${DRY:-}" ]; then
    echo "GPUs: $NGPU | ablation scopes: $ABLATION_SCOPES"
    echo "prereqs: stage0 make_nonce_arithmetic_splits (unless SKIP_DATASET/exists); stage1 score_heads x ${#DATASETS[@]}"
    echo "--- stage 2 fan-out jobs ($(build_jobs | wc -l)) ---"; build_jobs; exit 0
fi
if [ -n "${NO_TMUX:-}" ]; then run_pipeline; exit $?; fi
if ! command -v tmux >/dev/null 2>&1; then echo "tmux not found; use NO_TMUX=1" >&2; exit 1; fi
if [ -z "${INSIDE_TMUX:-}" ]; then
    tmux kill-session -t "$SESSION" 2>/dev/null
    tmux new-session -d -s "$SESSION" \
        "INSIDE_TMUX=1 NGPU='$NGPU' N_PROMPTS='${N_PROMPTS:-}' ABLATION_SCOPES='$ABLATION_SCOPES' SKIP_DATASET='${SKIP_DATASET:-}' bash '$SCRIPT_DIR/run_all.sh'"
    echo "Launched tmux '$SESSION': prereqs then $(build_jobs | wc -l) jobs on $NGPU GPU(s)."
    echo "  attach: tmux attach -t $SESSION | detach: Ctrl-b d | logs: tail -f $LOG_DIR/*.log"
    exit 0
fi
run_pipeline
