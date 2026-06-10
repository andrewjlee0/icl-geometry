#!/usr/bin/env python
"""
MLP necessity at the FINAL (query-arrow) position, per layer, per task.
Ablates the MLP output at the final position one layer at a time (demos in context)
and measures the drop in multi-token ICL accuracy. Runs ALL tasks, N prompts each.

Self-contained. Run on the pod:
    nohup python -u mlp_necessity.py > results/logs/mlp_necessity.out 2>&1 &
Outputs:
    results/mlp_necessity__df.csv     (task, layer, baseline, acc, drop)
    results/mlp_necessity__pivot.csv  (layer x task drop table)
    results/mlp_necessity.png         (drop vs layer, one line per task)
"""
import os, sys, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))  # repo root, two up from experiments/interplay/
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import torch

import experiments.pairing._common as C
from data.loaders import load_dataset
from utils.eval import check_correct_multitoken

# ---------------- config ----------------
DATASET   = 'nonce+arithmetic'
N         = 50           # prompts per task
CUDA      = '0'
RES       = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(os.path.join(RES, 'logs'), exist_ok=True)

t0 = time.time()
print(f'[{time.time()-t0:.0f}s] loading model...', flush=True)
model  = C.load_model(cuda_visible=CUDA)
splits = load_dataset(DATASET)
tasks  = sorted(splits.keys())
n_layers = model.cfg.n_layers
print(f'[{time.time()-t0:.0f}s] {DATASET}: {len(tasks)} tasks, {n_layers} layers, N={N}', flush=True)

@torch.no_grad()
def acc_final_mlp_ablated(task, L_ablate):
    ok = n = 0
    for pd_ in splits[task]['icl_prompts'][:N]:
        prompt = pd_['prompt']; ans = pd_['query_output']
        arrow = C.query_arrow_position(model, prompt)
        toks  = model.to_tokens(prompt, prepend_bos=True)
        hooks = []
        if L_ablate is not None:
            def hook(mlp_out, hook, pos=arrow):
                mlp_out[0, pos, :] = 0.0
                return mlp_out
            hooks = [(f'blocks.{L_ablate}.hook_mlp_out', hook)]
        ok += int(check_correct_multitoken(model, toks, ans, hooks=(hooks or None)))
        n  += 1
    return ok / n

rows = []
for ti, task in enumerate(tasks):
    base = acc_final_mlp_ablated(task, None)
    print(f'[{time.time()-t0:.0f}s] ({ti+1}/{len(tasks)}) {task} baseline={base:.3f}', flush=True)
    for L in range(n_layers):
        a = acc_final_mlp_ablated(task, L)
        rows.append({'task': task, 'layer': L, 'baseline': base, 'acc': a, 'drop': base - a})
    # checkpoint after each task so an interruption keeps progress
    pd.DataFrame(rows).to_csv(os.path.join(RES, 'mlp_necessity__df.csv'), index=False)

M = pd.DataFrame(rows)
M.to_csv(os.path.join(RES, 'mlp_necessity__df.csv'), index=False)
piv = M.pivot_table(index='layer', columns='task', values='drop').round(3)
piv.to_csv(os.path.join(RES, 'mlp_necessity__pivot.csv'))

fig, ax = plt.subplots(figsize=(13, 7))
for t, g in M.groupby('task'):
    g = g.sort_values('layer')
    ax.plot(g['layer'].values, g['drop'].values, 'o-', label=t, ms=2, lw=1)
ax.axhline(0, color='gray', lw=.8)
ax.set(xlabel='layer (final-position MLP ablated)', ylabel='accuracy drop from baseline',
       title=f'{DATASET}: final-position MLP necessity by layer (N={N})')
ax.legend(fontsize=6, ncol=2)
ax.grid(alpha=.3)
fig.tight_layout()
fig.savefig(os.path.join(RES, 'mlp_necessity.png'), dpi=130)
print(f'[{time.time()-t0:.0f}s] DONE. wrote results/mlp_necessity__df.csv, __pivot.csv, .png', flush=True)
