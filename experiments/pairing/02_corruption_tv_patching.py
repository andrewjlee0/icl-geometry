"""02 — Corruption -> task-vector patching  (matrix conditions 3 & 4)

Extract the task vector (resid at query-arrow) under each corruption condition,
patch into a held-out zero-shot query, peak over layers. Conditions:
orig + {shuffle,random,mean,star} x {input,output}.

--save-activations stores the extracted TVs (key scope|cond|prompt_idx -> array).

Usage:
    python 02_corruption_tv_patching.py --dataset hendel --save-activations
"""
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from tqdm import tqdm

import _common as C
from data.prompts import build_zero_shot_prompt
from data.loaders import load_dataset


def main():
    p = C.base_parser(__doc__)
    p.add_argument('--save-activations', action='store_true')
    args = p.parse_args()
    ds = args.dataset.replace('+', '_')
    rng = random.Random(args.seed)
    model = C.load_model(cuda_visible=args.cuda)
    n_layers = model.cfg.n_layers

    splits = load_dataset(args.dataset)
    tasks = sorted(splits.keys())
    pools = C.make_pools(splits, model)
    conds = C.condition_list()
    print(f'{ds}: {len(tasks)} tasks | {len(conds)} conditions')

    jobs = [(t, i, pd_) for t in tasks
            for i, pd_ in enumerate(splits[t]['icl_prompts'][:args.n_prompts])]

    rec = []
    tv_store = {} if args.save_activations else None
    tv_meta = []
    for t, i, pd_ in tqdm(jobs, desc=f'{ds} TV patch'):
        zs_in, zs_out = C.held_out_query(splits, t, i)
        if zs_out is None:
            continue
        zs_prompt = build_zero_shot_prompt(zs_in)
        for cond, target, mode in conds:
            cpd, chooks = C.build_corruption(pd_, target, mode, rng, pools, model)
            arrow = C.query_arrow_position(model, cpd['prompt'])
            tv = C.extract_tv_all_layers(model, cpd['prompt'], arrow,
                                         fwd_hooks=(chooks or None))
            if tv_store is not None:
                key = f'{t}|{cond}|{i}'
                tv_store[key] = np.stack([tv[L] for L in range(n_layers)])
                tv_meta.append({'key': key, 'task': t, 'cond': cond, 'prompt_idx': i,
                                'query_input': zs_in, 'answer': zs_out})
            peak = max(C.patch_and_score(model, zs_prompt, tv[L], L, zs_out)[0]
                       for L in range(n_layers))
            rec.append({'task': t, 'cond': cond, 'peak': peak})
            torch.cuda.empty_cache()

    df = pd.DataFrame(rec)
    cond_order = [c[0] for c in conds]
    overall = df.groupby('cond')['peak'].mean().reindex(cond_order)
    task_peak = df.groupby(['task', 'cond'])['peak'].mean().unstack()[cond_order]
    print('\n=== mean peak TV recovery by condition ===')
    print(overall.round(3).to_string())
    print('\n=== per-task ===')
    print(task_peak.round(3).to_string())

    plot_df = overall.reset_index()
    fig, ax = plt.subplots(figsize=(max(8, 0.8 * len(cond_order)), 4.5))
    sns.barplot(data=plot_df, x='cond', y='peak', order=cond_order, ax=ax)
    ax.set_ylim(0, 1.05); ax.set_ylabel('peak TV recovery'); ax.set_xlabel('')
    ax.set_title(f'{ds}: TV recovery by corruption condition')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    tag = f'02_corruption_tvpatch_{ds}'
    C.save_fig(fig, tag)
    payload = {'df': df, 'overall': overall.reset_index(), 'task_peak': task_peak,
               'args': vars(args)}
    if args.save_activations:
        payload['tv_meta'] = pd.DataFrame(tv_meta)
        C.save_activations(tag, tv_store)
    C.save_results(tag, payload)


if __name__ == '__main__':
    main()
