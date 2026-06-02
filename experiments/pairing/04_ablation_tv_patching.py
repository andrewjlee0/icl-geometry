"""04 — Ablate heads -> task-vector patching  (matrix conditions 7 & 8)

Extract the task vector (resid at query-arrow) while a head set is ablated AND the
demos are corrupted, patch into a held-out zero-shot query, peak over layers.
Conditions (internal, per prompt): orig + {shuffle,random,mean,star} x {input,output}.
Variants per condition: intact / ablated / rand_ablated (head set ablated during
extraction). Scope selects which head set (see 03 for --scope semantics).

Requires score_heads.py first. --save-activations stores the extracted TVs.

Usage:
    python 04_ablation_tv_patching.py --dataset hendel --head-set pairing --scope pooled --save-activations
    python 04_ablation_tv_patching.py --dataset hendel --head-set pairing --scope task
"""
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

import _common as C
from data.prompts import build_zero_shot_prompt
from data.loaders import load_dataset
from utils.heads import make_ablation_hooks, select_scope, categorize_tasks


def parser():
    p = C.base_parser(__doc__)
    p.add_argument('--head-set', choices=['pairing', 'aggregation'], default='pairing')
    p.add_argument('--scope', default='pooled',
                   help="pooled | nonce | arithmetic | category:<c> | task:<t> | task")
    p.add_argument('--save-activations', action='store_true',
                   help='also save extracted task vectors to a .npz')
    return p


def run_one_scope(model, splits, task_subset, heads, rand_heads, pools, conds,
                  args, scope_label, tv_store, tv_meta):
    n_layers = model.cfg.n_layers
    abl = make_ablation_hooks(heads)
    rabl = make_ablation_hooks(rand_heads)
    variants = [('intact', []), ('ablated', abl), ('rand_ablated', rabl)]
    rng = random.Random(args.seed)
    rec = []
    jobs = [(t, i, pd_) for t in task_subset
            for i, pd_ in enumerate(splits[t]['icl_prompts'][:args.n_prompts])]
    for t, i, pd_ in tqdm(jobs, desc=f'{scope_label} TVpatch'):
        zs_in, zs_out = C.held_out_query(splits, t, i)
        if zs_out is None:
            continue
        zs_prompt = build_zero_shot_prompt(zs_in)
        for cond, target, mode in conds:
            cpd, chooks = C.build_corruption(pd_, target, mode, rng, pools, model)
            arrow = C.query_arrow_position(model, cpd['prompt'])
            for vname, vh in variants:
                tv = C.extract_tv_all_layers(model, cpd['prompt'], arrow,
                                             fwd_hooks=(chooks + vh) or None)
                if tv_store is not None and vname == 'ablated':
                    key = f'{scope_label}|{t}|{cond}|{i}'
                    tv_store[key] = np.stack([tv[L] for L in range(n_layers)])
                    tv_meta.append({'key': key, 'scope': scope_label, 'task': t,
                                    'cond': cond, 'prompt_idx': i,
                                    'query_input': zs_in, 'answer': zs_out})
                per_layer = C.patch_all_layers_batched(model, zs_prompt, tv, zs_out)
                peak = max(per_layer.values())
                rec.append({'scope': scope_label, 'task': t, 'cond': cond,
                            'variant': vname, 'peak': peak})
            torch.cuda.empty_cache()
    return rec


def main():
    args = parser().parse_args()
    ds = args.dataset.replace('+', '_')
    model = C.load_model(cuda_visible=args.cuda)

    splits = load_dataset(args.dataset)
    tasks = sorted(splits.keys())
    _, _, ms = C.load_scope_heads(args.dataset, args.head_pct, 'pooled', args.head_set)
    pools = C.make_pools(splits, model)
    conds = C.condition_list()
    print(f'{ds}: {len(tasks)} tasks | head-set={args.head_set} | scope={args.scope} '
          f'| {len(conds)} conditions')

    tv_store = {} if args.save_activations else None
    tv_meta = []
    rec = []
    if args.scope == 'task':
        for t in tasks:
            entry = select_scope(ms, f'task:{t}')
            rec += run_one_scope(model, splits, [t], entry[args.head_set],
                                 entry[f'{args.head_set}_rand'], pools, conds, args,
                                 f'task:{t}', tv_store, tv_meta)
    else:
        entry = select_scope(ms, args.scope)
        if args.scope.startswith('task:'):
            subset = [args.scope.split(':', 1)[1]]
        elif args.scope in ('nonce', 'arithmetic') or args.scope.startswith('category:'):
            cat = args.scope.split(':', 1)[1] if ':' in args.scope else args.scope
            subset = categorize_tasks(splits)[cat]
        else:
            subset = tasks
        rec = run_one_scope(model, splits, subset, entry[args.head_set],
                            entry[f'{args.head_set}_rand'], pools, conds, args,
                            args.scope, tv_store, tv_meta)

    df = pd.DataFrame(rec)
    order = ['intact', 'ablated', 'rand_ablated']
    summ = df.groupby(['cond', 'variant'])['peak'].mean().unstack('variant')
    summ = summ[[c for c in order if c in summ.columns]]
    print('\n=== peak TV recovery by condition (mean over tasks/prompts) ===')
    print(summ.round(3).to_string())

    plot_df = df.groupby(['cond', 'variant'])['peak'].mean().reset_index()
    cond_order = [c[0] for c in conds]
    fig, ax = plt.subplots(figsize=(max(9, 0.8 * len(cond_order)), 4.8))
    import seaborn as sns
    sns.barplot(data=plot_df, x='cond', y='peak', hue='variant',
                order=cond_order, hue_order=order, ax=ax)
    ax.set_ylim(0, 1.05); ax.set_xlabel(''); ax.set_ylabel('peak TV recovery')
    ax.set_title(f'{ds}: TV patch, ablate {args.head_set} ({args.scope})')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right', fontsize=7)
    ax.legend(title=''); ax.grid(True, alpha=0.3, axis='y')

    sc = args.scope.replace(':', '-')
    tag = f'04_ablation_tvpatch_{ds}_{args.head_set}_{sc}'
    C.save_fig(fig, tag)
    payload = {'df': df, 'summary': summ.reset_index(), 'args': vars(args)}
    if args.save_activations:
        payload['tv_meta'] = pd.DataFrame(tv_meta)
        C.save_activations(tag, tv_store)
    C.save_results(tag, payload)


if __name__ == '__main__':
    main()
