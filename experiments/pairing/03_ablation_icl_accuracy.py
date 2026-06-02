"""03 — Ablate heads -> ICL accuracy  (matrix conditions 5 & 6)

Zero-ablate a head set during the forward pass and measure greedy ICL accuracy
across the full corruption sweep. Conditions (run internally, per prompt):
    orig + {shuffle,random,mean,star} x {input,output}   (9 conditions)

Head set chosen by --head-set {pairing,aggregation} and --scope:
    pooled            heads scored over all prompts
    nonce|arithmetic  category-specific heads (nonce+arithmetic dataset)
    category:<c>      explicit category
    task:<t>          one task's heads, run on that task only
    task              loop every task, each on its own head set

Requires score_heads.py to have been run first (provides the head-set cache).
For each condition we compare: intact / ablated / rand_ablated.

Usage:
    python 03_ablation_icl_accuracy.py --dataset nonce+arithmetic --head-set pairing --scope pooled
    python 03_ablation_icl_accuracy.py --dataset nonce+arithmetic --head-set pairing --scope task
"""
import random
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from tqdm import tqdm

import _common as C
from data.loaders import load_dataset
from utils.eval import check_correct_multitoken
from utils.heads import make_ablation_hooks, select_scope


def parser():
    p = C.base_parser(__doc__)
    p.add_argument('--head-set', choices=['pairing', 'aggregation'], default='pairing')
    p.add_argument('--scope', default='pooled',
                   help="pooled | nonce | arithmetic | category:<c> | task:<t> | task")
    return p


def run_one_scope(model, splits, task_subset, heads, rand_heads, pools, conds,
                  args, scope_label):
    """Run the full condition sweep for one (head set, task subset)."""
    abl = make_ablation_hooks(heads)
    rabl = make_ablation_hooks(rand_heads)
    rng = random.Random(args.seed)
    rec = []
    jobs = [(t, i, pd_) for t in task_subset
            for i, pd_ in enumerate(splits[t]['icl_prompts'][:args.n_prompts])]
    for t, i, pd_ in tqdm(jobs, desc=f'{scope_label} ICL'):
        ans = pd_.get('query_output')
        if ans is None:
            continue
        for cond, target, mode in conds:
            cpd, chooks = C.build_corruption(pd_, target, mode, rng, pools, model)
            tok = model.to_tokens(cpd['prompt'], prepend_bos=True)
            for variant, vh in [('intact', []), ('ablated', abl), ('rand_ablated', rabl)]:
                ok = check_correct_multitoken(model, tok, ans, hooks=chooks + vh)
                rec.append({'scope': scope_label, 'task': t, 'cond': cond,
                            'variant': variant, 'correct': ok})
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

    rec = []
    if args.scope == 'task':
        # loop every task on its own per-task head set
        for t in tasks:
            entry = select_scope(ms, f'task:{t}')
            rec += run_one_scope(model, splits, [t], entry[args.head_set],
                                 entry[f'{args.head_set}_rand'], pools, conds, args,
                                 f'task:{t}')
    else:
        entry = select_scope(ms, args.scope)
        if args.scope.startswith('task:'):
            subset = [args.scope.split(':', 1)[1]]
        elif args.scope in ('nonce', 'arithmetic') or args.scope.startswith('category:'):
            from utils.heads import categorize_tasks
            cat = args.scope.split(':', 1)[1] if ':' in args.scope else args.scope
            subset = categorize_tasks(splits)[cat]
        else:
            subset = tasks
        rec = run_one_scope(model, splits, subset, entry[args.head_set],
                            entry[f'{args.head_set}_rand'], pools, conds, args, args.scope)

    df = pd.DataFrame(rec)
    acc = df.groupby(['cond', 'variant'])['correct'].mean().unstack('variant')
    order = ['intact', 'ablated', 'rand_ablated']
    acc = acc[[c for c in order if c in acc.columns]]
    acc['specific_effect'] = acc['intact'] - acc['ablated']
    print('\n=== accuracy by condition (mean over tasks/prompts) ===')
    print(acc.round(3).to_string())

    plot_df = df.groupby(['cond', 'variant'])['correct'].mean().reset_index()
    cond_order = [c[0] for c in conds]
    fig, ax = plt.subplots(figsize=(max(9, 0.8 * len(cond_order)), 4.8))
    sns.barplot(data=plot_df, x='cond', y='correct', hue='variant',
                order=cond_order, hue_order=order, ax=ax)
    ax.set_ylim(0, 1.05); ax.set_xlabel(''); ax.set_ylabel('ICL accuracy')
    ax.set_title(f'{ds}: ablate {args.head_set} ({args.scope})')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right', fontsize=7)
    ax.legend(title=''); ax.grid(True, alpha=0.3, axis='y')

    sc = args.scope.replace(':', '-')
    tag = f'03_ablation_icl_{ds}_{args.head_set}_{sc}'
    C.save_fig(fig, tag)
    C.save_results(tag, {'df': df, 'acc': acc, 'args': vars(args)})


if __name__ == '__main__':
    main()
