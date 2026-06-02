"""01 — Corruption -> ICL accuracy  (matrix conditions 1 & 2)

Greedy ICL accuracy across the full corruption sweep, evaluated on the same
prompts. Conditions: orig + {shuffle,random,mean,star} x {input,output}.
'mean' is a global mean-embed (see utils.corruptions); the ladder runs
orig -> shuffle -> random -> mean -> star in increasing severity.

Usage:
    python 01_corruption_icl_accuracy.py --dataset hendel
    python 01_corruption_icl_accuracy.py --dataset nonce+arithmetic
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


def main():
    args = C.base_parser(__doc__).parse_args()
    ds = args.dataset.replace('+', '_')
    rng = random.Random(args.seed)
    model = C.load_model(cuda_visible=args.cuda)

    splits = load_dataset(args.dataset)
    tasks = sorted(splits.keys())
    pools = C.make_pools(splits, model)
    conds = C.condition_list()
    print(f'{ds}: {len(tasks)} tasks | {len(conds)} conditions')

    jobs = [(t, i, pd_) for t in tasks
            for i, pd_ in enumerate(splits[t]['icl_prompts'][:args.n_prompts])]

    rec = []
    for t, i, pd_ in tqdm(jobs, desc=f'{ds} ICL acc'):
        ans = pd_.get('query_output')
        if ans is None:
            continue
        for cond, target, mode in conds:
            cpd, chooks = C.build_corruption(pd_, target, mode, rng, pools, model)
            tok = model.to_tokens(cpd['prompt'], prepend_bos=True)
            ok = check_correct_multitoken(model, tok, ans, hooks=(chooks or None))
            rec.append({'task': t, 'cond': cond, 'correct': ok})
            torch.cuda.empty_cache()

    df = pd.DataFrame(rec)
    cond_order = [c[0] for c in conds]
    acc = df.groupby(['task', 'cond'])['correct'].mean().unstack()[cond_order]
    overall = df.groupby('cond')['correct'].mean().reindex(cond_order)
    print('\n=== mean accuracy by condition ===')
    print(overall.round(3).to_string())
    print('\n=== per-task accuracy ===')
    print(acc.round(3).to_string())

    plot_df = df.groupby(['cond'])['correct'].mean().reindex(cond_order).reset_index()
    fig, ax = plt.subplots(figsize=(max(8, 0.8 * len(cond_order)), 4.5))
    sns.barplot(data=plot_df, x='cond', y='correct', order=cond_order, ax=ax)
    ax.set_ylim(0, 1.05); ax.set_ylabel('ICL accuracy'); ax.set_xlabel('')
    ax.set_title(f'{ds}: ICL accuracy by corruption condition')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    tag = f'01_corruption_icl_{ds}'
    C.save_fig(fig, tag)
    C.save_results(tag, {'df': df, 'acc': acc, 'overall': overall.reset_index(),
                         'args': vars(args)})


if __name__ == '__main__':
    main()
