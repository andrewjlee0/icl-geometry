"""bridge — link the pairing and aggregation head sets.

Two experiments, both asking how the two populations interact:

(1) WRITE-COSINE UNDER PAIRING ABLATION.
    For each prompt, take every aggregation head's write to the residual at the
    final (query) position — z_h @ W_O[h] — under a normal forward pass vs a pass
    with the pairing heads zero-ablated. Cosine(intact, pairing-ablated) per head.
    Low cosine => ablating pairing heads changes WHAT the aggregation heads write,
    i.e. aggregation depends on pairing upstream.

(2) PAIRING AMPLIFICATION.
    Scale the pairing heads' outputs (x2, x2.5, x3) during the forward pass and
    measure (a) ICL accuracy (should stay high) and (b) peak task-vector patching
    into a held-out zero-shot query (predicted to drop if amplifying pairing heads
    contaminates the otherwise-abstract task vector). Matched random heads are the
    control. Hendel prompts are split into survives/fails under shuffled inputs.

Usage:
    python bridge.py --dataset hendel
    python bridge.py --dataset hendel --scales 2 2.5 3 --n-prompts 50
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
from data.prompts import build_icl_prompt, build_zero_shot_prompt
from data.loaders import load_dataset
from utils.corruptions import apply_corruption
from utils.eval import check_correct_multitoken
from utils.heads import get_head_sets, make_ablation_hooks, make_amplify_hooks


def parser():
    p = C.base_parser(__doc__)
    p.add_argument('--scales', type=float, nargs='+', default=[2.0, 2.5, 3.0],
                   help='amplification factors for experiment (2)')
    return p


@torch.no_grad()
def aggregation_write_vectors(model, prompt, agg_heads, fwd_hooks=None):
    """Each aggregation head's residual write (z_h @ W_O[h]) at the final position."""
    toks = model.to_tokens(prompt, prepend_bos=True)
    last = toks.shape[1] - 1
    zfilter = lambda n: 'hook_z' in n
    if fwd_hooks:
        with model.hooks(fwd_hooks=fwd_hooks):
            _, cache = model.run_with_cache(toks, names_filter=zfilter)
    else:
        _, cache = model.run_with_cache(toks, names_filter=zfilter)
    writes = {}
    for (L, h) in agg_heads:
        z = cache[f'blocks.{L}.attn.hook_z'][0, last, h]      # (d_head,)
        W_O = model.blocks[L].attn.W_O[h]                     # (d_head, d_model)
        writes[(L, h)] = (z.float() @ W_O.float()).cpu().numpy()
    del cache
    torch.cuda.empty_cache()
    return writes


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float('nan')
    return float(np.dot(a, b) / (na * nb))


def main():
    args = parser().parse_args()
    ds = args.dataset.replace('+', '_')
    rng = random.Random(args.seed)
    model = C.load_model(cuda_visible=args.cuda)
    n_layers = model.cfg.n_layers

    splits = load_dataset(args.dataset)
    tasks = sorted(splits.keys())
    print(f'{args.dataset}: {len(tasks)} tasks')

    # head sets on clean prompts (cached, shared with scripts 03/04)
    clean = [p for t in tasks for p in splits[t]['icl_prompts'][:args.n_prompts]]
    cache = C.RESULTS_DIR / f'head_sets_{ds}_pct{args.head_pct}.pkl'
    hs = get_head_sets(model, clean, pct=args.head_pct, cache_path=cache)
    pairing, aggregation = hs['pairing'], hs['aggregation']
    pairing_rand = hs['pairing_rand']
    pairing_ablation = make_ablation_hooks(pairing)
    print(f'{len(pairing)} pairing, {len(aggregation)} aggregation heads')

    jobs = [(t, i, pd_)
            for t in tasks
            for i, pd_ in enumerate(splits[t]['icl_prompts'][:args.n_prompts])]

    # ------------------------------------------------------------------ #
    # classify survives/fails under shuffled (input-corrupted) demos
    # ------------------------------------------------------------------ #
    print('classifying survives/fails under input shuffle...')
    group = {}
    for t, i, pd_ in tqdm(jobs, desc='classify'):
        ans = pd_.get('query_output')
        if ans is None:
            group[(t, i)] = None
            continue
        tok = model.to_tokens(pd_['prompt'], prepend_bos=True)
        orig_ok = check_correct_multitoken(model, tok, ans)
        cpd = apply_corruption(pd_, target='input', mode='shuffle', rng=rng,
                               build_prompt=build_icl_prompt)
        shuf_ok = check_correct_multitoken(
            model, model.to_tokens(cpd['prompt'], prepend_bos=True), ans)
        group[(t, i)] = ('survives' if (orig_ok and shuf_ok)
                         else 'fails' if orig_ok else None)
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # (1) write-cosine under pairing ablation
    # ------------------------------------------------------------------ #
    print('\n(1) aggregation-write cosine under pairing ablation...')
    cos_rows = []
    for t, i, pd_ in tqdm(jobs, desc='write cosine'):
        intact = aggregation_write_vectors(model, pd_['prompt'], aggregation)
        ablated = aggregation_write_vectors(model, pd_['prompt'], aggregation,
                                            fwd_hooks=pairing_ablation)
        for (L, h) in aggregation:
            cos_rows.append({'task': t, 'layer': L, 'head': h,
                             'group': group[(t, i)],
                             'cos': cosine(intact[(L, h)], ablated[(L, h)])})
    cos_df = pd.DataFrame(cos_rows)
    print('mean aggregation-write cosine (intact vs pairing-ablated): '
          f'{cos_df["cos"].mean():.3f}')
    print('per aggregation head (mean over prompts), lowest 10:')
    per_head = (cos_df.groupby(['layer', 'head'])['cos'].mean()
                      .sort_values().head(10))
    print(per_head.round(3).to_string())

    fig1, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(cos_df['cos'].dropna(), bins=30, ax=ax)
    ax.axvline(cos_df['cos'].mean(), color='r', ls='--',
               label=f'mean={cos_df["cos"].mean():.3f}')
    ax.set_xlabel('cosine(intact write, pairing-ablated write)')
    ax.set_title(f'{args.dataset}: aggregation-write change under pairing ablation')
    ax.legend()
    C.save_fig(fig1, f'bridge_writecos_{ds}')

    # ------------------------------------------------------------------ #
    # (2) pairing amplification: ICL accuracy + peak TV patching
    # ------------------------------------------------------------------ #
    print('\n(2) pairing amplification...')
    conditions = [('normal', None)]
    for s in args.scales:
        conditions.append((f'pairing_{s}x', make_amplify_hooks(pairing, s)))
    for s in args.scales:
        conditions.append((f'rand_{s}x', make_amplify_hooks(pairing_rand, s)))

    amp_rows = []
    for t, i, pd_ in tqdm(jobs, desc='amplification'):
        g = group[(t, i)]
        if g is None:
            continue
        ans = pd_['query_output']
        tok_icl = model.to_tokens(pd_['prompt'], prepend_bos=True)
        zs_in, zs_out = C.held_out_query(splits, t, i)
        if zs_out is None:
            continue
        zs_prompt = build_zero_shot_prompt(zs_in)
        arrow = C.query_arrow_position(model, pd_['prompt'])
        for cond, hooks in conditions:
            icl_ok = check_correct_multitoken(model, tok_icl, ans, hooks=hooks)
            tv = C.extract_tv_all_layers(model, pd_['prompt'], arrow, fwd_hooks=hooks)
            peak = max(C.patch_all_layers_batched(model, zs_prompt, tv, zs_out).values())
            amp_rows.append({'task': t, 'group': g, 'cond': cond,
                             'icl': icl_ok, 'patch_peak': peak})
        torch.cuda.empty_cache()

    amp_df = pd.DataFrame(amp_rows)
    cond_order = [c for c, _ in conditions]
    summ = amp_df.groupby('cond')[['icl', 'patch_peak']].mean().reindex(cond_order)
    print('\n=== amplification: ICL acc & peak patching (mean over prompts) ===')
    print(summ.round(3).to_string())
    print('\nby survives/fails:')
    print(amp_df.groupby(['group', 'cond'])[['icl', 'patch_peak']]
                .mean().round(3).to_string())

    fig2, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, col, ttl in [(axes[0], 'icl', 'ICL accuracy (should stay high)'),
                         (axes[1], 'patch_peak', 'Peak TV patching (predicted to drop)')]:
        sns.barplot(data=amp_df, x='cond', y=col, order=cond_order,
                    errorbar=('ci', 95), ax=ax)
        ax.set_ylim(0, 1.05); ax.set_xlabel(''); ax.set_title(ttl)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
    fig2.suptitle(f'{args.dataset}: pairing amplification')
    C.save_fig(fig2, f'bridge_amplification_{ds}')

    C.save_results(f'bridge_{ds}', {
        'cos_df': cos_df, 'per_head_cos': per_head,
        'amp_df': amp_df, 'amp_summary': summ,
        'pairing': pairing, 'aggregation': aggregation, 'args': vars(args)})


if __name__ == '__main__':
    main()
