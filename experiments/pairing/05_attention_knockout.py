"""05 — Attention knockout: what are the pairing heads doing?   (pairing-mechanism)

More specific than 03/04 (which ablate a head's whole output). Here we knock out
individual EDGES in the attention pattern of the selected heads — setting chosen
query->key pre-softmax scores to -inf and letting softmax renormalize — to ask
which connections the heads actually rely on. Necessity test only:


  NECESSITY: knock out each source->target edge type. If accuracy drops for the
    head set but not random, that edge is necessary.

Slot taxonomy (exhaustive): BOS, I (demo input), O (demo output), arr (arrow),
nl (newline between demos), Q (query input), F (final position). The newline slot
is included here — it was omitted from the earlier notebook version, so attention
to/from the \\n separators is now a first-class, testable slot rather than being
silently swept into the -inf mask.

Usage:
    python 05_attention_knockout.py --dataset hendel --head-set pairing
    python 05_attention_knockout.py --dataset nonce+arithmetic --head-set pairing --mode necessity
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
from data.loaders import load_dataset
from utils.positions import find_slot_positions
from utils.eval import check_correct_multitoken
from utils.heads import make_ablation_hooks, make_batched_attn_knockout_hooks

NEG_INF = float('-inf')


def parser():
    p = C.base_parser(__doc__)
    p.add_argument('--head-set', choices=['pairing', 'aggregation'], default='pairing')
    p.add_argument('--scope', default='pooled',
                   help="pooled | nonce | arithmetic | category:<c> | task:<t> | task "
                        "(which head set to knock out; slot edges are per-prompt)")
    p.add_argument('--n-per-task', type=int, default=10,
                   help='prompts per task for the knockout sweep (kept small; batched)')
    return p


# --------------------------------------------------------------------------- #
# mask-function factories (closures over a slot dict S = find_slot_positions(...))
# --------------------------------------------------------------------------- #
def _neginf(scores, srcs, tgts):
    s = scores.clone()
    K = s.shape[-1]
    for src in srcs:
        for tgt in tgts:
            if tgt < K:
                s[src, tgt] = NEG_INF
    return s


def necessity_fns(S):
    """knock out source->target edge types across the full slot taxonomy."""
    per_demo = S['per_demo']
    out_all, in_all = S['out_all'], S['in_all']
    arr, nl, q = S['arrow_all'], S['nl_all'], S['query']
    final = S['final']
    fns = {}

    # O (output) row
    fns['O->BOS'] = lambda s: _neginf(s, out_all, [0])
    def _o_paired(s):
        s = s.clone()
        for d in per_demo:
            for o in d['out']:
                for i in d['in']:
                    s[o, i] = NEG_INF
        return s
    fns['O->I_paired'] = _o_paired
    def _o_unpaired(s):
        s = s.clone()
        for d_idx, d in enumerate(per_demo):
            others = [i for j_idx, j in enumerate(per_demo) if j_idx != d_idx for i in j['in']]
            for o in d['out']:
                for i in others:
                    s[o, i] = NEG_INF
        return s
    fns['O->I_unpaired'] = _o_unpaired
    fns['O->arr'] = lambda s: _neginf(s, out_all, arr)
    fns['O->nl'] = lambda s: _neginf(s, out_all, nl)            # newline slot

    # F (final) row
    fns['F->BOS'] = lambda s: _neginf(s, [final], [0])
    fns['F->I'] = lambda s: _neginf(s, [final], in_all)
    fns['F->O'] = lambda s: _neginf(s, [final], out_all)
    fns['F->arr'] = lambda s: _neginf(s, [final], arr)
    fns['F->nl'] = lambda s: _neginf(s, [final], nl)            # newline slot
    fns['F->Q'] = lambda s: _neginf(s, [final], q)

    # Q (query) row
    if q:
        fns['Q->BOS'] = lambda s: _neginf(s, q, [0])
        fns['Q->I'] = lambda s: _neginf(s, q, in_all)
        fns['Q->O'] = lambda s: _neginf(s, q, out_all)
        fns['Q->arr'] = lambda s: _neginf(s, q, arr)
        fns['Q->nl'] = lambda s: _neginf(s, q, nl)              # newline slot

    # nl (newline) row — newlines as a SOURCE, also previously untested
    fns['nl->BOS'] = lambda s: _neginf(s, nl, [0])
    fns['nl->I'] = lambda s: _neginf(s, nl, in_all)
    fns['nl->O'] = lambda s: _neginf(s, nl, out_all)
    fns['nl->arr'] = lambda s: _neginf(s, nl, arr)
    return fns


# --------------------------------------------------------------------------- #
# batched greedy accuracy under a list of mask fns
# --------------------------------------------------------------------------- #
@torch.no_grad()
def batched_knockout_acc(model, tokens, answer, heads, mask_fns, max_new_tokens=20):
    """Greedy-decode under each mask fn (one batch row each); return list of {0,1}."""
    n = len(mask_fns)
    target = str(answer).strip()
    hooks = make_batched_attn_knockout_hooks(heads, mask_fns)
    cur = tokens.expand(n, -1).clone()
    gen = [[] for _ in range(n)]
    done = [False] * n
    val = [0] * n
    for _ in range(max_new_tokens):
        if all(done):
            break
        logits = model.run_with_hooks(cur, fwd_hooks=hooks)
        new = []
        for b in range(n):
            nt = logits[b, -1].argmax().item()
            gen[b].append(nt)
            new.append(nt)
            if not done[b]:
                dec = model.tokenizer.decode(gen[b]).strip()
                if dec == target:
                    val[b] = 1; done[b] = True
                elif len(dec) >= len(target) or (dec and not target.startswith(dec)):
                    done[b] = True
        cur = torch.cat([cur, torch.tensor(new, device=cur.device).unsqueeze(1)], dim=1)
    return val


def run_block(model, splits, tasks, heads, rand_heads, full_abl, fn_factory,
              n_per_task, desc):
    """Run one knockout block (sufficiency or necessity) -> long-form records."""
    rec = []
    for t in tqdm(tasks, desc=desc):
        for pd_ in splits[t]['icl_prompts'][:n_per_task]:
            tokens = model.to_tokens(pd_['prompt'], prepend_bos=True)
            ans = pd_['query_output']
            S = find_slot_positions(model, pd_['prompt'], pd_['demo_pairs'],
                                    query_input=pd_.get('query_input'))
            fns = fn_factory(S)
            names = list(fns.keys())
            fn_list = list(fns.values())

            rec.append({'task': t, 'cond': 'unablated', 'group': '-',
                        'correct': check_correct_multitoken(model, tokens, ans)})
            rec.append({'task': t, 'cond': 'full_ablation', 'group': '-',
                        'correct': check_correct_multitoken(model, tokens, ans, hooks=full_abl)})

            for grp, hl in [('head_set', heads), ('random', rand_heads)]:
                vals = batched_knockout_acc(model, tokens, ans, hl, fn_list)
                for nm, v in zip(names, vals):
                    rec.append({'task': t, 'cond': nm, 'group': grp, 'correct': v})
            torch.cuda.empty_cache()
    return pd.DataFrame(rec)


def summarize_and_plot(df, title, tag):
    """head_set vs random per condition; 'specific' = random - head_set."""
    piv = (df[df['group'].isin(['head_set', 'random'])]
           .groupby(['cond', 'group'])['correct'].mean().unstack('group'))
    if 'head_set' in piv and 'random' in piv:
        piv['specific'] = piv['random'] - piv['head_set']
    base = df[df['cond'].isin(['unablated', 'full_ablation'])].groupby('cond')['correct'].mean()
    print(f'\n=== {title} ===')
    print('baselines:', {k: round(v, 3) for k, v in base.items()})
    print(piv.round(3).to_string())

    plot_src = df[df['group'].isin(['head_set', 'random'])]
    order = [c for c in df['cond'].unique() if c not in ('unablated', 'full_ablation')]
    fig, ax = plt.subplots(figsize=(max(9, 0.6 * len(order)), 4.8))
    sns.barplot(data=plot_src, x='cond', y='correct', hue='group',
                order=order, hue_order=['head_set', 'random'],
                errorbar=('ci', 95), ax=ax)
    ax.set_ylim(0, 1.05); ax.set_xlabel(''); ax.set_ylabel('accuracy')
    ax.set_title(title)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right', fontsize=7)
    ax.grid(True, alpha=0.3, axis='y'); ax.legend(title='')
    C.save_fig(fig, tag)
    return piv


def main():
    args = parser().parse_args()
    ds = args.dataset.replace('+', '_')
    model = C.load_model(cuda_visible=args.cuda)

    splits = load_dataset(args.dataset)
    tasks = sorted(splits.keys())
    print(f'{ds}: {len(tasks)} tasks | head-set={args.head_set} | scope={args.scope} | necessity')

    from utils.heads import select_scope, categorize_tasks
    _, _, ms = C.load_scope_heads(args.dataset, args.head_pct, 'pooled', args.head_set)

    payload = {'args': vars(args)}
    all_dfs = []

    def block_for(subset, heads, rand_heads, label):
        full_abl = make_ablation_hooks(heads)
        df = run_block(model, splits, subset, heads, rand_heads, full_abl,
                       necessity_fns, args.n_per_task, f'necessity[{label}]')
        df['scope'] = label
        return df

    if args.scope == 'task':
        for t in tasks:
            e = select_scope(ms, f'task:{t}')
            all_dfs.append(block_for([t], e[args.head_set], e[f'{args.head_set}_rand'], f'task:{t}'))
    else:
        e = select_scope(ms, args.scope)
        if args.scope.startswith('task:'):
            subset = [args.scope.split(':', 1)[1]]
        elif args.scope in ('nonce', 'arithmetic') or args.scope.startswith('category:'):
            cat = args.scope.split(':', 1)[1] if ':' in args.scope else args.scope
            subset = categorize_tasks(splits)[cat]
        else:
            subset = tasks
        all_dfs.append(block_for(subset, e[args.head_set], e[f'{args.head_set}_rand'], args.scope))

    nec_df = pd.concat(all_dfs, ignore_index=True)
    sc = args.scope.replace(':', '-')
    nec_piv = summarize_and_plot(
        nec_df, f'{ds}: attention necessity ({args.head_set}, {args.scope})',
        f'05_knockout_necessity_{ds}_{args.head_set}_{sc}')
    payload['necessity_df'] = nec_df
    payload['necessity_summary'] = nec_piv.reset_index()

    C.save_results(f'05_attention_knockout_{ds}_{args.head_set}_{sc}', payload)


if __name__ == '__main__':
    main()
