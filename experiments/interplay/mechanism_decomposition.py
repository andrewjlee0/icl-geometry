"""Where does the transplantable cross-task task-code live? Residual-component decomposition.

Motivating puzzle (from crosstask_ov_patching results, arithmetic):
  - Patching the FULL residual at demo-output positions installs the donor task (~0.93 acc_donor).
  - Patching ALL attention heads' output there does NOT transfer (breaks: ~0.97 other).
  - Pairing/aggregation heads are causally NECESSARY (removing/replacing them breaks the source
    task) but NOT SUFFICIENT (their OV write doesn't carry the donor task).
  => Something assembles a transplantable code in the residual that no single head-set carries.
     This script localizes it.

Method
------
Same cross-task setup (within arithmetic, donor -> source at demo-output positions, mean over each
demo's output span, query position untouched, KV-cached prefill-only patch). We swap DONOR content
into the source, but vary WHICH residual component and WHICH layers:

  modes (each = one batched generation per (pair, source prompt)):
    resid_layer : row L patches ONLY blocks.L.hook_resid_post           (28 rows: per-layer sweep)
    resid_cumul : row L patches resid_post at layers 0..L               (where does it saturate?)
    attn_layer  : row L patches ONLY blocks.L.hook_attn_out             (attention contribution)
    mlp_layer   : row L patches ONLY blocks.L.hook_mlp_out              (MLP contribution)
    components  : full_resid_all / attn_all / mlp_all / attn+mlp_all    (which component, all layers)

Readout: acc_donor / acc_src / acc_leak / other (exact match), donor target = donor_rule(source qi).
A component/layer that lifts acc_donor is (part of) the carrier; one that only raises 'other' is
necessary-but-not-sufficient. Comparing resid_layer (assembled state after L) vs attn/mlp_layer
(individual writes at L) separates "the code is read here" from "this component writes it".

Run:
  python mechanism_decomposition.py --cuda 0                 # arithmetic, all pairs, N=15
  python mechanism_decomposition.py --cuda 0 --pilot
Results -> experiments/interplay/results/mechanism_decomp_*.{pkl,csv}
"""
import os
import sys
import time
import pickle
import argparse
import itertools
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch
from tqdm import tqdm

from data.loaders import load_dataset
from data.tasks import NONCE_TASKS, ARITH_TASKS
from transformer_lens import TransformerLensKeyValueCache as KVCache

# reuse validated helpers
import crosstask_ov_patching as X
from crosstask_ov_patching import apply_rule, per_demo_output_positions, load_model

INTERPLAY_RESULTS = REPO_ROOT / 'experiments' / 'interplay' / 'results'
POINT_HOOK = {'resid': 'hook_resid_post', 'attn': 'hook_attn_out', 'mlp': 'hook_mlp_out'}


# --------------------------------------------------------------------------- #
# donor cache: mean-over-output-span of resid_post / attn_out / mlp_out per layer
# --------------------------------------------------------------------------- #
@torch.no_grad()
def build_donor_cache(model, prompts, points, device, extract='mean'):
    """extract='mean': mean over each demo's output span (default, original).
       extract='last': the LAST output token of each demo (no multi-token smear)."""
    n_layers, d_model = model.cfg.n_layers, model.cfg.d_model
    n_demos = len(prompts[0]['demo_pairs'])
    cache = {pt: {L: torch.zeros(len(prompts), n_demos, d_model, dtype=torch.float16,
                                 device=device) for L in range(n_layers)} for pt in points}
    hooknames = {f'blocks.{L}.{POINT_HOOK[pt]}' for pt in points for L in range(n_layers)}
    ok = []
    for j, p in enumerate(prompts):
        pos = per_demo_output_positions(model, p)
        if pos is None:
            ok.append(False); continue
        ok.append(True)
        idx = [torch.tensor(ps, device=device) for ps in pos]
        toks = model.to_tokens(p['prompt'], prepend_bos=True)
        _, c = model.run_with_cache(toks, names_filter=lambda n: n in hooknames)
        for pt in points:
            for L in range(n_layers):
                v = c[f'blocks.{L}.{POINT_HOOK[pt]}'][0]            # (seq, d_model)
                if extract == 'last':
                    cache[pt][L][j] = torch.stack([v[ix[-1]] for ix in idx]).to(torch.float16)
                else:
                    cache[pt][L][j] = torch.stack([v[ix].mean(0) for ix in idx]).to(torch.float16)
        del c
        torch.cuda.empty_cache()
    return cache, ok


# --------------------------------------------------------------------------- #
# batched patched generation over conditions (each = a set of (point, layer) patches)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def patched_generate(model, src_prompt, flat_pos, demo_idx, donor_row, conditions,
                     targets, max_new):
    """conditions: list of dict point -> list[layer] to patch (donor content) for that row.
       donor_row:  dict point -> dict layer -> (n_demos, d_model)."""
    device = next(model.parameters()).device
    n_cond = len(conditions)
    base = model.to_tokens(src_prompt, prepend_bos=True)
    cur = base.repeat(n_cond, 1)
    flat_max = int(flat_pos.max())

    # (point, layer) -> list of rows that patch it
    pl_rows = defaultdict(list)
    for b, cond in enumerate(conditions):
        for pt, layers in cond.items():
            for L in layers:
                pl_rows[(pt, L)].append(b)

    def mk_hook(pt, L, rows):
        dv = donor_row[pt][L][demo_idx]            # (n_flat, d_model)
        def hook(v, hook):
            if v.shape[1] <= flat_max:             # prefill only
                return v
            for b in rows:
                v[b, flat_pos, :] = dv.to(v.dtype)
            return v
        return hook

    hooks = [(f'blocks.{L}.{POINT_HOOK[pt]}', mk_hook(pt, L, rows))
             for (pt, L), rows in pl_rows.items()]

    kv = KVCache.init_cache(model.cfg, device, n_cond)
    gen = torch.zeros(n_cond, max_new, dtype=torch.long, device=device)
    with model.hooks(fwd_hooks=hooks):
        logits = model(cur, past_kv_cache=kv)[:, -1, :]
        for t in range(max_new):
            nxt = logits.argmax(dim=-1)
            gen[:, t] = nxt
            if t < max_new - 1:
                logits = model(nxt[:, None], past_kv_cache=kv)[:, -1, :]

    src_t, dnr_t, leak_t = targets
    maxlen = max(len(src_t), len(dnr_t), len(leak_t))
    gen = gen.tolist()
    out = []
    for b in range(n_cond):
        ids, label, dec = [], 'other', ''
        for tid in gen[b]:
            ids.append(tid)
            dec = model.tokenizer.decode(ids).strip()
            if dec == src_t:
                label = 'acc_src'; break
            if dec == dnr_t:
                label = 'acc_donor'; break
            if dec == leak_t:
                label = 'acc_leak'; break
            if len(dec) >= maxlen or (dec and not any(
                    t.startswith(dec) for t in (src_t, dnr_t, leak_t))):
                break
        out.append((label, dec))
    return out


def build_modes(n_layers):
    """Return dict mode_name -> list of (cond_label, condition-dict)."""
    allL = list(range(n_layers))
    modes = {}
    modes['resid_layer'] = [(f'L{L}', {'resid': [L]}) for L in allL]
    modes['resid_cumul'] = [(f'L0-{L}', {'resid': list(range(L + 1))}) for L in allL]
    modes['attn_layer'] = [(f'L{L}', {'attn': [L]}) for L in allL]
    modes['mlp_layer'] = [(f'L{L}', {'mlp': [L]}) for L in allL]
    modes['components'] = [
        ('full_resid', {'resid': allL}),
        ('attn_all', {'attn': allL}),
        ('mlp_all', {'mlp': allL}),
        ('attn+mlp_all', {'attn': allL, 'mlp': allL}),
    ]
    return modes


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', default='nonce+arithmetic')
    ap.add_argument('--cuda', default='0')
    ap.add_argument('--families', default='arithmetic',
                    help='comma list; arithmetic is where transfer exists')
    ap.add_argument('--n-prompts', type=int, default=15)
    ap.add_argument('--max-pairs', type=int, default=0)
    ap.add_argument('--modes', default='components,resid_layer,mlp_layer,attn_layer,resid_cumul')
    ap.add_argument('--pilot', action='store_true')
    ap.add_argument('--max-new', type=int, default=10)
    ap.add_argument('--out-name', default='mechanism_decomp')
    ap.add_argument('--patch-pos', default='mean_all',
                    choices=['mean_all', 'last_all', 'last_last'],
                    help='mean_all: mean-over-span -> all output positions (original); '
                         'last_all: last output token -> all positions (clean extraction); '
                         'last_last: last output token -> last output position only (per-demo)')
    args = ap.parse_args()
    if args.pilot:
        args.max_pairs = 2; args.n_prompts = 6; args.families = 'arithmetic'
    EXTRACT = 'last' if args.patch_pos.startswith('last') else 'mean'
    WRITE = 'last' if args.patch_pos.endswith('last') else 'all'
    print(f'[patch-pos] {args.patch_pos}  (extract={EXTRACT}, write={WRITE})')

    t0 = time.time()
    print(f'[load] {X.MODEL_NAME}')
    model = load_model(args.cuda)
    device = next(model.parameters()).device
    splits = load_dataset(args.dataset)
    n_layers = model.cfg.n_layers

    all_modes = build_modes(n_layers)
    modes = {m: all_modes[m] for m in args.modes.split(',') if m in all_modes}
    points = {'resid', 'attn', 'mlp'}     # cache all; cheap relative to generation

    fam_tasks = {'nonce': [t for t in NONCE_TASKS if t in splits],
                 'arithmetic': [t for t in ARITH_TASKS if t in splits]}
    families = [f for f in args.families.split(',') if fam_tasks.get(f)]
    N = args.n_prompts

    def tlen(s):
        return int(model.to_tokens(s, prepend_bos=False).shape[1])

    rows = []
    src_pos_cache = {}
    for fam in families:
        tasks = fam_tasks[fam]
        pairs = list(itertools.permutations(tasks, 2))
        if args.max_pairs:
            pairs = pairs[:args.max_pairs]
        by_donor = defaultdict(list)
        for s, dn in pairs:
            by_donor[dn].append(s)

        for donor_task, src_tasks in tqdm(by_donor.items(), desc=f'{fam} donors'):
            donor_prompts = splits[donor_task]['icl_prompts'][:N]
            donor_cache, ok = build_donor_cache(model, donor_prompts, points, device, extract=EXTRACT)

            for src_task in src_tasks:
                src_prompts = splits[src_task]['icl_prompts'][:N]
                for j in range(min(N, len(src_prompts), len(donor_prompts))):
                    if not ok[j]:
                        continue
                    sp = src_prompts[j]
                    key = (src_task, j)
                    if key not in src_pos_cache:
                        s_out = per_demo_output_positions(model, sp)
                        if s_out is None:
                            src_pos_cache[key] = None
                        else:
                            flat, didx = [], []
                            for i, ps in enumerate(s_out):
                                if WRITE == 'last':
                                    flat.append(ps[-1]); didx.append(i)
                                else:
                                    flat.extend(ps); didx.extend([i] * len(ps))
                            src_pos_cache[key] = (torch.tensor(flat, device=device),
                                                  torch.tensor(didx, device=device))
                    if src_pos_cache[key] is None:
                        continue
                    flat_pos, demo_idx = src_pos_cache[key]
                    qi = sp['query_input']
                    src_tgt = str(sp['query_output'])
                    dnr_tgt = apply_rule(donor_task, qi)
                    leak_tgt = str(donor_prompts[j]['query_output'])
                    targets = (src_tgt, dnr_tgt, leak_tgt)
                    mn = min(args.max_new, max(tlen(src_tgt), tlen(dnr_tgt), tlen(leak_tgt)) + 2)
                    donor_row = {pt: {L: donor_cache[pt][L][j] for L in range(n_layers)}
                                 for pt in points}

                    try:
                        for mode, conds in modes.items():
                            labels = [lab for lab, _ in conds]
                            res = patched_generate(model, sp['prompt'], flat_pos, demo_idx,
                                                   donor_row, [c for _, c in conds], targets, mn)
                            for lab, (binl, dec) in zip(labels, res):
                                rows.append({'family': fam, 'mode': mode, 'cond': lab,
                                             'src': src_task, 'donor': donor_task,
                                             'pair': f'{src_task}->{donor_task}', 'prompt_idx': j,
                                             'bin': binl,
                                             'acc_src': int(binl == 'acc_src'),
                                             'acc_donor': int(binl == 'acc_donor'),
                                             'acc_leak': int(binl == 'acc_leak'),
                                             'other': int(binl == 'other'),
                                             'src_tgt': src_tgt, 'dnr_tgt': dnr_tgt,
                                             'leak_tgt': leak_tgt, 'decoded': dec})
                    except Exception as ex:
                        print(f'[skip] {src_task}->{donor_task} j={j}: {type(ex).__name__}: {ex}',
                              flush=True)
                        torch.cuda.empty_cache()
            del donor_cache
            torch.cuda.empty_cache()
            import pandas as pd
            pd.DataFrame(rows).to_csv(INTERPLAY_RESULTS / f'{args.out_name}__rows.csv', index=False)
            print(f'[checkpoint] donor={donor_task} done, {len(rows)} rows saved', flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    INTERPLAY_RESULTS.mkdir(parents=True, exist_ok=True)
    with open(INTERPLAY_RESULTS / f'{args.out_name}.pkl', 'wb') as f:
        pickle.dump({'rows': df, 'args': vars(args)}, f)
    df.to_csv(INTERPLAY_RESULTS / f'{args.out_name}__rows.csv', index=False)

    if len(df):
        summ = (df.groupby(['family', 'mode', 'cond'])
                  [['acc_src', 'acc_donor', 'acc_leak', 'other']].mean().reset_index())
        summ.to_csv(INTERPLAY_RESULTS / f'{args.out_name}__summary.csv', index=False)
        # headline: components + the best transferring layer in each per-layer sweep
        with pd.option_context('display.width', 200, 'display.max_rows', 60):
            comp = summ[summ['mode'] == 'components']
            if len(comp):
                print('\n=== components (mean acc over pairs*prompts) ===')
                print(comp[['family', 'cond', 'acc_src', 'acc_donor', 'acc_leak', 'other']].to_string(index=False))
            for m in ['resid_layer', 'mlp_layer', 'attn_layer']:
                sub = summ[summ['mode'] == m]
                if len(sub):
                    top = sub.sort_values('acc_donor', ascending=False).head(5)
                    print(f'\n=== {m}: top-5 layers by acc_donor ===')
                    print(top[['cond', 'acc_donor', 'acc_src', 'other']].to_string(index=False))
    print(f'\n[done] {len(df)} rows, {time.time()-t0:.0f}s')
    print(f'[saved] {INTERPLAY_RESULTS / (args.out_name + ".pkl")}')


if __name__ == '__main__':
    main()
