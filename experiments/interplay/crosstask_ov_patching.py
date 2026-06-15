"""Cross-task patching at demonstration-output positions: is the per-demo task code transplantable?

Question
--------
Take a DONOR task's content at the demonstration output positions and splice it into a
SOURCE prompt (demo i -> demo i). Does the model stop doing the source task and start doing
the DONOR task -- on the SOURCE's own query input? And if so, WHICH component carries it:
the pairing heads' OV writes, attention writes in general, or the full residual stream?

This is the "do the ingredients transfer" experiment, complementing the final-position
(aggregated-code) patching in crosstask_patch_allpairs.ipynb.

Conditions (one batched generation per (source prompt, donor prompt))
---------------------------------------------------------------------
  pairing / aggregation        : overwrite ONLY those heads' z (== their OV write) at the
                                 source demo-output positions with the donor's.
  pairing_rand / aggregation_rand : matched random-head controls (same per-layer counts,
                                 excluding the other functional population).
  all_heads                    : overwrite ALL heads' z at those positions (attention ceiling).
  full_resid                   : overwrite the FULL residual stream at those positions
                                 (absolute ceiling -- is ANY transfer achievable here?).
The ceilings are what make a pairing/aggregation NULL interpretable: full_resid says whether
the task code is present-and-transplantable at all; pairing vs all_heads vs full_resid says
which component carries it.

Position rule (cross-task alignment)
------------------------------------
Source and donor outputs differ in token length, so per-token alignment is impossible. We use
mean-over-span: for each donor demo i, average the donor's content over that demo's output
tokens -> one vector; write it into EVERY output token of source demo i. (Validated: full_resid
under this rule installs the donor arithmetic answer 10/10; pairing-head z does not.) The query
position is never patched. The patch is re-applied at the same fixed positions every greedy step.

Readout (exact full-string match, priority src > donor > leak)
  acc_src   : output == source_rule(source_query_input)  -> patch inert, model still does its task
  acc_donor : output == donor_rule (source_query_input)  -> TRANSFER: donor task installed on source
  acc_leak  : output == donor prompt's own stored answer -> copied content, not a rule
  other     : none -> broken

Results -> experiments/interplay/results/  (separate from the repo-root results/).

Run:
  python crosstask_ov_patching.py --cuda 0            # full overnight run
  python crosstask_ov_patching.py --cuda 0 --pilot    # 3 pairs/family smoke test
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

import numpy as np
import torch
from tqdm import tqdm

from configs import MODEL_NAME
from data.loaders import load_dataset
from data.tasks import _NONCE_TRANSFORMS, _ARITH_SPECS, NONCE_TASKS, ARITH_TASKS
from utils.positions import find_per_demo_positions_robust
from utils.heads import select_scope
from transformer_lens import TransformerLensKeyValueCache as HookedTransformerKeyValueCache

INTERPLAY_RESULTS = REPO_ROOT / 'experiments' / 'interplay' / 'results'
INTERPLAY_RESULTS.mkdir(parents=True, exist_ok=True)
ROOT_RESULTS = REPO_ROOT / 'results'

HEAD_CONDITIONS = ('pairing', 'aggregation', 'pairing_rand', 'aggregation_rand')
CEILING_CONDITIONS = ('all_heads', 'full_resid')


def apply_rule(task, query_input):
    if task in _NONCE_TRANSFORMS:
        return _NONCE_TRANSFORMS[task](str(query_input))
    fn, _lo, _hi = _ARITH_SPECS[task]
    return str(fn(int(query_input)))


def load_model(cuda_visible):
    if cuda_visible is not None and 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_visible)
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained(MODEL_NAME, device='cuda', dtype=torch.float16)
    model.eval()
    return model


def per_demo_output_positions(model, pdata):
    """List (len n_demos) of lists of token positions = ALL output tokens of each demo.

    Returns None if any demo's output span is unresolved (skip that prompt)."""
    per_demo = find_per_demo_positions_robust(model, pdata['prompt'], pdata['demo_pairs'])
    if len(per_demo) != len(pdata['demo_pairs']):
        return None
    out = []
    for d in per_demo:
        ops = d.get('output_positions', [])
        if not ops:
            return None
        out.append(ops)
    return out


# --------------------------------------------------------------------------- #
# donor cache: mean-over-output-span content per demo, per prompt
# --------------------------------------------------------------------------- #
@torch.no_grad()
def build_donor_cache(model, prompts, z_layers, need_resid, device):
    """Returns (zc, rc, ok):
      zc: dict L -> (n_prompts, n_demos, n_heads, d_head) fp16   (mean z over each demo span)
      rc: dict L -> (n_prompts, n_demos, d_model)        fp16   (mean resid over span) or None
      ok: list[bool] per prompt
    """
    n_heads, d_head = model.cfg.n_heads, model.cfg.d_head
    d_model = model.cfg.d_model
    n_layers = model.cfg.n_layers
    n_demos = len(prompts[0]['demo_pairs'])
    np_ = len(prompts)
    zc = {L: torch.zeros(np_, n_demos, n_heads, d_head, dtype=torch.float16, device=device)
          for L in z_layers}
    rc = ({L: torch.zeros(np_, n_demos, d_model, dtype=torch.float16, device=device)
           for L in range(n_layers)} if need_resid else None)

    def zfilter(n):
        return ('attn.hook_z' in n) or (need_resid and n.endswith('hook_resid_post'))

    ok = []
    for j, p in enumerate(prompts):
        pos = per_demo_output_positions(model, p)
        if pos is None:
            ok.append(False)
            continue
        ok.append(True)
        idx = [torch.tensor(ps, device=device) for ps in pos]
        toks = model.to_tokens(p['prompt'], prepend_bos=True)
        _, c = model.run_with_cache(toks, names_filter=zfilter)
        for L in z_layers:
            z = c[f'blocks.{L}.attn.hook_z'][0]                      # (seq, n_heads, d_head)
            zc[L][j] = torch.stack([z[ix].mean(0) for ix in idx]).to(torch.float16)
        if need_resid:
            for L in range(n_layers):
                r = c[f'blocks.{L}.hook_resid_post'][0]              # (seq, d_model)
                rc[L][j] = torch.stack([r[ix].mean(0) for ix in idx]).to(torch.float16)
        del c
        torch.cuda.empty_cache()
    return zc, rc, ok


# --------------------------------------------------------------------------- #
# batched patched generation across conditions
# --------------------------------------------------------------------------- #
@torch.no_grad()
def patched_generate(model, src_prompt, flat_pos, demo_idx, donor_zrow, donor_rrow,
                     cond_specs, targets, max_new):
    """KV-cached, prefill-only patched greedy generation across conditions (batch rows).

    The patch lands during prefill at the fixed source-output positions; the KV cache then
    carries it forward, so re-applying every step is unnecessary and the output is identical
    to the full-re-forward version (greedy). Tokens are collected on-GPU and classified once
    at the end -- no per-step host sync. Returns list over conditions of (bin_label, decoded)."""
    device = next(model.parameters()).device
    n_cond = len(cond_specs)
    base = model.to_tokens(src_prompt, prepend_bos=True)
    cur = base.repeat(n_cond, 1)
    flat_max = int(flat_pos.max())
    n_layers = model.cfg.n_layers
    all_heads = torch.arange(model.cfg.n_heads, device=device)

    z_rows = defaultdict(list)        # L -> list of (b, head_tensor)
    resid_rows = []
    for b, spec in enumerate(cond_specs):
        if spec['kind'] == 'resid':
            resid_rows.append(b)
        elif spec['kind'] == 'all_heads':
            for L in range(n_layers):
                z_rows[L].append((b, all_heads))
        else:
            for L, ht in spec['by_layer'].items():
                z_rows[L].append((b, ht))

    fp_col = flat_pos[:, None]
    def mk_zhook(L):
        rows = z_rows[L]
        dz = donor_zrow[L][demo_idx]               # (n_flat, n_heads, d_head)
        def hook(z, hook):
            if z.shape[1] <= flat_max:             # prefill only (steps are len-1)
                return z
            for b, ht in rows:
                z[b, fp_col, ht[None, :], :] = dz[:, ht, :].to(z.dtype)
            return z
        return hook

    def mk_rhook(L):
        dr = donor_rrow[L][demo_idx]               # (n_flat, d_model)
        def hook(v, hook):
            if v.shape[1] <= flat_max:
                return v
            for b in resid_rows:
                v[b, flat_pos, :] = dr.to(v.dtype)
            return v
        return hook

    hooks = [(f'blocks.{L}.attn.hook_z', mk_zhook(L)) for L in z_rows]
    if resid_rows:
        hooks += [(f'blocks.{L}.hook_resid_post', mk_rhook(L)) for L in range(n_layers)]

    kv = HookedTransformerKeyValueCache.init_cache(model.cfg, device, n_cond)
    gen = torch.zeros(n_cond, max_new, dtype=torch.long, device=device)
    with model.hooks(fwd_hooks=hooks):
        logits = model(cur, past_kv_cache=kv)[:, -1, :]        # prefill (patch lands)
        for t in range(max_new):
            nxt = logits.argmax(dim=-1)
            gen[:, t] = nxt
            if t < max_new - 1:
                logits = model(nxt[:, None], past_kv_cache=kv)[:, -1, :]

    src_t, dnr_t, leak_t = targets
    maxlen = max(len(src_t), len(dnr_t), len(leak_t))
    gen = gen.tolist()
    result = []
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
        result.append((label, dec))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', default='nonce+arithmetic')
    ap.add_argument('--cuda', default='0')
    ap.add_argument('--scope', default='pooled')
    ap.add_argument('--head-pct', type=int, default=10)
    ap.add_argument('--n-prompts', type=int, default=50)
    ap.add_argument('--families', default='nonce,arithmetic')
    ap.add_argument('--conditions',
                    default='pairing,aggregation,pairing_rand,aggregation_rand,all_heads,full_resid')
    ap.add_argument('--max-pairs', type=int, default=0, help='cap ordered pairs/family (0=all)')
    ap.add_argument('--pilot', action='store_true')
    ap.add_argument('--max-new', type=int, default=12)
    ap.add_argument('--out-name', default=None)
    args = ap.parse_args()
    if args.pilot:
        args.max_pairs = 3
        args.n_prompts = min(args.n_prompts, 10)

    t0 = time.time()
    print(f'[load] {MODEL_NAME}')
    model = load_model(args.cuda)
    device = next(model.parameters()).device
    splits = load_dataset(args.dataset)
    n_layers = model.cfg.n_layers

    ds = args.dataset.replace('+', '_')
    cache_path = ROOT_RESULTS / f'head_sets_{ds}_pct{args.head_pct}.pkl'
    if not cache_path.exists():
        raise FileNotFoundError(f'{cache_path} missing -- run score_heads.py first.')
    with open(cache_path, 'rb') as f:
        entry = select_scope(pickle.load(f), args.scope)

    conditions = args.conditions.split(',')
    # build per-condition spec
    cond_specs_template = []
    head_layers = set()
    for c in conditions:
        if c in HEAD_CONDITIONS:
            by_layer = defaultdict(list)
            for L, h in entry[c]:
                by_layer[L].append(h)
            ht = {L: torch.tensor(hs, device=device) for L, hs in by_layer.items()}
            head_layers.update(ht.keys())
            cond_specs_template.append({'kind': 'heads', 'by_layer': ht, 'name': c})
            print(f'[heads] {c:18s} {len(entry[c])} heads, layers {sorted(by_layer)}')
        elif c == 'all_heads':
            cond_specs_template.append({'kind': 'all_heads', 'by_layer': None, 'name': c})
        elif c == 'full_resid':
            cond_specs_template.append({'kind': 'resid', 'by_layer': None, 'name': c})
        else:
            raise ValueError(f'unknown condition {c}')

    need_all_heads = any(s['kind'] == 'all_heads' for s in cond_specs_template)
    need_resid = any(s['kind'] == 'resid' for s in cond_specs_template)
    z_layers = sorted(range(n_layers) if need_all_heads else head_layers)

    fam_tasks = {'nonce': [t for t in NONCE_TASKS if t in splits],
                 'arithmetic': [t for t in ARITH_TASKS if t in splits]}
    families = [f for f in args.families.split(',') if fam_tasks.get(f)]
    N = args.n_prompts

    rows = []
    n_skips = 0
    src_pos_cache = {}        # (src_task, j) -> (flat_pos, demo_idx) or None
    tlen_cache = {}           # target string -> token length

    def tok_len(s):
        if s not in tlen_cache:
            tlen_cache[s] = int(model.to_tokens(s, prepend_bos=False).shape[1])
        return tlen_cache[s]

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
            zc, rc, ok = build_donor_cache(model, donor_prompts, z_layers, need_resid, device)

            for src_task in src_tasks:
                src_prompts = splits[src_task]['icl_prompts'][:N]
                for j in range(min(N, len(src_prompts), len(donor_prompts))):
                    if not ok[j]:
                        n_skips += 1; continue
                    sp = src_prompts[j]
                    key = (src_task, j)
                    if key not in src_pos_cache:
                        s_out = per_demo_output_positions(model, sp)
                        if s_out is None:
                            src_pos_cache[key] = None
                        else:
                            flat, didx = [], []
                            for i, ps in enumerate(s_out):
                                flat.extend(ps); didx.extend([i] * len(ps))
                            src_pos_cache[key] = (torch.tensor(flat, device=device),
                                                  torch.tensor(didx, device=device))
                    if src_pos_cache[key] is None:
                        n_skips += 1; continue
                    flat_pos, demo_idx = src_pos_cache[key]
                    qi = sp['query_input']
                    src_tgt = str(sp['query_output'])
                    dnr_tgt = apply_rule(donor_task, qi)
                    leak_tgt = str(donor_prompts[j]['query_output'])
                    donor_zrow = {L: zc[L][j] for L in z_layers}
                    donor_rrow = {L: rc[L][j] for L in range(n_layers)} if need_resid else None
                    mn = min(args.max_new,
                             max(tok_len(src_tgt), tok_len(dnr_tgt), tok_len(leak_tgt)) + 2)

                    res = patched_generate(model, sp['prompt'], flat_pos, demo_idx,
                                           donor_zrow, donor_rrow, cond_specs_template,
                                           (src_tgt, dnr_tgt, leak_tgt), mn)
                    for spec, (label, dec) in zip(cond_specs_template, res):
                        rows.append({
                            'family': fam, 'src': src_task, 'donor': donor_task,
                            'pair': f'{src_task}->{donor_task}', 'prompt_idx': j,
                            'condition': spec['name'], 'bin': label,
                            'acc_src': int(label == 'acc_src'),
                            'acc_donor': int(label == 'acc_donor'),
                            'acc_leak': int(label == 'acc_leak'),
                            'other': int(label == 'other'),
                            'query_input': qi, 'src_tgt': src_tgt, 'dnr_tgt': dnr_tgt,
                            'leak_tgt': leak_tgt, 'decoded': dec,
                            'collision_src_donor': int(src_tgt == dnr_tgt)})
            del zc, rc
            torch.cuda.empty_cache()

    import pandas as pd
    df = pd.DataFrame(rows)
    name = args.out_name or f'crosstask_ov_{ds}_{args.scope.replace(":", "-")}'
    with open(INTERPLAY_RESULTS / f'{name}.pkl', 'wb') as f:
        pickle.dump({'rows': df, 'args': vars(args), 'conditions': conditions,
                     'n_skips': n_skips}, f)
    df.to_csv(INTERPLAY_RESULTS / f'{name}__rows.csv', index=False)

    if len(df):
        pair_means = (df.groupby(['family', 'condition', 'pair'])
                        [['acc_src', 'acc_donor', 'acc_leak', 'other']].mean().reset_index())
        summary = (pair_means.groupby(['family', 'condition'])
                     [['acc_src', 'acc_donor', 'acc_leak', 'other']]
                     .agg(['mean', 'sem']).round(4))
        pair_means.to_csv(INTERPLAY_RESULTS / f'{name}__pair_means.csv', index=False)
        summary.to_csv(INTERPLAY_RESULTS / f'{name}__summary.csv')
        print('\n=== family x condition (mean over pairs of per-pair accuracy) ===')
        with pd.option_context('display.width', 200):
            print(summary[[('acc_src', 'mean'), ('acc_donor', 'mean'),
                           ('acc_leak', 'mean'), ('other', 'mean')]])
    print(f'\n[done] {len(df)} rows, {n_skips} skips, {time.time()-t0:.0f}s')
    print(f'[saved] {INTERPLAY_RESULTS / (name + ".pkl")}')


if __name__ == '__main__':
    main()
