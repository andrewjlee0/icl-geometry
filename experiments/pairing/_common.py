"""Shared helpers for the systematic experiment scripts (01-04).

Keeps each script thin: model loading, CLI arg parsing, result/figure IO, and the
task-vector patching primitives that scripts 02 and 04 share.

Run any script as e.g.:
    python 01_corruption_icl_accuracy.py --dataset hendel
    python 04_ablation_tv_patching.py --dataset real --head-set pairing --corrupt input
"""
import os
import sys
import pickle
import argparse
from pathlib import Path


def _find_repo_root(start):
    """Walk up until we find the dir holding utils/ and configs/ (repo root)."""
    p = Path(start).resolve()
    for parent in [p, *p.parents]:
        if (parent / 'utils').is_dir() and (parent / 'configs').is_dir():
            return parent
    return p.parents[2]  # fallback: experiments/<folder>/_common.py -> repo root


REPO_ROOT = _find_repo_root(__file__)
sys.path.insert(0, str(REPO_ROOT))  # repo root on path for `from utils...`, `from configs...`

import numpy as np
import torch

# results/ lives at repo root, a sibling of experiments/
RESULTS_DIR = REPO_ROOT / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# model + args
# --------------------------------------------------------------------------- #
def load_model(device='cuda', cuda_visible=None):
    if cuda_visible is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_visible)
    from transformer_lens import HookedTransformer
    from configs import MODEL_NAME
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=device, dtype=torch.float16)
    model.eval()
    return model


def base_parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--dataset', choices=['hendel', 'nonce+arithmetic'], default='hendel')
    p.add_argument('--cuda', default='0', help='CUDA_VISIBLE_DEVICES value')
    p.add_argument('--n-prompts', type=int, default=20,
                   help='max ICL prompts per task')
    p.add_argument('--corrupt-mode', choices=['shuffle', 'random', 'star'],
                   default='shuffle')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--head-pct', type=int, default=10,
                   help='top-percent head selection for pairing/aggregation')
    return p


def save_results(name, payload, csv=True):
    """Pickle the full payload; also dump every DataFrame value to its own CSV.

    The .pkl holds everything (DataFrames, arrays, head lists, args). When csv=True
    each pandas DataFrame in the top level of `payload` is additionally written as
    `<name>__<key>.csv`, so plotting tools can read them directly.
    """
    out = RESULTS_DIR / f'{name}.pkl'
    with open(out, 'wb') as f:
        pickle.dump(payload, f)
    print(f'[saved] {out}')
    if csv:
        import pandas as pd
        for key, val in payload.items():
            if isinstance(val, pd.DataFrame):
                cpath = RESULTS_DIR / f'{name}__{key}.csv'
                val.to_csv(cpath)
                print(f'[saved] {cpath}')
            elif isinstance(val, pd.Series):
                cpath = RESULTS_DIR / f'{name}__{key}.csv'
                val.to_frame().to_csv(cpath)
                print(f'[saved] {cpath}')
    return out


def save_activations(name, arrays):
    """Save large activation arrays compactly via np.savez_compressed.

    `arrays` is a dict of str -> np.ndarray (e.g. task vectors, logits). Written to
    `<name>__activations.npz`. Kept separate from the .pkl so the behavioral results
    stay small and quick to load.
    """
    out = RESULTS_DIR / f'{name}__activations.npz'
    np.savez_compressed(out, **arrays)
    print(f'[saved] {out}  ({len(arrays)} arrays)')
    return out


def save_fig(fig, name):
    out = RESULTS_DIR / f'{name}.png'
    fig.savefig(out, dpi=130, bbox_inches='tight')
    print(f'[saved] {out}')
    return out


# --------------------------------------------------------------------------- #
# task-vector patching primitives (shared by 02 and 04)
# --------------------------------------------------------------------------- #
def query_arrow_position(model, prompt, sep=' →'):
    """Position of the LAST arrow token (the query arrow)."""
    arrow_tok = model.to_tokens(sep, prepend_bos=False)[0, 0].item()
    toks = model.to_tokens(prompt, prepend_bos=True)[0]
    pos = (toks == arrow_tok).nonzero().squeeze(-1).tolist()
    return pos[-1] if pos else toks.shape[0] - 1


@torch.no_grad()
def extract_tv_all_layers(model, prompt, arrow_pos, fwd_hooks=None):
    """resid_post at arrow_pos for every layer, optionally under ablation hooks.

    Returns dict layer -> np.ndarray (d_model,). When fwd_hooks is given the TV is
    extracted from a forward pass with those heads ablated (used by script 04).
    """
    rfilter = lambda n: 'resid_post' in n
    toks = model.to_tokens(prompt, prepend_bos=True)
    if fwd_hooks:
        with model.hooks(fwd_hooks=fwd_hooks):
            _, cache = model.run_with_cache(toks, names_filter=rfilter)
    else:
        _, cache = model.run_with_cache(toks, names_filter=rfilter)
    out = {L: cache['resid_post', L][0, arrow_pos].cpu().float().numpy()
           for L in range(model.cfg.n_layers)}
    del cache
    torch.cuda.empty_cache()
    return out


@torch.no_grad()
def patch_and_score(model, zs_prompt, tv_vec, layer, answer, return_logits=False,
                    topk=10):
    """Replace last-token resid_post at `layer` with tv_vec; return (correct, rank).

    correct = top-1 next token matches the answer's first content token.
    rank    = rank of that target token in the patched logits (0 = top).

    If return_logits=True, returns (correct, rank, info) where info is a dict with
    the target id/logit and the top-k (token_id, str, logit) predictions — useful
    for inspecting what the model predicted when patching failed.
    """
    device = next(model.parameters()).device
    toks = model.to_tokens(zs_prompt, prepend_bos=True)
    theta = torch.tensor(tv_vec, device=device, dtype=model.cfg.dtype)

    def hook(value, hook):
        value[0, -1, :] = theta
        return value

    logits = model.run_with_hooks(
        toks, fwd_hooks=[(f'blocks.{layer}.hook_resid_post', hook)])[0, -1]

    # target = first non-space token of " answer"
    target_toks = model.to_tokens(' ' + str(answer).strip(), prepend_bos=False)[0]
    space_id = model.to_tokens(' ', prepend_bos=False)[0, 0].item()
    target = next((t.item() for t in target_toks if t.item() != space_id),
                  target_toks[0].item())
    rank = int((logits > logits[target]).sum().item())
    correct = int(logits.argmax().item() == target)

    if not return_logits:
        return correct, rank

    tv, ti = torch.topk(logits, min(topk, logits.shape[-1]))
    info = {
        'target_id': target,
        'target_str': model.tokenizer.decode([target]),
        'target_logit': float(logits[target]),
        'topk': [(int(i), model.tokenizer.decode([int(i)]), float(v))
                 for v, i in zip(tv.tolist(), ti.tolist())],
    }
    return correct, rank, info


def held_out_query(splits, task, p_idx):
    """A query distinct from prompt p_idx's own query (no query reuse)."""
    ev = splits[task].get('eval_data') or []
    if ev:
        eq = ev[p_idx % len(ev)]
        return eq['query_input'], eq.get('query_output')
    prompts = splits[task]['icl_prompts']
    j = (p_idx + 1) % len(prompts)
    if j == p_idx:
        j = (p_idx + 2) % len(prompts)
    other = prompts[j]
    return other['query_input'], other.get('query_output')


# --------------------------------------------------------------------------- #
# corruption-condition sweep (shared by 01-04)
# --------------------------------------------------------------------------- #
# The condition axis: orig + {shuffle,random,mean,star} x {input,output}.
# 'orig' has no side; the four corruption modes each apply to input and output.
CORRUPTION_MODES = ['shuffle', 'random', 'mean', 'star']
CORRUPTION_SIDES = ['input', 'output']


def condition_list():
    """List of (cond_name, target, mode): the 9 conditions = orig + 4 modes x 2 sides."""
    conds = [('orig', 'none', 'orig')]
    for side in CORRUPTION_SIDES:
        for mode in CORRUPTION_MODES:
            conds.append((f'{mode}_{side}', side, mode))
    return conds


def make_pools(splits, model):
    """Build the global token pools once (for random/mean corruptions)."""
    from data.loaders import build_token_pools
    return build_token_pools(splits, model)


def build_corruption(pdata, target, mode, rng, pools, model):
    """corruption_plan wrapper with this folder's prompt builder + position finder."""
    from data.prompts import build_icl_prompt
    from utils.corruptions import corruption_plan
    from utils.positions import find_per_demo_positions_robust
    return corruption_plan(pdata, target=target, mode=mode, rng=rng,
                           build_prompt=build_icl_prompt, pools=pools, model=model,
                           find_positions=find_per_demo_positions_robust)


def load_scope_heads(dataset, head_pct, scope, head_set, results_dir=None):
    """Load the multiscope head cache and return (heads, rand_heads) for a scope.

    dataset: 'hendel' | 'nonce+arithmetic'; scope: 'pooled'|'nonce'|'arithmetic'|
    'category:<c>'|'task:<t>'. Requires score_heads.py to have been run first.
    """
    import pickle
    from utils.heads import select_scope
    rd = Path(results_dir) if results_dir else RESULTS_DIR
    ds = dataset.replace('+', '_')
    cache = rd / f'head_sets_{ds}_pct{head_pct}.pkl'
    if not cache.exists():
        raise FileNotFoundError(
            f'{cache} not found. Run `python score_heads.py --dataset {dataset}` first.')
    with open(cache, 'rb') as f:
        ms = pickle.load(f)
    entry = select_scope(ms, scope)
    return entry[head_set], entry[f'{head_set}_rand'], ms
