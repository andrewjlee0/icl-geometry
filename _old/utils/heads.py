"""Head populations for the pairing mechanism.

Two head scores, both defined exactly as in the repo's earlier head-scoring
notebooks so results are consistent:

  PAIRING heads: attention FROM demo output positions TO within-demo input
    positions, minus a baseline (attention from any non-output query position to
    inputs). Heads that route output->input score high.
    (Previously called "OI" / output->input heads.)

  AGGREGATION heads: attention FROM the final position TO output positions,
    minus a baseline (final position -> input positions). Heads that pull the
    demo outputs into the query score high.
    (Previously called "OA" / output-attention heads.)

get_head_sets(model, splits, ...) computes both, selects the top percentile,
caches to results/, and returns the (layer, head) lists plus a matched random
control set for each.
"""
import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

from .positions import find_per_demo_positions_robust

PATTERN_FILTER = lambda name: 'attn.hook_pattern' in name


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
@torch.no_grad()
def score_pairing_heads(model, prompts_data, desc='Pairing scoring'):
    """Output->input attention score per head, baseline-subtracted. (n_layers, n_heads)."""
    from tqdm import tqdm
    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
    num_sum = np.zeros((n_layers, n_heads), dtype=np.float64)
    base_sum = np.zeros((n_layers, n_heads), dtype=np.float64)
    cnt = 0
    for pdata in tqdm(prompts_data, desc=desc, leave=False):
        demos = pdata['demo_pairs']
        tokens = model.to_tokens(pdata['prompt'], prepend_bos=True)
        seq_len = tokens.shape[1]
        per_demo = find_per_demo_positions_robust(model, pdata['prompt'], demos)
        all_in, all_out = set(), set()
        for d in per_demo:
            all_in.update(d.get('input_positions', []))
            all_out.update(d.get('output_positions', []))
        num_mask = torch.zeros(seq_len, seq_len)
        for d in per_demo:
            for op in d.get('output_positions', []):
                for ip in d.get('input_positions', []):
                    num_mask[op, ip] = 1.0
        base_mask = torch.zeros(seq_len, seq_len)
        ins_list = sorted(all_in)
        for q in range(seq_len):
            if q in all_out:
                continue
            for ik in ins_list:
                if ik <= q:
                    base_mask[q, ik] = 1.0
        n_num = int(num_mask.sum().item())
        n_base = int(base_mask.sum().item())
        if n_num == 0 or n_base == 0:
            continue
        _, cache = model.run_with_cache(tokens, names_filter=PATTERN_FILTER)
        for L in range(n_layers):
            patt = cache['pattern', L][0].cpu().float()
            num_sum[L] += (patt * num_mask).sum(dim=(1, 2)).numpy() / n_num
            base_sum[L] += (patt * base_mask).sum(dim=(1, 2)).numpy() / n_base
        cnt += 1
        del cache
        torch.cuda.empty_cache()
    return (num_sum - base_sum) / max(cnt, 1)


@torch.no_grad()
def score_aggregation_heads(model, prompts_data, desc='Aggregation scoring'):
    """Final-position->output attention score per head, baseline-subtracted."""
    from tqdm import tqdm
    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
    num_sum = np.zeros((n_layers, n_heads), dtype=np.float64)
    base_sum = np.zeros((n_layers, n_heads), dtype=np.float64)
    cnt = 0
    for pdata in tqdm(prompts_data, desc=desc, leave=False):
        demos = pdata['demo_pairs']
        tokens = model.to_tokens(pdata['prompt'], prepend_bos=True)
        seq_len = tokens.shape[1]
        last_pos = seq_len - 1
        per_demo = find_per_demo_positions_robust(model, pdata['prompt'], demos)
        all_in, all_out = set(), set()
        for d in per_demo:
            all_in.update(d.get('input_positions', []))
            all_out.update(d.get('output_positions', []))
        out_idx, in_idx = sorted(all_out), sorted(all_in)
        if not out_idx or not in_idx:
            continue
        _, cache = model.run_with_cache(tokens, names_filter=PATTERN_FILTER)
        for L in range(n_layers):
            patt = cache['pattern', L][0].cpu().float()
            row = patt[:, last_pos, :]
            num_sum[L] += row[:, out_idx].sum(dim=1).numpy() / len(out_idx)
            base_sum[L] += row[:, in_idx].sum(dim=1).numpy() / len(in_idx)
        cnt += 1
        del cache
        torch.cuda.empty_cache()
    return (num_sum - base_sum) / max(cnt, 1)


# --------------------------------------------------------------------------- #
# selection + caching
# --------------------------------------------------------------------------- #
def select_top_pct(scores, pct=10):
    """Top `pct` percent of heads by score. Returns sorted list of (layer, head)."""
    n_layers, n_heads = scores.shape
    n_keep = max(1, int(round(n_layers * n_heads * pct / 100)))
    flat = sorted(((L, h, scores[L, h]) for L in range(n_layers) for h in range(n_heads)),
                  key=lambda x: -x[2])
    return [(L, h) for L, h, _ in flat[:n_keep]]


def matched_random_heads(top_heads, n_layers, n_heads, seed=0):
    """Random control: same count, same per-layer distribution as top_heads."""
    rng = np.random.RandomState(seed)
    by_layer = defaultdict(int)
    for L, _ in top_heads:
        by_layer[L] += 1
    chosen = []
    top_set = set(top_heads)
    for L, k in by_layer.items():
        avail = [h for h in range(n_heads) if (L, h) not in top_set]
        rng.shuffle(avail)
        chosen.extend((L, h) for h in avail[:k])
    return chosen


def get_head_sets(model, prompts_data, pct=10, cache_path=None, recompute=False,
                  rand_seed=0):
    """Compute/load pairing and aggregation head sets for a dataset.

    Returns dict:
      {'pairing': [...], 'aggregation': [...],
       'pairing_rand': [...], 'aggregation_rand': [...],
       'pairing_score': ndarray, 'aggregation_score': ndarray, 'pct': int}
    """
    cache_path = Path(cache_path) if cache_path else None
    if cache_path and cache_path.exists() and not recompute:
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
    pairing_score = score_pairing_heads(model, prompts_data)
    aggregation_score = score_aggregation_heads(model, prompts_data)
    pairing = select_top_pct(pairing_score, pct)
    aggregation = select_top_pct(aggregation_score, pct)
    out = {
        'pairing': pairing,
        'aggregation': aggregation,
        'pairing_rand': matched_random_heads(pairing, n_layers, n_heads, rand_seed),
        'aggregation_rand': matched_random_heads(aggregation, n_layers, n_heads, rand_seed + 1),
        'pairing_score': pairing_score,
        'aggregation_score': aggregation_score,
        'pct': pct,
    }
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(out, f)
    return out


# --------------------------------------------------------------------------- #
# ablation hooks
# --------------------------------------------------------------------------- #
def make_ablation_hooks(heads):
    """Zero-ablate the given (layer, head) list across ALL positions."""
    by_layer = defaultdict(list)
    for L, h in heads:
        by_layer[L].append(h)
    fwd_hooks = []
    for L, h_list in by_layer.items():
        def hook(z, hook, _h_list=h_list):
            for h in _h_list:
                z[0, :, h, :] = 0
            return z
        fwd_hooks.append((f'blocks.{L}.attn.hook_z', hook))
    return fwd_hooks


def make_amplify_hooks(heads, scale):
    """Multiply the given (layer, head) outputs by `scale` at ALL positions.

    Used by the bridge experiment to amplify pairing heads' contribution.
    """
    by_layer = defaultdict(list)
    for L, h in heads:
        by_layer[L].append(h)
    fwd_hooks = []
    for L, h_list in by_layer.items():
        def hook(z, hook, _h_list=h_list, _s=scale):
            for h in _h_list:
                z[0, :, h, :] *= _s
            return z
        fwd_hooks.append((f'blocks.{L}.attn.hook_z', hook))
    return fwd_hooks
