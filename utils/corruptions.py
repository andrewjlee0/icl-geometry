"""Demonstration corruptions.

A "corruption" breaks the input->output pairing in the demonstrations from one
side while leaving the other side intact. There are five severity levels, forming
a monotone ladder from least to most destructive of the corrupted side's info:

  orig     no corruption (baseline)
  shuffle  derange the corrupted side across demos (pairing broken, tokens kept)
  random   replace each corrupted item with a random draw from the GLOBAL pool
  mean     replace each corrupted item's embedding with the GLOBAL mean embedding
           of that pool (the 'average' of random; an embed-layer hook, not text)
  star     replace each corrupted item with a constant '*' placeholder

target='input' corrupts the demo inputs; target='output' corrupts the outputs.

'random' and 'mean' draw on the same global token pool (see data.loaders.
build_token_pools), so 'mean' is literally the mean of what 'random' samples —
giving a clean stepwise decrease across conditions.
"""
import random

MODES = ['orig', 'shuffle', 'random', 'mean', 'star']
STAR_TOKEN = '*'


def _derangement(items, rng, max_tries=100):
    """Return a permutation of items with no element in its original slot."""
    n = len(items)
    if n < 2:
        return list(items)
    idx = list(range(n))
    for _ in range(max_tries):
        perm = idx[:]
        rng.shuffle(perm)
        if all(perm[i] != i for i in range(n)):
            return [items[p] for p in perm]
    # fallback: cyclic shift guarantees no fixed point
    return [items[(i + 1) % n] for i in range(n)]


def corrupt_demos(demo_pairs, target='input', mode='shuffle', rng=None,
                  input_pool=None, output_pool=None, star_token=STAR_TOKEN):
    """Return a TEXT-corrupted copy of demo_pairs (modes: shuffle/random/star).

    'mean' is not handled here because it is applied at the embedding layer, not
    as text; for mean we place a star-like placeholder so positions are findable
    and overwrite their embeddings via a hook (see corruption_plan). 'orig'
    returns demo_pairs unchanged.
    """
    if rng is None:
        rng = random.Random()
    if target not in ('input', 'output'):
        raise ValueError(f"target must be 'input'|'output', got {target!r}")
    if mode in ('orig', 'none'):
        return [tuple(p) for p in demo_pairs]

    inputs = [i for i, _ in demo_pairs]
    outputs = [o for _, o in demo_pairs]
    side = inputs if target == 'input' else outputs

    if mode == 'shuffle':
        new_side = _derangement(side, rng)
    elif mode == 'random':
        pool = input_pool if target == 'input' else output_pool
        if not pool:
            raise ValueError("mode='random' requires the corresponding pool")
        new_side = [rng.choice(pool) for _ in side]
    elif mode in ('star', 'mean'):
        # mean uses the same placeholder text; its embedding is overwritten by hook
        new_side = [star_token for _ in side]
    else:
        raise ValueError(f"unknown mode {mode!r}")

    if target == 'input':
        return list(zip(new_side, outputs))
    return list(zip(inputs, new_side))


def _mean_embed_hook(positions, mean_vec):
    """hook_embed hook: overwrite `positions` with mean_vec (the global mean embed)."""
    def hook_fn(emb, hook, _pos=list(positions), _v=mean_vec):
        for p in _pos:
            emb[:, p, :] = _v.to(emb.dtype)
        return emb
    return ('hook_embed', hook_fn)


def corruption_plan(pdata, target, mode='shuffle', rng=None, build_prompt=None,
                    pools=None, model=None, find_positions=None):
    """Unified corruption -> (prompt_dict, fwd_hooks).

    Returns a (possibly corrupted) copy of pdata with its 'prompt' rebuilt, plus a
    list of forward hooks to apply during the model call. For all text modes the
    hook list is empty; for mode='mean' it contains a single hook_embed hook that
    overwrites the corrupted side's token positions with the global mean embedding.

    Args:
        target: 'none' | 'input' | 'output'
        mode:   one of MODES ('orig' is treated as no-op like target='none')
        pools:  dict from build_token_pools (required for 'random' and 'mean')
        model, find_positions: required for 'mean' to locate corrupted positions;
            find_positions(model, prompt, demos) -> per-demo dicts with
            'input_positions'/'output_positions' (pass utils.positions.find_per_demo_positions_robust)
    """
    out = dict(pdata)
    if target in (None, 'none') or mode in ('orig', 'none'):
        return out, []

    kwargs = {}
    if mode == 'random':
        if not pools:
            raise ValueError("mode='random' requires pools=build_token_pools(...)")
        kwargs['input_pool'] = pools['input_words']
        kwargs['output_pool'] = pools['output_words']

    new_demos = corrupt_demos(pdata['demo_pairs'], target=target, mode=mode,
                              rng=rng, **kwargs)
    out['demo_pairs'] = new_demos
    if build_prompt is not None:
        out['prompt'] = build_prompt(new_demos, pdata['query_input'])

    hooks = []
    if mode == 'mean':
        if not (pools and model and find_positions):
            raise ValueError("mode='mean' requires pools, model, and find_positions")
        per_demo = find_positions(model, out['prompt'], new_demos)
        key = 'input_positions' if target == 'input' else 'output_positions'
        positions = [p for d in per_demo for p in d.get(key, [])]
        mean_vec = pools['mean_input_embed'] if target == 'input' else pools['mean_output_embed']
        if positions:
            hooks.append(_mean_embed_hook(positions, mean_vec))
    return out, hooks


def apply_corruption(pdata, target, mode='shuffle', rng=None, build_prompt=None,
                     **kwargs):
    """Back-compat text-only wrapper: returns just the corrupted prompt dict.

    For 'mean' use corruption_plan (it needs an embed hook). target='none' returns
    an unchanged shallow copy.
    """
    out = dict(pdata)
    if target in (None, 'none') or mode in ('orig', 'none'):
        return out
    new_demos = corrupt_demos(pdata['demo_pairs'], target=target, mode=mode,
                              rng=rng, **kwargs)
    out['demo_pairs'] = new_demos
    if build_prompt is not None:
        out['prompt'] = build_prompt(new_demos, pdata['query_input'])
    return out
