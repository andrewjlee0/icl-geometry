"""Unified dataset access so the pairing experiment scripts treat both datasets
identically. Both are LOAD-ONLY: the run scripts never generate data.

    load_dataset('hendel')           -> data/hendel_splits.pkl
                                        (built once by data/make_hendel_splits.py)
    load_dataset('nonce+arithmetic') -> data/nonce_arithmetic_splits.pkl
                                        (built once by data/make_nonce_arithmetic_splits.py)

Both return: {task_name: {'icl_prompts': [...], 'eval_data': [...]}}

Split file locations come from configs.defaults (SPLITS_DIR), so relocating the
splits is a one-line change there.
"""
import pickle
from pathlib import Path

from configs.defaults import HENDEL_SPLITS, NONCE_ARITH_SPLITS

_FILES = {
    'hendel': HENDEL_SPLITS,
    'nonce+arithmetic': NONCE_ARITH_SPLITS,
}
_MAKERS = {
    'hendel': 'data/make_hendel_splits.py',
    'nonce+arithmetic': 'data/make_nonce_arithmetic_splits.py',
}


def load_dataset(kind, cache_path=None):
    if kind not in _FILES:
        raise ValueError(f"kind must be one of {list(_FILES)}, got {kind!r}")
    path = Path(cache_path) if cache_path else Path(_FILES[kind])
    if not path.exists():
        raise FileNotFoundError(
            f"{kind} dataset not found at {path}. "
            f"Build it once with `python {_MAKERS[kind]}` from the repo root.")
    with open(path, 'rb') as f:
        return pickle.load(f)


def build_token_pools(splits, model):
    """Global input/output token pools + mean embeddings, for corruption modes.

    Mirrors claim1/icl_accuracy: the pool is the FIRST token id of ' '+word for
    every demo input (resp. output) across ALL tasks in `splits`. Returns:
      {'input_words', 'output_words',           # str pools (for mode='random')
       'input_tokids', 'output_tokids',         # first-token-id pools
       'mean_input_embed', 'mean_output_embed'}  # W_E mean over each pool (tensors)

    Not category-aware on purpose: 'random' draws from this global pool and 'mean'
    is the mean of the SAME pool, giving a monotone severity ladder
    orig -> shuffle -> random -> mean -> star.
    """
    import torch
    W_E = model.W_E.detach()
    in_words, out_words, in_ids, out_ids = [], [], [], []
    for task in splits.values():
        for pd_ in task['icl_prompts']:
            for inp, out in pd_['demo_pairs']:
                in_words.append(inp)
                out_words.append(out)
                in_ids.append(model.to_tokens(' ' + str(inp), prepend_bos=False)[0, 0].item())
                out_ids.append(model.to_tokens(' ' + str(out), prepend_bos=False)[0, 0].item())
    return {
        'input_words': in_words, 'output_words': out_words,
        'input_tokids': in_ids, 'output_tokids': out_ids,
        'mean_input_embed': W_E[in_ids].mean(dim=0),
        'mean_output_embed': W_E[out_ids].mean(dim=0),
    }
