"""Real-ICL task generators (nonce-word + arithmetic) and dataset builder.

These are the "non-hendel" tasks: character-level transforms on CVCVC nonce words
and arithmetic maps. Outputs are fully determined by the input (no vocabulary
shortcut), so solving them requires reading the input->output mapping.

build_nonce_arithmetic_dataset(model, ...) generates many candidate prompts per task, keeps
only the ones the model already solves, and returns the SAME dict shape as the
Hendel splits:  {task_name: {'icl_prompts': [...], 'eval_data': [...]}}
so downstream code treats both datasets identically.
"""
import os
import pickle
import random
from pathlib import Path

import torch

from .prompts import build_icl_prompt, build_zero_shot_prompt
from utils.eval import check_correct_multitoken

CONSONANTS = 'bdfghjklmnprstvwz'
VOWELS = 'aeiou'


# --------------------------------------------------------------------------- #
# nonce-word generators
# --------------------------------------------------------------------------- #
def make_nonsense(length, rng):
    pattern = [CONSONANTS, VOWELS] * (length // 2 + 1)
    return ''.join(rng.choice(p) for p in pattern[:length])


def _gen_nonce(rng, transform_fn, n_demos, word_len=5):
    """Generic nonce-word task: n_demos demos + 1 query, all distinct words."""
    words = []
    while len(set(words)) < n_demos + 1:
        words = [make_nonsense(word_len, rng) for _ in range(n_demos + 1)]
    demo_pairs = [(w, transform_fn(w)) for w in words[:n_demos]]
    qi = words[n_demos]
    return demo_pairs, qi, transform_fn(qi)


_NONCE_TRANSFORMS = {
    'repetition':    lambda x: x + x,
    'prepend_first': lambda x: x[0] + x,
    'append_last':   lambda x: x + x[-1],
    'drop_first':    lambda x: x[1:],
    'drop_last':     lambda x: x[:-1],
    'reverse':       lambda x: x[::-1],
    'swap_ends':     lambda x: x[-1] + x[1:-1] + x[0],
    'rotate_left':   lambda x: x[1:] + x[0],
    'suffix_ed':     lambda x: x + 'ed',
    'double_vowels': lambda x: ''.join(c * 2 if c in VOWELS else c for c in x),
}


# --------------------------------------------------------------------------- #
# arithmetic generators
# --------------------------------------------------------------------------- #
def _gen_arith(rng, fn, lo, hi, n_demos):
    pool = list(range(lo, hi + 1))
    rng.shuffle(pool)
    nums = pool[:n_demos + 1]
    demo_pairs = [(str(n), str(fn(n))) for n in nums[:n_demos]]
    qi = str(nums[n_demos])
    return demo_pairs, qi, str(fn(nums[n_demos]))


# NOTE ON REPLACEMENTS:
# mul_2, mul_3, mul_10 were removed. They are heavily "prior-based": x2/x3 are
# fluent times-tables and x10 is trivial digit-append, so the model can answer
# from parametric knowledge / format alone instead of reading the demos. They
# are replaced with less-fluent maps that genuinely require the demonstrations:
#   mul_2  -> mul_7        (y = 7x)        7x table is far less rote
#   mul_3  -> mul_9        (y = 9x)        9x table is far less rote
#   mul_10 -> mul_4_add_7  (y = 4x + 7)    affine; offset blocks format shortcuts
_ARITH_SPECS = {
    'add_3':       (lambda x: x + 3,      1, 50),
    'add_7':       (lambda x: x + 7,      1, 50),
    'add_13':      (lambda x: x + 13,     1, 50),
    'mul_5':       (lambda x: 5 * x,      1, 20),
    'mul_7':       (lambda x: 7 * x,      1, 20),   # replaces mul_2
    'mul_9':       (lambda x: 9 * x,      1, 15),   # replaces mul_3
    'mul_4_add_7': (lambda x: 4 * x + 7,  1, 30),   # replaces mul_10
    'mul_2_add_1': (lambda x: 2 * x + 1,  1, 50),
    'mul_3_add_2': (lambda x: 3 * x + 2,  1, 30),
    'mul_2_sub_1': (lambda x: 2 * x - 1,  1, 50),
}

NONCE_TASKS = list(_NONCE_TRANSFORMS.keys())
ARITH_TASKS = list(_ARITH_SPECS.keys())
NONCE_ARITH_TASKS = NONCE_TASKS + ARITH_TASKS


def generate_one(task_name, rng, n_demos):
    """Return (demo_pairs, query_input, query_output) for a single sample."""
    if task_name in _NONCE_TRANSFORMS:
        return _gen_nonce(rng, _NONCE_TRANSFORMS[task_name], n_demos)
    fn, lo, hi = _ARITH_SPECS[task_name]
    return _gen_arith(rng, fn, lo, hi, n_demos)


# --------------------------------------------------------------------------- #
# dataset builder
# --------------------------------------------------------------------------- #
def build_nonce_arithmetic_dataset(model, n_demos=10, n_raw=200, n_keep=50, n_eval=20,
                       min_successes=5, cache_path=None, verbose=True):
    """Build the real-ICL dataset, filtered to prompts the model already solves.

    Returns dict: task_name -> {'icl_prompts': [...], 'eval_data': [...]}
      icl_prompts[i] = {'prompt', 'demo_pairs', 'query_input', 'query_output'}
      eval_data[i]   = {'demo_pairs', 'query_input', 'query_output',
                        'icl_prompt', 'zs_prompt'}   (held-out fresh queries)
    """
    if cache_path is None:
        cache_path = Path(__file__).parent.parent / 'configs' / 'nonce_arithmetic_splits.pkl'
    cache_path = Path(cache_path)

    if cache_path.exists():
        with open(cache_path, 'rb') as f:
            splits = pickle.load(f)
        if verbose:
            print(f'Loaded cached real dataset: {len(splits)} tasks from {cache_path}')
        return splits

    splits = {}
    if verbose:
        print(f'{"Task":<16s} {"orig_acc":>9s} {"n_correct":>10s} {"kept":>6s}')
        print('-' * 46)

    for name in NONCE_ARITH_TASKS:
        rng = random.Random(hash(name) % 10 ** 6)

        # generate candidate ICL prompts, keep the ones the model solves
        candidates = []
        for _ in range(n_raw):
            demos, qi, qo = generate_one(name, rng, n_demos)
            candidates.append({
                'prompt': build_icl_prompt(demos, qi),
                'demo_pairs': demos,
                'query_input': qi,
                'query_output': qo,
            })

        good = []
        for c in candidates:
            tok = model.to_tokens(c['prompt'], prepend_bos=True)
            if check_correct_multitoken(model, tok, c['query_output']):
                good.append(c)
            torch.cuda.empty_cache()

        acc = len(good) / len(candidates)
        if len(good) > n_keep:
            good = random.Random(42).sample(good, n_keep)

        # held-out eval queries: fresh samples the model also solves
        eval_data = []
        tries = 0
        while len(eval_data) < n_eval and tries < n_eval * 20:
            tries += 1
            demos, qi, qo = generate_one(name, rng, n_demos)
            icl_prompt = build_icl_prompt(demos, qi)
            tok = model.to_tokens(icl_prompt, prepend_bos=True)
            if check_correct_multitoken(model, tok, qo):
                eval_data.append({
                    'demo_pairs': demos,
                    'query_input': qi,
                    'query_output': qo,
                    'icl_prompt': icl_prompt,
                    'zs_prompt': build_zero_shot_prompt(qi),
                })
            torch.cuda.empty_cache()

        if verbose:
            print(f'{name:<16s} {acc:>9.3f} {len(good):>10d} {len(good):>6d}')

        if len(good) >= min_successes:
            splits[name] = {'icl_prompts': good, 'eval_data': eval_data}

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(splits, f)
    if verbose:
        dropped = [n for n in NONCE_ARITH_TASKS if n not in splits]
        if dropped:
            print(f'\nDropped (< {min_successes} successes): {dropped}')
        print(f'\nActive real tasks: {len(splits)}')
        print(f'Saved to {cache_path}')

    return splits
