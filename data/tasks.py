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
    'add_3':       (lambda x: x + 3,      1, 80),
    'add_7':       (lambda x: x + 7,      1, 80),
    'add_13':      (lambda x: x + 13,     1, 80),
    'mul_5':       (lambda x: 5 * x,      1, 40),
    'mul_7':       (lambda x: 7 * x,      1, 40),   # replaces mul_2
    'mul_9':       (lambda x: 9 * x,      1, 40),   # replaces mul_3
    'mul_4_add_7': (lambda x: 4 * x + 7,  1, 60),   # replaces mul_10
    'mul_2_add_1': (lambda x: 2 * x + 1,  1, 80),
    'mul_3_add_2': (lambda x: 3 * x + 2,  1, 60),
    'mul_2_sub_1': (lambda x: 2 * x - 1,  1, 80),
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
def build_nonce_arithmetic_dataset(model, n_demos=10, n_keep=50, n_eval=20,
                       max_attempts_per_prompt=40, min_successes=5,
                       cache_path=None, verbose=True, n_raw=None):
    """Build the nonce/arith dataset, filtered to prompts the model already solves.

    Generates candidates and keeps the solved ones until it has n_keep solved ICL
    prompts (plus n_eval solved held-out eval queries) per task, rather than
    drawing a fixed pool and keeping whatever fraction passes. This guarantees a
    uniform n_keep across tasks when the task is solvable at all.

    A per-prompt attempt cap (max_attempts_per_prompt) bounds the work: collection
    stops after n_keep*max_attempts_per_prompt tries even if n_keep isn't reached,
    so a genuinely-too-hard task can't loop forever. n_raw is accepted but ignored
    (kept for backward compatibility with older calls).

    Returns dict: task_name -> {'icl_prompts': [...], 'eval_data': [...]}
      icl_prompts[i] = {'prompt', 'demo_pairs', 'query_input', 'query_output'}
      eval_data[i]   = {'demo_pairs', 'query_input', 'query_output',
                        'icl_prompt', 'zs_prompt'}   (held-out fresh queries)
    """
    if cache_path is None:
        from configs.defaults import NONCE_ARITH_SPLITS
        cache_path = NONCE_ARITH_SPLITS
    cache_path = Path(cache_path)

    if cache_path.exists():
        with open(cache_path, 'rb') as f:
            splits = pickle.load(f)
        if verbose:
            print(f'Loaded cached nonce/arith dataset: {len(splits)} tasks from {cache_path}')
        return splits

    splits = {}
    if verbose:
        print(f'{"Task":<16s} {"hit_rate":>9s} {"icl_kept":>9s} {"eval_kept":>10s} {"attempts":>9s}')
        print('-' * 60)

    target_total = n_keep + n_eval                 # solved prompts needed per task
    max_attempts = target_total * max_attempts_per_prompt

    for name in NONCE_ARITH_TASKS:
        rng = random.Random(hash(name) % 10 ** 6)

        # keep generating until we have enough SOLVED prompts (or hit the cap)
        solved = []
        attempts = 0
        while len(solved) < target_total and attempts < max_attempts:
            attempts += 1
            demos, qi, qo = generate_one(name, rng, n_demos)
            prompt = build_icl_prompt(demos, qi)
            tok = model.to_tokens(prompt, prepend_bos=True)
            if check_correct_multitoken(model, tok, qo):
                solved.append({'prompt': prompt, 'demo_pairs': demos,
                               'query_input': qi, 'query_output': qo})
            torch.cuda.empty_cache()

        hit_rate = len(solved) / max(attempts, 1)
        # first n_keep -> icl prompts, next n_eval -> held-out eval queries
        good = solved[:n_keep]
        eval_pool = solved[n_keep:n_keep + n_eval]
        eval_data = [{'demo_pairs': s['demo_pairs'], 'query_input': s['query_input'],
                      'query_output': s['query_output'], 'icl_prompt': s['prompt'],
                      'zs_prompt': build_zero_shot_prompt(s['query_input'])}
                     for s in eval_pool]

        if verbose:
            flag = '' if len(good) >= n_keep else '  <-- under target'
            print(f'{name:<16s} {hit_rate:>9.3f} {len(good):>9d} '
                  f'{len(eval_data):>10d} {attempts:>9d}{flag}')

        if len(good) >= min_successes:
            splits[name] = {'icl_prompts': good, 'eval_data': eval_data}

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(splits, f)
    if verbose:
        dropped = [n for n in NONCE_ARITH_TASKS if n not in splits]
        if dropped:
            print(f'\nDropped (< {min_successes} solved): {dropped}')
        under = [n for n in splits if len(splits[n]['icl_prompts']) < n_keep]
        if under:
            print(f'Under target n_keep={n_keep} (task too hard to fill): {under}')
        print(f'\nActive nonce/arith tasks: {len(splits)}')
        print(f'Saved to {cache_path}')

    return splits
