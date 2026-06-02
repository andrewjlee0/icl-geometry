"""Run ONCE to freeze the nonce/arith ("real ICL") dataset:

    python data/make_nonce_arithmetic_splits.py            # builds data/nonce_arithmetic_splits.pkl
    python data/make_nonce_arithmetic_splits.py --rebuild  # overwrite an existing one

This is the only place real-task data is generated. The experiment scripts in
experiments/pairing_mechanism/ only LOAD the resulting pkl (via load_dataset),
exactly like the Hendel pipeline loads data/hendel_splits.pkl built by make_hendel_splits.py.

Generation = sample candidate prompts per task -> keep the ones the model already
solves (so corruption/ablation effects are measured against a working baseline)
-> subsample to N_KEEP, plus N_EVAL held-out queries. Seeded, so re-running with
the same model reproduces the same dataset.
"""
import argparse
from pathlib import Path

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
_ROOT = Path(__file__).resolve().parents[1]
import torch
from transformer_lens import HookedTransformer

from configs import MODEL_NAME, N_DEMOS, N_TV_PROMPTS, N_EVAL_QUERIES
from data.tasks import build_nonce_arithmetic_dataset


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cuda', default='0', help='CUDA_VISIBLE_DEVICES value')
    p.add_argument('--rebuild', action='store_true', help='overwrite existing pkl')
    p.add_argument('--n-keep', type=int, default=N_TV_PROMPTS, help='solved ICL prompts kept per task')
    p.add_argument('--n-eval', type=int, default=N_EVAL_QUERIES, help='held-out eval queries per task')
    p.add_argument('--max-attempts-per-prompt', type=int, default=40,
                   help='cap = (n_keep+n_eval)*this generation attempts per task')
    p.add_argument('--min-successes', type=int, default=5, help='drop tasks below this many solved prompts')
    args = p.parse_args()

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)

    from configs.defaults import NONCE_ARITH_SPLITS
    cache = NONCE_ARITH_SPLITS
    if cache.exists() and not args.rebuild:
        raise SystemExit(f'{cache} already exists. Use --rebuild to overwrite it.')
    if cache.exists() and args.rebuild:
        cache.unlink()

    model = HookedTransformer.from_pretrained(MODEL_NAME, device='cuda', dtype=torch.float16)
    model.eval()

    build_nonce_arithmetic_dataset(
        model,
        n_demos=N_DEMOS,
        n_keep=args.n_keep,
        n_eval=args.n_eval,
        max_attempts_per_prompt=args.max_attempts_per_prompt,
        min_successes=args.min_successes,
        cache_path=cache,
        verbose=True,
    )
    print(f'\nFrozen real dataset written to {cache}. Experiment scripts will load it.')


if __name__ == '__main__':
    main()
