"""score_heads — compute & cache pairing/aggregation head sets at all scopes.

Run once per dataset BEFORE the ablation/knockout scripts. Scores both head types
at pooled / per-category / per-task granularity in a single pass and caches the
whole thing to results/head_sets_<dataset>_pct<pct>.pkl. The experiment scripts
(03/04/05) then load this and pick a scope with --scope.

Usage (from experiments/pairing/):
    python score_heads.py --dataset hendel
    python score_heads.py --dataset nonce+arithmetic
"""
import _common as C
from data.loaders import load_dataset
from utils.heads import get_head_sets_multiscope


def main():
    p = C.base_parser(__doc__)
    p.add_argument('--recompute', action='store_true', help='ignore existing cache')
    args = p.parse_args()
    ds = args.dataset.replace('+', '_')
    model = C.load_model(cuda_visible=args.cuda)

    splits = load_dataset(args.dataset)
    tasks = sorted(splits.keys())
    print(f'{ds}: {len(tasks)} tasks — scoring pairing & aggregation at all scopes')

    cache = C.RESULTS_DIR / f'head_sets_{ds}_pct{args.head_pct}.pkl'
    ms = get_head_sets_multiscope(
        model, splits, pct=args.head_pct,
        n_prompts=(args.n_prompts if args.n_prompts else None),
        cache_path=cache, recompute=args.recompute, verbose=True)

    print('\n=== scopes cached ===')
    for sc in ms['scopes']:
        print('  ', sc)
    print(f'\nTop pooled pairing heads: {ms["pooled"]["pairing"][:8]}')
    print(f'Top pooled aggregation heads: {ms["pooled"]["aggregation"][:8]}')
    print(f'\nsaved -> {cache}')


if __name__ == '__main__':
    main()
