"""Unified dataset access so the four pairing_mechanism scripts treat both
datasets identically. Both are LOAD-ONLY: the run scripts never generate data.

    load_dataset('hendel') -> Hendel splits from configs/splits.pkl
                              (built once by make_splits.py)
    load_dataset('nonce+arithmetic') -> from configs/nonce_arithmetic_splits.pkl
                              (built once by make_nonce_arithmetic_splits.py)

Both return: {task_name: {'icl_prompts': [...], 'eval_data': [...]}}
"""
import pickle
from pathlib import Path

_CONFIG_DIR = Path(__file__).parent.parent / 'configs'
_FILES = {
    'hendel': 'splits.pkl',
    'nonce+arithmetic': 'nonce_arithmetic_splits.pkl',
}
_MAKERS = {
    'hendel': 'data/make_hendel_splits.py',
    'nonce+arithmetic': 'data/make_nonce_arithmetic_splits.py',
}


def load_dataset(kind, cache_path=None):
    if kind not in _FILES:
        raise ValueError(f"kind must be one of {list(_FILES)}, got {kind!r}")
    path = Path(cache_path) if cache_path else _CONFIG_DIR / _FILES[kind]
    if not path.exists():
        raise FileNotFoundError(
            f"{kind} dataset not found at {path}. "
            f"Build it once with `python {_MAKERS[kind]}` from the repo root.")
    with open(path, 'rb') as f:
        return pickle.load(f)
