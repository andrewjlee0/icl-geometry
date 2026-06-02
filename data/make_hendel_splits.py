"""Run once: python data/make_hendel_splits.py"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
_ROOT = Path(__file__).resolve().parents[1]
import random
from configs import *
from data.hendel import load_hendel_data, sample_splits
import pickle
from pathlib import Path

random.seed(SEED)

tasks = load_hendel_data(HENDEL_REPO)

all_splits = {}
for task_name, pairs in tasks.items():
    icl, ev = sample_splits(pairs, N_DEMOS, N_TV_PROMPTS, N_EVAL_QUERIES)
    all_splits[task_name] = {'icl_prompts': icl, 'eval_data': ev}
    print(f'{task_name}: {len(icl)} ICL, {len(ev)} eval')

from configs.defaults import HENDEL_SPLITS
out = HENDEL_SPLITS
with open(out, 'wb') as f:
    pickle.dump(all_splits, f)
print(f'\nSaved to {out}')
