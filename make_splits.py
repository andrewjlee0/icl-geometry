"""Run once: python make_splits.py"""
import random
from configs import *
from utils.data import load_hendel_data, sample_splits
import pickle
from pathlib import Path

random.seed(SEED)

tasks = load_hendel_data(HENDEL_REPO)

all_splits = {}
for task_name, pairs in tasks.items():
    icl, ev = sample_splits(pairs, N_DEMOS, N_TV_PROMPTS, N_EVAL_QUERIES)
    all_splits[task_name] = {'icl_prompts': icl, 'eval_data': ev}
    print(f'{task_name}: {len(icl)} ICL, {len(ev)} eval')

out = Path('configs/splits.pkl')
with open(out, 'wb') as f:
    pickle.dump(all_splits, f)
print(f'\nSaved to {out}')
