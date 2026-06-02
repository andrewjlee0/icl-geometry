"""data/ — dataset construction and loading.

  hendel.py   load Hendel tasks, build splits
  tasks.py    nonce/arith generators + build_nonce_arithmetic_dataset
  prompts.py  ICL / zero-shot prompt-string construction
  loaders.py  load_dataset('hendel' | 'nonce+arithmetic')  (load-only)

Frozen splits are created once by the make_*_splits.py scripts in this folder.
"""
from .prompts import build_icl_prompt, build_zero_shot_prompt
from .hendel import load_hendel_data, sample_splits, get_splits
from .tasks import (NONCE_TASKS, ARITH_TASKS, NONCE_ARITH_TASKS,
                    generate_one, build_nonce_arithmetic_dataset)
from .loaders import load_dataset
