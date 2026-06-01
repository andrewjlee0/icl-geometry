from .data import load_hendel_data, sample_splits, get_splits
from .prompts import build_icl_prompt, build_zero_shot_prompt, find_role_positions, find_per_demo_positions
from .extraction import (extract_hidden_states, extract_head_outputs,
                         extract_attention_patterns, extract_head_outputs_at_positions)
from .eval import (check_correct, eval_patched_resid, eval_patched_resid_add,
                   eval_with_head_replace, eval_with_head_ablation,
                   compute_task_vectors, run_patching_sweep)
from .scores import compute_induction_scores_rrt, compute_head_contributions, select_top_heads