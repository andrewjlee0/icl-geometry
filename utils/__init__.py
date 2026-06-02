"""utils/ — experiment-running machinery.

Dataset construction/loading lives in the data/ package, not here.
"""
from .positions import (find_role_positions, find_per_demo_positions,
                        find_per_demo_positions_robust, find_slot_positions)
from .extraction import (extract_hidden_states, extract_head_outputs,
                         extract_attention_patterns, extract_head_outputs_at_positions)
from .eval import (check_correct, check_correct_multitoken,
                   eval_patched_resid, eval_patched_resid_add,
                   eval_with_head_replace, eval_with_head_ablation,
                   compute_task_vectors, run_patching_sweep)
from .scores import compute_induction_scores_rrt, compute_head_contributions, select_top_heads
from .heads import (score_pairing_heads, score_aggregation_heads, select_top_pct,
                    matched_random_heads, get_head_sets, get_head_sets_multiscope, select_scope, categorize_tasks,
                    make_ablation_hooks, make_amplify_hooks,
                    make_attn_knockout_hooks, make_batched_attn_knockout_hooks)
from .corruptions import corrupt_demos, apply_corruption, corruption_plan, MODES
