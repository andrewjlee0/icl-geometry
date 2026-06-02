"""Project-wide configuration."""
from pathlib import Path

# Paths
HENDEL_REPO = Path("/home/cvllab/Documents/andrew/repos/icl_task_vectors")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Model
MODEL_NAME = "meta-llama/Llama-3.2-3B"

# Prompt format
SEPARATOR = " →"
DEMO_SEP = "\n"

# Experiment defaults
N_DEMOS = 10
N_TV_PROMPTS = 50
N_EVAL_QUERIES = 20
SEED = 42
