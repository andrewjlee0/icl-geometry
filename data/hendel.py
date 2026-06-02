"""Data loading (Hendel repo), train/eval splitting, and caching."""
import json
import random
import pickle
from pathlib import Path
from .prompts import build_icl_prompt, build_zero_shot_prompt


def parse_json_pairs(content):
    """Try multiple formats to extract (input, output) pairs from JSON."""
    if not isinstance(content, (list, dict)):
        return None
    if isinstance(content, list) and len(content) > 0:
        if isinstance(content[0], (list, tuple)) and len(content[0]) == 2:
            return [(str(x), str(y)) for x, y in content]
        if isinstance(content[0], dict):
            for k_in, k_out in [('input','output'),('x','y'),('source','target'),
                                ('word','translation'),('question','answer')]:
                if k_in in content[0] and k_out in content[0]:
                    return [(str(d[k_in]), str(d[k_out])) for d in content]
            str_keys = [k for k, v in content[0].items() if isinstance(v, str)]
            if len(str_keys) >= 2:
                return [(str(d[str_keys[0]]), str(d[str_keys[1]])) for d in content]
    if isinstance(content, dict):
        return [(str(k), str(v)) for k, v in content.items()]
    return None


def load_hendel_data(repo_path):
    """Load all tasks from Hendel et al. data directory."""
    data_dir = Path(repo_path) / "data"
    tasks = {}
    if not data_dir.exists():
        raise FileNotFoundError(f"Data dir not found: {data_dir}")
    for json_file in sorted(data_dir.rglob("*.json")):
        task_name = str(json_file.relative_to(data_dir).with_suffix(''))
        try:
            with open(json_file) as f:
                content = json.load(f)
        except Exception:
            continue
        pairs = parse_json_pairs(content)
        if pairs and len(pairs) > 0:
            tasks[task_name] = pairs
    return tasks


def sample_splits(pairs, n_demos, n_icl_prompts, n_eval):
    """Create ICL prompts and held-out eval queries for one task.

    Eval pairs are held out before ICL prompt sampling (no leakage).

    Returns:
        icl_prompts: list of dicts with keys prompt, demo_pairs, query_input, query_output
        eval_data: list of dicts with keys demo_pairs, query_input, query_output, icl_prompt, zs_prompt
    """
    all_pairs = list(pairs)
    random.shuffle(all_pairs)

    eval_pairs = all_pairs[:n_eval]
    train_pairs = all_pairs[n_eval:]

    icl_prompts = []
    for _ in range(n_icl_prompts):
        sampled = random.sample(train_pairs, min(n_demos + 1, len(train_pairs)))
        demos = sampled[:n_demos]
        query_in, query_out = sampled[n_demos]
        icl_prompts.append({
            'prompt': build_icl_prompt(demos, query_in),
            'demo_pairs': demos,
            'query_input': query_in,
            'query_output': query_out,
        })

    eval_data = []
    for query_in, query_out in eval_pairs:
        demos = random.sample(train_pairs, min(n_demos, len(train_pairs)))
        eval_data.append({
            'demo_pairs': demos,
            'query_input': query_in,
            'query_output': query_out,
            'icl_prompt': build_icl_prompt(demos, query_in),
            'zs_prompt': build_zero_shot_prompt(query_in),
        })

    return icl_prompts, eval_data


def get_splits(hendel_repo, n_demos, n_icl_prompts, n_eval, seed=42,
               cache_path=None):
    """Load or create splits for all tasks.

    If cache_path exists, loads from cache. Otherwise builds splits and saves.
    Default cache: data/hendel_splits.pkl (from configs.SPLITS_DIR).

    Returns:
        dict: task_name -> {'icl_prompts': [...], 'eval_data': [...]}
    """
    if cache_path is None:
        from configs.defaults import HENDEL_SPLITS
        cache_path = HENDEL_SPLITS
    cache_path = Path(cache_path)

    if cache_path.exists():
        with open(cache_path, 'rb') as f:
            all_splits = pickle.load(f)
        print(f'Loaded cached splits: {len(all_splits)} tasks from {cache_path}')
        return all_splits

    random.seed(seed)
    tasks = load_hendel_data(hendel_repo)

    all_splits = {}
    for task_name, pairs in tasks.items():
        icl, ev = sample_splits(pairs, n_demos, n_icl_prompts, n_eval)
        all_splits[task_name] = {'icl_prompts': icl, 'eval_data': ev}
        print(f'{task_name}: {len(icl)} ICL prompts, {len(ev)} eval queries')

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(all_splits, f)
    print(f'Saved splits to {cache_path}')

    return all_splits
