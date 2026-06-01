import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ICLExample:
    """A single input-output demonstration."""
    input_token: str
    output_token: str


@dataclass  
class ICLPrompt:
    """Full ICL prompt with demonstrations and query."""
    demonstrations: List[ICLExample]
    query_input: str
    expected_output: str
    relation_name: str


def create_icl_tasks() -> List[ICLPrompt]:
    """
    Create ICL tasks with MULTIPLE PROMPTS per task, each with 3 demonstrations.
    This gives independent samples across prompts.
    """
    tasks = []
    
    # Define example pools for each relation type
    # Each pool has (input, output) pairs and we'll sample different subsets
    
    # antonym_pairs = [
    #     ("hot", "cold"), ("big", "small"), ("fast", "slow"), ("up", "down"),
    #     ("left", "right"), ("old", "new"), ("rich", "poor"), ("dark", "light"),
    #     ("hard", "soft"), ("wet", "dry"), ("loud", "quiet"), ("early", "late"),
    #     ("good", "bad"), ("happy", "sad"), ("tall", "short"), ("thin", "thick"),
    # ]
    
    capital_pairs = [
        ("France", "Paris"), ("Japan", "Tokyo"), ("Italy", "Rome"), ("Egypt", "Cairo"),
        ("Cuba", "Havana"), ("Peru", "Lima"), ("Greece", "Athens"), ("Poland", "Warsaw"),
        ("Sweden", "Stockholm"), ("Norway", "Oslo"), ("Austria", "Vienna"), ("Spain", "Madrid"),
        ("China", "Beijing"), ("Russia", "Moscow"), ("Germany", "Berlin"), ("India", "Delhi"),
    ]
    
    past_tense_pairs = [
        ("run", "ran"), ("eat", "ate"), ("go", "went"), ("see", "saw"),
        ("take", "took"), ("give", "gave"), ("come", "came"), ("know", "knew"),
        ("think", "thought"), ("bring", "brought"), ("buy", "bought"), ("catch", "caught"),
        ("teach", "taught"), ("find", "found"), ("tell", "told"), ("sell", "sold"),
    ]
    
    plural_pairs = [
        ("cat", "cats"), ("dog", "dogs"), ("car", "cars"), ("tree", "trees"),
        ("house", "houses"), ("bird", "birds"), ("book", "books"), ("chair", "chairs"),
        ("table", "tables"), ("phone", "phones"), ("lamp", "lamps"), ("door", "doors"),
        ("ball", "balls"), ("cup", "cups"), ("hat", "hats"), ("key", "keys"),
    ]
    
    language_pairs = [
        ("France", "French"), ("Spain", "Spanish"), ("Germany", "German"), ("Italy", "Italian"),
        ("Portugal", "Portuguese"), ("Russia", "Russian"), ("Japan", "Japanese"), ("China", "Chinese"),
        ("Poland", "Polish"), ("Sweden", "Swedish"), ("Finland", "Finnish"), ("Turkey", "Turkish"),
        ("Greece", "Greek"), ("Denmark", "Danish"), ("Norway", "Norwegian"), ("Holland", "Dutch"),
    ]
    
    gender_pairs = [
        ("king", "queen"), ("man", "woman"), ("boy", "girl"), ("father", "mother"),
        ("son", "daughter"), ("brother", "sister"), ("uncle", "aunt"), ("husband", "wife"),
        ("actor", "actress"), ("prince", "princess"), ("hero", "heroine"), ("waiter", "waitress"),
        ("god", "goddess"), ("host", "hostess"), ("lion", "lioness"), ("emperor", "empress"),
    ]
    
    comparative_pairs = [
        ("big", "bigger"), ("small", "smaller"), ("fast", "faster"), ("slow", "slower"),
        ("tall", "taller"), ("short", "shorter"), ("old", "older"), ("young", "younger"),
        ("hot", "hotter"), ("cold", "colder"), ("loud", "louder"), ("soft", "softer"),
        ("hard", "harder"), ("weak", "weaker"), ("strong", "stronger"), ("bright", "brighter"),
    ]
    
    agent_pairs = [
        ("teach", "teacher"), ("write", "writer"), ("read", "reader"), ("play", "player"),
        ("sing", "singer"), ("dance", "dancer"), ("drive", "driver"), ("lead", "leader"),
        ("work", "worker"), ("build", "builder"), ("paint", "painter"), ("farm", "farmer"),
        ("hunt", "hunter"), ("bank", "banker"), ("deal", "dealer"), ("dream", "dreamer"),
    ]
    
    # Create multiple prompts per task type
    # Each prompt uses 3 demos and a different query
    
    def make_prompts(pairs, relation_name, n_prompts=8):
        """Create n_prompts different prompts from the pair pool."""
        prompts = []
        n_pairs = len(pairs)
        
        for i in range(n_prompts):
            # Select 3 demo pairs (non-overlapping with query)
            # Use different starting points to get variety
            start_idx = (i * 4) % n_pairs
            demo_indices = [(start_idx + j) % n_pairs for j in range(3)]
            query_idx = (start_idx + 3) % n_pairs
            
            demos = [ICLExample(pairs[idx][0], pairs[idx][1]) for idx in demo_indices]
            query_input, expected_output = pairs[query_idx]
            
            prompts.append(ICLPrompt(
                demonstrations=demos,
                query_input=query_input,
                expected_output=expected_output,
                relation_name=relation_name
            ))
        
        return prompts
    
    # Generate prompts for each relation type
    # tasks.extend(make_prompts(antonym_pairs, "antonym", n_prompts=8))
    tasks.extend(make_prompts(capital_pairs, "capital", n_prompts=8))
    tasks.extend(make_prompts(past_tense_pairs, "past_tense", n_prompts=8))
    tasks.extend(make_prompts(plural_pairs, "plural", n_prompts=8))
    tasks.extend(make_prompts(language_pairs, "language", n_prompts=8))
    tasks.extend(make_prompts(gender_pairs, "gender", n_prompts=8))
    tasks.extend(make_prompts(comparative_pairs, "comparative", n_prompts=8))
    tasks.extend(make_prompts(agent_pairs, "agent_noun", n_prompts=8))
    
    return tasks


def format_prompt(task: ICLPrompt) -> str:
    """Format ICL task as a string prompt."""
    lines = []
    for demo in task.demonstrations:
        lines.append(f"{demo.input_token} -> {demo.output_token}")
    lines.append(f"{task.query_input} ->")
    return "\n".join(lines)


def find_token_positions(
    prompt: str,
    task: ICLPrompt,
    tokenizer
) -> Dict[str, List[int]]:
    """
    Find token positions for inputs, outputs, query, and final position.
    """
    encoding = tokenizer(prompt, return_tensors="pt")
    input_ids = encoding.input_ids[0]
    
    tokens = []
    for i, tid in enumerate(input_ids):
        decoded = tokenizer.decode([tid])
        tokens.append(decoded)
    
    positions = {
        'input_positions': [],
        'output_positions': [],
        'query_position': None,
        'final_position': len(tokens) - 1
    }
    
    all_inputs = [demo.input_token for demo in task.demonstrations]
    all_outputs = [demo.output_token for demo in task.demonstrations]
    
    # Reconstruct string with token boundaries
    reconstructed = ""
    token_start_chars = []
    for i, tok in enumerate(tokens):
        token_start_chars.append(len(reconstructed))
        reconstructed += tok
    
    # Find each input word
    search_start = 0
    for inp in all_inputs:
        idx = reconstructed.lower().find(inp.lower(), search_start)
        if idx != -1:
            for i, start_char in enumerate(token_start_chars):
                if i + 1 < len(token_start_chars):
                    end_char = token_start_chars[i + 1]
                else:
                    end_char = len(reconstructed)
                if start_char <= idx < end_char:
                    positions['input_positions'].append(i)
                    search_start = end_char
                    break
    
    # Find each output word
    search_start = 0
    for out in all_outputs:
        idx = reconstructed.lower().find(out.lower(), search_start)
        if idx != -1:
            for i, start_char in enumerate(token_start_chars):
                if i + 1 < len(token_start_chars):
                    end_char = token_start_chars[i + 1]
                else:
                    end_char = len(reconstructed)
                if start_char <= idx < end_char:
                    positions['output_positions'].append(i)
                    search_start = end_char
                    break
    
    # Find query
    query_idx = reconstructed.lower().rfind(task.query_input.lower())
    if query_idx != -1:
        for i, start_char in enumerate(token_start_chars):
            if i + 1 < len(token_start_chars):
                end_char = token_start_chars[i + 1]
            else:
                end_char = len(reconstructed)
            if start_char <= query_idx < end_char:
                positions['query_position'] = i
                break
    
    return positions, tokens


class ActivationCache:
    """Hook-based activation caching."""
    
    def __init__(self, model):
        self.model = model
        self.activations = {}
        self.hooks = []
        
    def _make_hook(self, name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                self.activations[name] = output[0].detach().float().cpu()
            else:
                self.activations[name] = output.detach().float().cpu()
        return hook
    
    def register_hooks(self):
        """Register hooks on residual stream (layer outputs)."""
        if hasattr(self.model, 'transformer'):
            for i, layer in enumerate(self.model.transformer.h):
                hook = layer.register_forward_hook(self._make_hook(f'layer_{i}'))
                self.hooks.append(hook)
            self.n_layers = len(self.model.transformer.h)
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            for i, layer in enumerate(self.model.model.layers):
                hook = layer.register_forward_hook(self._make_hook(f'layer_{i}'))
                self.hooks.append(hook)
            self.n_layers = len(self.model.model.layers)
        else:
            raise ValueError("Unknown model architecture")
                
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
    def clear(self):
        self.activations = {}


def get_logit_lens_prediction(model, tokenizer, hidden_state, track_tokens=None):
    """
    Apply logit lens: project hidden state through unembedding to get top token.
    """
    with torch.no_grad():
        hidden = torch.tensor(hidden_state).float().to(model.device)
        
        if hasattr(model, 'dtype'):
            hidden = hidden.to(model.dtype)
        elif next(model.parameters()).dtype == torch.bfloat16:
            hidden = hidden.to(torch.bfloat16)
        
        if hasattr(model, 'model') and hasattr(model.model, 'norm'):
            hidden = model.model.norm(hidden)
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'ln_f'):
            hidden = model.transformer.ln_f(hidden)
        
        if hasattr(model, 'lm_head'):
            logits = model.lm_head(hidden)
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'wte'):
            logits = hidden @ model.transformer.wte.weight.T
        else:
            return None, 0.0, {}
        
        logits = logits.float()
        probs = torch.softmax(logits, dim=-1)
        top_idx = torch.argmax(logits).item()
        top_prob = probs[top_idx].item()
        top_token = tokenizer.decode([top_idx])
        
        tracked_probs = {}
        if track_tokens:
            for name, token_str in track_tokens.items():
                token_ids = tokenizer.encode(token_str, add_special_tokens=False)
                if token_ids:
                    tracked_probs[name] = probs[token_ids[0]].item()
    
    return top_token, top_prob, tracked_probs


def compute_clustering_score(input_acts: np.ndarray, output_acts: np.ndarray) -> float:
    """
    Compute role clustering score: between-role variance / within-role variance.
    """
    input_centroid = input_acts.mean(axis=0)
    output_centroid = output_acts.mean(axis=0)
    
    between_var = np.linalg.norm(output_centroid - input_centroid) ** 2
    within_input = np.mean([np.linalg.norm(x - input_centroid)**2 for x in input_acts])
    within_output = np.mean([np.linalg.norm(x - output_centroid)**2 for x in output_acts])
    within_var = (within_input + within_output) / 2 + 1e-10
    
    return between_var / within_var


def compute_random_baseline_clustering(all_acts: np.ndarray, n_group1: int, n_permutations: int = 100) -> Dict[str, float]:
    """
    Compute clustering score for random splits as a control.
    Returns mean and std of clustering scores under random assignment.
    """
    scores = []
    n_total = len(all_acts)
    
    for _ in range(n_permutations):
        # Random permutation of indices
        perm = np.random.permutation(n_total)
        group1 = all_acts[perm[:n_group1]]
        group2 = all_acts[perm[n_group1:]]
        scores.append(compute_clustering_score(group1, group2))
    
    return {
        'mean': np.mean(scores),
        'std': np.std(scores),
        'scores': scores
    }


def compute_cosine_similarities(input_acts: np.ndarray, output_acts: np.ndarray) -> Dict[str, float]:
    """
    Compute average cosine similarity within and between roles.
    """
    # Within input
    if len(input_acts) > 1:
        input_sim = cosine_similarity(input_acts)
        within_input = (input_sim.sum() - np.trace(input_sim)) / (len(input_acts) * (len(input_acts) - 1))
    else:
        within_input = 1.0
    
    # Within output
    if len(output_acts) > 1:
        output_sim = cosine_similarity(output_acts)
        within_output = (output_sim.sum() - np.trace(output_sim)) / (len(output_acts) * (len(output_acts) - 1))
    else:
        within_output = 1.0
    
    # Between roles
    between_sim = cosine_similarity(input_acts, output_acts)
    between = between_sim.mean()
    
    return {
        'within_input': within_input,
        'within_output': within_output,
        'within_avg': (within_input + within_output) / 2,
        'between': between
    }


def compute_random_baseline_cosine(all_acts: np.ndarray, n_group1: int, n_permutations: int = 100) -> Dict[str, float]:
    """
    Compute cosine similarity gap for random splits as a control.
    Returns mean and std of (within - between) under random assignment.
    """
    gaps = []
    n_total = len(all_acts)
    
    for _ in range(n_permutations):
        perm = np.random.permutation(n_total)
        group1 = all_acts[perm[:n_group1]]
        group2 = all_acts[perm[n_group1:]]
        cos_sims = compute_cosine_similarities(group1, group2)
        gap = cos_sims['within_avg'] - cos_sims['between']
        gaps.append(gap)
    
    return {
        'mean': np.mean(gaps),
        'std': np.std(gaps),
        'gaps': gaps
    }


def compute_linear_separability_cv(input_acts: np.ndarray, output_acts: np.ndarray) -> float:
    """
    Compute linear separability using cross-validation.
    """
    X = np.vstack([input_acts, output_acts])
    y = np.array([0]*len(input_acts) + [1]*len(output_acts))
    
    if len(X) < 4:
        return 0.5
    
    clf = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')
    
    # Use leave-one-out CV for small datasets
    n_splits = min(5, len(X))
    try:
        scores = cross_val_score(clf, X, y, cv=n_splits, scoring='accuracy')
        return scores.mean()
    except:
        # Fallback to train accuracy if CV fails
        clf.fit(X, y)
        return accuracy_score(y, clf.predict(X))


def run_analysis(model_name: str = "meta-llama/Llama-3.2-1B"):
    """Run full role geometry analysis."""
    
    print(f"{'='*60}")
    print(f"ICL ROLE GEOMETRY ANALYSIS v2")
    print(f"Model: {model_name}")
    print(f"{'='*60}\n")
    
    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Device: {device}")
    
    # Setup cache
    cache = ActivationCache(model)
    cache.register_hooks()
    n_layers = cache.n_layers
    print(f"Number of layers: {n_layers}\n")
    
    # Create tasks
    tasks = create_icl_tasks()
    n_relation_types = len(set(t.relation_name for t in tasks))
    prompts_per_type = len(tasks) // n_relation_types
    print(f"Number of relation types: {n_relation_types}")
    print(f"Prompts per relation: {prompts_per_type}")
    print(f"Total prompts: {len(tasks)}")
    print(f"Demos per prompt: {len(tasks[0].demonstrations)}")
    
    # Storage for results
    all_results = []
    
    # Also collect all activations across tasks for aggregate analysis
    all_input_acts_by_layer = {l: [] for l in range(n_layers)}
    all_output_acts_by_layer = {l: [] for l in range(n_layers)}
    
    for task_idx, task in enumerate(tasks):
        prompt = format_prompt(task)
        
        # Get positions
        positions, tokens = find_token_positions(prompt, task, tokenizer)
        
        n_inputs_found = len(positions['input_positions'])
        n_outputs_found = len(positions['output_positions'])
        
        # Validate
        if n_inputs_found < 2 or n_outputs_found < 2:
            print(f"  Task {task_idx} ({task.relation_name}): skipped - not enough positions")
            continue
        if positions['query_position'] is None:
            print(f"  Task {task_idx} ({task.relation_name}): skipped - no query position")
            continue
            
        # Forward pass
        cache.clear()
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Check prediction
        predicted_id = torch.argmax(outputs.logits[0, -1, :]).item()
        predicted_token = tokenizer.decode([predicted_id])
        correct = task.expected_output.lower() in predicted_token.lower()
        
        # Print progress every 8 tasks
        if task_idx % 8 == 0:
            print(f"  Processing {task.relation_name} prompts... (task {task_idx+1}/{len(tasks)})")
        
        # Extract and analyze activations at each layer
        results = {
            'task': task.relation_name,
            'n_inputs': n_inputs_found,
            'n_outputs': n_outputs_found,
            'correct': correct,
            'clustering_scores': [],
            'separability_scores': [],
            'cosine_within': [],
            'cosine_between': [],
            'final_query_prob': [],
            'final_output_prob': [],
            'final_relative_position': [],  # 0 = at input centroid, 1 = at output centroid
        }
        
        track_tokens = {
            'query': task.query_input,
            'output': task.expected_output
        }
        
        for layer in range(n_layers):
            layer_act = cache.activations[f'layer_{layer}'][0].numpy()
            
            input_acts = layer_act[positions['input_positions']]
            output_acts = layer_act[positions['output_positions']]
            final_act = layer_act[positions['final_position']]
            
            # Store for aggregate analysis
            all_input_acts_by_layer[layer].append(input_acts)
            all_output_acts_by_layer[layer].append(output_acts)
            
            # Clustering
            clustering = compute_clustering_score(input_acts, output_acts)
            results['clustering_scores'].append(clustering)
            
            # Linear separability with CV
            separability = compute_linear_separability_cv(input_acts, output_acts)
            results['separability_scores'].append(separability)
            
            # Cosine similarities
            cos_sims = compute_cosine_similarities(input_acts, output_acts)
            results['cosine_within'].append(cos_sims['within_avg'])
            results['cosine_between'].append(cos_sims['between'])
            
            # Final position relative distance: 0 = at input centroid, 1 = at output centroid
            input_centroid = input_acts.mean(axis=0)
            output_centroid = output_acts.mean(axis=0)
            dist_to_input = np.linalg.norm(final_act - input_centroid)
            dist_to_output = np.linalg.norm(final_act - output_centroid)
            total_dist = dist_to_input + dist_to_output + 1e-10
            relative_pos = dist_to_input / total_dist  # 0 = at input, 1 = at output
            results['final_relative_position'].append(relative_pos)
            
            # Logit lens
            _, _, tracked = get_logit_lens_prediction(model, tokenizer, final_act, track_tokens)
            results['final_query_prob'].append(tracked.get('query', 0))
            results['final_output_prob'].append(tracked.get('output', 0))
        
        all_results.append(results)
    
    cache.remove_hooks()
    
    if len(all_results) == 0:
        print("No tasks completed!")
        return
    
    # Print accuracy summary
    n_correct = sum(1 for r in all_results if r.get('correct', False))
    print(f"\n\nModel accuracy: {n_correct}/{len(all_results)} ({100*n_correct/len(all_results):.1f}%)")
    
    # Aggregate analysis across all tasks
    print(f"\n{'='*60}")
    print("AGGREGATE ANALYSIS (all prompts combined)")
    print(f"{'='*60}")
    
    aggregate_clustering = []
    aggregate_separability = []
    aggregate_cosine_within = []
    aggregate_cosine_between = []
    
    # Control baselines
    random_clustering_baseline = []
    random_cosine_baseline = []
    
    for layer in range(n_layers):
        # Combine all inputs and outputs across tasks
        all_inputs = np.vstack(all_input_acts_by_layer[layer])
        all_outputs = np.vstack(all_output_acts_by_layer[layer])
        
        if layer == 0:
            print(f"Aggregating: {len(all_inputs)} inputs, {len(all_outputs)} outputs per layer")
        
        aggregate_clustering.append(compute_clustering_score(all_inputs, all_outputs))
        aggregate_separability.append(compute_linear_separability_cv(all_inputs, all_outputs))
        
        cos_sims = compute_cosine_similarities(all_inputs, all_outputs)
        aggregate_cosine_within.append(cos_sims['within_avg'])
        aggregate_cosine_between.append(cos_sims['between'])
        
        # Compute random baselines
        all_acts = np.vstack([all_inputs, all_outputs])
        n_inputs = len(all_inputs)
        
        random_clust = compute_random_baseline_clustering(all_acts, n_inputs, n_permutations=100)
        random_clustering_baseline.append(random_clust)
        
        random_cos = compute_random_baseline_cosine(all_acts, n_inputs, n_permutations=100)
        random_cosine_baseline.append(random_cos)
    
    print(f"Aggregate analysis complete.")
    print(f"Random baseline controls computed (100 permutations per layer).")
    
    # Plotting - save individual plots
    print(f"\n{'='*60}")
    print("CREATING VISUALIZATIONS")
    print(f"{'='*60}")
    
    # Group results by relation type for cleaner plotting
    relation_types = list(set(r['task'] for r in all_results))
    results_by_relation = {rel: [r for r in all_results if r['task'] == rel] for rel in relation_types}
    
    # Compute average per relation type
    avg_by_relation = {}
    for rel, rels in results_by_relation.items():
        avg_by_relation[rel] = {
            'clustering': np.mean([r['clustering_scores'] for r in rels], axis=0),
            'separability': np.mean([r['separability_scores'] for r in rels], axis=0),
            'cosine_within': np.mean([r['cosine_within'] for r in rels], axis=0),
            'cosine_between': np.mean([r['cosine_between'] for r in rels], axis=0),
            'query_prob': np.mean([r['final_query_prob'] for r in rels], axis=0),
            'output_prob': np.mean([r['final_output_prob'] for r in rels], axis=0),
        }
    
    # Plot 1: Clustering scores WITH RANDOM BASELINE
    fig, ax = plt.subplots(figsize=(8, 6))
    for rel in relation_types:
        ax.plot(avg_by_relation[rel]['clustering'], alpha=0.5, label=rel)
    
    # Plot random baseline (mean +/- 2 std)
    random_mean = [r['mean'] for r in random_clustering_baseline]
    random_std = [r['std'] for r in random_clustering_baseline]
    ax.plot(random_mean, 'k--', linewidth=2, label='Random split (mean)')
    ax.fill_between(range(n_layers), 
                    [m - 2*s for m, s in zip(random_mean, random_std)],
                    [m + 2*s for m, s in zip(random_mean, random_std)],
                    color='gray', alpha=0.3, label='Random split (±2 std)')
    
    ax.set_xlabel('Layer')
    ax.set_ylabel('Clustering Score (between/within variance)')
    ax.set_title(f'Role Clustering Across Layers\n({len(all_results)} prompts, {sum(r["n_inputs"] for r in all_results)} input tokens)')
    ax.legend(fontsize=7, loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plot1_clustering.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: plot1_clustering.png")
    
    # Plot 2: Linear separability with CV
    fig, ax = plt.subplots(figsize=(8, 6))
    for rel in relation_types:
        ax.plot(avg_by_relation[rel]['separability'], alpha=0.5, label=rel)
    ax.plot(aggregate_separability, 'k-', linewidth=2.5, label='All combined')
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Chance')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Cross-Validated Accuracy')
    ax.set_title('Linear Separability of Roles (Cross-Validated)')
    ax.set_ylim([0.4, 1.05])
    ax.legend(fontsize=8, loc='lower right', ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plot2_separability.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: plot2_separability.png")
    
    # Plot 3: Cosine similarity (within vs between roles) WITH RANDOM BASELINE
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Actual gap
    cos_gap = np.array(aggregate_cosine_within) - np.array(aggregate_cosine_between)
    ax.plot(cos_gap, 'b-', linewidth=2, label='Role-based gap (within - between)')
    
    # Random baseline gap
    random_gap_mean = [r['mean'] for r in random_cosine_baseline]
    random_gap_std = [r['std'] for r in random_cosine_baseline]
    ax.plot(random_gap_mean, 'k--', linewidth=2, label='Random split gap (mean)')
    ax.fill_between(range(n_layers),
                    [m - 2*s for m, s in zip(random_gap_mean, random_gap_std)],
                    [m + 2*s for m, s in zip(random_gap_mean, random_gap_std)],
                    color='gray', alpha=0.3, label='Random split (±2 std)')
    
    ax.axhline(y=0, color='r', linestyle=':', alpha=0.5)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Cosine Similarity Gap (within - between)')
    ax.set_title('Cosine Similarity Gap: Role-Based vs Random Split')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plot3_cosine_similarity.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: plot3_cosine_similarity.png")
    
    # Plot 4: P(query) vs P(output) at final position - PANEL per relation type
    n_relations = len(relation_types)
    n_cols = 3
    n_rows = (n_relations + n_cols - 1) // n_cols  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
    axes = axes.flatten() if n_relations > 1 else [axes]
    
    sorted_relations = sorted(relation_types)
    
    for idx, rel in enumerate(sorted_relations):
        ax = axes[idx]
        query_prob = avg_by_relation[rel]['query_prob']
        output_prob = avg_by_relation[rel]['output_prob']
        
        ax.plot(query_prob, 'b--', linewidth=2, label='P(query)')
        ax.plot(output_prob, 'g-', linewidth=2, label='P(output)')
        
        # Find and mark crossover point (if any)
        crossover_layers = np.where(np.array(query_prob) > np.array(output_prob))[0]
        if len(crossover_layers) > 0:
            first_crossover = crossover_layers[0]
            ax.axvline(x=first_crossover, color='r', linestyle=':', alpha=0.5)
            ax.set_title(f'{rel}\n(P(query)>P(output) at L{first_crossover})', fontsize=10)
        else:
            ax.set_title(f'{rel}\n(P(output) always > P(query))', fontsize=10)
        
        ax.set_xlabel('Layer', fontsize=9)
        ax.set_ylabel('Probability', fontsize=9)
        ax.set_yscale('log')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
    
    # Hide unused subplots
    for idx in range(n_relations, len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle('Argument Formation: P(query) vs P(output) at Final Position', fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig('plot4_argument_formation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: plot4_argument_formation.png")
    
    # Plot 5: Final position relative distance to input vs output centroids
    fig, ax = plt.subplots(figsize=(8, 6))
    
    all_relative_distances = []
    for r in all_results:
        all_relative_distances.append(r['final_relative_position'])
    
    # Plot per relation type
    for rel in relation_types:
        rel_results = [r for r in all_results if r['task'] == rel]
        rel_avg = np.mean([r['final_relative_position'] for r in rel_results], axis=0)
        ax.plot(rel_avg, alpha=0.5, label=rel)
    
    # Plot overall average
    overall_avg = np.mean(all_relative_distances, axis=0)
    ax.plot(overall_avg, 'k-', linewidth=2.5, label='All combined')
    
    ax.axhline(y=0, color='b', linestyle='--', alpha=0.5, label='Input centroid')
    ax.axhline(y=1, color='g', linestyle='--', alpha=0.5, label='Output centroid')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Relative Position\n(0=input centroid, 1=output centroid)')
    ax.set_title('Final Token: Position Relative to Role Centroids')
    ax.legend(fontsize=8, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plot5_final_position_distance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: plot5_final_position_distance.png")
    
    # Plot 6: PCA visualization at key layers
    fig, ax = plt.subplots(figsize=(8, 6))
    layers_to_plot = [0, n_layers//2, n_layers-1]
    colors = ['red', 'green', 'blue']
    
    for idx, layer in enumerate(layers_to_plot):
        all_inputs = np.vstack(all_input_acts_by_layer[layer])
        all_outputs = np.vstack(all_output_acts_by_layer[layer])
        
        combined = np.vstack([all_inputs, all_outputs])
        pca = PCA(n_components=2)
        projected = pca.fit_transform(combined)
        
        n_in = len(all_inputs)
        inputs_pca = projected[:n_in]
        outputs_pca = projected[n_in:]
        
        offset = idx * 3
        ax.scatter(inputs_pca[:, 0], inputs_pca[:, 1] + offset, 
                  c=colors[idx], marker='o', alpha=0.5, s=30,
                  label=f'L{layer} inputs')
        ax.scatter(outputs_pca[:, 0], outputs_pca[:, 1] + offset,
                  c=colors[idx], marker='x', alpha=0.5, s=30,
                  label=f'L{layer} outputs')
    
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2 (offset by layer)')
    ax.set_title('PCA: Inputs (o) vs Outputs (x)\nat layers 0, mid, final')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plot6_pca.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: plot6_pca.png")
    
    # Plot 7: Summary metrics
    fig, ax = plt.subplots(figsize=(8, 6))
    clust_norm = (np.array(aggregate_clustering) - min(aggregate_clustering)) / (max(aggregate_clustering) - min(aggregate_clustering) + 1e-10)
    sep_norm = (np.array(aggregate_separability) - 0.5) / 0.5
    cos_gap = np.array(aggregate_cosine_within) - np.array(aggregate_cosine_between)
    cos_gap_norm = (cos_gap - min(cos_gap)) / (max(cos_gap) - min(cos_gap) + 1e-10)
    
    ax.plot(clust_norm, 'b-', linewidth=2, label='Clustering (normalized)')
    ax.plot(sep_norm, 'g-', linewidth=2, label='Separability (above chance)')
    ax.plot(cos_gap_norm, 'r-', linewidth=2, label='Cosine gap (normalized)')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Normalized Score')
    ax.set_title('Summary: All Role Geometry Metrics (Normalized)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plot7_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: plot7_summary.png")
    
    # Print summary
    print(f"\n\n{'='*60}")
    print("SUMMARY OF RESULTS")
    print(f"{'='*60}")
    
    print(f"\nData: {len(all_results)} prompts, {sum(r['n_inputs'] for r in all_results)} total input tokens, {sum(r['n_outputs'] for r in all_results)} total output tokens")
    
    # Accuracy by relation type
    print("\n" + "="*60)
    print("MODEL ACCURACY BY RELATION TYPE")
    print("="*60)
    print(f"{'Relation':<15} {'Correct':<10} {'Total':<10} {'Accuracy':<10}")
    print("-"*45)
    for rel in sorted(relation_types):
        rel_results = [r for r in all_results if r['task'] == rel]
        n_correct = sum(1 for r in rel_results if r['correct'])
        print(f"{rel:<15} {n_correct:<10} {len(rel_results):<10} {100*n_correct/len(rel_results):.0f}%")
    total_correct = sum(1 for r in all_results if r['correct'])
    print("-"*45)
    print(f"{'TOTAL':<15} {total_correct:<10} {len(all_results):<10} {100*total_correct/len(all_results):.0f}%")
    
    # Table: Layer-by-layer metrics with baselines
    print("\n" + "="*60)
    print("LAYER-BY-LAYER METRICS (with random baselines)")
    print("="*60)
    print(f"{'Layer':<6} {'Clustering':<12} {'Rand Clust':<12} {'Cos Gap':<10} {'Rand Gap':<10} {'Final Pos':<10}")
    print("-"*60)
    
    avg_final_pos = np.mean([r['final_relative_position'] for r in all_results], axis=0)
    cos_gap = np.array(aggregate_cosine_within) - np.array(aggregate_cosine_between)
    
    for layer in range(n_layers):
        rand_clust = random_clustering_baseline[layer]['mean']
        rand_gap = random_cosine_baseline[layer]['mean']
        print(f"{layer:<6} {aggregate_clustering[layer]:<12.3f} {rand_clust:<12.3f} {cos_gap[layer]:<10.4f} {rand_gap:<10.4f} {avg_final_pos[layer]:<10.3f}")
    
    # Summary statistics by layer region
    print("\n" + "="*60)
    print("SUMMARY BY LAYER REGION")
    print("="*60)
    
    print("\n1. ROLE CLUSTERING (higher = better separation):")
    early = np.mean(aggregate_clustering[:n_layers//3])
    mid = np.mean(aggregate_clustering[n_layers//3:2*n_layers//3])
    late = np.mean(aggregate_clustering[2*n_layers//3:])
    peak_layer = np.argmax(aggregate_clustering)
    rand_mean_overall = np.mean([r['mean'] for r in random_clustering_baseline])
    rand_std_overall = np.mean([r['std'] for r in random_clustering_baseline])
    print(f"   Early layers (0-{n_layers//3-1}): {early:.3f}")
    print(f"   Middle layers ({n_layers//3}-{2*n_layers//3-1}): {mid:.3f}")
    print(f"   Late layers ({2*n_layers//3}-{n_layers-1}): {late:.3f}")
    print(f"   Peak at layer: {peak_layer}")
    print(f"   Random baseline: {rand_mean_overall:.3f} ± {rand_std_overall:.3f}")
    
    # Statistical test: how many std above random?
    peak_clust = aggregate_clustering[peak_layer]
    peak_rand_mean = random_clustering_baseline[peak_layer]['mean']
    peak_rand_std = random_clustering_baseline[peak_layer]['std']
    z_score = (peak_clust - peak_rand_mean) / (peak_rand_std + 1e-10)
    print(f"   Peak clustering is {z_score:.1f} std above random baseline")
    
    print("\n2. COSINE SIMILARITY GAP (within - between, higher = more clustered):")
    cos_gap = np.array(aggregate_cosine_within) - np.array(aggregate_cosine_between)
    early = np.mean(cos_gap[:n_layers//3])
    mid = np.mean(cos_gap[n_layers//3:2*n_layers//3])
    late = np.mean(cos_gap[2*n_layers//3:])
    rand_gap_mean = np.mean([r['mean'] for r in random_cosine_baseline])
    rand_gap_std = np.mean([r['std'] for r in random_cosine_baseline])
    print(f"   Early layers: {early:.4f}")
    print(f"   Middle layers: {mid:.4f}")
    print(f"   Late layers: {late:.4f}")
    print(f"   Random baseline: {rand_gap_mean:.4f} ± {rand_gap_std:.4f}")
    
    # Statistical test for cosine gap
    peak_cos_layer = np.argmax(cos_gap)
    peak_cos = cos_gap[peak_cos_layer]
    peak_cos_rand_mean = random_cosine_baseline[peak_cos_layer]['mean']
    peak_cos_rand_std = random_cosine_baseline[peak_cos_layer]['std']
    z_score_cos = (peak_cos - peak_cos_rand_mean) / (peak_cos_rand_std + 1e-10)
    print(f"   Peak gap at layer {peak_cos_layer} is {z_score_cos:.1f} std above random baseline")
    
    print("\n3. FINAL POSITION RELATIVE DISTANCE (0=input, 1=output):")
    early = np.mean(avg_final_pos[:n_layers//3])
    mid = np.mean(avg_final_pos[n_layers//3:2*n_layers//3])
    late = np.mean(avg_final_pos[2*n_layers//3:])
    print(f"   Early layers: {early:.3f}")
    print(f"   Middle layers: {mid:.3f}")
    print(f"   Late layers: {late:.3f}")
    
    print("\n4. ARGUMENT FORMATION (P(query) vs P(output) at final position):")
    avg_query = np.mean([r['final_query_prob'] for r in all_results], axis=0)
    avg_output = np.mean([r['final_output_prob'] for r in all_results], axis=0)
    query_peak = np.argmax(avg_query)
    output_peak = np.argmax(avg_output)
    crossover = np.argmax(avg_output > avg_query) if np.any(avg_output > avg_query) else -1
    print(f"   P(query) peaks at layer: {query_peak}")
    print(f"   P(output) peaks at layer: {output_peak}")
    print(f"   P(output) > P(query) starting at layer: {crossover}")
    
    print("\n" + "="*60)
    print("HYPOTHESIS 1 ASSESSMENT")
    print("="*60)
    
    # Check if clustering is significantly above random
    if z_score > 2:
        print(f"   [✓] Role clustering significantly exceeds random baseline ({z_score:.1f} std)")
    else:
        print(f"   [✗] Role clustering NOT significantly above random ({z_score:.1f} std)")
    
    if peak_layer < n_layers // 3:
        print("   [✓] Role clustering peaks EARLY - supports H1")
    elif peak_layer < 2 * n_layers // 3:
        print("   [~] Role clustering peaks in MIDDLE layers - partial support")
    else:
        print("   [✗] Role clustering peaks LATE - contradicts H1")
    
    if z_score_cos > 2:
        print(f"   [✓] Cosine gap significantly exceeds random baseline ({z_score_cos:.1f} std)")
    else:
        print(f"   [✗] Cosine gap NOT significantly above random ({z_score_cos:.1f} std)")
    
    if query_peak < output_peak:
        print("   [✓] Query info arrives before output - supports argument formation")
    else:
        print("   [?] No clear argument formation stage")
    
    return all_results, {
        'aggregate_clustering': aggregate_clustering,
        'aggregate_separability': aggregate_separability,
        'aggregate_cosine_within': aggregate_cosine_within,
        'aggregate_cosine_between': aggregate_cosine_between,
        'random_clustering_baseline': random_clustering_baseline,
        'random_cosine_baseline': random_cosine_baseline,
    }


if __name__ == "__main__":
    results, aggregates = run_analysis("meta-llama/Llama-3.2-1B")