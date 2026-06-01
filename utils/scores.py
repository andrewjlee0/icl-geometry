"""Head scoring: induction scores, AIE, attention selectivity."""
import torch
import numpy as np
from tqdm import tqdm


@torch.no_grad()
def compute_induction_scores_rrt(model, seq_len=50, n_trials=10):
    """Standard induction + prev-token scores using repeated random tokens."""
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    device = next(model.parameters()).device
    filter_fn = lambda name: "pattern" in name

    induction = np.zeros((n_layers, n_heads))
    prev_token = np.zeros((n_layers, n_heads))

    for trial in tqdm(range(n_trials), desc="RRT trials"):
        rand_tokens = torch.randint(1000, 10000, (seq_len,))
        repeated = torch.cat([
            torch.tensor([model.tokenizer.bos_token_id]), rand_tokens, rand_tokens
        ]).unsqueeze(0).to(device)

        _, cache = model.run_with_cache(repeated, names_filter=filter_fn)
        for layer in range(n_layers):
            pat = cache['pattern', layer][0].cpu().float().numpy()
            for h in range(n_heads):
                induction[layer, h] += pat[h].diagonal(-seq_len + 1).mean()
                prev_token[layer, h] += pat[h].diagonal(-1).mean()
        del cache
        torch.cuda.empty_cache()

    return induction / n_trials, prev_token / n_trials


def compute_head_contributions(model, mean_head_outputs):
    """Compute W_O @ mean_z for all heads. Returns dict (layer, head) -> np.array (d_model,)."""
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    contribs = {}
    for layer in range(n_layers):
        W_O = model.blocks[layer].attn.W_O.detach().cpu().float()
        for h in range(n_heads):
            z_h = mean_head_outputs[layer, h].numpy()
            contribs[(layer, h)] = W_O[h].numpy().T @ z_h
    return contribs


def select_top_heads(scores, percentile=90):
    """Select heads above a percentile threshold. Returns list of (layer, head) tuples."""
    threshold = np.percentile(scores, percentile)
    n_layers, n_heads = scores.shape
    return [(l, h) for l in range(n_layers) for h in range(n_heads)
            if scores[l, h] >= threshold]
