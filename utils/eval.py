"""Evaluation (correctness checking) and patching/hooking utilities."""
import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm


def check_correct(model, pred_logits, target_output):
    """Check if top-1 prediction matches target (first token)."""
    pred_id = pred_logits.argmax().item()
    target_tokens = model.tokenizer.encode(" " + target_output)
    if target_tokens and pred_id == target_tokens[0]:
        return True
    pred_str = model.tokenizer.decode([pred_id]).strip().lower()
    target_str = target_output.strip().lower()
    if pred_str and target_str and (pred_str.startswith(target_str[:3]) or target_str.startswith(pred_str[:3])):
        return True
    return False


@torch.no_grad()
def eval_patched_resid(model, zs_prompt, theta, layer):
    """Patch last-token residual stream at layer with theta (replacement)."""
    device = next(model.parameters()).device
    tokens = model.to_tokens(zs_prompt, prepend_bos=True)
    theta_t = torch.tensor(theta, device=device, dtype=model.cfg.dtype)

    def hook(value, hook):
        value[0, -1, :] = theta_t
        return value

    logits = model.run_with_hooks(
        tokens, fwd_hooks=[(f'blocks.{layer}.hook_resid_post', hook)]
    )
    return logits[0, -1]


@torch.no_grad()
def eval_patched_resid_add(model, zs_prompt, vector, layer):
    """Add vector to last-token residual stream at layer (addition, not replacement)."""
    device = next(model.parameters()).device
    tokens = model.to_tokens(zs_prompt, prepend_bos=True)
    vec_t = torch.tensor(vector, device=device, dtype=model.cfg.dtype)

    def hook(value, hook):
        value[0, -1, :] += vec_t
        return value

    logits = model.run_with_hooks(
        tokens, fwd_hooks=[(f'blocks.{layer}.hook_resid_post', hook)]
    )
    return logits[0, -1]


@torch.no_grad()
def eval_with_head_replace(model, zs_prompt, mean_z, heads):
    """Replace specific heads' outputs at their native layers with mean ICL outputs.

    Args:
        mean_z: tensor (n_layers, n_heads, d_head) — mean head outputs from ICL
        heads: list of (layer, head) tuples
    """
    device = next(model.parameters()).device
    tokens = model.to_tokens(zs_prompt, prepend_bos=True)
    mean_z_device = mean_z.to(device).to(model.cfg.dtype)

    heads_by_layer = defaultdict(list)
    for (l, h) in heads:
        heads_by_layer[l].append(h)

    fwd_hooks = []
    for l, h_list in heads_by_layer.items():
        mz = mean_z_device[l]
        def hook(z, hook, _h_list=h_list, _mz=mz):
            for h in _h_list:
                z[0, -1, h, :] = _mz[h]
            return z
        fwd_hooks.append((f'blocks.{l}.attn.hook_z', hook))

    logits = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)
    return logits[0, -1]


@torch.no_grad()
def eval_with_head_ablation(model, prompt, heads):
    """Zero-ablate specific heads during a forward pass.

    Args:
        prompt: string (can be ICL or zero-shot)
        heads: list of (layer, head) tuples
    """
    tokens = model.to_tokens(prompt, prepend_bos=True)
    heads_by_layer = defaultdict(list)
    for (l, h) in heads:
        heads_by_layer[l].append(h)

    fwd_hooks = []
    for l, h_list in heads_by_layer.items():
        def hook(z, hook, _h_list=h_list):
            for h in _h_list:
                z[0, :, h, :] = 0
            return z
        fwd_hooks.append((f'blocks.{l}.attn.hook_z', hook))

    logits = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)
    return logits[0, -1]


def compute_task_vectors(hidden_states_by_layer):
    """Average hidden states across prompts to form task vectors.

    Args:
        hidden_states_by_layer: dict layer -> np.array (n_prompts, d_model)
    Returns:
        dict layer -> np.array (d_model,)
    """
    return {l: v.mean(axis=0) for l, v in hidden_states_by_layer.items()}


def run_patching_sweep(model, eval_data, task_vectors, n_layers, check_fn=check_correct):
    """Sweep patching across all layers, return accuracy per layer."""
    accs = {}
    for layer in range(n_layers):
        theta = task_vectors[layer]
        correct = []
        for eq in eval_data:
            logits = eval_patched_resid(model, eq['zs_prompt'], theta, layer)
            correct.append(float(check_fn(model, logits, eq['query_output'])))
        accs[layer] = np.mean(correct)
    return accs
