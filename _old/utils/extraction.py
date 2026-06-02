"""Extract hidden states, head outputs, and attention patterns from model runs."""
import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from .prompts import find_role_positions


@torch.no_grad()
def extract_hidden_states(model, prompts_data, positions='last'):
    """Extract residual stream vectors at specified positions.

    Args:
        model: HookedTransformer
        prompts_data: list of dicts with 'prompt' key (and optionally 'demo_pairs', 'query_input')
        positions: 'last' | 'all' | 'by_role' (requires demo_pairs and query_input)

    Returns:
        dict mapping layer -> np.array of shape (n_prompts, d_model) if positions='last'
        or more complex structure for other modes
    """
    n_layers = model.cfg.n_layers
    device = next(model.parameters()).device
    filter_fn = lambda name: "resid_post" in name

    if positions == 'last':
        vecs = defaultdict(list)
        for pdata in tqdm(prompts_data, desc="Extracting hidden states"):
            prompt = pdata['prompt']
            tokens = model.to_tokens(prompt, prepend_bos=True)
            last_pos = tokens.shape[1] - 1
            _, cache = model.run_with_cache(tokens, names_filter=filter_fn)
            for layer in range(n_layers):
                h = cache['resid_post', layer][0, last_pos].cpu().float().numpy()
                vecs[layer].append(h)
            del cache
            torch.cuda.empty_cache()
        return {l: np.stack(v) for l, v in vecs.items()}

    elif positions == 'by_role':
        last_vecs = defaultdict(list)
        input_vecs = defaultdict(list)
        output_vecs = defaultdict(list)
        sep_vecs = defaultdict(list)

        for pdata in tqdm(prompts_data, desc="Extracting by role"):
            prompt = pdata['prompt']
            tokens = model.to_tokens(prompt, prepend_bos=True)
            last_pos = tokens.shape[1] - 1
            roles = find_role_positions(model, prompt, pdata['demo_pairs'], pdata['query_input'])

            _, cache = model.run_with_cache(tokens, names_filter=filter_fn)
            for layer in range(n_layers):
                h = cache['resid_post', layer][0].cpu().float()
                last_vecs[layer].append(h[last_pos].numpy())
                if roles['input_positions']:
                    input_vecs[layer].append(h[roles['input_positions']].mean(0).numpy())
                if roles['output_positions']:
                    output_vecs[layer].append(h[roles['output_positions']].mean(0).numpy())
                if roles['separator_positions']:
                    sep_vecs[layer].append(h[roles['separator_positions']].mean(0).numpy())
            del cache
            torch.cuda.empty_cache()

        return {
            'last': {l: np.stack(v) for l, v in last_vecs.items()},
            'input': {l: np.stack(v) for l, v in input_vecs.items()},
            'output': {l: np.stack(v) for l, v in output_vecs.items()},
            'separator': {l: np.stack(v) for l, v in sep_vecs.items()},
        }
    else:
        raise ValueError(f"Unknown positions mode: {positions}")


@torch.no_grad()
def extract_head_outputs(model, prompts_data, position='last'):
    """Extract attention head outputs (hook_z) at specified position.

    Returns:
        mean_z: tensor of shape (n_layers, n_heads, d_head), averaged across prompts
        all_z: list of tensors if return_all=True
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    d_head = model.cfg.d_head
    filter_fn = lambda name: "hook_z" in name

    accum = torch.zeros(n_layers, n_heads, d_head)
    n = 0

    for pdata in tqdm(prompts_data, desc="Extracting head outputs"):
        prompt = pdata['prompt']
        tokens = model.to_tokens(prompt, prepend_bos=True)
        last_pos = tokens.shape[1] - 1
        _, cache = model.run_with_cache(tokens, names_filter=filter_fn)

        for layer in range(n_layers):
            z = cache[f'blocks.{layer}.attn.hook_z'][0, last_pos]  # (n_heads, d_head)
            accum[layer] += z.cpu().float()

        n += 1
        del cache
        torch.cuda.empty_cache()

    return accum / n


@torch.no_grad()
def extract_head_outputs_at_positions(model, prompts_data, position_type='output'):
    """Extract head outputs at specific position types (output tokens, separators, etc).

    Args:
        position_type: 'output' | 'separator' | 'input'

    Returns:
        Per-prompt list of (n_layers, n_positions, n_heads, d_head) tensors
    """
    n_layers = model.cfg.n_layers
    filter_fn = lambda name: "hook_z" in name
    all_outputs = []

    for pdata in tqdm(prompts_data, desc=f"Head outputs at {position_type}"):
        prompt = pdata['prompt']
        tokens = model.to_tokens(prompt, prepend_bos=True)
        roles = find_role_positions(model, prompt, pdata['demo_pairs'], pdata['query_input'])
        positions = roles[f'{position_type}_positions']

        _, cache = model.run_with_cache(tokens, names_filter=filter_fn)
        prompt_z = []
        for layer in range(n_layers):
            z = cache[f'blocks.{layer}.attn.hook_z'][0]  # (seq, n_heads, d_head)
            prompt_z.append(z[positions].cpu().float())  # (n_pos, n_heads, d_head)
        all_outputs.append(prompt_z)
        del cache
        torch.cuda.empty_cache()

    return all_outputs


@torch.no_grad()
def extract_attention_patterns(model, prompts_data):
    """Extract attention from last position to input/output/separator positions.

    Returns:
        attn_to_input: (n_layers, n_heads), averaged across prompts
        attn_to_output: (n_layers, n_heads)
        attn_to_sep: (n_layers, n_heads)
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    filter_fn = lambda name: "pattern" in name

    attn_to_input = np.zeros((n_layers, n_heads))
    attn_to_output = np.zeros((n_layers, n_heads))
    attn_to_sep = np.zeros((n_layers, n_heads))
    n = 0

    for pdata in tqdm(prompts_data, desc="Attention patterns"):
        prompt = pdata['prompt']
        tokens = model.to_tokens(prompt, prepend_bos=True)
        last_pos = tokens.shape[1] - 1
        roles = find_role_positions(model, prompt, pdata['demo_pairs'], pdata['query_input'])

        _, cache = model.run_with_cache(tokens, names_filter=filter_fn)
        for layer in range(n_layers):
            pat = cache['pattern', layer][0].cpu().float().numpy()  # (n_heads, seq, seq)
            row = pat[:, last_pos, :]  # (n_heads, seq)
            in_idx = roles['input_positions']
            out_idx = roles['output_positions']
            sep_idx = roles['separator_positions']
            if in_idx:
                attn_to_input[layer] += row[:, in_idx].sum(axis=1)
            if out_idx:
                attn_to_output[layer] += row[:, out_idx].sum(axis=1)
            if sep_idx:
                attn_to_sep[layer] += row[:, sep_idx].sum(axis=1)
        n += 1
        del cache
        torch.cuda.empty_cache()

    return attn_to_input / n, attn_to_output / n, attn_to_sep / n
