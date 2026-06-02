"""Prompt-string construction (the 'data' side of prompts).

Position-finding (which token is which role) lives in utils/positions.py since
that is experiment-analysis machinery, not dataset construction.
"""
from configs.defaults import SEPARATOR, DEMO_SEP


def build_icl_prompt(demo_pairs, query_input):
    """Build: inp1 → out1\\ninp2 → out2\\n...\\nquery →"""
    lines = [f"{inp}{SEPARATOR} {out}" for inp, out in demo_pairs]
    lines.append(f"{query_input}{SEPARATOR}")
    return DEMO_SEP.join(lines)


def build_zero_shot_prompt(query_input):
    """Build: query →"""
    return f"{query_input}{SEPARATOR}"
