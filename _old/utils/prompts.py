"""Prompt construction and token position identification."""
from configs.defaults import SEPARATOR, DEMO_SEP


def build_icl_prompt(demo_pairs, query_input):
    """Build: inp1 → out1\\ninp2 → out2\\n...\\nquery →"""
    lines = [f"{inp}{SEPARATOR} {out}" for inp, out in demo_pairs]
    lines.append(f"{query_input}{SEPARATOR}")
    return DEMO_SEP.join(lines)


def build_zero_shot_prompt(query_input):
    """Build: query →"""
    return f"{query_input}{SEPARATOR}"


def find_role_positions(model, prompt, demo_pairs, query_input):
    """Find token positions corresponding to input words, output words, and separators.

    Returns dict with keys:
        input_positions: list of int
        output_positions: list of int
        separator_positions: list of int (the → tokens)
    """
    input_pos = []
    output_pos = []
    separator_pos = []
    current_text = ""

    for inp, out in demo_pairs:
        prefix_len = len(model.to_tokens(current_text, prepend_bos=True)[0]) if current_text else 1

        # Input tokens
        with_input = current_text + inp
        with_input_len = len(model.to_tokens(with_input, prepend_bos=True)[0])
        input_pos.extend(range(prefix_len, with_input_len))

        # Separator tokens
        with_sep = with_input + SEPARATOR
        with_sep_len = len(model.to_tokens(with_sep, prepend_bos=True)[0])
        separator_pos.extend(range(with_input_len, with_sep_len))

        # Output tokens
        with_output = with_sep + " " + out
        with_output_len = len(model.to_tokens(with_output, prepend_bos=True)[0])
        output_pos.extend(range(with_sep_len, with_output_len))

        current_text = with_output + DEMO_SEP

    # Final query separator
    full = current_text + query_input + SEPARATOR
    full_len = len(model.to_tokens(full, prepend_bos=True)[0])
    # The last separator is the final → (query position)
    query_sep_start = len(model.to_tokens(current_text + query_input, prepend_bos=True)[0])
    separator_pos.extend(range(query_sep_start, full_len))

    return {
        'input_positions': input_pos,
        'output_positions': output_pos,
        'separator_positions': separator_pos,
    }


def find_per_demo_positions(model, prompt, demo_pairs):
    """Find positions for EACH demonstration separately.

    Returns list of dicts, one per demo, each with:
        input_positions, output_positions, separator_positions
    """
    demos = []
    current_text = ""

    for inp, out in demo_pairs:
        prefix_len = len(model.to_tokens(current_text, prepend_bos=True)[0]) if current_text else 1
        demo_info = {}

        with_input = current_text + inp
        with_input_len = len(model.to_tokens(with_input, prepend_bos=True)[0])
        demo_info['input_positions'] = list(range(prefix_len, with_input_len))

        with_sep = with_input + SEPARATOR
        with_sep_len = len(model.to_tokens(with_sep, prepend_bos=True)[0])
        demo_info['separator_positions'] = list(range(with_input_len, with_sep_len))

        with_output = with_sep + " " + out
        with_output_len = len(model.to_tokens(with_output, prepend_bos=True)[0])
        demo_info['output_positions'] = list(range(with_sep_len, with_output_len))

        demos.append(demo_info)
        current_text = with_output + DEMO_SEP

    return demos
