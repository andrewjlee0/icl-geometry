"""Token-position identification: which positions are inputs, outputs, separators.

This is experiment-analysis machinery (used by extraction and head scoring), so it
lives in utils/ rather than data/. Prompt *construction* is in data/prompts.py.
"""
from configs.defaults import SEPARATOR, DEMO_SEP


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


def find_per_demo_positions_robust(model, prompt, demos, sep_char='\u2192'):
    """Token positions of each demo's input/output spans, via char->token map.

    More robust than incremental retokenization for multi-token / no-leading-space
    outputs (nonce words, numbers). Returns list of dicts with
    'input_positions' and 'output_positions'.
    """
    tokens = model.to_tokens(prompt, prepend_bos=True)[0]
    full_decoded = model.tokenizer.decode(tokens)
    char_to_tok = []
    for i in range(len(tokens)):
        s = model.tokenizer.decode([tokens[i].item()])
        char_to_tok.extend([i] * len(s))

    def span_to_toks(start, end):
        if start < 0 or end > len(char_to_tok):
            return []
        return sorted(set(char_to_tok[start:end]))

    result = []
    search_from = 0
    for inp, out in demos:
        demo = {'input_positions': [], 'output_positions': []}
        inp_s, out_s = str(inp), str(out)
        idx = full_decoded.find(inp_s, search_from)
        if idx >= 0:
            demo['input_positions'] = span_to_toks(idx, idx + len(inp_s))
            search_from = idx + len(inp_s)
        arrow = full_decoded.find(sep_char, search_from)
        if arrow >= 0:
            out_char = full_decoded.find(out_s, arrow)
            if out_char >= 0:
                demo['output_positions'] = span_to_toks(out_char, out_char + len(out_s))
                search_from = out_char + len(out_s)
        result.append(demo)
    return result


def find_slot_positions(model, prompt, demos, query_input=None, sep_char='→'):
    """Full slot taxonomy for attention-knockout experiments.

    Extends find_per_demo_positions_robust with the structural tokens the original
    knockout omitted, so the source/target grid is genuinely exhaustive:

      per_demo : list of {'in', 'out', 'arrow', 'nl'} token-position lists per demo
      in_all   : all demo input positions
      out_all  : all demo output positions
      arrow_all: all arrow (separator) positions
      nl_all   : all newline (DEMO_SEP) positions  <-- previously missing
      bos      : [0]
      query    : query-input token positions (if query_input given / found)
      final    : last position (seq_len - 1)
      seq_len  : sequence length

    Newlines are located as the token positions whose decoded text contains the
    DEMO_SEP character, excluding any already claimed by input/output/arrow spans.
    """
    from .positions import find_per_demo_positions_robust  # same module; explicit
    toks = model.to_tokens(prompt, prepend_bos=True)[0]
    seq_len = len(toks)
    per_demo_io = find_per_demo_positions_robust(model, prompt, demos, sep_char=sep_char)

    # arrows: positions strictly between a demo's last input and first output
    per_demo = []
    for d in per_demo_io:
        ins = d.get('input_positions', [])
        outs = d.get('output_positions', [])
        arrow = list(range(max(ins) + 1, min(outs))) if (ins and outs) else []
        per_demo.append({'in': ins, 'out': outs, 'arrow': arrow, 'nl': []})

    # newlines: any token whose decoded string contains the separator newline char
    nl_char = DEMO_SEP if DEMO_SEP.strip('\n') == '' else '\n'
    claimed = set()
    for d in per_demo:
        claimed.update(d['in']); claimed.update(d['out']); claimed.update(d['arrow'])
    nl_all = []
    for i in range(seq_len):
        s = model.tokenizer.decode([toks[i].item()])
        if ('\n' in s) and (i not in claimed):
            nl_all.append(i)
    # assign each newline to the demo it terminates (nearest preceding output)
    for nl in nl_all:
        best = None
        for d_idx, d in enumerate(per_demo):
            if d['out'] and max(d['out']) < nl:
                best = d_idx
        if best is not None:
            per_demo[best]['nl'].append(nl)

    # query positions
    query_pos = []
    if query_input is not None:
        full = model.tokenizer.decode(toks)
        last_arrow = full.rfind(sep_char)
        qi = full.rfind(str(query_input), 0, last_arrow if last_arrow > 0 else None)
        if qi >= 0:
            char_to_tok = []
            for i in range(seq_len):
                char_to_tok.extend([i] * len(model.tokenizer.decode([toks[i].item()])))
            for ci in range(qi, min(qi + len(str(query_input)), len(char_to_tok))):
                query_pos.append(char_to_tok[ci])
            query_pos = sorted(set(query_pos))

    return {
        'per_demo': per_demo,
        'in_all': sorted({p for d in per_demo for p in d['in']}),
        'out_all': sorted({p for d in per_demo for p in d['out']}),
        'arrow_all': sorted({p for d in per_demo for p in d['arrow']}),
        'nl_all': sorted(nl_all),
        'bos': [0],
        'query': query_pos,
        'final': seq_len - 1,
        'seq_len': seq_len,
    }
