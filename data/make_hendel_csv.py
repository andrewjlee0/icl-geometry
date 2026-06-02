"""Export the Hendel ICL splits to a CSV.

Each 10-shot prompt becomes one row with columns:
  - prompt_id            : unique row id
  - kind                 : 'icl' or 'query_only'
  - task                 : task name (e.g. 'translation/en_fr')
  - prompt_idx           : index of the prompt within the task
  - partner_id           : prompt_id of the paired row (an icl row's query-only partner, or vice versa)
  - input_1 ... input_10 : the demo inputs (empty for query-only rows)
  - output_1 ... output_10: the demo outputs (empty for query-only rows)
  - query_input          : the query word
  - query_output         : the expected answer
  - prompt               : the full constructed prompt string

Pairing rule for query-only partners: each ICL prompt at index i in a task is paired with
a query-only row whose query is taken from the ICL prompt at index (i+1) % N within the same
task. This guarantees the query-only's query is a held-out word relative to its partner.
"""
from __future__ import annotations

import csv
import pickle
from pathlib import Path

import sys; from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
_ROOT = _P(__file__).resolve().parents[1]
from configs import SEPARATOR, DEMO_SEP


def build_icl_prompt(demos, query):
    parts = [f'{i}{SEPARATOR} {o}' for i, o in demos]
    parts.append(f'{query}{SEPARATOR}')
    return DEMO_SEP.join(parts)


def main():
    from configs.defaults import HENDEL_SPLITS
    splits_path = HENDEL_SPLITS
    out_path = _ROOT / 'hendel_dataset.csv'

    with splits_path.open('rb') as f:
        all_splits = pickle.load(f)

    n_demos_max = max(
        len(p['demo_pairs'])
        for splits in all_splits.values()
        for p in splits['icl_prompts']
    )

    fieldnames = (
        ['prompt_id', 'kind', 'task', 'prompt_idx', 'partner_id']
        + [f'input_{i+1}' for i in range(n_demos_max)]
        + [f'output_{i+1}' for i in range(n_demos_max)]
        + ['query_input', 'query_output', 'prompt']
    )

    rows = []
    pid = 0

    # Pass 1: assign IDs to all (task, prompt_idx, kind) combos so we can fill partner_id.
    id_lookup = {}  # (task, prompt_idx, kind) -> pid
    for task in sorted(all_splits.keys()):
        prompts = all_splits[task]['icl_prompts']
        for i in range(len(prompts)):
            id_lookup[(task, i, 'icl')] = pid; pid += 1
            id_lookup[(task, i, 'query_only')] = pid; pid += 1

    # Pass 2: build rows.
    for task in sorted(all_splits.keys()):
        prompts = all_splits[task]['icl_prompts']
        n = len(prompts)
        for i, pdata in enumerate(prompts):
            partner_idx = (i + 1) % n
            partner_pdata = prompts[partner_idx]

            # ICL row
            icl_id = id_lookup[(task, i, 'icl')]
            qo_id = id_lookup[(task, i, 'query_only')]

            row_icl = {
                'prompt_id': icl_id,
                'kind': 'icl',
                'task': task,
                'prompt_idx': i,
                'partner_id': qo_id,
                'query_input': pdata['query_input'],
                'query_output': pdata.get('query_output', ''),
                'prompt': pdata['prompt'],
            }
            for k, (inp, out) in enumerate(pdata['demo_pairs']):
                row_icl[f'input_{k+1}'] = inp
                row_icl[f'output_{k+1}'] = out
            rows.append(row_icl)

            # Query-only partner row: query taken from a DIFFERENT prompt's query.
            qo_query_input = partner_pdata['query_input']
            qo_query_output = partner_pdata.get('query_output', '')
            qo_prompt = f'{qo_query_input}{SEPARATOR}'

            row_qo = {
                'prompt_id': qo_id,
                'kind': 'query_only',
                'task': task,
                'prompt_idx': i,
                'partner_id': icl_id,
                'query_input': qo_query_input,
                'query_output': qo_query_output,
                'prompt': qo_prompt,
            }
            rows.append(row_qo)

    # Write CSV.
    with out_path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

    n_icl = sum(1 for r in rows if r['kind'] == 'icl')
    n_qo = sum(1 for r in rows if r['kind'] == 'query_only')
    print(f'Wrote {len(rows)} rows ({n_icl} icl, {n_qo} query_only) to {out_path}')


if __name__ == '__main__':
    main()
