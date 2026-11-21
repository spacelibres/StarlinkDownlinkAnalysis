"""Propose actual single-erasure values using frozen MCS 70 rows."""

from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from project_paths import experiment_output, require_run_context
from audit_mcs66_single_erasures import evaluate
from audit_qam_pn15 import equality_predictions
from audit_ut_repeated_data import checked_artifact, read
from receiver_models import pn_bits
from recover_mcs66_systematic_checks import packed_arrays


def repeated_terminal_links(data, rows):
    indices = {(r['frame'], r['resource']): i for i, r in enumerate(rows)}
    links = defaultdict(set)
    for pair in data['pairs']:
        if not pair['same_first_code5']:
            continue
        for allocation in pair['allocations']:
            if allocation['label'] != 70:
                continue
            for word in allocation['repeated_words']:
                keys = [(frame, word['resource']) for frame in pair['frames']]
                if all(key in indices for key in keys):
                    a, b = [indices[key] for key in keys]
                    links[a].add(b)
                    links[b].add(a)
    return links


def main():
    require_run_context()
    hashes = {}

    def artifact(run, exp, name):
        return checked_artifact(run, exp, name, hashes)

    a, b = [artifact('v118-mcs70-sparse'+s, 'E120', 'mcs70_sparse_checks.npz') for s in ('','-confirm')]
    if hashes[a] != hashes[b]:
        raise ValueError('Frozen check matrices differ')
    with np.load(a, allow_pickle=False) as z:
        support = np.stack([np.flatnonzero(row) for row in np.unpackbits(z['check_matrix_packed'], axis=1)])
    meta = read(artifact('v118-mcs70-terminal', 'E120', 'mcs70_terminal_words.json'))
    with np.load(artifact('v118-mcs70-terminal', 'E120', 'mcs70_terminal_words.npz'), allow_pickle=False) as z:
        bits, mask = packed_arrays(z, ('bits', 'mask'))
    pattern = equality_predictions(np.arange(meta['systematic_end']), pn_bits(), meta['pn_model'])
    repeats = read(artifact('v079-code5-gain-masked', 'E081', 'code5_gain_transfer.json'))
    partners = repeated_terminal_links(repeats, meta['rows'])
    rows, summary = evaluate(meta['rows'], bits, mask, support, pattern, partners)
    output = {'schema': 'starlink-mcs70-single-erasures/v1', 'summary': summary,
              'observed_repeat_pairs': sum(map(len, partners.values()))//2,
              'rows': rows, 'input_sha256': {p.as_posix(): h for p, h in hashes.items()}}
    out = experiment_output('E120')
    out.mkdir(parents=True, exist_ok=True)
    (out/'mcs70_erasures.json').write_text(json.dumps(output, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(summary, flush=True)


if __name__ == '__main__':
    main()
