"""Save observed MCS 70 terminal words without filling any erasures."""

from functools import lru_cache
import json
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from project_paths import experiment_output, require_run_context
from audit_code5_gain_transfer import decisions, four_corner_slots
from audit_pn_padding_starts import CacheOnly
from audit_qam_pn15 import equality_predictions
from audit_qin_exemplars import remove_header_gain
from audit_ut_repeated_data import checked_artifact, read
from educational_fec_demo import encode_block, word_bits
from extract_gain_padded_words import constellation_models
from extract_mcs66_padding_sweep import measure_tail
from layout_core import NONPILOT
from receiver_models import pn_bits, pn_reference
from snapshot_qin_exemplars import ETAG, SIZE, URL, as_complex


def selected_terminal_words(observations, allocations):
    result = []
    for obs, alloc in zip(observations, allocations, strict=True):
        if obs['start'] != alloc['start'] or obs['label'] != alloc['label']:
            raise ValueError('Archive observations and allocation geometry differ')
        index = alloc['count']-1
        if alloc['label'] == 70 and index in obs['masked_transfer_stop_words']:
            result.append((alloc, index))
    return result


def extract(source, gain, layout, models, pn):
    rotation = np.column_stack([pn_reference(int(k), np.arange(2, 302)) for k in NONPILOT]).conj()
    start = np.r_[word_bits(0x3048)[16:], np.zeros(16, np.uint8)]
    signs = 1-2*encode_block(start)[:48].astype(float)
    edge = layout['marker_systematic_ends']['70']
    rows, bits, masks = [], [], []

    @lru_cache(maxsize=1)
    def block(first):
        end = min(first+source.chunks[0], len(source))
        return as_complex(source[first:end, 1:, :])[..., NONPILOT % 1024]

    for row, geometry in zip(gain['frames'], layout['word_windows_and_onsets'], strict=True):
        frame = row['frame']
        if frame != geometry['frame']:
            raise ValueError('Archive frame order differs')
        targets = selected_terminal_words(row['allocations'], geometry['allocations'])
        if not targets:
            continue
        first = frame//source.chunks[0]*source.chunks[0]
        flat = remove_header_gain(block(first)[frame-first]*rotation, signs)
        axis = complex(*row['axis'])
        corners = four_corner_slots(flat, axis)
        for alloc, index in targets:
            begin = alloc['start'] + index*alloc['width_resources']
            rx, visible, _, _ = decisions(flat, axis, {**alloc, 'start': begin, 'count': 1}, models, edge, corners)
            y, mask = rx[0], visible[0]
            bits.append(np.packbits(y & mask))
            masks.append(np.packbits(mask))
            rows.append({'frame': frame, 'group': row['group'], 'calibration': row['calibration'],
                         'resource': begin, 'codeword_index': index, **measure_tail(y, mask, pn, edge)})
    return edge, rows, np.asarray(bits), np.asarray(masks)


def main():
    require_run_context()
    hashes = {}

    def doc(run, experiment, filename):
        return read(checked_artifact(run, experiment, filename, hashes))

    gain = doc('v079-code5-gain-masked', 'E081', 'code5_gain_transfer.json')
    layout = doc('v068-pn-padding-starts', 'E070', 'pn_padding_starts.json')
    labels = doc('v028-ut-cross32-labels-final', 'E030', 'ut_cross32_labels.json')
    model = labels['frozen_pn_model']
    pn = equality_predictions(np.arange(16384), pn_bits(), model)
    models = constellation_models(labels, hashes)
    reader = CacheOnly(ROOT/'.cache/qin-mat-pages', False)
    with reader, h5py.File(reader, 'r') as archive:
        edge, rows, bits, mask = extract(archive['yDataDec'], gain, layout, models, pn)
    out = experiment_output('E120')
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out/'mcs70_terminal_words.npz', bits=bits, mask=mask)
    result = {'schema': 'starlink-mcs70-terminal-words/v1', 'systematic_end': edge, 'codeword_bits': 16384,
              'rows': rows, 'pn_model': model, 'source_archive': {'url': URL, 'etag': ETAG, 'bytes': SIZE,
                 'pages': [reader.receipts[k] for k in sorted(reader.receipts)]},
              'input_sha256': {p.as_posix(): d for p, d in hashes.items()}}
    (out/'mcs70_terminal_words.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print('Terminal MCS 70 words', len(rows), flush=True)


if __name__ == '__main__':
    main()
