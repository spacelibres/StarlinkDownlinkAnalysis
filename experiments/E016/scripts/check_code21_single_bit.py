"""Audit a bounded one-bit correction using the frozen Code 21 generator."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'project.json').is_file())
sys.path.insert(0, str(ROOT))
from project_paths import experiment_output, require_run_context
from code21_ring import single_bit_candidates
from validate_code21 import digest

RUN = ROOT / 'runs/v014-code21-trained-labels/artifacts/experiments/E016/results'
GENERATOR_HASH = '65107672775f5d53fcf6f58f3390b26483c553320ef1fe1c0536768e49ac6c1b'


def main():
    require_run_context()
    destination = experiment_output('E016')
    destination.mkdir(parents=True, exist_ok=True)
    record_path = RUN / 'code21.json'
    input_path = RUN / 'code21_observations.npz'
    source = json.loads(record_path.read_text(encoding='utf-8'))
    with np.load(input_path, allow_pickle=False) as data:
        words = data['hard_bits']
        symbols = data['observations_with_edge_controls'][:, 1:-1]
        p = data['parity_generator']
        raw_prediction = data['predicted_parity']
    if not source['equal_generators'] or hashlib.sha256(p.tobytes()).hexdigest() != GENERATOR_HASH:
        raise RuntimeError('Expected the frozen generator from both assignments')
    d, parity_length = p.shape
    syndromes = raw_prediction ^ words[:, d:]
    corrected = words.copy()
    details = []
    direction, shift = source['published_group']['selected_convention']
    quarters = np.arange(4)
    phase = source['phase_zero_radians'] + quarters * np.pi / 2
    label_to_phase = np.empty(4)
    label_to_phase[np.array(source['quadrant_labels'])[(direction * quarters + shift) % 4]] = phase
    constellation = np.exp(1j * label_to_phase)
    for i in np.flatnonzero(np.any(syndromes, axis=1)):
        syndrome = ((words[i, :d] @ p) % 2) ^ words[i, d:]
        if not np.array_equal(syndrome, syndromes[i]):
            raise RuntimeError('Frozen parity prediction does not reproduce')
        candidates = single_bit_candidates(p, syndrome)
        case = {'codeword_index': int(i), **source['words'][i], 'initial_syndrome_weight': int(syndrome.sum()),
                'tested_systematic_positions': d, 'tested_parity_positions': parity_length,
                'all_single_bit_candidates': candidates}
        if len(candidates) == 1:
            bit = candidates[0]
            corrected[i, bit] ^= 1
            if np.any(((corrected[i, :d] @ p) % 2) != corrected[i, d:]):
                raise RuntimeError('Candidate failed re-encoding check')
            labels = words[i, ::2] * 2 + words[i, 1::2]
            hard_distance = abs(symbols[i] - constellation[labels]) ** 2
            changed_labels = labels[bit // 2] ^ (2 if bit % 2 == 0 else 1)
            distance = abs(symbols[i, bit // 2] - constellation[changed_labels]) ** 2
            all_deltas = np.column_stack([
                abs(symbols[i] - constellation[labels ^ mask]) ** 2 - hard_distance
                for mask in [2, 1]
            ]).ravel()
            delta = float(distance - hard_distance[bit // 2])
            actual = symbols[i, bit // 2]
            case.update(status='unique_within_one_bit', changed_bit=bit, final_syndrome_weight=0,
                        symbol_index=bit // 2, symbol_in_raw_iq_grid_resource=case['start_resource'] + bit // 2,
                        observed_iq=[float(actual.real), float(actual.imag)],
                        original_label=int(labels[bit // 2]), corrected_label=int(changed_labels),
                        squared_distance_increment=delta,
                        increment_rank_among_all_single_bit_flips=int(np.sum(all_deltas < all_deltas[bit])) + 1)
        else:
            case['status'] = 'ambiguous' if candidates else 'outside_one_bit_model'
        details.append(case)
    result = {'schema': 'starlink-code21-one-bit/v1', 'source_json_sha256': digest(record_path),
              'source_npz_sha256': digest(input_path), 'generator_sha256': GENERATOR_HASH,
              'words': len(words), 'initial_nonzero_syndromes': len(details), 'cases': details,
              'resolved_under_one_bit_model': sum(r['status'] == 'unique_within_one_bit' for r in details),
              'scope': 'Post hoc, bounded to one changed hard bit. A zero syndrome does not certify transmitted or application data.'}
    np.savez_compressed(destination / 'code21_one_bit.npz', corrected_hard_bits=corrected)
    (destination / 'code21_one_bit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('Enumerated individually:', result['initial_nonzero_syndromes'],
          'unique one-bit solutions:', result['resolved_under_one_bit_model'], flush=True)


if __name__ == '__main__':
    main()
