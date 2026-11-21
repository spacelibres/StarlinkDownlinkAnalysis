"""Decode a punctured H1 prefix from LEO's existing dual-receiver sign exports."""

from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from project_paths import EXTERNAL_LEO_ROOT, PROJECT_ROOT, experiment_output, require_run_context
from audit_code_observability import encoder_matrix, gf2_rank
from audit_leo_affine_transfer import resource_role
from audit_leo_early_symbol_roles import REFERENCE
from audit_leo_lower_edge_relation import resource_coordinates, score_sign_export, signed_bin
from audit_ut_repeated_data import checked_artifact, read
from educational_fec_demo import word_bits
from receiver_models import pn_reference


DS_PATH = Path('reports/2026_09_28_sequence_semantics/local')


def minimum_distance_prefix(candidates, observations, width):
    keep = observations >= 0
    distances = np.count_nonzero(candidates[:, keep] != observations[keep], axis=1)
    winners = np.flatnonzero(distances == distances.min())
    size = 1 << width
    messages = np.unique(winners % size).tolist()
    signs = sorted(set((winners // size).tolist()))
    common = messages[0]
    varied = 0
    for value in messages[1:]:
        varied |= common ^ value
    bits = ''.join('?' if varied & (1 << k) else str((common >> k) & 1) for k in range(width-1, -1, -1))
    return {'distance': int(distances.min()), 'messages': messages, 'signs': signs, 'bits': bits}


def codebooks(matrix, bias):
    width = int(np.flatnonzero(matrix.any(axis=0))[-1]) + 1
    messages = (np.arange(1 << width)[:, None] >> np.arange(width-1, -1, -1)) & 1
    if gf2_rank(matrix) != width:
        raise ValueError('H1 projection no longer uniquely encodes the visible prefix')
    for shift in range(len(matrix)):
        first = ((messages @ np.roll(matrix, shift, axis=0)[:, :width].T) & 1) ^ bias
        yield shift, np.concatenate((first, first ^ 1)), width


def decode_visit(export, coordinates, offsets, generator, template):
    matrix = generator[offsets]
    phase = np.array([pn_reference(signed_bin(k), np.array([s]))[0] * np.exp(-.5j*np.pi*template[k, s-1])
                      for s, k in coordinates])
    if not np.allclose(abs(phase.real), 1) or np.any(abs(phase.imag) > 1e-10):
        raise ValueError('H1 export phase convention changed')
    books = list(codebooks(matrix, phase.real > 0))
    _, candidates, width = books[0]
    if len({bytes(row) for row in candidates}) != len(candidates):
        raise ValueError('An H1 prefix can be confused with another prefix by a global sign flip')
    indices = [export['bins'].index(k) for _, k in coordinates]
    frames = []
    for row in export['raw_signs']:
        symbol = next(s for s in row['symbols'] if s['ofdm_symbol'] == 2)
        seen = {kind: np.array([-1 if symbol[kind][i]=='?' else int(symbol[kind][i]) for i in indices])
                for kind in ('rx0', 'rx1', 'consensus')}
        seen['agree'] = np.where(seen['rx0'] == seen['rx1'], seen['rx0'], -1)
        decode = {key: {**minimum_distance_prefix(candidates, value, width),
                        'observed': int(np.count_nonzero(value >= 0))} for key, value in seen.items()}
        controls = [minimum_distance_prefix(book, seen['agree'], width) for _, book, _ in books[1:]]
        frames.append({'frame': row['frame'], 'projected': decode, 'agree_shift_controls': [
            {'shift': shift, **control} for (shift, _, _), control in zip(books[1:], controls, strict=True)]})
    return {'coordinates': coordinates, 'h1_offsets': offsets, 'width': width,
            'checks_known_polarity': len(matrix) - width, 'checks_unknown_polarity': len(matrix) - width - 1,
            'minimum_distance': int(np.count_nonzero(candidates[1:] != candidates[0], axis=1).min()),
            'phase_signs': phase.real.round().astype(int).tolist(), 'frames': frames}


def code5_geometry(resources, code):
    words = {}
    for resource in resources:
        role = resource_role(resource)
        if role['part'].startswith('Code 5'):
            words.setdefault(role['word'], []).append(role['offset'])
    return [{'word': word, 'observed': len(offsets), 'offsets': offsets, 'rank': gf2_rank(code[offsets])}
            for word, offsets in sorted(words.items())]


def main():
    require_run_context()
    external, inputs = {}, {}

    def leo(name):
        path = DS_PATH / name
        data = (EXTERNAL_LEO_ROOT / path).read_bytes()
        external[path.as_posix()] = sha256(data).hexdigest()
        return json.loads(data)

    recovered = leo('local_header_recovery.json')['rows'] + [leo('ds9_last_header_audit.json')]
    prior = read(checked_artifact('v127-leo-lower-edge', 'E129', 'leo_lower_edge_relation.json', inputs))
    if any(prior['leo_inputs_sha256'][path] != value for path, value in external.items()):
        raise ValueError('Native signs differ from E129 archive receipts')
    received = checked_artifact('v010-code5-parity-02', 'E012', 'code5_observations.npz', inputs)
    with np.load(received, allow_pickle=False) as src:
        code5 = np.r_[np.eye(1024, dtype=np.uint8), src['parity_generator'].T]
    generator = encoder_matrix(2)
    inputs[REFERENCE] = sha256(REFERENCE.read_bytes()).hexdigest()
    template = loadmat(REFERENCE)['referenceTemplateRotations']

    results = []
    for export in recovered:
        entry = {'signal': export['signal'], 'expected_soft_sha256': export['sha256'],
                 'evaluation_frames': export.get('evaluation_frames', [])}
        results.append(entry)
        if not export.get('raw_signs'):
            continue
        score_sign_export(export, [(2, export['bins'][0])], 0)
        coordinates = [(s, k) for s in range(2, 8) for k in export['bins']]
        resources = resource_coordinates(coordinates)
        entry['code5_projection'] = code5_geometry(resources, code5)
        h1 = [(coord, r) for coord, r in zip(coordinates, resources, strict=True) if resource_role(r)['part']=='H1']
        if h1:
            places, offsets = zip(*h1)
            entry['h1'] = decode_visit(export, list(places), list(offsets), generator, template)

    # This public prefix is used to score the projections, after prediction.
    width = {row['h1']['width'] for row in results if 'h1' in row}
    if len(width) != 1:
        raise ValueError('Expected a common H1 prefix width for the two lower-edge visits')
    width = width.pop()
    public = ''.join(map(str, word_bits(0x3048)[16:16+width]))
    result = {'schema': 'starlink-leo-projected-h1/v1', 'public_reference_bits': public,
              'external_inputs_sha256': external,
              'project_inputs_sha256': {str(path.relative_to(PROJECT_ROOT).as_posix()): h for path, h in inputs.items()},
              'visits': results}
    output = experiment_output('E144')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'leo_projected_h1.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf8')
    for visit in results:
        if 'h1' in visit:
            candidates = [r['projected']['agree']['bits'] for r in visit['h1']['frames']]
            print(visit['signal'], candidates.count(public), '/', len(candidates), 'unique public prefix', flush=True)


if __name__ == '__main__':
    main()
