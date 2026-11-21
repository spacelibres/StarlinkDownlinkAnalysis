"""Map existing narrow IQ observations to published PDU-header resources.

No missing carrier, coded bit, or on-air field is filled from the publication.
PN alignment: Liu et al., arXiv:2610.29387v2, equations 7--8.
"""

# Repository layout adapter; numerical methods below retain their prior equations.
import sys as _sys
from pathlib import Path as _Path
from functools import partial as _partial
_PROJECT_ROOT = next(p for p in _Path(__file__).resolve().parents if (p / "project.json").is_file())
_sys.path.insert(0, str(_PROJECT_ROOT))
from project_paths import PROJECT_ROOT, RAW_WAV, LEO_SNAPSHOT, EXTERNAL_LEO_ROOT, VI_ROOT, legacy_artifact, experiment_output, require_run_context
artifact_path = _partial(legacy_artifact, prefix='', experiment='E003')
if __name__ == "__main__":
    require_run_context()
from pathlib import Path
import csv, hashlib, json
import numpy as np
import audit_nonpilot_leakage as core

OUT = experiment_output('E003', '')
SOURCE = 'https://arxiv.org/html/2610.29387v2'
pn = np.zeros(32767, dtype=np.uint8)
pn[:15] = [int(x) for x in '000000100000010']
for j in range(15, len(pn)):
    pn[j] = pn[j-14] ^ pn[j-15]

def reference(k, slots):
    signed = int(k) if int(k) < 512 else int(k)-1024
    ordinal = signed+512 if signed < 0 else signed+508
    d = 1020*(np.asarray(slots)-17)+ordinal
    rho = (2*pn[(d+16327)%32767].astype(int) + (1 ^ pn[(d+16320)%32767]) + 1) % 4
    return np.exp(.5j*np.pi*rho)

def rank_gf2(a):
    a = np.array(a, dtype=np.uint8).copy()
    row = 0
    for col in range(a.shape[1]):
        pivots = np.flatnonzero(a[row:, col])
        if not len(pivots):
            continue
        pivot = row+int(pivots[0]); a[[row, pivot]] = a[[pivot, row]]
        for r in range(a.shape[0]):
            if r != row and a[r, col]: a[r] ^= a[row]
        row += 1
        if row == a.shape[0]: break
    return row

def encoder_matrix():
    # Affine output polarity cannot change this dependency/rank calculation.
    matrix = np.zeros((228, 64), dtype=np.uint8)
    for block in range(2):
        for n in range(38):
            for output, generator in enumerate([0o155, 0o123, 0o137]):
                for delay in range(7):
                    input_index = n-delay
                    if (generator >> delay) & 1 and 0 <= input_index < 32:
                        matrix[block*114+n*3+output, block*32+input_index] ^= 1
    return matrix

pilots = []
for k in range(528, 536):
    ratio = core.expected[:, k-528] / reference(k, core.symbol)
    pilots.append({'k': k, 'coherence': float(abs(ratio.mean())),
                   'constant_phase_radians': float(np.angle(ratio.mean()))})
assert min(x['coherence'] for x in pilots) > 1-1e-12

loaded = [k for k in range(-512, 512) if k not in [-2, -1, 0, 1]]
nonpilot = [k for k in loaded if not (-496 <= k <= -489 or 488 <= k <= 495)]
header = nonpilot[:228]
assert header[0] == -512 and header[-1] == -277
matrix = encoder_matrix()
coverage = []; rows = []; differences = []
with np.load(artifact_path('projected_nonpilot_observations.npz'), allow_pickle=False) as z:
    for window in range(8):
        key = f'iq_{window}'; ids = z[key+'_unknown_indices'].astype(int)
        values = z[key+'_values']; prediction = z[key+'_pilot_prediction']
        positions = []
        for col, k in enumerate(ids):
            signed = int(k)-1024
            if signed not in header: continue
            q = header.index(signed); positions.append(q)
            # Pilot expected values differ from the new PN reference by -3pi/4.
            # Equalization extrapolates the pilot channel; absolute reliability
            # has not been established, so export complex observations only.
            observed = values[:, 0, col]/prediction[:, col]/reference(k, 2)*np.exp(.75j*np.pi)
            for frame, value in enumerate(observed):
                rows.append({'window': key, 'frame': frame, 'slot': 2,
                             'subcarrier_signed': signed, 'coded_resource_ordinal': q,
                             'pn_removed_real': float(value.real), 'pn_removed_imag': float(value.imag),
                             'channel_extrapolated': True, 'decoded_bit': '',
                             'header_crc_passed': False})
            delta = reference(k, core.symbol)/np.exp(.5j*np.pi*core.rot[k, core.symbol-1])
            differences.append({'window': key, 'k': int(k),
                                'reference_difference_is_binary': bool(np.max(abs(delta.imag)) < 1e-10),
                                'reference_difference_period15_agreement': float(np.mean(abs(delta[15:]-delta[:-15]) < 1e-10))})
        deps = np.flatnonzero(np.any(matrix[positions], axis=0)).tolist()
        coverage.append({'window': key, 'observed_header_resources': positions,
                         'count_per_frame': len(positions), 'header_total': 228,
                         'ideal_binary_constraint_rank': rank_gf2(matrix[positions]),
                         'dependent_retained_input_positions': deps,
                         'observes_mcs_input_positions16_to35': bool(set(deps) & set(range(16, 36))),
                         'complete_header_observed': False})

union = sorted({q for r in coverage for q in r['observed_header_resources']})
assert union == [13, 14, 15, 16, 17]
assert not any(r['observes_mcs_input_positions16_to35'] for r in coverage)
paired_coverage = []
with np.load(artifact_path('paired_tcode_observations.npz'), allow_pickle=False) as z:
    for key in ['iq_2', 'iq_3']:
        positions = [header.index(int(k)-1024) for k in z[key+'_indices']
                     if int(k)-1024 in header and not 528 <= int(k) <= 535]
        paired_coverage.append({'array': key, 'header_resource_ordinals': positions,
                                'count_per_frame': len(positions), 'complete_header': False})
pss = json.loads((artifact_path('wav_pss_specificity_extended_results.json')).read_text())
exact = pss['results'][0]; controls = pss['results'][1:]
result = {
    'source': SOURCE, 'source_scope': 'Published configuration; applying its header map to local pilot-indexed snippets is a conditional test, not proof of an identical format.',
    'reference_pilot_checks': pilots, 'template_difference_checks': differences,
    'header1_coverage': coverage,
    'paired_rx_header1_coverage': paired_coverage,
    'optimistic_union_not_a_valid_simultaneous_observation': union,
    'optimistic_union_constraint_rank': rank_gf2(matrix[union]),
    'optimistic_union_input_positions': np.flatnonzero(np.any(matrix[union], axis=0)).tolist(),
    'interpretation': 'Only early prefix dependencies are observed. MCS/length/CRC inputs are unobserved in PDU1. Combining different frames or unvalidated receiver pairs cannot fill the missing columns.',
    'prior_15_symbol_signs_status': 'Unvalidated relative observations. The two public references themselves differ by a near-15-symbol binary pattern; these signs must not be described as newly decoded message bits.',
    'wav_pss': {'exact_heldout_ratio': exact['test_same_phase_ratio'],
                'control_count': len(controls),
                'max_control_heldout_ratio': max(r['test_same_phase_ratio'] for r in controls),
                'controls_exceeding_exact': sum(r['test_same_phase_ratio'] >= exact['test_same_phase_ratio'] for r in controls),
                'post_event_ratio': exact['post_same_phase_ratio'],
                'interpretation': 'Noncoherent PSS-compatible feature; not per-frame complex synchronization or PDU decoding.'},
    'validated_headers': 0, 'validated_messages': 0, 'satellite_identifier': None,
}
(artifact_path('header_observability_audit.json')).write_text(json.dumps(result, indent=2), encoding='utf8')
with (artifact_path('partial_header_complex_observations.csv')).open('w', encoding='utf-8-sig', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
ref = artifact_path('reference/misaka_README.md')
(artifact_path('reference/misaka_provenance.json')).write_text(json.dumps({'url': 'https://github.com/misaka666/Starlink-Downlink-Analysis/blob/main/README.md', 'retrieval_date': '2026-11-20', 'sha256': hashlib.sha256(ref.read_bytes()).hexdigest(), 'bytes': ref.stat().st_size, 'paper': SOURCE, 'third_party_capture_not_local_recording': True}, indent=2), encoding='utf8')
print(json.dumps({'coverage': coverage, 'optimistic_union': union, 'union_rank': result['optimistic_union_constraint_rank'], 'union_input_positions': result['optimistic_union_input_positions'], 'complex_observations_exported': len(rows), 'wav_pss': result['wav_pss'], 'validated_messages': 0}, indent=2))
