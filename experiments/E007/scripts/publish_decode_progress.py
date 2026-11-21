"""Publish only supported decoding stages and reproducible candidate slices."""

# Repository layout adapter; numerical methods below retain their prior equations.
import sys as _sys
from pathlib import Path as _Path
from functools import partial as _partial
_PROJECT_ROOT = next(p for p in _Path(__file__).resolve().parents if (p / "project.json").is_file())
_sys.path.insert(0, str(_PROJECT_ROOT))
from project_paths import PROJECT_ROOT, RAW_WAV, LEO_SNAPSHOT, EXTERNAL_LEO_ROOT, VI_ROOT, legacy_artifact, experiment_output, require_run_context
artifact_path = _partial(legacy_artifact, prefix='', experiment='E007')
if __name__ == "__main__":
    require_run_context()
from pathlib import Path
import csv,json
import numpy as np
from scipy.io import wavfile

OUT=experiment_output('E007', '')
sync=json.loads((artifact_path('narrowband_sync_prefix_extended_test.json')).read_text(encoding='utf8'))
gutter=json.loads((artifact_path('central_gutter_grid_test.json')).read_text(encoding='utf8'))
fold=json.loads((artifact_path('recovered_cyclic_waveform.json')).read_text(encoding='utf8'))
data=json.loads((artifact_path('central_ofdm_candidates.json')).read_text(encoding='utf8'))
control=json.loads((artifact_path('central_narrowband_model_control.json')).read_text(encoding='utf8'))
fs,raw=wavfile.read(RAW_WAV,mmap=True)
with np.load(artifact_path('narrowband_sync_prefix_extended_observations.npz'),allow_pickle=False) as z:
    epochs=z['frame_epoch_s']+sync['exact']['offset_from_power_guard_epoch_s'];blocks=z['block_index'];energy=abs(z['selected_projection'])**2
rows=[]
for i,(epoch,block,value) in enumerate(zip(epochs,blocks,energy)):
    group='calibration' if int(block) in sync['calibration_blocks'] else 'heldout' if int(block) in sync['heldout_blocks'] else 'post_reference' if int(block) in sync['post_noise_calibration_blocks'] else 'post_heldout' if int(block) in sync['post_heldout_blocks'] else 'transition_not_tested'
    rows.append({'prefix_source_row':i,'frame_epoch_candidate_s':float(epoch),'block':int(block),'group':group,'known_prefix_whitened_energy':float(value),'epoch_is_candidate':True,'message_crc_passed':False})
with (artifact_path('frame_candidate_catalog.csv')).open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
folder=artifact_path('frame_candidates');folder.mkdir(exist_ok=True)
held=np.flatnonzero(np.isin(blocks,sync['heldout_blocks']))
selected=held[np.argsort(energy[held])[-8:]][::-1];slices=[]
clock=1-fold['inherited_phase_drift_samples_s']/fs
for rank,i in enumerate(selected):
    first=max(0,int(np.floor(epochs[i]*fs))-64);last=min(len(raw),first+int(np.ceil(3200/clock))+128)
    path=folder/f'candidate_{rank:02d}_row_{int(i)}.npz'
    np.savez_compressed(path,raw_iq_int16=raw[first:last],sample_rate_hz=np.array(fs),source_first_sample=np.array(first),frame_epoch_candidate_s=epochs[i],prefix_source_row=np.array(i))
    slices.append({'file':str(path),'source_first_sample':first,'source_last_sample_exclusive':last,'epoch_candidate_s':float(epochs[i]),'prefix_energy':float(energy[i]),'message_decoded':False})
summary={'scope':'Original WAV, waveform/frame acquisition and conditional symbols; no validated payload fields.',
         'stages':{'periodic_power_profiles':'recovered and tested on whole heldout blocks',
                   'central_four_null_grid':'supported by conditional spectral model',
                   'known_pss_sss_prefix_match':'supported relative to tested randomized templates; conditional receiver model',
                   'six_carrier_complex_symbol_estimates':'extracted; approximate narrowband receiver model',
                   'unknown_repeated_binary_pattern':'not supported in selected signal frames',
                   'message_fec_crc':'not reached'},
         'period_us_under_clock_model':3200/fs/clock*1e6,
         'subcarrier_spacing_hypothesis_hz':234375.,'subcarrier_grid_half_shift_hz':117187.5,
         'gutter_model_heldout_shape_correlation':gutter['calibration_selected_model']['heldout_shape_correlation'],
         'prefix_control_counts':{'spectrum_preserving':sync['control_count'],'structure_preserving':sync['structured_control_count']},
         'prefix_controls_exceeding_exact':{'spectrum_preserving':sync['controls_exceeding_exact'],'structure_preserving':sync['structured_controls_exceeding_exact']},
         'frame_catalog_entries':len(rows),'raw_frame_candidate_slices':slices,
         'signal_region_candidate_frames':sum(data['group_summaries'][key]['frames'] for key in ['calibration','heldout']),
         'post_event_control_frames':data['group_summaries']['post']['frames'],
         'signal_region_complex_data_symbol_estimates':192*300*6,
         'symbol_phase_convention':'Half-subcarrier offset is continuous; deterministic slot rotation is removed before PN/data analysis.',
         'synthetic_forward_model_heldout_sign_agreement_by_carrier':control['noiseless_heldout_sign_agreement_by_carrier'],
         'validated_unknown_bits':0,'validated_messages':0,'decoded_semantic_fields':{},
         'exploratory_bit_candidates_in_post_control':data['group_summaries']['post']['supported_relative_bits'],
         'report':str(artifact_path('分步解码进展.txt'))}
(artifact_path('decode_progress_summary.json')).write_text(json.dumps(summary,indent=2),encoding='utf8')
print(json.dumps({'catalog_entries':len(rows),'raw_candidate_slices':len(slices),'signal_frames':summary['signal_region_candidate_frames'],'complex_data_estimates':summary['signal_region_complex_data_symbol_estimates'],'validated_unknown_bits':0},indent=2))
