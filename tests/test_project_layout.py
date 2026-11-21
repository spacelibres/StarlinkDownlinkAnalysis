from pathlib import Path
import importlib.util
import json
import unittest
from unittest.mock import patch
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]

class ProjectLayoutTests(unittest.TestCase):
    def test_raw_input_is_classified_without_byte_changes(self):
        target=ROOT/'data/raw/wav/USER_IQ.wav'
        self.assertTrue(target.is_file(),'raw WAV must be classified under data/raw/wav')

    def test_cross_experiment_artifact_and_script_paths_resolve(self):
        spec=importlib.util.spec_from_file_location('project_paths',ROOT/'project_paths.py')
        self.assertTrue((ROOT/'project_paths.py').is_file(),'portable project path resolver is missing')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.assertTrue(module.legacy_artifact('refined_track.npz').is_file())
        self.assertIn('E001',module.legacy_artifact('refined_track.npz').as_posix())
        self.assertTrue(module.legacy_artifact('extract_central_ofdm_candidates.py').is_file())
        self.assertTrue(module.legacy_artifact('reference/qin/sssVec.mat').is_file())

    def test_every_original_evidence_file_has_a_new_location(self):
        target=ROOT/'provenance/layout_manifest.json'
        self.assertTrue(target.is_file(),'migration manifest is missing')
        manifest=json.loads(target.read_text(encoding='utf8'))
        baseline=json.loads((ROOT/'provenance/baseline_manifest.json').read_text(encoding='utf8'))
        self.assertEqual({r['path'] for r in baseline['files']},{r['original_path'] for r in manifest['files']})
        for row in manifest['files']:
            self.assertTrue((ROOT/row['current_path']).is_file(),row['current_path'])

    def test_learning_and_reports_have_distinct_locations(self):
        self.assertTrue((ROOT/'learning/通信公式与证据学习.ipynb').is_file())
        self.assertTrue((ROOT/'reports/E007/分步解码进展.txt').is_file())

    def test_run_redirects_cross_experiment_evidence_and_reports(self):
        import project_paths
        run=ROOT/'.cache/test-run/workspace'
        # Patch this module's dependency, not the process environment. On Windows,
        # restoring empty environment values can remove inherited native values.
        with patch.object(project_paths,'os',SimpleNamespace(environ={'STARLINK_RUN_DIR':str(run),'STARLINK_EXPERIMENT_ID':'E007'})):
            self.assertTrue(project_paths.legacy_artifact('refined_track.npz').is_relative_to(run))
            self.assertTrue(project_paths.legacy_artifact('分步解码进展.txt').is_relative_to(run))
            self.assertEqual(project_paths.RAW_WAV,ROOT/'data/raw/wav'/project_paths.RAW_WAV.name)
            self.assertFalse(project_paths.legacy_artifact('extract_central_ofdm_candidates.py').is_relative_to(run))

    def test_paths_reject_traversal(self):
        import project_paths
        with self.assertRaises(ValueError):project_paths.legacy_artifact('../escape.json')

if __name__=='__main__':unittest.main()
