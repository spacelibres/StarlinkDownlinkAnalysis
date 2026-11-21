"""Numerical source regressions that a syntax-only migration check cannot catch."""
import ast
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import project_paths

class ReceiverRegressionTests(unittest.TestCase):
    def test_known_pilot_only_null_generator_returns_complex_samples(self):
        import audit_nonpilot_leakage as core
        samples=core.synth(core.meta['windows'][0]['winner'])
        self.assertIsInstance(samples,np.ndarray)
        self.assertEqual(samples.shape,(50000,))
        self.assertTrue(np.iscomplexobj(samples))
        self.assertTrue(np.isfinite(samples).all())
        self.assertGreater(np.mean(abs(samples)**2),0)

    def test_migration_rewrites_path_but_preserves_local_numeric_out(self):
        spec=importlib.util.spec_from_file_location('organizer',ROOT/'tools/organize_legacy_workspace.py')
        organizer=importlib.util.module_from_spec(spec);spec.loader.exec_module(organizer)
        with tempfile.TemporaryDirectory(prefix='starlink-ast-regression-') as temp:
            path=Path(temp)/'sample.py'
            path.write_text('from pathlib import Path\nimport numpy as np\nOUT=Path(__file__).parent\ndef synth():\n    out=np.zeros(8,complex)\n    return out\n',encoding='utf8')
            organizer.rewrite_script(path,'E003')
            tree=ast.parse(path.read_text(encoding='utf8'))
            function=next(n for n in tree.body if isinstance(n,ast.FunctionDef))
            self.assertEqual(ast.unparse(function.body[0].value),'np.zeros(8, complex)')

if __name__=='__main__':unittest.main()
