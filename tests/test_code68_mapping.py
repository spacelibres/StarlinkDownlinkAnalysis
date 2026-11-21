import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/E023/scripts'))
from code68_mapping_core import nullspace, forced_collisions, contract_checks


class Code68MappingTests(unittest.TestCase):
    def test_nullspace_detects_symbol_pairs_forced_to_same_two_bits(self):
        equations=[0b0011,0b1100]
        basis,rank=nullspace(equations,width=4)
        self.assertEqual(rank,2)
        self.assertEqual(len(basis),2)
        self.assertEqual(forced_collisions(basis,points=2,bits=2),[(0,1)])

    def test_check_contraction_preserves_bit_serial_order_and_xor(self):
        out=contract_checks(np.array([1,0,1]),np.array([[0,1,2],[0,1,4]]),2,2)
        self.assertEqual(out.tolist(),[11,8])


if __name__=='__main__':
    unittest.main()
