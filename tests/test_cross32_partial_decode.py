import unittest

import numpy as np
import project_paths
from audit_cross32_partial_decode import parity_messages


class Cross32PartialDecodeTests(unittest.TestCase):
    def test_extrinsic_message_ignores_its_recipient_and_handles_zero(self):
        incoming = np.array([[2.0, -3.0, 0.0], [0.0, -3.0, 0.0]])
        result = parity_messages(incoming)
        expected = 2 * np.arctanh(np.tanh(1.0) * np.tanh(-1.5))
        np.testing.assert_allclose(result, [[0.0, 0.0, expected], [0.0, 0.0, 0.0]])
