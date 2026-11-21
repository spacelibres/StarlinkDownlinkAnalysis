import unittest

import numpy as np
import project_paths

from audit_leo_layout_relation import layout_role, predict_by_layout, template_signs


class LeoLayoutRelationTests(unittest.TestCase):
    def test_scrambler_and_template_restore_qin_real_signs(self):
        words = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.uint8)
        pn = np.array([[0, 1, 1], [1, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(template_signs(words, pn, np.array([1, -1, 1])),
                                      [[1, 0, 0], [1, 0, 0]])

    def test_position_resolves_against_directory_and_record_boundaries(self):
        self.assertEqual(layout_role(168, 24, 135), 'directory_crc24')
        self.assertEqual(layout_role(168, 20, 135), 'record_1')
        self.assertEqual(layout_role(1240, 20, 135), 'following_record')

    def test_layout_predictions_use_only_training_and_abstain_on_unseen_or_tied(self):
        keys = [(24, 135), (24, 135), (24, 163), (24, 163), (20, 135), (24, 135)]
        self.assertEqual(predict_by_layout(keys, '111001', [1, 0, 1, 1, 0, 0]), '11???1')
