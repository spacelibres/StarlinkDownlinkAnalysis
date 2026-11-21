import unittest

import project_paths

from audit_code5_record_countdown import countdown, decode_timing_word, tail_resource_positions


class Code5RecordCountdownTests(unittest.TestCase):
    def test_word_reads_six_countdown_bits_and_three_group_bits(self):
        self.assertEqual(decode_timing_word(0x0e6a), {'countdown': 53, 'groups': 7, 'other_bits': 0})
        self.assertEqual(decode_timing_word(0x0181), {'countdown': 0, 'groups': 0, 'other_bits': 0x0181})

    def test_countdown_reloads_on_observed_boundary_branch(self):
        self.assertEqual(countdown(5564, 53, 0, 4), 1)
        self.assertEqual(countdown(5569, 53, 0, 4), 53)
        self.assertEqual(countdown(5573, 53, 0, 4), 45)
        self.assertEqual(countdown(4334, 53, 34, None), 46)

    def test_field_moves_across_codeword_boundary_with_the_28_byte_option(self):
        self.assertEqual(tail_resource_positions(24, 135, 1, 6), tuple(range(1221, 1227)))
        self.assertEqual(tail_resource_positions(24, 163, 1, 6), tuple(range(2469, 2475)))
