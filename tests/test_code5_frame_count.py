from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/E079/scripts"))
from audit_code5_frame_count import counter_at, fit_offset, measure_timing, off_grid_links
from audit_leo_code5_projection import field_rank


class Code5FrameCountTests(unittest.TestCase):
    def test_dynamic_unaligned_counter_is_little_endian(self):
        for length in (20, 24, 60):
            value = 0x20DE
            word = ((value << (421 + 8 * length)) | length).to_bytes(128, "little")
            self.assertEqual(counter_at(word), value)
            self.assertEqual(counter_at(word, delta=1), value >> 1)

    def test_counter_offset_and_sampling_grid(self):
        phase, coordinates = measure_timing(np.array([1.412, 3.414, 5.413, 6.412, 10.91]),
                                            np.array([True, True, True, True, False]))
        self.assertAlmostEqual(phase, .4125, places=3)
        self.assertEqual(coordinates[2][0], 5)
        self.assertFalse(coordinates[4][2])
        self.assertEqual(fit_offset([65535, 0, 2], [0, 1, 3], modulus=65536), 65535)

    def test_parity_can_expose_a_field_only_after_cancelling_nuisance_bits(self):
        equations = np.array([[1, 1, 0], [0, 1, 0]], dtype=np.uint8)
        target = np.array([True, False, False])
        self.assertEqual(field_rank(equations[:1], target), 0)
        self.assertEqual(field_rank(equations, target), 1)

    def test_off_grid_destination_is_chosen_by_clock_before_header_comparison(self):
        rows = [{"on_waveform_grid": False, "record_index": 0, "counter": 9},
                {"on_waveform_grid": True, "record_index": 1, "counter": 9}]
        headers = [{"toa_s": t, "h1": {"words_hex": "AA"}, "h2": {"words_hex": h2}}
                   for t, h2 in ((0.002, "BB"), (0.004, "BB"), (0.005, "CC"))]
        timing = [(2, .3, False), (4, .001, True), (5, .001, True)]
        result = off_grid_links(rows, headers, timing, 5, [{"frames": [0, 1], "repeated_words": [{}]}])
        self.assertEqual(result["both_headers_by_frame_shift"]["0"], 1)
        self.assertEqual(result["both_headers_by_frame_shift"]["1"], 0)
        self.assertEqual(result["links"][0]["matches"]["0"][0]["previous_repeated_words"], 1)
