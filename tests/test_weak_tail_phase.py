import unittest

import numpy as np

import project_paths
from audit_weak_tail_phase import deinterleave_slots, projection_bank, select_tail
from audit_weak_tail_paths import collect_candidates, eligible_words, endpoint, onset_range, rank
from audit_ut_offset_constrained_h2 import coded_bits
from decode_ut_offset_body_first import CRC_RESOURCES, header_candidates
from educational_fec_demo import crc8


class WeakTailPhaseTests(unittest.TestCase):
    def test_absolute_phase_is_invariant_to_slot_index_and_complex_gain(self):
        rng = np.random.default_rng(111)
        word = np.where(rng.integers(2, size=60), 1, -1)
        slots = np.arange(286, 302)
        absolute = (slots[:, None] - 2)*1004 + np.arange(1004)
        gains = np.exp(1j*np.arange(len(slots))*.32)
        samples = word[(absolute - 37) % 60]*gains[:, None]
        blocks = deinterleave_slots(samples, slots)
        basis = projection_bank(word[None])
        score = select_tail(blocks, basis, 0)
        self.assertEqual(score[0]['phase'], 37)
        self.assertAlmostEqual(score[0]['held_improvement'], 1., places=10)

    def test_calibration_does_not_use_complementary_frequency_blocks(self):
        rng = np.random.default_rng(112)
        word = np.where(rng.integers(2, size=60), 1, -1)
        source = rng.normal(size=(16, 4, 4, 60)) + 1j*rng.normal(size=(16, 4, 4, 60))
        source[:, :, ::2] += 4*word[(np.arange(60)-51) % 60]
        bank = projection_bank(word[None])
        first = select_tail(source, bank, 0)
        source[:, :, 1::2] *= -1
        second = select_tail(source, bank, 0)
        self.assertEqual(first[0]['phase'], 51)
        self.assertEqual(first[0]['phase'], second[0]['phase'])
        self.assertEqual(first[0]['calibration_improvement'], second[0]['calibration_improvement'])
        self.assertNotEqual(first[0]['held_improvement'], second[0]['held_improvement'])

    def test_end_paths_can_add_one_missing_terminal_allocation(self):
        widths = {5: 2048, 57: 2048, 71: 3277}
        start = 8420
        prefix = (3, [(57, 2)])
        finish = endpoint(prefix, start, widths) + 3277
        picks = collect_candidates([prefix], start, widths, (finish, finish), finish % 60)
        self.assertIn((3, ((57, 2), (71, 1))), picks)
        self.assertTrue(all(endpoint(p, start, widths) == finish for p in picks))
        self.assertEqual(collect_candidates([prefix], start, widths, (finish, finish), (finish+1) % 60), [])

    def test_inconsistent_onset_is_erased(self):
        fits = [[{'onset': {'first_periodic_slot': x}} for x in group] for group in ((29, 32), (29, 32))]
        self.assertIsNone(onset_range({'fits': fits, 'accepted_phase': 20}))

    def test_changing_crc_samples_cannot_change_the_pre_crc_ranking(self):
        desired = (2, ((57, 2),))
        other = (2, ((70, 1),))
        widths = {57: 2048, 70: 4096}
        bits = header_candidates(desired)[0]
        bits[-8:] = np.unpackbits(np.array([crc8(bits[:-8])], dtype=np.uint8))
        soft = 1-2*coded_bits(bits).astype(float)
        candidates = eligible_words([desired, other])
        for mode in ('joint', 'prefix_first'):
            chosen = rank(candidates, soft, 8420, widths, mode)['selected']
            self.assertEqual(chosen['path'], desired)
            self.assertTrue(chosen['crc_ok'])
            received = soft.copy()
            received[-CRC_RESOURCES:] *= -1
            altered = rank(candidates, received, 8420, widths, mode)['selected']
            self.assertEqual(altered['path'], desired)
            self.assertEqual(altered['pre_crc_score'], chosen['pre_crc_score'])
