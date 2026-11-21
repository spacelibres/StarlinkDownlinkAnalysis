import unittest

import project_paths  # Registers experiment modules.

from audit_leo_lower_edge_relation import resource_coordinates, score_sign_export


class LeoLowerEdgeRelationTests(unittest.TestCase):
    def test_lower_edge_bins_enter_two_code5_systematic_words(self):
        self.assertEqual(resource_coordinates(((3, 524), (3, 537), (5, 524))), [1016, 1021, 3024])

    def test_selected_disagreement_and_unselected_agreement_remain_erased(self):
        export = {
            "bins": [524, 537], "evaluation_frames": [1, 2], "matched": {"count": 6, "agreement": 5 / 6},
            "raw_signs": [
                {"frame": 1, "symbols": [
                    {"ofdm_symbol": 3, "rx0": "10", "rx1": "00", "selected": "11", "consensus": "?0"},
                    {"ofdm_symbol": 5, "rx0": "00", "rx1": "00", "selected": "11", "consensus": "00"},
                ]},
                {"frame": 2, "symbols": [
                    {"ofdm_symbol": 3, "rx0": "10", "rx1": "10", "selected": "11", "consensus": "10"},
                    {"ofdm_symbol": 5, "rx0": "00", "rx1": "00", "selected": "00", "consensus": "??"},
                ]},
            ],
        }
        result = score_sign_export(export, ((3, 524), (3, 537), (5, 524)), 1)
        self.assertEqual(result["selected_signs"], 6)
        self.assertEqual(result["agreeing_selected_signs"], 5)
        self.assertEqual(result["frames"][0]["candidate_consensus"], "?00")
        self.assertEqual(result["frames"][1]["candidate_consensus"], "10?")
        self.assertEqual(result["complete_consensus_frames"], 0)


if __name__ == "__main__":
    unittest.main()
