import unittest

import project_paths
from audit_mcs85_archive_records import link_tags


class Mcs85ArchiveRecordTests(unittest.TestCase):
    def test_same_tag_links_skip_recognized_other_tag_and_break_on_bad_boundary(self):
        rows = [
            {"byte_offset": n, "label_hex": tag}
            for n, tag in [(0, "aa"), (10, "bb"), (20, "aa"), (30, "aa"),
                           (51, "aa"), (61, "aa"), (71, "aa")]
        ]
        self.assertEqual(link_tags(rows, {"aa": 10, "bb": 10}), [(0, 2, 2), (4, 5, 1)])
