import unittest

from annotator.db import PROTOCOL_VERSION, normalize_annotation


class AnnotationProtocolTests(unittest.TestCase):
    def test_non_new_growth_is_saved_as_skip_without_curl(self):
        self.assertEqual(normalize_annotation("mature", None, "yes"), ("mature", "skip", None))
        self.assertEqual(normalize_annotation("tube", "severe", "yes"), ("tube", "skip", None))

    def test_new_growth_accepts_the_four_damage_classes(self):
        for damage in ("healthy", "mild", "severe", "uncertain"):
            self.assertEqual(normalize_annotation("flush", damage, "yes"), ("flush", damage, None))

    def test_legacy_binary_injured_is_rejected_for_new_labels(self):
        with self.assertRaisesRegex(ValueError, "healthy, mild, severe, or uncertain"):
            normalize_annotation("flush", "injured", None)

    def test_protocol_has_a_stable_export_name(self):
        self.assertEqual(PROTOCOL_VERSION, "visible_new_growth_damage_v2")


if __name__ == "__main__":
    unittest.main()
