"""Verify lookup proposals stay within their allowed, already-verified source."""

import unittest

from exl3_ssd_stream.cli import parser
from exl3_ssd_stream.optimized.ngram import lookup_draft


class LookupTests(unittest.TestCase):
    def test_continuation_is_bounded_by_verified_source(self):
        tokens = list(range(1, 13)) + [99] + list(range(1, 9))
        self.assertEqual(lookup_draft(tokens, 20, source_limit=12), [9, 10, 11, 12])
        self.assertEqual(lookup_draft(tokens, 20, source_limit=0), [])
        self.assertEqual(lookup_draft(tokens, 2, source_limit=12), [9, 10])

    def test_minimum_match_prevents_short_accidental_repetition(self):
        self.assertEqual(lookup_draft([1, 2, 3, 4, 5, 8, 1, 2, 3], 20), [])
        self.assertEqual(lookup_draft(list(range(20)), 8), [])

    def test_prefers_longest_matching_suffix(self):
        tokens = list(range(1, 11)) + [77, 66] + list(range(1, 9))
        self.assertEqual(lookup_draft(tokens, 2), [9, 10])
        self.assertEqual(lookup_draft(tokens, 0), [])

    def test_optimized_cli_can_parse_without_gpu_dependencies(self):
        args = parser().parse_args(["--model", "example", "--optimized", "--mtp", "8"])
        self.assertTrue(args.optimized)
        self.assertEqual(args.mtp, 8)


if __name__ == "__main__":
    unittest.main()
