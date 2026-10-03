import unittest

from exl3_ssd_stream.cli import parser
from exl3_ssd_stream.validation import cache_budget, final_answer, validate_generation


class ValidationTests(unittest.TestCase):
    def test_unfinished_reasoning_is_not_a_final_answer(self):
        self.assertEqual(final_answer("Let me think. 2 + 2 is"), "")
        self.assertEqual(final_answer("</think>"), "")
        self.assertEqual(final_answer("2 + 2 = 4"), "")
        self.assertEqual(final_answer("Reasoning.</think> 2 + 2 = 4\n"), "2 + 2 = 4")

    def test_invalid_output_limits_rejected_before_model_load(self):
        for context, tokens in [(0, 8), (257, 8), (512, 0), (512, -1), (512, 512)]:
            with self.subTest(context=context, tokens=tokens), self.assertRaises(ValueError):
                validate_generation(context, tokens)
        validate_generation(512, 256)

    def test_automatic_budget_requires_exact_sentinel(self):
        for value in ["-2", "-0.5", "nan", "inf", "-inf"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                cache_budget(value)
        for value in ["-1", "0", "1.5"]:
            self.assertEqual(cache_budget(value), float(value))

    def test_cli_can_parse_without_gpu_dependencies(self):
        args = parser().parse_args(
            ["--model", "example", "--cache-gib", "0", "--context", "2048", "--interactive"]
        )
        self.assertTrue(args.interactive)
        self.assertEqual(args.context, 2048)
        self.assertEqual(args.cache_gib, 0)


if __name__ == "__main__":
    unittest.main()
