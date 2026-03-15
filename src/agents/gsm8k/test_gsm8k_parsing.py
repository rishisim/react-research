"""Unit tests for GSM8K parsing helpers."""

import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../shared")))
import wrappers


class GSM8KParsingTests(unittest.TestCase):
    def test_ground_truth_extraction_integer(self):
        raw = "Some reasoning here.\n#### 18"
        self.assertEqual(wrappers.parse_gsm8k_ground_truth_answer(raw), "18")

    def test_ground_truth_extraction_negative_decimal(self):
        raw = "Work shown\n#### -3.5"
        self.assertEqual(wrappers.parse_gsm8k_ground_truth_answer(raw), "-3.5")

    def test_ground_truth_extraction_commas(self):
        raw = "Detailed steps\n#### 1,234"
        self.assertEqual(wrappers.parse_gsm8k_ground_truth_answer(raw), "1234")

    def test_prediction_normalization_equivalent(self):
        self.assertEqual(wrappers.parse_gsm8k_prediction_answer("1,234"), "1234")
        self.assertEqual(wrappers.parse_gsm8k_prediction_answer("1234.0"), "1234")
        self.assertEqual(wrappers.parse_gsm8k_prediction_answer("+1234"), "1234")

    def test_prediction_from_freeform_text(self):
        pred = "I think the answer is 56 after calculation."
        self.assertEqual(wrappers.parse_gsm8k_prediction_answer(pred), "56")

    def test_non_numeric_prediction(self):
        self.assertEqual(wrappers.gsm8k_answers_match("", "Reason\n#### 42"), 0)
        self.assertEqual(wrappers.gsm8k_answers_match("null", "Reason\n#### 42"), 0)


if __name__ == "__main__":
    unittest.main()
