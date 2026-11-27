#!/usr/bin/env python3
"""Unit tests for the pure alignment measurer in test_table_alignment.py.

Runs with: python3 -m unittest discover -s tests
No network: these pin the measurement function, not the model.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qc_table import extract_table_lines, measure_alignment  # noqa: E402

ALIGNED = """\
Here is the table:
+------------+--------+----------------------+
| component  | qty    | note                 |
+------------+--------+----------------------+
| resistor   | 4      | 10k ohm, 0.1W        |
| capacitor  | 12     | 100uF electrolytic   |
| LED        | 2      | red, 5mm diffused    |
+------------+--------+----------------------+
"""

# Row 3's second bar sits one column early; row 4's third bar two late.
MISALIGNED = """\
| component  | qty    | note                 |
|------------|--------|----------------------|
| resistor   | 4      | 10k ohm, 0.1W        |
| capacitor  | 12     | 100uF electrolytic   |
| LED        | 2     | red, 5mm diffused     |
| diode      | 7      | 1N4148 small signal   |
"""


class TestExtract(unittest.TestCase):
    def test_picks_only_table_lines(self):
        lines = extract_table_lines(ALIGNED)
        self.assertTrue(all(("|" in l or "\u2502" in l or "+" in l)
                            for l in lines))
        self.assertEqual(len(lines), 7)  # 3 border + header + 3 data

    def test_ignores_prose(self):
        self.assertEqual(extract_table_lines("no table here\njust prose"), [])


class TestMeasure(unittest.TestCase):
    def test_aligned_table_zero_misaligned(self):
        m = measure_alignment(ALIGNED)
        self.assertTrue(m["found"])
        self.assertEqual(m["rows"], 4)          # data rows (header + 3 parts)
        self.assertEqual(m["columns"], 3)
        self.assertEqual(m["misaligned_rows"], 0)
        self.assertEqual(m["misaligned_line_indices"], [])

    def test_counts_misaligned_lines(self):
        m = measure_alignment(MISALIGNED)
        self.assertTrue(m["found"])
        self.assertEqual(m["columns"], 3)
        self.assertEqual(m["misaligned_rows"], 2)
        self.assertEqual(len(m["misaligned_line_indices"]), 2)

    def test_majority_reference_not_header(self):
        # The header deviates from the pattern the majority of lines share,
        # so exactly one data row (the header) is misaligned.
        text = (
            "|a|b|\n"
            "|--|--|\n"
            "|xx |yy |\n"
            "|zz |ww |\n"
        )
        m = measure_alignment(text)
        self.assertEqual(m["columns"], 2)
        # Lines 0 (header) and 1 (rules) both deviate from the data majority;
        # only the header is a data row, so misaligned_rows == 1.
        self.assertEqual(m["misaligned_rows"], 1)
        self.assertEqual(m["misaligned_line_indices"], [0, 1])

    def test_tab_expanded_to_8_cols(self):
        # A tab can smuggle false alignment; columns are counted after
        # expanding tabs at 8 columns.
        good = "| aa | bb |\n| cc\t| dd |"
        m = measure_alignment(good)
        self.assertEqual(m["misaligned_rows"], 1)

    def test_no_table(self):
        m = measure_alignment("I cannot produce a table.")
        self.assertFalse(m["found"])
        self.assertIsNone(m["misaligned_rows"])

    def test_duplicate_cells_reported(self):
        text = ("| name | qty |\n|------|-----|\n"
                "| bolt | 3   |\n| bolt | 5   |\n")
        m = measure_alignment(text)
        self.assertEqual(m["duplicate_cells"], 1)

    def test_uniform_column_widths_reported(self):
        # Uneven widths are required by the prompt; the measurer reports
        # whether every content column came out the same width.
        uniform = ("| aaa | bbb |\n|-----|-----|\n"
                   "| 1   | 2   |\n| 33  | 44  |\n")
        m = measure_alignment(uniform)
        self.assertTrue(m["uniform_widths"])
        self.assertFalse(measure_alignment(ALIGNED)["uniform_widths"])


if __name__ == "__main__":
    unittest.main()
