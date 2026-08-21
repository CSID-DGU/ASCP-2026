"""V2 full-flight master 단위 테스트."""

import unittest

from evaluation.full_flight_master import FullFlightInputError, validate_master_inputs


def _column(column_id="p0", legs=None, **overrides):
    value = {
        "column_id": column_id,
        "legs": [10, 20] if legs is None else legs,
        "cost": 3.0,
        "source_type": "policy",
        "is_legal": True,
    }
    value.update(overrides)
    return value


class InputContractTests(unittest.TestCase):
    def test_accepts_non_contiguous_global_ids(self):
        columns, universe = validate_master_inputs([_column()], [10, 20, 30])
        self.assertEqual(universe, (10, 20, 30))
        self.assertEqual(columns[0]["column_id"], "p0")

    def test_rejects_invalid_columns(self):
        cases = [
            (_column(legs=[10, 10]), [10], "중복 flight"),
            (_column(legs=[10, 99]), [10], "universe 밖"),
            (_column(source_type="unknown"), [10, 20], "source_type"),
            (_column(is_legal=False), [10, 20], "is_legal"),
            (_column(cost=float("inf")), [10, 20], "유한값"),
        ]
        for column, universe, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(FullFlightInputError, message):
                    validate_master_inputs([column], universe)

    def test_rejects_duplicate_universe_and_column_ids(self):
        with self.assertRaisesRegex(FullFlightInputError, "중복 ID"):
            validate_master_inputs([], [10, 10])
        with self.assertRaisesRegex(FullFlightInputError, "중복 column_id"):
            validate_master_inputs([_column(), _column()], [10, 20])


if __name__ == "__main__":
    unittest.main()
