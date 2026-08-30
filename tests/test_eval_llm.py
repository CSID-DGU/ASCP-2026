import os
import sys
import tempfile
import unittest

import pandas as pd


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "RL"))

from evaluation.eval_llm import (  # noqa: E402
    evaluate_llm_text,
    infer_airline,
    load_bts_instance,
    replace_symbolic_flight_ids,
)


CONSTRAINT = {
    "base_airport": 0,
    "min_conn": 0.5,
    "max_conn": 5.0,
    "max_legs": 8,
    "min_rest": 10.0,
    "max_duty_periods": 2,
    "max_pairing_days": 5,
    "min_pairing_legs": 3,
    "max_duty": 13.0,
}


class EvalLlmTests(unittest.TestCase):
    def test_infers_airline_like_previous_cli(self):
        self.assertEqual(infer_airline("RL/data/delta_2019_01.csv"), "delta")
        self.assertEqual(infer_airline("RL/data/jetblue_2019_01.csv"), "jetblue")
        self.assertEqual(infer_airline("tt201401.legs"), "turkish")

    def test_symbolic_ids_are_only_converted_not_rearranged(self):
        text = "Pairing 1: [DL00003, DL00001, DL00002]\nUncovered: []"
        self.assertEqual(
            replace_symbolic_flight_ids(text, "DL"),
            "Pairing 1: [3, 1, 2]\nUncovered: []",
        )

    def test_direct_and_forced_100_are_returned_together(self):
        instance = {
            "flights": {
                1: {"id": 1, "origin": 0, "dest": 1, "dep_time": 0.0, "arr_time": 1.0},
                2: {"id": 2, "origin": 1, "dest": 2, "dep_time": 2.0, "arr_time": 3.0},
                3: {"id": 3, "origin": 2, "dest": 0, "dep_time": 4.0, "arr_time": 5.0},
                4: {"id": 4, "origin": 0, "dest": 0, "dep_time": 6.0, "arr_time": 7.0},
            },
            "constraint": CONSTRAINT,
            "base_ids": [0],
            "airport_map": {"ATL": 0, "A": 1, "B": 2},
            "metadata": {"fixture": True},
        }
        text = "Pairing 1 (base=ATL): [1, 2, 3]\nUncovered: [4]"
        result = evaluate_llm_text(text, instance, airline="delta")
        self.assertFalse(result["solution_feasible"])
        self.assertEqual(result["legal_union_coverage"], 0.75)
        self.assertEqual(result["legacy_forced_100"]["n_forced_pairings"], 1)
        self.assertFalse(result["legacy_forced_100"]["use_as_primary_result"])

    def test_bts_loader_preserves_raw_row_ids_after_dropping_invalid_rows(self):
        frame = pd.DataFrame([
            {
                "ORIGIN": "ATL", "DEST": "JFK", "CRS_DEP_TIME": 800,
                "CRS_ARR_TIME": 1000, "CRS_ELAPSED_TIME": 120,
                "FL_DATE": "2019-01-01",
            },
            {
                "ORIGIN": "ATL", "DEST": "JFK", "CRS_DEP_TIME": None,
                "CRS_ARR_TIME": 1000, "CRS_ELAPSED_TIME": 120,
                "FL_DATE": "2019-01-01",
            },
            {
                "ORIGIN": "JFK", "DEST": "ATL", "CRS_DEP_TIME": 1200,
                "CRS_ARR_TIME": 1400, "CRS_ELAPSED_TIME": 120,
                "FL_DATE": "2019-01-01",
            },
        ])
        with tempfile.NamedTemporaryFile(suffix=".csv") as handle:
            frame.to_csv(handle.name, index=False)
            loaded = load_bts_instance(handle.name)
        self.assertEqual(sorted(loaded["flights"]), [1, 3])
        self.assertEqual(loaded["metadata"]["flight_id_basis"], "raw-row-1-based")
        self.assertEqual(loaded["metadata"]["time_basis"], "utc")


if __name__ == "__main__":
    unittest.main()
