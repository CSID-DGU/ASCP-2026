import json
import tempfile
import unittest
from pathlib import Path

from evaluation.completion_report import build_completion_report, render_completion_table, save_completion_report
from evaluation.completion_runner import solve_completion_stages


def column(column_id, legs, source_type):
    return {"column_id": column_id, "legs": legs, "cost": 2, "source_type": source_type, "is_legal": True}


class CompletionReportTests(unittest.TestCase):
    def setUp(self):
        self.ids = [10, 20, 30, 40]
        self.stages = solve_completion_stages(
            [column("p", [10], "policy"), column("s", [20], "salvage"), column("r", [30], "rescue")],
            self.ids, reposition_flight_ids=[], reserve_flight_ids=[], artificial_penalty=100,
        )

    def test_report_keeps_direct_and_completion_metrics_separate(self):
        report = build_completion_report(self.stages, self.ids)
        self.assertEqual(report["direct_candidate_coverage"], 0.25)
        self.assertEqual(report["post_rescue_candidate_coverage"], 0.75)
        self.assertEqual(report["completion_coverage"], 1.0)
        self.assertEqual(report["artificial_flight_ids"], [40])

    def test_json_and_table_output(self):
        report = build_completion_report(self.stages, self.ids)
        table = render_completion_table(report)
        self.assertIn("policy | Infeasible | 25.00", table)
        self.assertIn("artificial | Optimal", table)
        with tempfile.TemporaryDirectory() as directory:
            path = save_completion_report(report, Path(directory) / "result.json")
            loaded = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(loaded["schema_version"], "v2-completion-1.0")
        self.assertEqual(loaded["artificial_count"], 1)

    def test_wrong_stage_order_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "순서"):
            build_completion_report(self.stages[:-1], self.ids)


if __name__ == "__main__":
    unittest.main()
