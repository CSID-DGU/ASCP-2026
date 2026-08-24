import unittest
from unittest.mock import patch

from evaluation import evaluate_ip


class MultiAirlineEvalTests(unittest.TestCase):
    def test_runs_each_bts_airline_with_own_window_and_paths(self):
        calls = []

        def fake_evaluate_full(**kwargs):
            calls.append(kwargs)
            return {"status": "Optimal", "coverage": 1.0}

        with patch.object(evaluate_ip, "evaluate_full", side_effect=fake_evaluate_full):
            result = evaluate_ip.evaluate_multi_airline(
                "model.pt",
                completion_report_path="result/completion.json",
                dual_trace_path="result/trace.json",
                save_json_path="result/selected.json",
                data_path=None,
            )

        self.assertEqual([call["airline"] for call in calls], ["delta", "alaska", "jetblue"])
        self.assertEqual([call["window_days"] for call in calls], [6, 6, 8])
        self.assertEqual(calls[0]["completion_report_path"], "result/completion_delta.json")
        self.assertEqual(calls[1]["dual_trace_path"], "result/trace_alaska.json")
        self.assertEqual(calls[2]["save_json_path"], "result/selected_jetblue.json")
        self.assertEqual(set(result), {"delta", "alaska", "jetblue"})

    def test_rejects_single_data_path_for_multi(self):
        with self.assertRaises(ValueError):
            evaluate_ip.evaluate_multi_airline("model.pt", data_path="one.csv")


if __name__ == "__main__":
    unittest.main()
