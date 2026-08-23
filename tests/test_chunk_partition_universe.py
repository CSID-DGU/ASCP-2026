import unittest
from unittest.mock import patch

from evaluation.evaluate_ip import collect_pool_full


class ChunkUniverseTest(unittest.TestCase):
    @patch("evaluation.evaluate_ip.rollout_subset_global_batch")
    @patch("evaluation.evaluate_ip.partition_connected_chunks")
    def test_base_injection_does_not_remove_original_flight(
        self, partition_mock, rollout_mock
    ):
        base_flight = {
            "id": 0, "global_id": 0, "origin": 0, "dest": 1,
            "dep_time": 1.0, "arr_time": 2.0,
        }
        tail = {
            "id": 1, "global_id": 1, "origin": 2, "dest": 3,
            "dep_time": 3.0, "arr_time": 4.0,
        }
        partition_mock.return_value = [[dict(base_flight)], [dict(tail)]]
        seen_chunks = []

        def capture(chunk, *args, **kwargs):
            seen_chunks.append({f["global_id"] for f in chunk})
            return [[] for _ in range(kwargs["B"])]

        rollout_mock.side_effect = capture
        constraint = {
            "base_airport": 0, "min_conn": 0.5, "max_conn": 3.0,
            "min_rest": 10.0, "max_duty": 13.0, "max_legs": 8,
            "max_duty_periods": 2, "max_pairing_days": 5,
            "min_pairing_legs": 2,
        }
        collect_pool_full(
            [[base_flight, tail]], [0], constraint, object(), object(),
            n_rollouts_per_chunk=1, subset_size=1,
        )
        self.assertIn({0, 1}, seen_chunks)


if __name__ == "__main__":
    unittest.main()
