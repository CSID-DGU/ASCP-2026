import random
import unittest

import torch

from evaluation.dual_feedback import (
    normalize_dual,
    build_dual_signal,
    run_iterative_dual_feedback,
    solve_full_universe_lp,
)
from model.decoder import PointerDecoder


class FullUniverseDualTests(unittest.TestCase):
    def test_unknown_solver_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "지원하지 않는 LP solver"):
            solve_full_universe_lp(
                [{"legs": [0], "cost": 1.0}], [0], solver="unknown"
            )

    def test_uncovered_flight_remains_in_lp_with_artificial_dual(self):
        result = solve_full_universe_lp(
            [{"legs": [0], "cost": 2.0}], [0, 1], artificial_penalty=100.0
        )
        self.assertEqual(result["artificial_flight_ids"], [1])
        self.assertEqual(set(result["net_dual"]), {0, 1})
        self.assertGreater(result["net_dual"][1], result["net_dual"][0])

    def test_solver_and_formula_reduced_costs_match(self):
        result = solve_full_universe_lp(
            [
                {"legs": [0], "cost": 2.0},
                {"legs": [0, 1], "cost": 3.0},
            ],
            [0, 1],
            lambda_excess=1.0,
            artificial_penalty=100.0,
        )
        for solver_value, formula_value in zip(
            result["reduced_costs"], result["formula_reduced_costs"]
        ):
            self.assertAlmostEqual(solver_value, formula_value, places=6)

    def test_iterative_generation_reduces_artificial_need(self):
        def generate(signal, iteration):
            self.assertGreater(signal[1], signal[0])
            return [{"legs": [1], "cost": 3.0}]

        result = run_iterative_dual_feedback(
            [{"legs": [0], "cost": 2.0}], [0, 1], generate,
            max_iterations=2, artificial_penalty=100.0,
        )
        self.assertEqual(result["trace"][0]["artificial_count"], 1)
        self.assertEqual(result["last_lp"]["artificial_count"], 0)

    def test_normalization_preserves_keys_and_bounds(self):
        result = normalize_dual({10: -20.0, 20: 10.0})
        self.assertEqual(set(result), {10, 20})
        self.assertEqual(result[10], -1.0)
        self.assertEqual(result[20], 0.5)


    def test_dual_control_modes_preserve_experimental_contract(self):
        lp_result = {
            "net_dual": {10: 8.0, 20: 2.0, 30: -4.0},
            "artificial_flight_ids": [20],
        }
        real = build_dual_signal(lp_result, "real")
        self.assertEqual(build_dual_signal(lp_result, "zero"), {10: 0.0, 20: 0.0, 30: 0.0})
        self.assertEqual(build_dual_signal(lp_result, "uncovered-only"), {10: 0.0, 20: 1.0, 30: 0.0})
        uniform = build_dual_signal(lp_result, "uniform")
        self.assertEqual(len(set(uniform.values())), 1)
        shuffled = build_dual_signal(lp_result, "shuffled", rng=random.Random(7))
        self.assertEqual(set(shuffled), set(real))
        self.assertCountEqual(shuffled.values(), real.values())


    def test_robust_modes_separate_artificial_scale(self):
        lp_result = {
            "net_dual": {10: 1000.0, 20: 8.0, 30: 4.0, 40: 2.0},
            "artificial_flight_ids": [10],
        }
        robust = build_dual_signal(lp_result, "robust-real")
        self.assertEqual(robust, {10: 1.0, 20: 1.0, 30: 0.5, 40: 0.25})
        shuffled = build_dual_signal(
            lp_result, "robust-shuffled", rng=random.Random(7)
        )
        self.assertEqual(set(shuffled), set(robust))
        self.assertCountEqual(shuffled.values(), robust.values())


class DecoderDualBiasTests(unittest.TestCase):
    def test_action_bias_changes_preference_without_unmasking(self):
        torch.manual_seed(0)
        decoder = PointerDecoder(d_model=8, airport_emb_dim=2, constraint_dim=1, n_scalars=1)
        encoded = torch.zeros(2, 8)
        state = torch.zeros(6)
        mask = torch.tensor([1.0, 1.0, 0.0, 1.0])
        base_logits = decoder(encoded, state, mask, return_logits=True)
        biased_logits = decoder(
            encoded, state, mask,
            action_bias=torch.tensor([0.0, 3.0, 100.0, 0.0]),
            return_logits=True,
        )
        self.assertTrue(torch.isneginf(biased_logits[2]))
        self.assertAlmostEqual(
            float(biased_logits[1] - base_logits[1]), 3.0, places=5
        )


if __name__ == "__main__":
    unittest.main()
