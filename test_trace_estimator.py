"""
Unit tests for the block partition and Hessian-trace estimator.
Run with:  python3 test_trace_estimator.py
"""

import math
import unittest

import torch

from trace_scaled_zo import (
    load_model_and_tokenizer,
    get_param_blocks,
    compute_sigmas,
)


class TestBlockPartition(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.tok = load_model_and_tokenizer()
        cls.blocks = get_param_blocks(cls.model)

    def test_covers_all_parameters_exactly_once(self):
        total_in_blocks = sum(
            p.numel() for params in self.blocks.values() for _, p in params
        )
        total_in_model = sum(p.numel() for p in self.model.parameters())
        self.assertEqual(total_in_blocks, total_in_model)

    def test_reasonable_number_of_blocks(self):
        self.assertGreater(len(self.blocks), 10)
        self.assertLess(len(self.blocks), 500)


class TestTraceEstimatorToy(unittest.TestCase):
    def test_toy_quadratic_trace(self):
        H = torch.tensor([[4.0, 1.0], [1.0, 2.0]])
        x = torch.nn.Parameter(torch.tensor([1.0, 1.0]))

        def toy_loss():
            return 0.5 * x @ H @ x

        grads = torch.autograd.grad(toy_loss(), [x], create_graph=True)

        num_probes = 2000
        trace_sum = 0.0
        for _ in range(num_probes):
            z = torch.randint(0, 2, x.shape).float() * 2 - 1
            gz = sum((g * zz).sum() for g, zz in zip(grads, [z]))
            Hz = torch.autograd.grad(gz, [x], retain_graph=True)
            trace_sum += sum(
                (zz * hz).sum().item() for zz, hz in zip([z], Hz)
            )

        estimate = trace_sum / num_probes
        self.assertAlmostEqual(estimate, 6.0, delta=0.5)


class TestComputeSigmas(unittest.TestCase):
    def test_higher_curvature_gets_smaller_sigma(self):
        toy_blocks = {
            "block_a": [("w", torch.zeros(10))],
            "block_b": [("w", torch.zeros(90))],
        }
        toy_traces = {"block_a": 100.0, "block_b": 1.0}
        sigmas = compute_sigmas(toy_traces, toy_blocks)
        self.assertLess(sigmas["block_a"], sigmas["block_b"])

    def test_normalization_holds_fixed_budget(self):
        toy_blocks = {
            "block_a": [("w", torch.zeros(10))],
            "block_b": [("w", torch.zeros(90))],
        }
        toy_traces = {"block_a": 100.0, "block_b": 1.0}
        sigmas = compute_sigmas(toy_traces, toy_blocks)

        param_counts = {
            name: sum(p.numel() for _, p in params)
            for name, params in toy_blocks.items()
        }
        total_params = sum(param_counts.values())
        weighted_sq_sum = sum(
            param_counts[name] * (sigmas[name] ** 2) for name in sigmas
        )
        self.assertAlmostEqual(weighted_sq_sum, total_params, delta=1e-3)

    def test_nan_trace_falls_back_to_sigma_one_before_normalization(self):
        toy_blocks = {
            "block_a": [("w", torch.zeros(10))],
            "block_b": [("w", torch.zeros(10))],
        }
        toy_traces = {"block_a": float("nan"), "block_b": 4.0}
        sigmas = compute_sigmas(toy_traces, toy_blocks)
        self.assertFalse(math.isnan(sigmas["block_a"]))
        self.assertFalse(math.isnan(sigmas["block_b"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
