"""
Tests for the Week 7-12 additions to trace_scaled_zo.py, run on a tiny fake
model so they finish in seconds on a laptop (no download, no GPU).

    python -m unittest test_week7_12 -v
"""
import io
import json
import math
import contextlib
import tempfile
import types
import unittest
import os

import torch
import torch.nn as nn

import trace_scaled_zo as tz

V = 12
WORDS = {" terrible": 1, " great": 2}


class FakeTok:
    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        if text in WORDS:
            ids = [WORDS[text]]
        else:
            ids = [3 + (sum(map(ord, w)) % (V - 3)) for w in text.split()] or [3]
        if return_tensors == "pt":
            return types.SimpleNamespace(input_ids=torch.tensor([ids]))
        return types.SimpleNamespace(input_ids=ids)  # plain list, like HF


class FakeLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, 6)
        self.fc1 = nn.Linear(6, 8)
        self.fc2 = nn.Linear(8, V)

    def forward(self, ids):
        h = torch.tanh(self.fc1(self.emb(ids)))
        return types.SimpleNamespace(logits=self.fc2(h))


def make_model_and_tok():
    torch.manual_seed(1234)
    model = FakeLM()
    model.eval()
    return model, FakeTok()


def make_data():
    sents = ["good movie great fun", "bad boring awful film", "lovely story",
             "terrible plot", "wonderful acting", "dull and slow", "a joy",
             "a mess", "so fun", "so bad"] * 6
    train = [{"sentence": s, "label": i % 2} for i, s in enumerate(sents)]
    val = [{"sentence": s, "label": (i + 1) % 2} for i, s in enumerate(sents[:20])]
    return train, val


def quiet(fn, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(**kwargs)


class Week7to12Tests(unittest.TestCase):
    def setUp(self):
        self._saved = (tz.load_model_and_tokenizer, tz.load_sst2, tz.DEVICE)
        tz.load_model_and_tokenizer = make_model_and_tok
        tz.load_sst2 = make_data
        tz.DEVICE = "cpu"

    def tearDown(self):
        tz.load_model_and_tokenizer, tz.load_sst2, tz.DEVICE = self._saved

    def run_train(self, **kw):
        base = dict(use_trace_scaling=True, num_steps=6, seed=0,
                    trace_refresh_every=3, batch_size=4)
        base.update(kw)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "log.json")
            quiet(tz.train, log_path=path, **base)
            with open(path) as f:
                return json.load(f)

    # --- evaluation harness -------------------------------------------------
    def test_accuracy_matches_manual_count(self):
        model, tok = make_model_and_tok()
        _, val = make_data()
        result = tz.evaluate_accuracy(model, tok, val)
        correct = 0
        for ex in val:
            ids = tok(tz.build_prompt(ex["sentence"]), return_tensors="pt").input_ids
            logits = model(ids).logits[0, -1, :]
            correct += int(int(logits[[1, 2]].argmax()) == ex["label"])
        self.assertAlmostEqual(result["accuracy"], correct / len(val))
        self.assertEqual(result["n"], len(val))

    def test_eval_is_deterministic_and_does_not_change_the_run(self):
        a = self.run_train(eval_steps=(0, 6))
        b = self.run_train()
        self.assertEqual(a["losses"], b["losses"])      # eval changes nothing
        self.assertEqual([e["step"] for e in a["eval"]], [0, 6])
        again = self.run_train(eval_steps=(0, 6))
        self.assertEqual(a["eval"], again["eval"])

    # --- per-block logging --------------------------------------------------
    def test_block_details_are_strict_json_and_consistent(self):
        log = self.run_train(log_block_details=True)
        self.assertEqual([d["step"] for d in log["block_details"]], [0, 3])
        json.dumps(log["block_details"], allow_nan=False)  # no NaN literals
        total = sum(log["block_sizes"].values())
        for d in log["block_details"]:
            budget = sum(log["block_sizes"][n] * s ** 2
                         for n, s in d["sigmas"].items())
            self.assertAlmostEqual(budget / total, 1.0, places=5)

    # --- shuffled-sigma control --------------------------------------------
    def test_shuffled_control_uses_same_traces_but_different_sigmas(self):
        real = self.run_train(log_block_details=True, sigma_mode="trace")
        ctrl = self.run_train(log_block_details=True, sigma_mode="shuffled")
        self.assertEqual(real["block_details"][0]["traces"],
                         ctrl["block_details"][0]["traces"])
        self.assertNotEqual(real["block_details"][0]["sigmas"],
                            ctrl["block_details"][0]["sigmas"])

    # --- running-mean traces -----------------------------------------------
    def test_trace_average_is_the_mean_of_the_refreshes(self):
        log = self.run_train(log_block_details=True, trace_average=True)
        d0, d1 = log["block_details"]
        for n in d0["traces"]:
            vals = [x for x in (d0["traces"][n], d1["traces"][n]) if x is not None]
            if vals:
                self.assertAlmostEqual(d1["traces_used"][n], sum(vals) / len(vals),
                                       places=4)

    # --- rule / delta / probes options reach compute_sigmas -----------------
    def test_rule_option_changes_sigmas(self):
        a = self.run_train(log_block_details=True, sigma_rule="inv_sqrt")
        b = self.run_train(log_block_details=True, sigma_rule="inv")
        self.assertNotEqual(a["block_details"][0]["sigmas"],
                            b["block_details"][0]["sigmas"])

    # --- several perturbations per step ------------------------------------
    def test_multi_step_with_one_direction_equals_mezo_step(self):
        train, _ = make_data()
        batch = train[:4]

        def final_weights(step_fn):
            model, tok = make_model_and_tok()
            blocks = tz.get_param_blocks(model)
            sigmas = {n: 1.0 for n in blocks}
            import random
            random.seed(7)
            step_fn(model, tok, batch, blocks, sigmas, 1e-3, 1e-2)
            return torch.cat([p.detach().flatten() for p in model.parameters()])

        w1 = final_weights(tz.mezo_step)
        w2 = final_weights(lambda *a: tz.mezo_step_multi(*a, 1))
        self.assertTrue(torch.allclose(w1, w2, atol=1e-7))

    def test_multi_step_restores_weights_when_lr_is_zero(self):
        train, _ = make_data()
        model, tok = make_model_and_tok()
        before = torch.cat([p.detach().flatten() for p in model.parameters()]).clone()
        blocks = tz.get_param_blocks(model)
        tz.mezo_step_multi(model, tok, train[:4], blocks,
                           {n: 1.0 for n in blocks}, 1e-3, 0.0, 4)
        after = torch.cat([p.detach().flatten() for p in model.parameters()])
        self.assertTrue(torch.allclose(before, after, atol=1e-6))

    def test_multi_run_is_logged_and_runs(self):
        log = self.run_train(use_trace_scaling=False, num_perturbations=3)
        self.assertEqual(log["config"]["num_perturbations"], 3)
        self.assertEqual(len(log["losses"]), 6)
        self.assertTrue(all(math.isfinite(x) for x in log["losses"]))


if __name__ == "__main__":
    unittest.main()
