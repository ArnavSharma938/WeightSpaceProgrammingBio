"""Exporter verification only: no CNN inference, training, or Codex requests."""
import contextlib
import io
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_reasoning_inputs as exporter


class ExportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = exporter.verify_repository()
        cls.state = {}
        offset = 0
        for _, name, shape, _ in exporter.TENSORS:
            size = cls.model.state_dict()[name].numel()
            cls.state[name] = (torch.arange(offset, offset + size, dtype=torch.float32) / 1024 - 0.25).reshape(shape)
            offset += size

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        # These guards fail if exporter code ever starts a process or forwards
        # data through any CNN layer. Architecture inspection remains permitted.
        self.guards = contextlib.ExitStack()
        self.guards.enter_context(patch.object(subprocess, "Popen", side_effect=AssertionError("No subprocesses")))
        self.guards.enter_context(patch.object(torch.nn.Conv1d, "forward", side_effect=AssertionError("No inference")))
        self.guards.enter_context(patch.object(torch.nn.Linear, "forward", side_effect=AssertionError("No inference")))

    def tearDown(self):
        self.guards.close()
        self.temp.cleanup()

    def records(self):
        return [{"task": task, "checkpoint": f"private/{task}/best_state_dict.pt",
                 "checkpoint_sha256": str(index) * 64,
                 "state": {key: value + index / 8 for key, value in self.state.items()}}
                for index, task in enumerate(exporter.common.TASKS)]

    def prepare(self):
        with patch.object(exporter, "load_checkpoints", return_value=self.records()), contextlib.redirect_stdout(io.StringIO()):
            return exporter.prepare(self.root / "exp1")

    def test_repository_architecture_without_forward(self):
        self.assertEqual(exporter.verify_repository().parameter_count, 16418)

    def test_actual_best_checkpoints_cpu_weights_only(self):
        with patch.object(exporter.torch, "load", wraps=torch.load) as loading:
            records = exporter.load_checkpoints()
        self.assertEqual(len(records), 3)
        for call in loading.call_args_list:
            self.assertEqual(Path(call.args[0]).name, "best_state_dict.pt")
            self.assertEqual(call.kwargs, {"map_location": "cpu", "weights_only": True})
        for record in records:
            self.assertEqual(len(record["state"]), 6)
            self.assertEqual(sum(value.numel() for value in record["state"].values()), 16418)

    def test_roundtrip_all_six_tensors_and_scalar_coverage(self):
        text = exporter.render(self.state)
        self.assertEqual(exporter.verify_export(text, self.state),
                         {"W1": 4704, "b1": 32, "W2": 7168, "b2": 32, "W3": 4480, "b3": 2})
        self.assertNotIn("{", text)

    def test_fixed_precision_ties_and_negative_zero(self):
        state = {key: value.clone() for key, value in self.state.items()}
        state["classifier.bias"] = torch.tensor([0.0625, -0.0001])
        text = exporter.render(state)
        self.assertIn("b3[0] = 0.062\nb3[1] = -0.000", text)
        exporter.verify_export(text, state)

    def test_anonymity_saved_permutation_and_identical_descriptions(self):
        manifest = self.prepare()
        self.assertEqual(sorted(manifest["permutation"]), [0, 1, 2])
        self.assertEqual(random.Random(int(manifest["permutation_seed_hex"], 16)).sample(range(3), 3), manifest["permutation"])
        self.assertEqual(sorted(path.name for path in (self.root / "exp1").iterdir()),
                         ["model_001.txt", "model_002.txt", "model_003.txt"])
        self.assertEqual(json.loads((self.root / "private/exp1_manifest.json").read_text()), manifest)
        for entry, source_index in zip(manifest["models"], manifest["permutation"]):
            text = (self.root / "exp1" / entry["prompt_file"]).read_text()
            record = self.records()[source_index]
            self.assertEqual(text, exporter.render(record["state"]))
            self.assertEqual(entry["prompt_sha256"], exporter.sha(text.encode()))
            for forbidden in (*exporter.common.TASKS, "SignalP", "CoCoNat", "best_state_dict", "private/", "training_summary"):
                self.assertNotIn(forbidden, text)

    def test_refuses_existing_output_without_changing_any_bytes(self):
        self.prepare()
        before = {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        with self.assertRaises(FileExistsError):
            exporter.prepare(self.root / "exp1")
        self.assertEqual(before, {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()})

    def test_refuses_existing_private_manifest_even_without_prompts(self):
        (self.root / "private").mkdir()
        path = self.root / "private/exp1_manifest.json"
        path.write_bytes(b"private original")
        with self.assertRaises(FileExistsError):
            exporter.prepare(self.root / "exp1")
        self.assertEqual(path.read_bytes(), b"private original")
        self.assertFalse((self.root / "exp1").exists())

    def test_rejects_missing_extra_or_reordered_tensor(self):
        for state in ({key: value for key, value in self.state.items() if key != "classifier.bias"},
                      {**self.state, "extra": torch.zeros(1)}, dict(reversed(list(self.state.items())))):
            with self.assertRaises(ValueError):
                exporter.validate_state(state, self.model)

    def test_rejects_shape_dtype_and_nonfinite_values(self):
        for value in (torch.zeros(1), torch.zeros(32, dtype=torch.float64),
                      torch.full((32,), float("nan")), torch.full((32,), float("inf"))):
            with self.assertRaises(ValueError):
                exporter.validate_state({**self.state, "conv1.bias": value}, self.model)

    def test_rejects_changed_prose_and_placeholders(self):
        text = exporter.render(self.state)
        for modified in (text.replace("THE OBSERVATION", "CHANGED", 1), text + "{PLACEHOLDER}"):
            with self.assertRaises(ValueError):
                exporter.verify_export(modified, self.state)

    def test_rejects_filter_duplication_kernel_reversal_and_row_loss(self):
        text = exporter.render(self.state)
        lines = text.splitlines()
        index = next(i for i, line in enumerate(lines) if line.startswith("W1[out=0, in=0,"))
        duplicate = lines.copy()
        duplicate[index + 1] = duplicate[index]
        reverse = lines.copy()
        label, values = reverse[index].split(" = ")
        reverse[index] = label + " = [" + ", ".join(reversed(values[1:-1].split(", "))) + "]"
        dropped = [line for line in lines if not line.startswith("W3[row=1,")]
        for modified in (duplicate, reverse, dropped):
            with self.assertRaises(ValueError):
                exporter.verify_export("\n".join(modified) + "\n", self.state)

    def test_export_does_not_mutate_checkpoint_tensors(self):
        original = {name: tensor.clone() for name, tensor in self.state.items()}
        exporter.verify_export(exporter.render(self.state), self.state)
        for name, value in self.state.items():
            self.assertTrue(torch.equal(value, original[name]))


if __name__ == "__main__":
    unittest.main()
