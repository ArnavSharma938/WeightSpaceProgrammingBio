"""Export Experiment 1 text inputs; never evaluate a CNN or invoke the harness.

Only checkpoint scalars replace the six placeholders in the supplied prompt.
The random assignment and all provenance remain in a separate private manifest.
"""
from __future__ import annotations

import argparse
import ast
import datetime as dt
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import re
import secrets
import sys
import textwrap

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from protein_cnns import common, training
from protein_cnns.model import ProteinCNN

# Alias, authoritative state_dict name, shape, user-supplied placeholder.
TENSORS = (
    ("W1", "conv1.weight", (32, 21, 7), "COMPLETE_INDEXED_CONV1_WEIGHTS"),
    ("b1", "conv1.bias", (32,), "COMPLETE_INDEXED_CONV1_BIASES"),
    ("W2", "conv2.weight", (32, 32, 7), "COMPLETE_INDEXED_CONV2_WEIGHTS"),
    ("b2", "conv2.bias", (32,), "COMPLETE_INDEXED_CONV2_BIASES"),
    ("W3", "classifier.weight", (2, 2240), "COMPLETE_INDEXED_CLASSIFIER_WEIGHTS"),
    ("b3", "classifier.bias", (2,), "COMPLETE_INDEXED_CLASSIFIER_BIASES"),
)
COUNT = 16_418
PROMPT = """THE OBSERVATION
The network receives a sequence of 70 amino acids, ordered from the N-terminal side of the supplied segment to its C-terminal side. Positions are numbered 0 through 69.

The sequence is represented by a 21 x 70 array x, with channels first and sequence positions second. Each position is one-hot encoded. Channels 0 through 19 correspond, in order, to:
A C D E F G H I K L M N P Q R S T V W Y

Channel 20 is reserved for padding. The training inputs contain 70 canonical amino acids, so this channel is zero for those inputs.

THE OUTPUTS
The network produces two logits, z[0] and z[1], for two classes. Softmax converts the logits into class probabilities. The biological meanings of the classes are not supplied.

THE NETWORK
A convolutional neural network with two convolutional layers, ReLU activations, and a linear output layer:

h1 = ReLU(Conv1d(x, W1, b1, padding=3))
h2 = ReLU(Conv1d(h1, W2, b2, padding=3))
v[70*c + p] = h2[c, p]
z = W3 @ v + b3

Tensor shapes:
W1: 32 x 21 x 7
b1: 32
W2: 32 x 32 x 7
b2: 32
W3: 2 x 2240
b3: 2

For each convolution, weight axes are output channel, input channel, and kernel index. Stride and dilation are both 1, with one group. Kernel index k contributes from input position p+k-3 to output position p. Values outside the sequence boundaries are zero. This convolutional zero-padding is distinct from input channel 20.

Both hidden arrays have 32 channels and 70 positions. Flattening is channel-major: all 70 positions of channel 0 come first, followed by all positions of channel 1, and so on. Row j of W3 produces output logit j.

There are 16,418 parameters in total. The network was trained with supervised cross-entropy loss to distinguish two classes of protein sequences.

QUESTION
The numerical parameters below come from a trained protein-sequence classifier. Identify the classification task it learned by examining those parameters. No computational assistance is available: do not execute code, use a calculator, or run the network. Give your answer in a single response.

Provide the following fields in order:

Observations: Describe structure in the amino-acid channel weights, convolutional filters, position-dependent output weights, and biases. Cite specific tensor indices and numerical values supporting your observations.

Reasoning: Explain which protein classification tasks are consistent with those observations, which alternatives are inconsistent, and why.

Task: Identify the learned classification task as specifically as the evidence allows, using its established name where applicable.

Description: Explain what the network distinguishes in one or two sentences.

Confidence: State your confidence as a percentage from 0 to 100.

THE NETWORK'S WEIGHTS
All parameter values below are rounded to three decimal places. Indices are zero-based.

W1:
{COMPLETE_INDEXED_CONV1_WEIGHTS}

b1:
{COMPLETE_INDEXED_CONV1_BIASES}

W2:
{COMPLETE_INDEXED_CONV2_WEIGHTS}

b2:
{COMPLETE_INDEXED_CONV2_BIASES}

W3:
{COMPLETE_INDEXED_CLASSIFIER_WEIGHTS}

b3:
{COMPLETE_INDEXED_CLASSIFIER_BIASES}
"""


def sha(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def function_body(function):
    return ast.dump(ast.Module(body=ast.parse(textwrap.dedent(inspect.getsource(function))).body[0].body, type_ignores=[]))


def verify_repository():
    """Inspect modules and source, without executing the forward computation."""
    require(common.ALPHABET == "ACDEFGHIKLMNPQRSTVWY" and common.PAD_INDEX == 20
            and common.SEQUENCE_LENGTH == 70, "Repository encoding differs from supplied prompt")
    require(common.encode(common.ALPHABET).tolist() == list(range(20)), "Alphabet encoder changed")
    with torch.random.fork_rng(devices=[]):
        model = ProteinCNN().cpu()
    require(list(dict(model.named_children())) == ["conv1", "conv2", "classifier"], "Architecture modules differ")
    for layer, inputs in ((model.conv1, 21), (model.conv2, 32)):
        require(type(layer) is nn.Conv1d and layer.in_channels == inputs and layer.out_channels == 32
                and layer.kernel_size == (7,) and layer.padding == (3,) and layer.stride == (1,)
                and layer.dilation == (1,) and layer.groups == 1 and layer.padding_mode == "zeros"
                and layer.bias is not None, "Convolution differs from supplied prompt")
    head = model.classifier
    require(type(head) is nn.Linear and head.in_features == 2240 and head.out_features == 2
            and head.bias is not None, "Output layer differs from supplied prompt")
    expected_forward = ast.parse("""def forward(self, x):
    x = torch.relu(self.conv1(x))
    x = torch.relu(self.conv2(x))
    return self.classifier(x.flatten(start_dim=1))
""").body[0].body
    require(function_body(ProteinCNN.forward) == ast.dump(ast.Module(body=expected_forward, type_ignores=[])), "Forward computation changed; review scientific prompt")
    expected_encoding = ast.parse("""def one_hot(encoded):
    return functional.one_hot(encoded.long(), num_classes=21).float().transpose(1, 2).contiguous()
""").body[0].body
    require(function_body(training.one_hot) == ast.dump(ast.Module(body=expected_encoding, type_ignores=[])), "Training one-hot orientation changed")
    source = ast.parse(inspect.getsource(training.train_task))
    require(any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "nn"
                and node.func.attr == "CrossEntropyLoss" for node in ast.walk(source)), "Cross-entropy training not found")
    require(model.parameter_count == COUNT and not dict(model.named_buffers()), "Unexpected inference parameters/buffers")
    require(list(model.state_dict()) == [name for _, name, _, _ in TENSORS], "Repository parameter ordering changed")
    return model


def validate_state(state, model):
    require(isinstance(state, dict) and list(state) == [name for _, name, _, _ in TENSORS],
            "Checkpoint must contain exactly six tensors in repository order")
    for _, name, shape, _ in TENSORS:
        value = state[name]
        require(isinstance(value, torch.Tensor) and tuple(value.shape) == shape, f"Unexpected shape/type: {name}")
        require(value.device.type == "cpu" and value.dtype == torch.float32 and value.layout == torch.strided,
                f"Expected CPU full-precision float32 tensor: {name}")
        require(bool(torch.isfinite(value).all()), f"Non-finite checkpoint value: {name}")
    require(sum(t.numel() for t in state.values()) == COUNT, "Checkpoint scalar count differs")
    model.load_state_dict(state, strict=True)


def load_checkpoints():
    model = verify_repository()
    require(tuple(common.TASKS) == ("signal_peptide", "transmembrane", "coiled_coil"), "Expected exactly three trained tasks")
    records = []
    for task in common.TASKS:
        relative = Path("runs/seed_20261103") / task / "checkpoints/best_state_dict.pt"
        path = ROOT / relative
        before = sha(path.read_bytes())
        state = torch.load(path, map_location="cpu", weights_only=True)
        validate_state(state, model)
        require(sha(path.read_bytes()) == before, "Checkpoint changed while being loaded")
        summary = json.loads((path.parent.parent / "training_summary.json").read_text(encoding="utf-8"))
        require(summary["best_checkpoint_sha256"] == before, "Checkpoint hash differs from training record")
        records.append({"task": task, "checkpoint": relative.as_posix(), "checkpoint_sha256": before, "state": state})
    return records


def format_tensor(alias, tensor):
    if tensor.ndim == 3:
        return "\n".join(f"{alias}[out={out}, in={inp}, k=0:7] = [" +
                         ", ".join(f"{v:.3f}" for v in tensor[out, inp].tolist()) + "]"
                         for out in range(tensor.shape[0]) for inp in range(tensor.shape[1]))
    if tensor.ndim == 2:
        return "\n".join(f"{alias}[row={row}, v=0:2240] = [" +
                         ", ".join(f"{v:.3f}" for v in tensor[row].tolist()) + "]"
                         for row in range(tensor.shape[0]))
    return "\n".join(f"{alias}[{index}] = {v:.3f}" for index, v in enumerate(tensor.tolist()))


def render(state):
    text = PROMPT
    for alias, name, _, placeholder in TENSORS:
        text = text.replace("{" + placeholder + "}", format_tensor(alias, state[name]))
    return text


def verify_export(text, state):
    """Parse every emitted index/value independently; compare all rounded scalars."""
    require("{" not in text and "}" not in text and "COMPLETE_INDEXED" not in text, "Unresolved placeholder")
    pattern = re.escape(PROMPT)
    for _, _, _, placeholder in TENSORS:
        pattern = pattern.replace(re.escape("{" + placeholder + "}"), "(.*?)")
    match = re.fullmatch(pattern, text, flags=re.DOTALL)
    require(match is not None, "Descriptive text differs from the supplied prompt")
    coverage = {}
    for (alias, name, shape, _), payload in zip(TENSORS, match.groups()):
        values = []
        lines = payload.splitlines()
        if len(shape) == 3:
            require(len(lines) == shape[0] * shape[1], f"Missing/duplicate {alias} filter lines")
            for line_index, line in enumerate(lines):
                expected = f"{alias}[out={line_index // shape[1]}, in={line_index % shape[1]}, k=0:7] = ["
                require(line.startswith(expected) and line.endswith("]"), f"Incorrect {alias} filter index/order")
                tokens = line[len(expected):-1].split(", ")
                require(len(tokens) == 7, f"Incorrect {alias} kernel coverage")
                values.extend(tokens)
        elif len(shape) == 2:
            require(len(lines) == shape[0], f"Missing/duplicate {alias} rows")
            for row, line in enumerate(lines):
                expected = f"{alias}[row={row}, v=0:2240] = ["
                require(line.startswith(expected) and line.endswith("]"), f"Incorrect {alias} row/order")
                tokens = line[len(expected):-1].split(", ")
                require(len(tokens) == 2240, "Missing classifier coefficients")
                values.extend(tokens)
        else:
            require(len(lines) == shape[0], f"Missing/duplicate {alias} biases")
            for index, line in enumerate(lines):
                expected = f"{alias}[{index}] = "
                require(line.startswith(expected), f"Incorrect {alias} bias index/order")
                values.append(line[len(expected):])
        require(all(re.fullmatch(r"-?\d+\.\d{3}", value) for value in values), f"Noncanonical decimal in {alias}")
        expected_values = [f"{value:.3f}" for value in state[name].reshape(-1).tolist()]
        require(values == expected_values, f"Rounded values/order differ from checkpoint: {alias}")
        coverage[alias] = len(values)
    require(len(coverage) == 6 and sum(coverage.values()) == COUNT, "Incomplete parameter coverage")
    return coverage


def exclusive_write(path, data):
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def prepare(out):
    out = Path(out).resolve()
    private = out.parent / "private" / f"{out.name}_manifest.json"
    if out.exists() or private.exists():
        raise FileExistsError(f"Refusing to overwrite prepared output or private manifest: {out}; {private}")
    records = load_checkpoints()
    seed = secrets.randbits(256)
    permutation = random.Random(seed).sample(range(3), 3)
    manifest = {
        "private_do_not_submit": True,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "permutation_seed_hex": f"{seed:064x}",
        "permutation": permutation,
        "rounding": "Python float .3f formatting; nearest, ties-to-even; negative zero retained",
        "template_sha256": sha(PROMPT.encode("utf-8")),
        "exporter_sha256": sha(Path(__file__).read_bytes()),
        "repository_source_sha256": {name: sha((ROOT / "src/protein_cnns" / name).read_bytes())
                                     for name in ("model.py", "common.py", "training.py", "data.py")},
        "torch_version": torch.__version__,
        "models": [],
    }
    prepared = []
    for index, checkpoint_index in enumerate(permutation, 1):
        record = records[checkpoint_index]
        text = render(record["state"])
        coverage = verify_export(text, record["state"])
        raw = text.encode("utf-8")
        name = f"model_{index:03d}.txt"
        manifest["models"].append({
            "id": f"model_{index:03d}", "prompt_file": name,
            "task": record["task"], "checkpoint": record["checkpoint"],
            "checkpoint_sha256": record["checkpoint_sha256"],
            "prompt_sha256": sha(raw), "prompt_bytes": len(raw), "prompt_characters": len(text),
            "coverage": coverage, "tensor_count": 6, "scalar_count": sum(coverage.values()),
        })
        prepared.append((name, raw, record["state"]))
    # Reserve the private manifest exclusively before publishing any prompts.
    # A partial interrupted preparation is preserved and never silently replaced.
    private.parent.mkdir(parents=True, exist_ok=True)
    exclusive_write(private, (json.dumps(manifest, indent=2) + "\n").encode("utf-8"))
    out.mkdir(parents=True, exist_ok=False)
    for name, raw, state in prepared:
        path = out / name
        exclusive_write(path, raw)
        saved = path.read_bytes()
        require(saved == raw, "Written prompt differs from verified bytes")
        verify_export(saved.decode("utf-8"), state)
        print(f"{name}: 6 tensors; {COUNT:,} scalars; {len(saved):,} UTF-8 bytes; SHA-256 {sha(saved)}")
    print(f"Private mapping (never submit): {private}")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        prepare(args.out)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"Export stopped: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
