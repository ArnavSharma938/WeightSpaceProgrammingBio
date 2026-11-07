# Protein weight-space experiments

The repository contains three trained 16,418-parameter protein CNNs, two completed
weight-reading experiments, and a generic reasoning harness.

## Repository layout

~~~text
data/                                  prepared datasets, audits, and provenance
prompts/experiment_1_initial/           initial adapted question, three models
prompts/experiment_2_original_style/    original-style question, same models
prompts/private/                       checkpoint mappings and hashes; never submit
runs/seed_20261103/                     CNN checkpoints, training and evaluation
runs/reasoning_harness/experiment_1_initial/
runs/reasoning_harness/experiment_2_original_style/
runner/reasoning_harness.py
runner/reasoning_harness.toml
reports/reasoning_experiments_1_and_2.md
reports/reasoning_harness.md
reports/cnn_training_report.md
~~~

The experiment report includes all six exact responses. Experiments 1 and 2
here label the initial and original-style prompt runs of the same CNN task;
they do not refer to the separate RBM experiment in the original project packet.

## Reasoning harness

The harness submits prepared UTF-8 files through fresh codex exec processes,
using gpt-6.1-sol, high reasoning effort, neutral instructions, and no native
tools. It has no Python package dependencies or CNN imports.

See [the harness documentation](reports/reasoning_harness.md) for commands,
isolation, logging, retries, quota waiting, interruption recovery, and locking.

~~~powershell
.\.venv\Scripts\python.exe runner\reasoning_harness.py check
.\.venv\Scripts\python.exe runner\reasoning_harness.py status --run-dir runs\reasoning_harness\experiment_2_original_style
~~~

## Prepared prompts

Each model file contains identical factual prose and all six indexed tensors
rounded to three decimal places. Anonymous model identities match across the
two prompt folders. The corrected question is the exporter's current template.

For a new output directory while retaining the established identities:
~~~powershell
.\.venv\Scripts\python.exe scripts\prepare_reasoning_inputs.py --out prompts\new_batch --reuse-mapping prompts\private\experiment_1_initial_manifest.json
~~~

Output and a separate private manifest are created exclusively; existing files
are never overwritten. This command neither evaluates a CNN nor invokes Codex.

~~~powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_prepare_reasoning_inputs.py -v
~~~

## Training and data

The trained tasks are signal-peptide detection, transmembrane-segment detection,
and coiled-coil detection. All inputs have 70 canonical amino acids in the order
ACDEFGHIKLMNPQRSTVWY, followed by an unused padding channel. The exact CNN is
Conv1d(21,32,7,padding=3), ReLU, Conv1d(32,32,7,padding=3), ReLU,
channel-major flatten, and Linear(2240,2).

Prepared arrays, source IDs, split assignments, audits, and manifests are directly
in data/. The NPZ arrays and private example lists remain ignored by Git, as
before; source hashes, URLs, and preparation rules provide reproducibility.

Raw source downloads are needed only to rebuild the prepared datasets. They live
outside the repository in the local cache:
%LOCALAPPDATA%/protein_weight_space_cnns/source_data on Windows, or
~/.cache/protein_weight_space_cnns/source_data otherwise. The downloader verifies
pinned SHA-256 hashes and reuses valid cached files. Training/evaluation from
existing prepared arrays does not require this cache.

To recreate the environment and dataset, then train and evaluate:
~~~powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
.\.venv\Scripts\python.exe scripts\download_data.py
.\.venv\Scripts\python.exe scripts\cnns.py prepare
.\.venv\Scripts\python.exe scripts\cnns.py train --device cpu
.\.venv\Scripts\python.exe scripts\cnns.py evaluate --device cpu
~~~

The editable install can regenerate ignored .egg-info packaging metadata; those
generated files do not belong in source control. Alternatively run
scripts/reproduce.ps1 for the complete workflow. Existing trained checkpoints
remain under runs/seed_20261103/.

## Frozen dataset policy

- SignalP partitions 0, 1, and 2 are train, validation, and test.
- CoCoNat folds 1-9 are train, fold 0 is validation, and the publisher blind set
  is test.
- CoCoNat positive windows are centered on the longest register run of at least
  21 residues; negative windows are centered, one per source protein.
- Transmembrane positives require at least 15 contiguous M annotations.
- Training examples are deterministically balanced; held-out prevalence is kept.
- Training windows at 80% or greater aligned identity to held-out windows are
  removed; duplicate IDs/sequences are audited.

See [the training report](reports/cnn_training_report.md) for counts, metrics,
controls, hashes, and limitations.
