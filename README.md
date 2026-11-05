# Protein weight-space experiments

The initial pure-reasoning runner is `runner/pure_reasoning.py`. It submits
separately prepared UTF-8 prompt files through fresh `codex exec` processes,
using `gpt-6.1-sol`, high reasoning effort, neutral instructions, and no native
tools. It has no Python package dependencies and does not import the CNN code.
See [runner/README.md](runner/README.md) for exact PowerShell commands,
isolation checks, logging, retries, and restart recovery.

## Prepare Experiment 1 inputs

```powershell
.\.venv\Scripts\python.exe scripts\prepare_reasoning_inputs.py --out prompts/exp1
```

This exports the three existing best checkpoints as `model_001.txt` through
`model_003.txt`, each with the supplied experimental prose and all 16,418
parameters rounded to three decimals. It verifies architecture and encoding,
loads only CPU weights, and parses every exported index and scalar back against
the checkpoint. It does not evaluate a CNN or invoke Codex.

The random permutation, checkpoint identities/hashes, prompt hashes, and coverage
are saved separately in `prompts/private/exp1_manifest.json`. Keep that manifest
private and never submit it as an experimental message. Existing output or a
private manifest causes the exporter to stop rather than overwrite either.

To prepare the corrected original-style question while preserving the original
anonymous checkpoint identities:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_reasoning_inputs.py --out prompts/exp1_original_style --reuse-mapping prompts/private/exp1_manifest.json
```

The corrected prompts have their own private manifest at
`prompts/private/exp1_original_style_manifest.json`; original inputs and runs
remain separate.

Exporter tests (no network, inference, or Codex):

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_prepare_reasoning_inputs.py -v
```

This project implements and trains only the three CNN classifiers specified in
`Experiment 1 CNN (also experiment 4)` of the supplied experiment packet:

1. N-terminal cleavable signal peptide vs. no signal peptide.
2. Annotated transmembrane segment (at least 15 contiguous `M` labels) vs. no
   `M` label, among SignalP `NO_SP` records.
3. Coiled-coil window (at least 21 contiguous register labels `a`-`g`) vs.
   windows from CoCoNat negative proteins.

Every input is exactly 70 canonical amino acids. The model is exactly
Conv1d(21,32,7,padding=3), ReLU, Conv1d(32,32,7,padding=3), ReLU, channel-major
flatten, and Linear(2240,2), for 16,418 trainable parameters.

## Reproduce

Windows PowerShell, Python 3.10-3.12, and an internet connection are required
for a clean run. The checked-in raw files can also be reused offline.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
.\.venv\Scripts\python.exe scripts\download_data.py
.\.venv\Scripts\python.exe scripts\cnns.py prepare
.\.venv\Scripts\python.exe scripts\cnns.py train --device cpu
.\.venv\Scripts\python.exe scripts\cnns.py evaluate --device cpu
```

Or run `powershell -ExecutionPolicy Bypass -File scripts\reproduce.ps1`.

The download script verifies pinned SHA-256 hashes. Preparation writes the
resolved examples, private identifiers, extraction decisions, split audits,
and dataset manifest under `data/processed/`. Training writes one immutable run
directory per task under `runs/`. The training report is `reports/cnn_training_report.md`.

## Frozen split and extraction policy

- SignalP partitions 0, 1, and 2 are train, validation, and test respectively.
- CoCoNat folds 1-9 are training, fold 0 is validation, and the publisher's
  blind-test files are test.
- CoCoNat positive windows are centered on the midpoint of the longest
  contiguous register (`a`-`g`) run; ties use the earliest run. Bounds are
  shifted only enough to obtain 70 residues. Negative windows are centered.
- Training sets are downsampled deterministically to equal positive/negative
  counts. Validation and test prevalence is retained.
- Exact ID/sequence overlap is audited. Training windows with at least 80%
  aligned identity to a validation or test window are removed before balancing.

See the final report for counts, metrics, controls, limitations, and provenance.
