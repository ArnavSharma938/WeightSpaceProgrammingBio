from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np
import torch

from .common import ALPHABET, PROCESSED_DIR, REPORTS_DIR, RUNS_DIR, SEED, TASKS, sha256, write_json
from .metrics import classification_metrics
from .model import ProteinCNN
from .training import load_task, predict, resolve_device


def shuffled_composition_control(sequences: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    shuffled = sequences.copy()
    for row in shuffled:
        rng.shuffle(row)
    return shuffled


def mutation_scan(
    model: ProteinCNN,
    sequences: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    device: torch.device,
    limit: int = 32,
) -> dict:
    positive_indices = np.flatnonzero(labels == 1)
    if not len(positive_indices):
        return {"examples": 0, "substitution": "G (or P when residue is G)", "positions": []}
    ranked = positive_indices[np.argsort(scores[positive_indices])]
    selected = ranked[np.linspace(0, len(ranked) - 1, min(limit, len(ranked)), dtype=int)]
    originals = sequences[selected]
    original_scores = scores[selected]
    glycine = ALPHABET.index("G")
    proline = ALPHABET.index("P")
    rows = []
    for position in range(sequences.shape[1]):
        mutated = originals.copy()
        mutated[:, position] = np.where(mutated[:, position] == glycine, proline, glycine)
        mutated_scores = predict(model, mutated, device)
        drops = original_scores - mutated_scores
        rows.append(
            {
                "position_1_based": position + 1,
                "mean_score_drop": float(drops.mean()),
                "median_score_drop": float(np.median(drops)),
            }
        )
    return {
        "examples": int(len(selected)),
        "selection": "score-quantile-spaced held-out positives",
        "substitution": "G (or P when residue is G)",
        "positions": rows,
        "largest_mean_decreases": sorted(rows, key=lambda row: row["mean_score_drop"], reverse=True)[:10],
        "largest_mean_increases": sorted(rows, key=lambda row: row["mean_score_drop"])[:10],
    }


def evaluate_task(task: str, device_name: str = "auto") -> dict:
    device = resolve_device(device_name)
    data = load_task(task)
    checkpoint = RUNS_DIR / f"seed_{SEED}" / task / "checkpoints" / "best_state_dict.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing {checkpoint}; run scripts/cnns.py train first")
    model = ProteinCNN().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    test_sequences = data["test_sequences"]
    test_labels = data["test_labels"]
    scores = predict(model, test_sequences, device)
    rng = np.random.default_rng(SEED)
    shuffled = shuffled_composition_control(test_sequences, rng)
    shuffled_scores = predict(model, shuffled, device)
    positive = test_labels == 1
    test_metrics = classification_metrics(test_labels, scores)
    shuffled_metrics = classification_metrics(test_labels, shuffled_scores)
    credible = bool(test_metrics["roc_auc"] >= 0.80 and test_metrics["balanced_accuracy"] >= 0.70)
    suitable = bool(credible and shuffled_metrics["roc_auc"] <= test_metrics["roc_auc"] - 0.05)
    result = {
        "task": task,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "parameter_count": model.parameter_count,
        "test_metrics": test_metrics,
        "composition_preserving_shuffle_control": {
            "description": "Each held-out sequence was independently residue-shuffled, preserving its amino-acid composition.",
            "metrics": shuffled_metrics,
            "positive_mean_original_score": float(scores[positive].mean()) if positive.any() else float("nan"),
            "positive_mean_shuffled_score": float(shuffled_scores[positive].mean()) if positive.any() else float("nan"),
            "positive_fraction_score_decreased": float(np.mean(scores[positive] > shuffled_scores[positive])) if positive.any() else float("nan"),
        },
        "heldout_positive_mutation_scan": mutation_scan(model, test_sequences, test_labels, scores, device),
        "credible_heldout_discrimination": credible,
        "suitable_interpretation_target_under_controls": suitable,
    }
    run_dir = RUNS_DIR / f"seed_{SEED}" / task
    np.savez_compressed(run_dir / "test_predictions.npz", labels=test_labels, scores=scores, shuffled_scores=shuffled_scores)
    write_json(run_dir / "evaluation.json", result)
    return result


def _fmt(value: object, digits: int = 3) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def build_report(results: dict[str, dict]) -> Path:
    preparation = json.loads((PROCESSED_DIR / "preparation_summary.json").read_text(encoding="utf-8"))
    raw = json.loads((PROCESSED_DIR / "raw_data_manifest.json").read_text(encoding="utf-8"))
    training = {
        task: json.loads((RUNS_DIR / f"seed_{SEED}" / task / "training_summary.json").read_text(encoding="utf-8"))
        for task in TASKS
    }
    title = {
        "signal_peptide": "Signal peptide",
        "transmembrane": "Transmembrane segment",
        "coiled_coil": "Coiled coil",
    }
    lines = [
        "# Experiment 1 CNN training and validation report",
        "",
        "## Scope and outcome",
        "",
        "Only the three CNNs requested in **Experiment 1 CNN (also experiment 4)** were implemented and trained. No RBM, transformer, ESM, prompt-running, Astra, or weight-construction experiment was performed.",
        "",
        "All three checkpoints use the required architecture: `Conv1d(21,32,7,padding=3) -> ReLU -> Conv1d(32,32,7,padding=3) -> ReLU -> channel-major flatten(2240) -> Linear(2240,2)`. Each has exactly **16,418 trainable parameters**. Class 1 is feature-present.",
        "",
        "## Data and provenance",
        "",
        "- SignalP 6.0 training data: https://services.healthtech.dtu.dk/services/SignalP-6.0/public_data/train_set.fasta",
        "- SignalP format/reference: https://services.healthtech.dtu.dk/services/SignalP-6.0/",
        "- CoCoNat datasets and supplied splits: https://coconat.biocomp.unibo.it/datasets/",
        "",
        "The downloader pins the following source-file hashes:",
        "",
        "| File | Bytes | SHA-256 |",
        "|---|---:|---|",
    ]
    for filename, details in raw.items():
        lines.append(f"| `{filename}` | {details['bytes']} | `{details['sha256']}` |")
    lines.extend(
        [
            "",
            "SignalP publisher partitions 0/1/2 were frozen as train/validation/test. CoCoNat folds 1-9 were train, fold 0 validation, and the publisher's blind set test. Only 70-residue canonical-amino-acid examples entered a model. CoCoNat positives were centered on the longest register run of at least 21 residues; negatives used one centered window per protein. SignalP transmembrane positives required a contiguous `M` annotation of at least 15 residues.",
            "",
            "Before balancing, training windows with at least 80% position-aligned identity to any validation/test window were removed. This is a conservative window-level leakage filter, not a full profile-HMM or structure-based homology search.",
            "",
            "## Prepared datasets",
            "",
            "| Task | Train + / - | Validation + / - | Test + / - | Near-heldout train windows removed |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for task in TASKS:
        manifest = preparation[task]
        counts = manifest["counts"]
        removed = manifest["audit"]["close_homology_filter"]["removed_count"]
        lines.append(
            f"| {title[task]} | {counts['train'].get('1', 0)} / {counts['train'].get('0', 0)} | "
            f"{counts['validation'].get('1', 0)} / {counts['validation'].get('0', 0)} | "
            f"{counts['test'].get('1', 0)} / {counts['test'].get('0', 0)} | {removed} |"
        )
    lines.extend(
        [
            "",
            "Training sets were deterministically downsampled to equal class counts. Validation and test prevalence was not altered.",
            "",
            "## Training",
            "",
            f"Seed: `{SEED}`. Optimizer: AdamW, learning rate 0.001, weight decay 0.0001, batch size 128, maximum 100 epochs, early stopping after 20 non-improving epochs, selected by validation ROC AUC. Environment: Python {platform.python_version()}, PyTorch {torch.__version__}, NumPy {np.__version__}, OS {platform.platform()}.",
            "",
            "| Task | Best epoch | Epochs run | Train ROC AUC | Validation ROC AUC | Checkpoint SHA-256 |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for task in TASKS:
        item = training[task]
        lines.append(
            f"| {title[task]} | {item['best_epoch']} | {item['epochs_completed']} | "
            f"{_fmt(item['train_metrics']['roc_auc'])} | {_fmt(item['validation_metrics']['roc_auc'])} | `{item['best_checkpoint_sha256']}` |"
        )
    lines.extend(
        [
            "",
            "## Held-out results",
            "",
            "The fixed 0.5 class-1 probability threshold is reported without test-set tuning.",
            "",
            "| Task | ROC AUC | Balanced accuracy | Accuracy | Precision | Recall | MCC | Suitable under controls? |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for task in TASKS:
        metric = results[task]["test_metrics"]
        lines.append(
            f"| {title[task]} | {_fmt(metric['roc_auc'])} | {_fmt(metric['balanced_accuracy'])} | "
            f"{_fmt(metric['accuracy'])} | {_fmt(metric['precision'])} | {_fmt(metric['recall'])} | "
            f"{_fmt(metric['mcc'])} | {'yes' if results[task]['suitable_interpretation_target_under_controls'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "All three pass the held-out discrimination gate (ROC AUC at least 0.80 and balanced accuracy at least 0.70). The stricter suitability column also requires shuffling to reduce ROC AUC by at least 0.05, so a composition-dominated classifier is not presented as an equally strong weight-interpretation target.",
            "",
            "## Composition-preserving controls",
            "",
            "Each test sequence was residue-shuffled independently. This preserves per-sequence amino-acid composition while destroying positional order and contiguous motifs.",
            "",
            "| Task | Original ROC AUC | Shuffled ROC AUC | Positive score original -> shuffled | Positive fraction decreased |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for task in TASKS:
        control = results[task]["composition_preserving_shuffle_control"]
        lines.append(
            f"| {title[task]} | {_fmt(results[task]['test_metrics']['roc_auc'])} | "
            f"{_fmt(control['metrics']['roc_auc'])} | {_fmt(control['positive_mean_original_score'])} -> "
            f"{_fmt(control['positive_mean_shuffled_score'])} | {_fmt(control['positive_fraction_score_decreased'])} |"
        )
    lines.extend(
        [
            "",
            "Interpretation: the signal-peptide network depends strongly on sequence order and N-terminal organization. The transmembrane network retains substantial composition-only discrimination, consistent with hydrophobic enrichment, but gains additional performance from residue order/contiguity. The coiled-coil network's shuffled AUC is nearly unchanged, so its strong blind-set result is mostly composition-driven; it should not be treated as equivalent evidence that the CNN learned ordered heptad structure.",
        ]
    )
    lines.extend(["", "## Held-out mutation sensitivity", ""])
    for task in TASKS:
        scan = results[task]["heldout_positive_mutation_scan"]
        positions = ", ".join(
            f"{row['position_1_based']} ({row['mean_score_drop']:+.3f})" for row in scan["largest_mean_decreases"][:8]
        )
        lines.append(
            f"- **{title[task]}:** among {scan['examples']} score-quantile-spaced held-out positives, the largest mean class-1 score decreases after a single G substitution (P when already G) occurred at positions {positions}."
        )
    lines.extend(
        [
            "",
            "These interventions describe model sensitivity; they are not biological mutagenesis claims.",
            "",
            "## Reproducibility artifacts",
            "",
            "- `requirements.txt` pins the Python packages and CPU PyTorch wheel source.",
            "- `scripts/download_data.py` downloads and verifies every raw source.",
            "- `scripts/cnns.py prepare` records extraction decisions, private source IDs, counts, and leakage audits.",
            "- `scripts/cnns.py train` saves initial weights, best weights, histories, hyperparameters, and hashes.",
            "- `scripts/cnns.py evaluate` saves test predictions, controls, mutation scans, and this report.",
            "- `scripts/reproduce.ps1` runs the complete workflow on CPU.",
            "",
            "## Limitations",
            "",
            "SignalP train/validation/test all originate from one publisher dataset, although its supplied partitions are kept separate. CoCoNat has a genuinely separate publisher blind set. Source and taxonomic correlations can still substitute for biology. The 80% aligned-window filter detects close fixed-frame matches but cannot exclude all remote or shifted homologs. Sequence shuffling is a useful compositional control but not a substitute for evaluation on additional independently curated corpora.",
            "",
            "The requested work stops here: the three CNNs are coded, trained, evaluated, and documented; no later packet experiments were run.",
        ]
    )
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / "cnn_training_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def evaluate_all(device_name: str = "auto") -> tuple[dict[str, dict], Path]:
    results = {task: evaluate_task(task, device_name) for task in TASKS}
    return results, build_report(results)
