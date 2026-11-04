from __future__ import annotations

import json
import tarfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .common import ALPHABET, PROCESSED_DIR, RAW_DIR, SEED, SEQUENCE_LENGTH, TASKS, encode, sha256, write_json

CANONICAL = set(ALPHABET)
SP_ANNOTATION = {"SP": "S", "TAT": "T", "TATLIPO": "T", "LIPO": "L", "PILIN": "P"}


@dataclass(frozen=True)
class Example:
    identifier: str
    sequence: str
    label: int
    source: str
    extraction: str


def longest_run(text: str, accepted: set[str]) -> tuple[int, int]:
    best_start = best_end = current_start = 0
    in_run = False
    for index, character in enumerate(text + "\0"):
        if character in accepted:
            if not in_run:
                current_start = index
                in_run = True
        elif in_run:
            if index - current_start > best_end - best_start:
                best_start, best_end = current_start, index
            in_run = False
    return best_start, best_end


def parse_signalp(path: Path) -> list[tuple[str, str, str, int, str]]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) % 3:
        raise ValueError(f"SignalP file is not three-line FASTA: {path}")
    records = []
    for offset in range(0, len(lines), 3):
        header, sequence, annotation = lines[offset : offset + 3]
        fields = header.removeprefix(">").split("|")
        if len(fields) != 4 or len(sequence) != len(annotation):
            raise ValueError(f"Malformed SignalP record at line {offset + 1}")
        records.append((fields[0], sequence.upper(), fields[2], int(fields[3]), annotation))
    return records


def parse_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    identifier = None
    chunks: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if identifier is not None:
                records.append((identifier, "".join(chunks).upper()))
            identifier = line[1:].split()[0]
            chunks = []
        else:
            chunks.append(line)
    if identifier is not None:
        records.append((identifier, "".join(chunks).upper()))
    return records


def parse_annotated_fasta(path: Path) -> list[tuple[str, str, str]]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = []
    offset = 0
    while offset < len(lines):
        if not lines[offset].startswith(">") or offset + 2 >= len(lines):
            raise ValueError(f"Malformed annotated CoCoNat record near line {offset + 1}: {path}")
        identifier = lines[offset][1:].split()[0]
        sequence = lines[offset + 1].upper()
        registers = lines[offset + 2]
        if len(sequence) != len(registers):
            raise ValueError(f"Sequence/register length mismatch for {identifier}")
        records.append((identifier, sequence, registers))
        offset += 4 if offset + 3 < len(lines) and not lines[offset + 3].startswith(">") else 3
    return records


def load_cv_folds(path: Path) -> dict[str, int]:
    assignments: dict[str, int] = {}
    with tarfile.open(path, "r:gz") as archive:
        for fold in range(10):
            member = archive.extractfile(f"cv_sets/set{fold}")
            if member is None:
                raise ValueError(f"Missing cv_sets/set{fold}")
            for line in member.read().decode("utf-8").splitlines():
                identifier = line.strip()
                if identifier:
                    if identifier in assignments:
                        raise ValueError(f"Duplicate CoCoNat CV assignment: {identifier}")
                    assignments[identifier] = fold
    return assignments


def signalp_examples() -> tuple[dict[str, dict[str, list[Example]]], dict]:
    records = parse_signalp(RAW_DIR / "signalp6_train_set.fasta")
    result = {task: {split: [] for split in ("train", "validation", "test")} for task in TASKS[:2]}
    split_for_partition = {0: "train", 1: "validation", 2: "test"}
    audit = Counter()
    for identifier, sequence, kind, partition, annotation in records:
        if len(sequence) != SEQUENCE_LENGTH or not set(sequence) <= CANONICAL:
            audit["discarded_noncanonical_or_wrong_length"] += 1
            continue
        split = split_for_partition[partition]
        expected = SP_ANNOTATION.get(kind)
        if expected is not None and expected in annotation:
            result["signal_peptide"][split].append(
                Example(identifier, sequence, 1, "SignalP6", f"{kind}; annotation contains {expected}")
            )
        elif kind == "NO_SP":
            result["signal_peptide"][split].append(
                Example(identifier, sequence, 0, "SignalP6", "NO_SP")
            )
            start, end = longest_run(annotation, {"M"})
            if end - start >= 15:
                result["transmembrane"][split].append(
                    Example(identifier, sequence, 1, "SignalP6", f"longest M run={end - start}")
                )
            elif "M" not in annotation:
                result["transmembrane"][split].append(
                    Example(identifier, sequence, 0, "SignalP6", "NO_SP; no M annotation")
                )
            else:
                audit["discarded_tm_short_M_run"] += 1
        else:
            audit["discarded_inconsistent_sp_annotation"] += 1
    return result, dict(audit)


def centered_window(sequence: str, center_start: int | None = None, center_end: int | None = None) -> tuple[str, int]:
    if len(sequence) < SEQUENCE_LENGTH:
        raise ValueError("Sequence is shorter than 70")
    if center_start is None or center_end is None:
        start = (len(sequence) - SEQUENCE_LENGTH) // 2
    else:
        start = (center_start + center_end - SEQUENCE_LENGTH) // 2
        start = min(max(start, 0), len(sequence) - SEQUENCE_LENGTH)
    return sequence[start : start + SEQUENCE_LENGTH], start


def coconat_examples() -> tuple[dict[str, list[Example]], dict]:
    folds = load_cv_folds(RAW_DIR / "coconat_cv_sets.tgz")
    result = {split: [] for split in ("train", "validation", "test")}
    audit = Counter()

    def add_positive(identifier: str, sequence: str, registers: str, split: str) -> None:
        start, end = longest_run(registers, set("abcdefg"))
        if end - start < 21:
            audit[f"{split}_positive_no_21_register_run"] += 1
            return
        if len(sequence) < SEQUENCE_LENGTH:
            audit[f"{split}_positive_noncanonical_or_short"] += 1
            return
        window, window_start = centered_window(sequence, start, end)
        if not set(window) <= CANONICAL:
            audit[f"{split}_positive_noncanonical_or_short"] += 1
            return
        result[split].append(
            Example(identifier, window, 1, "CoCoNat", f"window_start={window_start}; longest_register_run={end-start}")
        )

    def add_negative(identifier: str, sequence: str, split: str) -> None:
        if len(sequence) < SEQUENCE_LENGTH or not set(sequence) <= CANONICAL:
            audit[f"{split}_negative_noncanonical_or_short"] += 1
            return
        window, window_start = centered_window(sequence)
        result[split].append(
            Example(identifier, window, 0, "CoCoNat", f"centered_negative_window_start={window_start}")
        )

    for identifier, sequence, registers in parse_annotated_fasta(RAW_DIR / "coconat_training_positive.annot.fasta"):
        if identifier not in folds:
            audit["training_positive_missing_cv_assignment"] += 1
            continue
        add_positive(identifier, sequence, registers, "validation" if folds[identifier] == 0 else "train")
    for identifier, sequence in parse_fasta(RAW_DIR / "coconat_training_negative.fasta"):
        if identifier not in folds:
            audit["training_negative_missing_cv_assignment"] += 1
            continue
        add_negative(identifier, sequence, "validation" if folds[identifier] == 0 else "train")
    for identifier, sequence, registers in parse_annotated_fasta(RAW_DIR / "coconat_blindtest_positive.annot.fasta"):
        add_positive(identifier, sequence, registers, "test")
    for identifier, sequence in parse_fasta(RAW_DIR / "coconat_blindtest_negative.fasta"):
        add_negative(identifier, sequence, "test")
    return result, dict(audit)


def leakage_audit(splits: dict[str, list[Example]]) -> dict:
    output: dict[str, object] = {}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        left_ids = {example.identifier for example in splits[left]}
        right_ids = {example.identifier for example in splits[right]}
        left_sequences = {example.sequence for example in splits[left]}
        right_sequences = {example.sequence for example in splits[right]}
        output[f"{left}_vs_{right}"] = {
            "shared_ids": len(left_ids & right_ids),
            "identical_sequences": len(left_sequences & right_sequences),
        }
    return output


def remove_close_heldout_neighbors(splits: dict[str, list[Example]], threshold: float = 0.80) -> tuple[dict[str, list[Example]], dict]:
    heldout = splits["validation"] + splits["test"]
    reference = np.stack([encode(example.sequence) for example in heldout])
    kept: list[Example] = []
    removed: list[dict[str, object]] = []
    maximum_identities: list[float] = []
    for example in splits["train"]:
        encoded = encode(example.sequence)
        maximum = float(np.mean(reference == encoded, axis=1).max()) if len(reference) else 0.0
        maximum_identities.append(maximum)
        if maximum >= threshold:
            removed.append({"id": example.identifier, "label": example.label, "max_aligned_identity": maximum})
        else:
            kept.append(example)
    filtered = {**splits, "train": kept}
    audit = {
        "threshold": threshold,
        "removed_count": len(removed),
        "removed_by_label": dict(Counter(item["label"] for item in removed)),
        "removed": removed,
        "pre_filter_train_max_identity_summary": {
            "mean": float(np.mean(maximum_identities)),
            "median": float(np.median(maximum_identities)),
            "max": float(np.max(maximum_identities)),
        },
    }
    return filtered, audit


def balance_training(examples: list[Example], seed: int = SEED) -> list[Example]:
    rng = np.random.default_rng(seed)
    positives = [example for example in examples if example.label == 1]
    negatives = [example for example in examples if example.label == 0]
    count = min(len(positives), len(negatives))
    positive_indices = rng.choice(len(positives), size=count, replace=False)
    negative_indices = rng.choice(len(negatives), size=count, replace=False)
    balanced = [positives[index] for index in positive_indices] + [negatives[index] for index in negative_indices]
    rng.shuffle(balanced)
    return balanced


def save_task(task: str, splits: dict[str, list[Example]], audit: dict) -> dict:
    arrays: dict[str, np.ndarray] = {}
    private_records: list[dict[str, object]] = []
    counts: dict[str, dict[str, int]] = {}
    for split, examples in splits.items():
        arrays[f"{split}_sequences"] = np.stack([encode(example.sequence) for example in examples])
        arrays[f"{split}_labels"] = np.asarray([example.label for example in examples], dtype=np.int64)
        counts[split] = dict(Counter(str(example.label) for example in examples))
        for row, example in enumerate(examples):
            private_records.append(
                {
                    "split": split,
                    "row": row,
                    "id": example.identifier,
                    "label": example.label,
                    "source": example.source,
                    "extraction": example.extraction,
                }
            )
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PROCESSED_DIR / f"{task}.npz", **arrays)
    with (PROCESSED_DIR / f"{task}_private_examples.jsonl").open("w", encoding="utf-8") as handle:
        for record in private_records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    manifest = {"task": task, "counts": counts, "audit": audit}
    write_json(PROCESSED_DIR / f"{task}_manifest.json", manifest)
    return manifest


def prepare_all() -> dict:
    signal_tasks, signal_extraction_audit = signalp_examples()
    coconat, coconat_extraction_audit = coconat_examples()
    all_tasks = {**signal_tasks, "coiled_coil": coconat}
    manifests = {}
    for task, raw_splits in all_tasks.items():
        audit = {
            "raw_counts": {split: dict(Counter(str(example.label) for example in examples)) for split, examples in raw_splits.items()},
            "overlap_before_filtering": leakage_audit(raw_splits),
            "extraction": coconat_extraction_audit if task == "coiled_coil" else signal_extraction_audit,
        }
        filtered, homology_audit = remove_close_heldout_neighbors(raw_splits)
        audit["close_homology_filter"] = homology_audit
        filtered["train"] = balance_training(filtered["train"], SEED)
        audit["overlap_after_filtering"] = leakage_audit(filtered)
        manifests[task] = save_task(task, filtered, audit)
    raw_manifest = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(RAW_DIR.iterdir())
        if path.is_file()
    }
    write_json(PROCESSED_DIR / "raw_data_manifest.json", raw_manifest)
    write_json(PROCESSED_DIR / "preparation_summary.json", manifests)
    return manifests
