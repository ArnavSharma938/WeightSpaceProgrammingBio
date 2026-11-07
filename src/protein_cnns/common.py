from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
PAD_INDEX = 20
SEQUENCE_LENGTH = 70
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Downloaded sources are a reproducible local cache, not repository artifacts.
RAW_DIR = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".cache")) / "protein_weight_space_cnns" / "source_data"
DATA_DIR = PROJECT_ROOT / "data"
RUNS_DIR = PROJECT_ROOT / "runs"
REPORTS_DIR = PROJECT_ROOT / "reports"
TASKS = ("signal_peptide", "transmembrane", "coiled_coil")
SEED = 20261103

AA_TO_INDEX = {aa: index for index, aa in enumerate(ALPHABET)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def encode(sequence: str) -> np.ndarray:
    return np.asarray([AA_TO_INDEX[aa] for aa in sequence], dtype=np.uint8)


def decode(encoded: np.ndarray) -> str:
    return "".join(ALPHABET[int(index)] for index in encoded)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
