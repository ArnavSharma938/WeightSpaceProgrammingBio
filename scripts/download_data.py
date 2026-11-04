from __future__ import annotations

import time
import urllib.request
from pathlib import Path

from protein_cnns.common import RAW_DIR, sha256

FILES = {
    "signalp6_train_set.fasta": (
        "https://services.healthtech.dtu.dk/services/SignalP-6.0/public_data/train_set.fasta",
        "18ebf6dcc43ed4d9197155d9e9a405e3ac040ec3bcee05fc570d0e88f439642c",
    ),
    "coconat_training_positive.annot.fasta": (
        "https://coconat.biocomp.unibo.it/static/data/training.positive.annot.fasta",
        "059dcdcbdec481be5160b5893f354ea163edfdcd22faa364d38ec5aaffa53ae4",
    ),
    "coconat_training_negative.fasta": (
        "https://coconat.biocomp.unibo.it/static/data/training.negative.fasta",
        "5b408e95bfb86bab06a5b96fec1ce8e344483221fd391351ecb16575265ed09e",
    ),
    "coconat_cv_sets.tgz": (
        "https://coconat.biocomp.unibo.it/static/data/cv_sets.tgz",
        "844cf7b442d4bfe6836648c42fc0bcae052aa940c54da9c7b6f24171baf5a99f",
    ),
    "coconat_blindtest_positive.annot.fasta": (
        "https://coconat.biocomp.unibo.it/static/data/dataset_718.biounit.positive.annot.fasta",
        "222cf64f306594f2d87250ddd469ac9b8d1bf714c1d4dd6f02e299c4ea493e97",
    ),
    "coconat_blindtest_negative.fasta": (
        "https://coconat.biocomp.unibo.it/static/data/dataset_718.negative.fasta",
        "42d3826c61619210e5a53e932f624f8f4c759c7654fcad9b0e752bc2f39304f0",
    ),
}


def fetch(destination: Path, url: str, expected_hash: str) -> None:
    if destination.exists() and sha256(destination) == expected_hash:
        print(f"verified existing {destination.name}")
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(1, 6):
        try:
            print(f"downloading {destination.name} (attempt {attempt})")
            request = urllib.request.Request(url, headers={"User-Agent": "protein-weight-space-cnns/0.1"})
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
                while block := response.read(1024 * 1024):
                    output.write(block)
            actual_hash = sha256(temporary)
            if actual_hash != expected_hash:
                raise RuntimeError(f"SHA-256 mismatch for {destination.name}: {actual_hash}")
            temporary.replace(destination)
            return
        except Exception:
            if attempt == 5:
                raise
            time.sleep(2**attempt)


if __name__ == "__main__":
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for filename, (source_url, pinned_hash) in FILES.items():
        fetch(RAW_DIR / filename, source_url, pinned_hash)
