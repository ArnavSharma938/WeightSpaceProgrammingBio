# Experiment 1 CNN training and validation report

## Scope and outcome

Only the three CNNs requested in **Experiment 1 CNN (also experiment 4)** were implemented and trained. No RBM, transformer, ESM, prompt-running, Astra, or weight-construction experiment was performed.

All three checkpoints use the required architecture: `Conv1d(21,32,7,padding=3) -> ReLU -> Conv1d(32,32,7,padding=3) -> ReLU -> channel-major flatten(2240) -> Linear(2240,2)`. Each has exactly **16,418 trainable parameters**. Class 1 is feature-present.

## Data and provenance

- SignalP 6.0 training data: https://services.healthtech.dtu.dk/services/SignalP-6.0/public_data/train_set.fasta
- SignalP format/reference: https://services.healthtech.dtu.dk/services/SignalP-6.0/
- CoCoNat datasets and supplied splits: https://coconat.biocomp.unibo.it/datasets/

The downloader pins the following source-file hashes:

| File | Bytes | SHA-256 |
|---|---:|---|
| `coconat_blindtest_negative.fasta` | 95124 | `42d3826c61619210e5a53e932f624f8f4c759c7654fcad9b0e752bc2f39304f0` |
| `coconat_blindtest_positive.annot.fasta` | 431701 | `222cf64f306594f2d87250ddd469ac9b8d1bf714c1d4dd6f02e299c4ea493e97` |
| `coconat_cv_sets.tgz` | 44393 | `844cf7b442d4bfe6836648c42fc0bcae052aa940c54da9c7b6f24171baf5a99f` |
| `coconat_training_negative.fasta` | 3719675 | `5b408e95bfb86bab06a5b96fec1ce8e344483221fd391351ecb16575265ed09e` |
| `coconat_training_positive.annot.fasta` | 3430458 | `059dcdcbdec481be5160b5893f354ea163edfdcd22faa364d38ec5aaffa53ae4` |
| `signalp6_train_set.fasta` | 3357917 | `18ebf6dcc43ed4d9197155d9e9a405e3ac040ec3bcee05fc570d0e88f439642c` |

SignalP publisher partitions 0/1/2 were frozen as train/validation/test. CoCoNat folds 1-9 were train, fold 0 validation, and the publisher's blind set test. Only 70-residue canonical-amino-acid examples entered a model. CoCoNat positives were centered on the longest register run of at least 21 residues; negatives used one centered window per protein. SignalP transmembrane positives required a contiguous `M` annotation of at least 15 residues.

Before balancing, training windows with at least 80% position-aligned identity to any validation/test window were removed. This is a conservative window-level leakage filter, not a full profile-HMM or structure-based homology search.

## Prepared datasets

| Task | Train + / - | Validation + / - | Test + / - | Near-heldout train windows removed |
|---|---:|---:|---:|---:|
| Signal peptide | 1381 / 1381 | 1667 / 5665 | 1487 / 5432 | 1 |
| Transmembrane segment | 521 / 521 | 549 / 5081 | 1037 / 4337 | 0 |
| Coiled coil | 427 / 427 | 43 / 892 | 47 / 316 | 0 |

Training sets were deterministically downsampled to equal class counts. Validation and test prevalence was not altered.

## Training

Seed: `20261103`. Optimizer: AdamW, learning rate 0.001, weight decay 0.0001, batch size 128, maximum 100 epochs, early stopping after 20 non-improving epochs, selected by validation ROC AUC. Environment: Python 3.12.10, PyTorch 2.15.0+cpu, NumPy 2.5.4, OS Windows-11-10.0.26200-SP0.

| Task | Best epoch | Epochs run | Train ROC AUC | Validation ROC AUC | Checkpoint SHA-256 |
|---|---:|---:|---:|---:|---|
| Signal peptide | 11 | 31 | 0.996 | 0.977 | `ac473296ef25f7dd3a7b6ef9fd355aad1550aceec69b67447437b08a174b142b` |
| Transmembrane segment | 20 | 40 | 1.000 | 0.997 | `b8a05efc618d5b0f511b8b508bfd3c09961d5e181db4624e31b0ad77e69c710c` |
| Coiled coil | 12 | 32 | 0.971 | 0.966 | `7445e085784d6a3891fbe640ba2c788c3b679464b292d30968c652aea921db73` |

## Held-out results

The fixed 0.5 class-1 probability threshold is reported without test-set tuning.

| Task | ROC AUC | Balanced accuracy | Accuracy | Precision | Recall | MCC | Suitable under controls? |
|---|---:|---:|---:|---:|---:|---:|---|
| Signal peptide | 0.973 | 0.928 | 0.942 | 0.839 | 0.903 | 0.834 | yes |
| Transmembrane segment | 0.996 | 0.973 | 0.978 | 0.927 | 0.963 | 0.931 | yes |
| Coiled coil | 0.950 | 0.885 | 0.862 | 0.483 | 0.915 | 0.600 | no |

All three pass the held-out discrimination gate (ROC AUC at least 0.80 and balanced accuracy at least 0.70). The stricter suitability column also requires shuffling to reduce ROC AUC by at least 0.05, so a composition-dominated classifier is not presented as an equally strong weight-interpretation target.

## Composition-preserving controls

Each test sequence was residue-shuffled independently. This preserves per-sequence amino-acid composition while destroying positional order and contiguous motifs.

| Task | Original ROC AUC | Shuffled ROC AUC | Positive score original -> shuffled | Positive fraction decreased |
|---|---:|---:|---:|---:|
| Signal peptide | 0.973 | 0.603 | 0.857 -> 0.080 | 0.976 |
| Transmembrane segment | 0.996 | 0.920 | 0.953 -> 0.678 | 0.853 |
| Coiled coil | 0.950 | 0.948 | 0.858 -> 0.833 | 0.617 |

Interpretation: the signal-peptide network depends strongly on sequence order and N-terminal organization. The transmembrane network retains substantial composition-only discrimination, consistent with hydrophobic enrichment, but gains additional performance from residue order/contiguity. The coiled-coil network's shuffled AUC is nearly unchanged, so its strong blind-set result is mostly composition-driven; it should not be treated as equivalent evidence that the CNN learned ordered heptad structure.

## Held-out mutation sensitivity

- **Signal peptide:** among 32 score-quantile-spaced held-out positives, the largest mean class-1 score decreases after a single G substitution (P when already G) occurred at positions 13 (+0.086), 11 (+0.084), 9 (+0.082), 7 (+0.082), 10 (+0.062), 12 (+0.059), 8 (+0.056), 14 (+0.053).
- **Transmembrane segment:** among 32 score-quantile-spaced held-out positives, the largest mean class-1 score decreases after a single G substitution (P when already G) occurred at positions 46 (+0.029), 42 (+0.023), 37 (+0.022), 38 (+0.022), 44 (+0.021), 45 (+0.021), 49 (+0.020), 40 (+0.018).
- **Coiled coil:** among 32 score-quantile-spaced held-out positives, the largest mean class-1 score decreases after a single G substitution (P when already G) occurred at positions 27 (+0.082), 43 (+0.080), 24 (+0.079), 25 (+0.077), 44 (+0.075), 45 (+0.075), 26 (+0.075), 32 (+0.075).

These interventions describe model sensitivity; they are not biological mutagenesis claims.

## Reproducibility artifacts

- `requirements.txt` pins the Python packages and CPU PyTorch wheel source.
- `scripts/download_data.py` downloads and verifies every raw source.
- `scripts/cnns.py prepare` records extraction decisions, private source IDs, counts, and leakage audits.
- `scripts/cnns.py train` saves initial weights, best weights, histories, hyperparameters, and hashes.
- `scripts/cnns.py evaluate` saves test predictions, controls, mutation scans, and this report.
- `scripts/reproduce.ps1` runs the complete workflow on CPU.

## Limitations

SignalP train/validation/test all originate from one publisher dataset, although its supplied partitions are kept separate. CoCoNat has a genuinely separate publisher blind set. Source and taxonomic correlations can still substitute for biology. The 80% aligned-window filter detects close fixed-frame matches but cannot exclude all remote or shifted homologs. Sequence shuffling is a useful compositional control but not a substitute for evaluation on additional independently curated corpora.

The requested work stops here: the three CNNs are coded, trained, evaluated, and documented; no later packet experiments were run.
