# Experiment 1: reading the protein CNN weights

All three prepared prompts completed on November 4, 2026, using
`gpt-6.1-sol` with high reasoning effort through Codex CLI 0.164.0. Each prompt
contained the same architecture description and all 16,418 parameters of one
trained CNN, rounded to three decimal places. Task identities were withheld.

| Input | Actual task | Response's leading interpretation | Confidence | Assessment |
|---|---|---|---:|---|
| model_001 | Transmembrane-segment detection | Membrane/transmembrane-related classification, without a unique task identification | 60% for the broad interpretation | Correct direction, qualified |
| model_002 | Signal-peptide detection | N-terminal secretory signal peptide versus non-signal peptide | 65% | Correct leading hypothesis |
| model_003 | Coiled-coil detection | Coiled-coil versus non-coiled-coil sequence segments | 65% | Correct leading hypothesis |

The result is **two explicit correct leading hypotheses and one correct broad
interpretation**, rather than three unambiguous exact identifications. All three
responses qualified their conclusions and discussed alternatives. These are
qualitative judgments against the private checkpoint mapping, not a preregistered
numerical score or evidence that every explanation is correct.

The transmembrane response emphasized hydrophobic stretches and charge exclusion,
but retained signal peptides as an alternative. The signal-peptide response
combined hydrophobicity, charge asymmetry, and early-versus-late output weighting.
The coiled-coil response emphasized leucine/glutamine/charged-residue composition
and proline/glycine contrasts. Its caution about unproven heptad periodicity is
consistent with the earlier shuffle control: this classifier retained almost all
its discrimination after sequence order was destroyed.

## Execution checks

The batch is saved in [`runs/pure_reasoning/exp1_001`](runs/pure_reasoning/exp1_001/).
All three inputs completed on their first attempt in distinct conversations,
with exit status 0, complete event streams, and no native tool use. Independent
re-auditing found zero violations. The five diagnostic items in each stream are
CLI startup/configuration warnings, not tool executions.

Submitted prompt hashes match the prepared files and private manifest. Source
checkpoint hashes also match the manifest, and final answer files match the
responses recorded in the JSONL streams.

| Input | Duration | Input tokens | Output tokens | Of which reasoning tokens |
|---|---:|---:|---:|---:|
| model_001 | 59.09 s | 114,251 | 2,316 | 1,377 |
| model_002 | 63.81 s | 114,251 | 2,389 | 1,494 |
| model_003 | 47.98 s | 114,251 | 1,892 | 1,034 |

Full responses:
[model_001](runs/pure_reasoning/exp1_001/001-2f1599520e5a/attempt-001/answer.txt),
[model_002](runs/pure_reasoning/exp1_001/002-b254574c4836/attempt-001/answer.txt),
[model_003](runs/pure_reasoning/exp1_001/003-b08926ec158d/attempt-001/answer.txt).

## Git visibility

All saved files under this batch were checked with
`git check-ignore --no-index --stdin`; none are ignored. Git currently reports
the batch as untracked, so its artifacts are eligible to be added but have not
yet been committed. No ignore-rule change was required. This report was written
from existing artifacts; no further model requests were launched.
