# Initial pure-reasoning Codex runner implementation report

Implemented for Experiment 1 of the weight-space project on 2026-11-03;
recovery fixes and a real smoke validation completed on 2026-11-04
(America/New_York; verification timestamps are recorded in UTC).

The runner is generic: it receives separately prepared UTF-8 prompt files and
obtains one response per fresh Codex conversation. It does not import CNN code,
inspect weights, create prompts, grade answers, or supply experimental feedback.
No real experimental model requests were launched during development.

## Implementation and execution policy

`runner/pure_reasoning.py` contains the complete runner using the Python 3.11+
standard library. `runner/pure_reasoning.toml` contains execution policy, and
`runner/neutral.md` contains the exact neutral model instruction:
`Answer the user's request.` No third-party dependency or API SDK is required.

The installed CLI was inspected directly: `codex-cli 0.164.0`. Its help,
feature inventory, bundled model catalog, and local prompt-inspection command
were checked. The catalog contains exactly the requested `gpt-6.1-sol` model
and supports high reasoning effort. Authentication and model access for this
account were validated by the harmless live smoke request described below.
The implementation rejects other CLI versions pending explicit revalidation.

Real requests invoke native `codex.exe exec` with `--ignore-user-config`,
`--ignore-rules`, `--ephemeral`, `--strict-config`, JSONL output, and a final-answer
file. Each process starts in its own temporary empty working directory with a
root marker. User instructions have a zero-byte document budget. Built-in model
instructions are replaced by the neutral file; developer instructions, skills,
memories, apps, collaboration, permissions, and environment-context injection
are disabled by supported settings. Every active native feature from this CLI
is disabled, except the switch whose purpose is to skip host skill discovery.
This removes shell, file tools, web, browser, integrations, plugins, subagents,
and other native tools. Read-only sandboxing is an additional constraint, not
the mechanism used to remove tools.

Existing authentication is used at its original location. The runner does not
read or copy credential files, log environment values, or create a separate
credential-bearing Codex home. It removes inherited parent-session `CODEX_*`
variables except the authentication home/API-key variables, removes unrelated
`OPENAI_*` overrides except the API key, and removes Rust diagnostic switches.

Local inspection with the complete isolation settings produced exactly one
synthetic user message and no additional tool definitions, instructions,
skills, memories, project text, or environment context. This inspection is
local and does not invoke inference. The probe subcommand does not support
`--ignore-user-config` or `--strict-config`: configured MCP servers are explicitly
disabled for this probe. Actual exec ignores user config and uses strict parsing.
Administrative configuration that conflicts with the policy blocks execution;
the runner does not weaken isolation to accommodate it.

## Input, logging, completion, and recovery

Input files are read as bytes, validated as UTF-8, snapshotted, SHA-256 hashed,
and passed unchanged as binary stdin to the positional `-` argument. There is
no prompt argument containing extra instructions and no newline/BOM rewrite.
Multiple input files are processed sequentially. Every request and retry uses
a new `codex exec`, never conversation resume or a follow-up message.

Every attempt has its own directory containing the submitted prompt, raw JSONL
events, stderr, answer (or separate partial-answer file), and metadata. The
metadata records timestamps, exact command/settings, prompt hash, process exit
status, reported usage, timeout/interruption flags, and the event audit. Run-level
execution metadata records the CLI version and complete settings. An atomic
ledger tracks per-input status and next allowed attempt time.

Agent-message, reasoning, and CLI diagnostic error items are allowed. Diagnostic
messages are retained and inspected for failure classification. Tool items/call
metadata, unknown item/event types, malformed complete JSONL records, multiple
threads/turns, or an unexpected reported model invalidate the attempt. A truncated
final JSONL/UTF-8 fragment is separately recorded as incomplete output and is
recoverable if no completed answer or confirmed violation exists. A completed
answer with an incomplete audit remains terminal-invalid to prevent resubmission.
Completed and invalid inputs are terminal
and are skipped after restart. A wrong answer, refusal, or uncertainty does not
trigger a retry. A CLI final-answer file, or an agent message followed by
`turn.completed`, ends the request even if process cleanup exits nonzero.

Transient infrastructure failures use exponential backoff, capped at five
minutes. Usage-limit failures wait for a parseable reported epoch/ISO reset or
relative delay; otherwise the default conservative interval is one hour. This
deadline survives restarting the runner and exhausting a retry budget. Timeouts,
retry count, base backoff, and unknown-reset interval are configurable. Quota
waits are unlimited and do not spend the transient retry allowance. Transient
failures are counted across restarts; increasing `--retries` on resume extends
that allowance. Expired reset timestamps use the conservative fallback interval.
Authentication, unsupported configuration, and unexplained incomplete failures
stop for inspection. Retrying after a manual resume creates a separate attempt.

Ctrl+C stops the child process tree, records the attempt, and exits 130. Crash
recovery reconciles final artifacts before scheduling a new attempt. If a
recorded child PID is still live after a forced parent termination, resume
stops until it exits, avoiding an overlapping request. An OS-backed exclusive
lock prevents two runners from owning the same run directory and is released
automatically on process exit. As with local process orchestration generally,
a forced kill can leave temporary files; normal Ctrl+C is the supported clean
interruption path.

## Recovery corrections

The user reported two reproducible crash/recovery defects in the original
implementation. Both are corrected on the actual installed CLI 0.164.0; the
friend's 0.166.0-alpha.1 environment was not installed or validated here.

1. A process interrupted mid-JSONL record was permanently marked invalid. The
   parser now distinguishes an unterminated final fragment from confirmed tool
   events or malformed complete records. Interrupted fragments remain in the
   original raw stream and are eligible for appropriate recovery. Legacy attempts
   affected by the old parser are re-audited, retaining their previous audit.
2. Failed attempts did not carry their scheduling state into the ledger during
   recovery. A single `outcome()` function now classifies and schedules both live
   execution and recovery. `attempt.json` is authoritative and stores the finish
   timestamp, failure status, retry policy, transient count, and next permitted
   attempt time before `state.json` is updated. Recovery reconstructs all outcomes,
   not just completed/invalid ones. If a crash occurs before classification, raw
   errors and the original failure timestamp reconstruct the deadline. An
   unavailable timestamp causes a conservative later wait, never an early retry.

Quota waiting now uses that same durable state, while remaining independent of
the bounded transient-error allowance. Repeated quota failures can recover
unattended without requiring manual resume after three waits.

## Verification

The offline isolation preflight passed against the actual installed CLI.
Thirty-nine runner tests use mocked Codex processes/local inspection results;
the existing architecture test also passes (40 tests total). Coverage includes
UTF-8/BOM/CRLF preservation, sequential files, event parsing and usage, native
tool and unknown-event detection, terminal answers after nonzero exits, partial
messages, refusals, timeout/Ctrl+C handling, error classification, retry limits,
reset waiting and fallback, persisted deadlines, restart skipping, crash
reconciliation, live-process protection, changed-prompt rejection, unsupported
CLI rejection, isolation failure rejection, and duplicate-run locking. Added
regressions exercise partial JSON/UTF-8 tails, confirmed tool use before a partial
tail, legacy-invalid migration, quotas saved before a ledger commit, raw quota
failure recovery, stable reconstruction from original failure time, no submission
before a recovered reset, unlimited quota waits with `--retries 0`, durable
transient budgets, diagnostic error items, and preservation of superseded audits.

## Live smoke validation

Exactly one real inference request was made through the actual runner. No
experimental prompt or model weights were submitted. The smoke fixture asks
only: `Reply with exactly WEIGHT_SPACE_SMOKE_OK.`

| Measurement | Observed value |
|---|---|
| CLI | `codex-cli 0.164.0` |
| Requested model / reasoning | `gpt-6.1-sol` / `high` |
| Started / finished (UTC) | 2026-11-04 13:53:44.945 / 13:53:48.894 |
| Final answer | `WEIGHT_SPACE_SMOKE_OK` |
| Process exit status | 0 |
| Input / cached input tokens | 2,238 / 2,048 |
| Output / reported reasoning output tokens | 10 / 0 |
| Final audit | `tool_free: true`, zero violations, complete JSONL |
| Requests / attempt directories | 1 / 1 |

This verified existing authentication, access to the requested model, real event
compatibility, usage parsing, final-answer capture, and the native-tool audit.
The stream also contains five startup/config diagnostic items of type `error`.
The initial parser rejected those conservatively as unknown items. After adding
support for this observed CLI diagnostic type, the same saved response was
re-audited and marked completed. The initial audit remains in `audit_history`.
No answer was retried. Running `resume` printed that all inputs were finished
and made no Codex request.

Evidence is under `runs/pure_reasoning/smoke_20261104`, including the exact prompt,
execution settings, raw events, answer, stderr, attempt metadata, and batch ledger.
Quota waiting and crash windows were tested with mocks rather than deliberately
exhausting live quota or interrupting another paid request.

## Codebase simplification and reports

The three fragmented CNN command wrappers were replaced by `scripts/cnns.py`
with `prepare`, `train`, and `evaluate` commands. `config.py` and `io.py` were
combined into `src/protein_cnns/common.py`, and imports and reproduction commands
were updated. Data preparation, architecture, training, metrics, and evaluation
remain separate substantive modules. The generic runner has no coupling to them.

The previous `reports/final_report.md` was renamed to
`reports/cnn_training_report.md`; the generator now uses that descriptive name.
Existing datasets, model checkpoints, metrics, and predictions were preserved.
The runner instructions and exact PowerShell commands are in `runner/README.md`.

## References consulted

- Original execution configuration:
  https://github.com/RikiyaT/lm_understand-logs/blob/main/harness/codex_config.toml
- Original neutral instruction:
  https://github.com/RikiyaT/lm_understand-logs/blob/main/harness/instructions.md
- Original Experiment 1 protocol:
  https://github.com/RikiyaT/lm_understand-logs/blob/main/exp1_which_game/README.md
- Official configuration reference and schema:
  https://developers.openai.com/codex/config-reference/
  and https://developers.openai.com/codex/config-schema.json
- Official non-interactive execution and JSONL documentation:
  https://developers.openai.com/codex/noninteractive/
- Official requested model/effort documentation:
  https://developers.openai.com/api/docs/models/gpt-6.1-sol
- Local authoritative behavior: `codex --version`, `codex exec --help`,
  `codex features list`, `codex debug models --bundled`, and
  `codex debug prompt-input` from version 0.164.0.
