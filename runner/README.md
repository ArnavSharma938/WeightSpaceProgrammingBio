# Initial pure-reasoning experiment runner

Python 3.11+ standard library only. This runner knows only prompt files,
processes, and events. Prepare prompts separately. It neither imports the CNN
package nor reads model weights. One new `codex exec` process per input/attempt;
no follow-ups, feedback, handoffs, or Codex conversation resume.

## Commands for this repository (Windows PowerShell)

```powershell
Set-Location 'C:\Users\arnav.DHEERAJACER\protein_weight_space_cnns'
```

The existing `.venv` contains Python 3.12. No pip installation is required for
the runner or its mocked tests. On a new machine, install Python 3.12 and create
the environment using `py -3.12 -m venv .venv`.

These are **Codex CLI commands**. The CLI is already installed at the verified
version here. Install only if absent on another machine:

```powershell
npm install -g @openai/codex@0.164.0
codex --version
codex login status
```

The first installs the pinned CLI; the second prints its version; the third
checks existing authentication. If authentication is missing, run `codex login`
yourself. The runner uses existing authentication in place and never copies
credentials into run artifacts. `CODEX_HOME`, if already set, stays unchanged.

These are **custom runner and Python test commands**:

```powershell
.\.venv\Scripts\python.exe runner\pure_reasoning.py check
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_pure_reasoning.py -v
```

`check` performs offline version/flag/model-catalog checks and local prompt
inspection with a synthetic message. It makes no inference request and sends
no experimental prompt. Account access to the requested model is determined
only by a live submission. A harmless smoke submission passed on this machine
on 2026-11-04; its artifacts are in `runs/pure_reasoning/smoke_20261104`.
The test command mocks every Codex process and needs no login or internet.

Submit your separately prepared prompt (the file must already exist):

```powershell
.\.venv\Scripts\python.exe runner\pure_reasoning.py run --run-dir runs\pure_reasoning\batch01 'C:\Users\arnav.DHEERAJACER\Downloads\prepared_prompt.txt'
```

For sequential prompts, append more file paths to that same command. Every
file receives a fresh conversation, including repeated copies of a file.
Prompts are snapshotted once as exact bytes, validated as UTF-8, hashed with
SHA-256, and sent as binary stdin without newline/BOM normalization or appended
instructions. The command uses the positional `-` argument to `codex exec`.

Inspect progress and final answers:

```powershell
.\.venv\Scripts\python.exe runner\pure_reasoning.py status --run-dir runs\pure_reasoning\batch01
Get-ChildItem runs\pure_reasoning\batch01 -Recurse -Filter answer.txt | ForEach-Object { $_.FullName; Get-Content -LiteralPath $_.FullName -Raw -Encoding UTF8 }
```

`status` reads persisted progress without starting Codex. The PowerShell command
prints answer files for every attempt; use each sibling `attempt.json` to check
whether that answer was accepted, incomplete, or invalid.

Interrupt with Ctrl+C, then resume using **our runner**:

```powershell
.\.venv\Scripts\python.exe runner\pure_reasoning.py resume --run-dir runs\pure_reasoning\batch01
```

This resumes the batch ledger, not a Codex conversation. It skips completed and
invalid inputs and creates fresh attempts for unfinished inputs. It does not
use `codex exec resume`. If a previous Codex process survived a forced parent
termination, recovery stops while that recorded PID remains live; wait for it
to exit before resuming. Ordinary Ctrl+C terminates the child process tree.

Optional process/retry controls (also accepted by `resume`):

```powershell
.\.venv\Scripts\python.exe runner\pure_reasoning.py run --run-dir runs\pure_reasoning\batch02 --timeout 1800 --retries 3 --backoff 10 --usage-recheck 3600 'C:\Users\arnav.DHEERAJACER\Downloads\prepared_prompt.txt'
```

Timeout is seconds per process. `--retries` limits additional transient-failure
attempts per input across restarts; increasing it on `resume` extends the saved
allowance. **Quota waits are unlimited and do not consume that allowance.**
Transient failures use exponential backoff capped at 300 seconds. A reported usage reset epoch,
timezone-qualified ISO timestamp, or explicit relative delay is honored. If
the reset cannot be parsed unambiguously, wait one hour before submitting the
same prompt in a fresh attempt. This conservative policy is configurable with
`--usage-recheck`; it is not a polling conversation. Already-expired reported
resets also use this conservative interval to avoid rapid repeated requests.
Wait deadlines survive restart, including when transient retries are exhausted. Authentication/configuration
failures stop immediately; unexplained incomplete requests stop for inspection.
A refusal, uncertain response, or wrong answer is completed and never retried.

## Execution policy and validation

`pure_reasoning.toml` holds execution settings; `neutral.md` replaces built-in
model instructions. The runner freezes the required model/effort and exact
neutral text, disables all active native feature flags discovered from the
verified CLI (except the flag that suppresses host skill discovery), disables
project/personal document loading, skill instruction injection, bundled skills,
memory use, plan/question tools, and context injection. Native shell, editing,
web, browser, apps, plugins, integrations, and subagent tools are absent from
the locally inspected input. Removed flags are ignored because they no longer
represent capabilities. A new CLI version requires revalidation; it fails closed.

Real execution adds `--ignore-user-config --ignore-rules --ephemeral
--strict-config --skip-git-repo-check --json --color never
--output-last-message <attempt>/answer.txt` plus explicit settings and stdin
`-`, from a new empty temporary working directory with a root marker. Existing
authentication stays in the original Codex home. Parent `CODEX_*` environment
variables are removed except `CODEX_HOME` and `CODEX_API_KEY`. No environment
values or credential files are logged. `OPENAI_*` overrides are removed except
`OPENAI_API_KEY`; inherited Rust diagnostic logging switches are also removed.

The local inspection subcommand does not support `--ignore-user-config` or
`--strict-config`. For that probe only, configured MCP servers are explicitly
disabled and all protocol overrides applied. The inspection must contain
exactly the synthetic user message; otherwise setup stops. Real `exec` ignores
the user config and uses strict parsing. Administrative requirements that
conflict with settings therefore block execution instead of weakening it.

The installed CLI's stdin and logging flags are documented by `codex exec
--help`; behavior was checked against the official configuration schema and
documentation. Source references are recorded in the task report.

## Artifacts and completion

```
batch01/
  execution.json                  CLI version, effective settings, local audit
  state.json                      atomic progress ledger and wait deadlines
  .runner.lock                    OS lock (released by exit/crash)
  001-<prompt-hash-prefix>/
    prompt.txt                    immutable exact input snapshot
    attempt-001/
      prompt.txt                  exact submitted bytes for this attempt
      events.jsonl                raw Codex stdout
      answer.txt                  completed final answer
      partial_answer.txt          partial message, only for incomplete requests
      stderr.txt                  raw process stderr
      attempt.json                status, hash, timestamps, exit code, usage, audit
```

`agent_message`, `reasoning`, and CLI diagnostic `error` items pass the event
audit. Diagnostic messages are retained for failure classification. Native call
items/metadata, unknown events, and malformed complete JSONL records invalidate
the attempt. A truncated final record (including partial UTF-8) makes the audit
incomplete and is recoverable when no completed answer or confirmed violation
exists. A completed answer with an incomplete audit is terminal-invalid and
is never resubmitted. Invalid inputs are terminal and require manual review; they are never
automatically rerun. Every attempt directory is preserved. An answer file
written by Codex or an agent message plus `turn.completed` ends the input even
if cleanup later exits nonzero. Partial messages without a completion signal
remain incomplete and use a separate `partial_answer.txt` file.

Each attempt record commits its outcome, failure timestamp, retry policy,
transient-failure count, and next permitted attempt time before the batch ledger
is updated. Execution and recovery share the same outcome/scheduling function.
The ledger is reconstructed from attempt records after a crash, including quota
failures saved before a ledger update and raw failures captured before outcome
classification. Older invalid attempts with only a truncated tail are re-audited
and become recoverable; superseded audits remain in `audit_history`. Recovery
checks attempt artifacts before scheduling another request. An OS-backed
exclusive lock prevents two instances from processing the same run directory.
Do not manually modify run artifacts while resuming.

Exit codes: 0 finished/check passed; 2 setup/infrastructure/manual-inspection
stop; 130 interrupted. Status and validity must still be read in the ledger.

## Recorded live smoke test

Exactly one live request was submitted through this runner with CLI 0.164.0,
the required model/effort, and the unchanged isolation policy:
`tests/fixtures/smoke_prompt.txt` asks for `WEIGHT_SPACE_SMOKE_OK`.
It returned that exact answer, exited 0, and passed the final tool audit.
CLI startup diagnostics initially caused a conservative audit rejection; the
parser was corrected to recognize diagnostic items and the existing response
was re-audited. It was not submitted again. `resume` confirmed that the finished
input was skipped; the run contains one attempt directory.

To inspect the saved smoke evidence without launching a request:

```powershell
.\.venv\Scripts\python.exe runner\pure_reasoning.py status --run-dir runs\pure_reasoning\smoke_20261104
Get-Content runs\pure_reasoning\smoke_20261104\001-3e6cfb79cb03\attempt-001\attempt.json -Raw
```
