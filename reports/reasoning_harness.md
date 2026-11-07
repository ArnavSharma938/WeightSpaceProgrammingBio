# Reasoning harness

The reasoning harness submits prepared UTF-8 prompt files to fresh Codex CLI
conversations. Each input receives one response. It knows nothing about CNNs,
protein sequences, checkpoints, or grading. It does not generate messages or
supply follow-up instructions.

## Requirements and configuration

Python 3.11+ standard library only. The CNN package is not required by the
harness. Codex CLI 0.164.0 is pinned; other versions stop pending revalidation.
Existing Codex authentication is reused in place. Credentials are not copied
to run directories or printed.

Execution settings are in runner/reasoning_harness.toml:
gpt-6.1-sol, high reasoning effort, and the neutral instruction in
runner/neutral.md: "Answer the user's request." Model substitution is prohibited.

Native capabilities are disabled through configuration: shell and file tools,
web, browser, apps, plugins, integrations, subagents, skills, memories, and
plan/question tools. Context injection and project/personal instruction loading
are disabled. Read-only sandboxing is an additional restriction. Every active
native feature switch is disabled except the one suppressing host skill discovery.

Requests use --ignore-user-config, --ignore-rules, --ephemeral, --strict-config,
--skip-git-repo-check, --json, --color never, --output-last-message, and stdin
argument -. Each request starts in a new empty temporary working directory.
Parent CODEX_* variables are stripped except CODEX_HOME and CODEX_API_KEY;
OPENAI_* overrides are stripped except OPENAI_API_KEY. Rust diagnostic switches
are also removed. Environment values and credential files are never logged.

## PowerShell commands

Run from the repository root. No pip installation is needed for the harness.
The existing .venv supplies Python 3.12.

Codex commands inspect the CLI and authentication:
~~~powershell
codex --version
codex login status
~~~

Install the pinned CLI only if needed:
~~~powershell
npm install -g @openai/codex@0.164.0
~~~

Custom harness commands check local isolation and run mocked tests:
~~~powershell
.\.venv\Scripts\python.exe runner\reasoning_harness.py check
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_reasoning_harness.py -v
~~~

The check command performs no inference. It checks the CLI version, required
flags, bundled model catalog, and locally inspected input. The inspection must
contain exactly the synthetic check message, without a tool catalog or extra
context. The inspection subcommand lacks --ignore-user-config and --strict-config,
so configured MCP servers are explicitly disabled for that local probe. Actual
exec ignores user configuration and uses strict parsing. Conflicting administrative
requirements stop execution.

Submit one or several existing prepared files:
~~~powershell
.\.venv\Scripts\python.exe runner\reasoning_harness.py run --run-dir runs\reasoning_harness\batch03 prompts\experiment_2_original_style\model_001.txt prompts\experiment_2_original_style\model_002.txt prompts\experiment_2_original_style\model_003.txt
~~~

Use a new run directory for new requests. Files run sequentially, each in a
fresh conversation. UTF-8 bytes, whitespace, BOM, and line endings are preserved.
No instructions are appended. The --codex option, placed before the subcommand,
can select an explicit native codex.exe executable.

Inspect results without launching requests:
~~~powershell
.\.venv\Scripts\python.exe runner\reasoning_harness.py status --run-dir runs\reasoning_harness\batch03
Get-ChildItem runs\reasoning_harness\batch03 -Recurse -Filter answer.txt | ForEach-Object { $_.FullName; Get-Content -LiteralPath $_.FullName -Raw -Encoding UTF8 }
~~~

Interrupt with Ctrl+C and resume the batch:
~~~powershell
.\.venv\Scripts\python.exe runner\reasoning_harness.py resume --run-dir runs\reasoning_harness\batch03
~~~

This resumes the saved batch ledger, not a Codex conversation. Completed and
invalid inputs are skipped. Unfinished inputs use fresh exec processes.
Recorded commands and source paths describe the original execution; they are
provenance fields. Resume reads the run directory's saved prompt snapshots.

## Timeouts and waiting

Both run and resume accept:
- --timeout: seconds per process, default 1800.
- --retries: additional transient retries per input across restarts, default 3.
  Increase it on resume to extend a saved allowance.
- --backoff: initial transient backoff in seconds, default 10, doubling to a
  maximum of 300 seconds.
- --usage-recheck: fallback quota interval in seconds, default 3600.

Quota waits are unlimited and do not consume transient retries. A parseable
reported reset epoch, timezone-qualified ISO timestamp, or relative delay is
honored. Missing, ambiguous, or already-expired resets use the conservative
fallback interval. Wait deadlines survive interruption and restarting.

Authentication failures, unsupported configuration, and unexplained incomplete
requests stop for inspection. There are no follow-up messages, automatic
"continue" requests, handoffs, or retries to improve answer quality.

## Artifacts and audit

~~~text
run-directory/
  execution.json
  state.json
  .runner.lock
  001-<prompt-hash-prefix>/
    prompt.txt
    attempt-001/
      prompt.txt
      events.jsonl
      answer.txt
      partial_answer.txt  (only when incomplete)
      stderr.txt
      attempt.json
~~~

Execution metadata records the CLI version and settings. Attempt metadata
records exact command/settings, prompt SHA-256, timestamps, exit status, reported
usage, timeout/interruption flags, and audit. Each attempt is preserved separately.

Only agent-message, reasoning, and diagnostic error items pass the event audit.
Native tool items/call metadata, unknown events, malformed complete JSONL records,
multiple conversations/turns, or an unexpected reported model invalidate a response.
Diagnostic items are retained for failure classification; they are not tool use.

A truncated final JSONL record is incomplete output, not proof of tool use.
Without a completed answer or confirmed violation it is recoverable. A completed
answer with an incomplete audit is terminal-invalid and is not resubmitted.
A CLI final-answer file, or an agent message followed by turn.completed, ends
the request even if cleanup later exits nonzero. A refusal, uncertainty, or
incorrect answer is still completed.

## Durable recovery and concurrency

attempt.json is authoritative; state.json is a reconstructable index. The same
outcome function handles live execution and recovery. Failure status, finish
timestamp, retry policy, transient count, and next permitted attempt time are
committed to the attempt before the batch ledger. Recovery reconciles all outcomes,
including failures and raw streams captured before classification. Superseded
audits are retained in audit_history.

An OS-backed exclusive lock prevents two instances from processing one run
directory and is released on exit/crash. Ctrl+C terminates the child process tree
and preserves progress. If a recorded child PID remains live after forced parent
termination, resume stops until it exits. Do not edit run artifacts while resuming.

Exit codes: 0 finished/check passed; 2 setup/infrastructure/inspection stop;
130 interrupted. Read the ledger and attempt audit to distinguish valid completed
responses from terminal-invalid responses.

## References

- [Codex non-interactive execution](https://developers.openai.com/codex/noninteractive/)
- [Codex configuration reference](https://developers.openai.com/codex/config-reference/)
- [Original experiment harness](https://github.com/RikiyaT/lm_understand-logs/blob/main/harness/codex_config.toml)
