"""Offline protocol tests. Every experimental Codex process is mocked."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runner"))
import reasoning_harness as runner


def events(answer="I am uncertain.", tool=None, complete=True):
    rows = [{"type": "thread.started", "thread_id": "fresh"}, {"type": "turn.started"}]
    if tool:
        rows.append({"type": "item.completed", "item": {"type": tool}})
    rows.append({"type": "item.completed", "item": {"type": "agent_message", "text": answer}})
    if complete:
        rows.append({"type": "turn.completed", "usage": {"input_tokens": 7, "output_tokens": 9}})
    return b"".join(json.dumps(row).encode() + b"\n" for row in rows)


class MockCodex:
    def __init__(self, command, **kwargs):
        self.command = command
        self.kwargs = kwargs
        self.returncode = 0
        self.pid = 99999999
        self.received = None

    def communicate(self, input, timeout):
        self.received = input
        self.kwargs["stdout"].write(events())
        self.kwargs["stdout"].flush()
        return None, None

    def poll(self):
        return self.returncode


class RunnerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.args = argparse.Namespace(codex=None, retries=2, timeout=2, backoff=0, usage_recheck=3600)
        self.verified = {"settings": {"model": runner.MODEL}, "cli_version": runner.VERIFIED_VERSION}

    def tearDown(self):
        self.tmp.cleanup()

    def test_exact_utf8_stdin_and_fresh_exec(self):
        raw = b"\xef\xbb\xbf  \xce\xb1\r\n\r\nlast line  \n"
        created = []
        def factory(*args, **kwargs):
            process = MockCodex(*args, **kwargs)
            created.append(process)
            return process
        with patch.object(runner.subprocess, "Popen", factory):
            result = runner.attempt("codex.exe", {}, raw, self.root / "attempt-001", 1)
        self.assertEqual(created[0].received, raw)
        self.assertEqual((self.root / "attempt-001/prompt.txt").read_bytes(), raw)
        self.assertEqual(result["prompt_sha256"], runner.digest(raw))
        self.assertEqual(result["status"], "completed")
        self.assertEqual(created[0].command[-1], "-")
        self.assertNotIn("resume", created[0].command)
        self.assertEqual(result["audit"]["usage"], [{"input_tokens": 7, "output_tokens": 9}])

    def test_detect_all_native_and_unknown_items(self):
        path = self.root / "events.jsonl"
        for item in ("command_execution", "file_change", "web_search", "mcp_tool_call", "collab_tool_call", "browser", "plan", "future_tool"):
            path.write_bytes(events(tool=item))
            self.assertFalse(runner.audit(path)["tool_free"], item)

    def test_malformed_and_unknown_events_fail_closed(self):
        path = self.root / "events.jsonl"
        path.write_bytes(b'{"type":"future.event"}\n{broken\n')
        self.assertEqual(len(runner.audit(path)["violations"]), 2)

    def test_partial_answer_incomplete(self):
        class Partial(MockCodex):
            def communicate(self, input, timeout):
                self.kwargs["stdout"].write(events(complete=False))
                self.returncode = 1
        with patch.object(runner.subprocess, "Popen", Partial):
            result = runner.attempt("codex", {}, b"prompt", self.root / "a", 1)
        self.assertEqual(result["status"], "incomplete")

    def test_completed_nonzero_exit_never_retried(self):
        class Completed(MockCodex):
            def communicate(self, input, timeout):
                super().communicate(input, timeout)
                self.returncode = 1
                self.kwargs["stderr"].write(b"connection lost after answer")
        with patch.object(runner.subprocess, "Popen", Completed):
            result = runner.attempt("codex", {}, b"prompt", self.root / "a", 1)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["process_exit_status"], 1)

    def test_final_file_is_terminal_without_completed_event(self):
        class Final(MockCodex):
            def communicate(self, input, timeout):
                Path(self.command[self.command.index("--output-last-message") + 1]).write_text("Refusal.")
                self.returncode = 1
        with patch.object(runner.subprocess, "Popen", Final):
            result = runner.attempt("codex", {}, b"prompt", self.root / "a", 1)
        self.assertEqual(result["status"], "completed")

    def test_tool_using_answer_is_terminal_invalid(self):
        class Tool(MockCodex):
            def communicate(self, input, timeout):
                self.kwargs["stdout"].write(events(tool="web_search"))
        with patch.object(runner.subprocess, "Popen", Tool):
            result = runner.attempt("codex", {}, b"prompt", self.root / "a", 1)
        self.assertEqual(result["status"], "invalid")

    def test_timeout_and_ctrl_c_recorded(self):
        for index, error in enumerate((subprocess.TimeoutExpired("codex", 1), KeyboardInterrupt())):
            class Broken(MockCodex):
                def communicate(self, input, timeout):
                    raise error
            with patch.object(runner.subprocess, "Popen", Broken), patch.object(runner, "stop_process"):
                result = runner.attempt("codex", {}, b"prompt", self.root / str(index), 1)
            self.assertEqual(result["status"], "transient" if index == 0 else "interrupted")

    def test_error_classes(self):
        for text, expected in (("HTTP 503 overloaded", "transient"), ("401 unauthorized", "authentication"),
                               ("usage_limit_reached", "usage_limit"), ("unknown field tools", "unsupported_configuration"),
                               ("model_not_found", "unsupported_configuration"), ("unexplained exit", "incomplete")):
            self.assertEqual(runner.classify(text), expected)

    def test_reset_times_and_conservative_fallback(self):
        self.assertEqual(runner.reset_time('{"resets_at":1900000000}', 100, 3600), 1900000000)
        self.assertEqual(runner.reset_time("try again in 2 hours", 100, 3600), 7300)
        self.assertEqual(runner.reset_time("unknown", 100, 3600), 3700)
        self.assertEqual(runner.reset_time("2026-11-04T00:00:00Z", 100, 3600), 1793750400)

    def test_environment_strips_parent_context_preserves_auth(self):
        with patch.dict(os.environ, {"CODEX_THREAD_ID": "parent", "CODEX_HOME": "auth-place", "CODEX_API_KEY": "secret"}):
            env = runner.environment()
        self.assertNotIn("CODEX_THREAD_ID", env)
        self.assertEqual(env["CODEX_HOME"], "auth-place")
        self.assertEqual(env["CODEX_API_KEY"], "secret")

    def test_duplicate_lock(self):
        with runner.locked(self.root):
            with self.assertRaises(runner.ProtocolError):
                with runner.locked(self.root):
                    self.fail("Second lock acquired")
        with runner.locked(self.root):
            pass

    def run_with(self, statuses, prompts=True):
        prompt = self.root / "input.txt"
        prompt.write_bytes(b"exact input\r\n")
        calls = []
        def fake(cli, settings, raw, directory, timeout, policy=None, transient_failures=0):
            directory.mkdir()
            status = statuses[len(calls)]
            result = {"status": status}
            (directory / "stderr.txt").write_text({"transient": "503", "usage_limit": "usage limit", "authentication": "401 unauthorized"}.get(status, ""))
            if status in runner.TERMINAL:
                (directory / "events.jsonl").write_bytes(events(tool="web_search" if status == "invalid" else None))
            result = runner.outcome(directory, result, policy, transient_failures)
            calls.append(raw)
            return result
        with patch.object(runner, "executable", return_value="mock"), patch.object(runner, "check", return_value=self.verified), patch.object(runner, "attempt", fake):
            runner.run(self.root / "run", [prompt] if prompts else [], self.args)
        return calls

    def test_retry_then_completed_and_restart_skip(self):
        self.assertEqual(len(self.run_with(["transient", "completed"])), 2)
        with patch.object(runner, "attempt", side_effect=AssertionError("must skip")):
            runner.run(self.root / "run", [], self.args)
        folders = list((self.root / "run").glob("*/attempt-*"))
        self.assertEqual(len(folders), 2)

    def test_invalid_skipped_on_restart(self):
        self.run_with(["invalid"])
        with patch.object(runner, "attempt", side_effect=AssertionError("must skip")):
            runner.run(self.root / "run", [], self.args)

    def test_auth_stops_without_retry(self):
        with self.assertRaisesRegex(runner.ProtocolError, "authentication"):
            self.run_with(["authentication"])
        self.assertEqual(len(list((self.root / "run").glob("*/attempt-*"))), 1)

    def test_retry_limit_persisted(self):
        self.args.retries = 1
        with self.assertRaisesRegex(runner.ProtocolError, "Retry limit"):
            self.run_with(["transient", "transient"])
        self.assertEqual(len(list((self.root / "run").glob("*/attempt-*"))), 2)

    def test_usage_waits_do_not_spend_transient_budget(self):
        self.args.retries = 0
        clock = [100.0]
        with patch.object(runner.time, "time", side_effect=lambda: clock[0]), patch.object(runner.time, "sleep", side_effect=lambda seconds: clock.__setitem__(0, clock[0] + seconds)):
            calls = self.run_with(["usage_limit"] * 5 + ["completed"])
        self.assertEqual(len(calls), 6)
        self.assertEqual(clock[0], 18100)
        item = runner.load(self.root / "run/state.json")["inputs"][0]
        self.assertEqual(item["status"], "completed")
        self.assertEqual(item["transient_failures"], 0)

    def test_crash_recovery_completed_answer(self):
        folder = self.root / "id/attempt-001"
        folder.mkdir(parents=True)
        runner.save(folder / "attempt.json", {"status": "running"})
        (folder / "events.jsonl").write_bytes(events())
        state = {"inputs": [{"id": "id", "status": "pending"}]}
        runner.recover(self.root, state)
        self.assertEqual(state["inputs"][0]["status"], "completed")
        self.assertTrue((folder / "answer.txt").exists())

    def test_changed_saved_prompt_rejected(self):
        with self.assertRaises(runner.ProtocolError):
            self.run_with(["authentication"])
        folder = next((self.root / "run").glob("001-*"))
        (folder / "prompt.txt").write_bytes(b"modified")
        with patch.object(runner, "executable", return_value="mock"), patch.object(runner, "check", return_value=self.verified):
            with self.assertRaisesRegex(runner.ProtocolError, "Saved prompt changed"):
                runner.run(self.root / "run", [], self.args)

    def test_sequential_files_each_start_fresh(self):
        prompts = [self.root / "one.txt", self.root / "two.txt"]
        for index, path in enumerate(prompts):
            path.write_bytes(f"prompt {index}\r\n".encode())
        calls = []
        def fake(cli, settings, raw, directory, timeout, **kwargs):
            calls.append(raw)
            directory.mkdir()
            result = {"status": "completed"}
            runner.save(directory / "attempt.json", result)
            return result
        with patch.object(runner, "executable", return_value="mock"), patch.object(runner, "check", return_value=self.verified), patch.object(runner, "attempt", fake):
            runner.run(self.root / "run", prompts, self.args)
        self.assertEqual(calls, [p.read_bytes() for p in prompts])

    def test_usage_wait_honored_then_new_attempt(self):
        prompt = self.root / "prompt.txt"
        prompt.write_bytes(b"p")
        clock = [100.0]
        seen = []
        def fake(cli, settings, raw, directory, timeout, policy=None, transient_failures=0):
            directory.mkdir()
            status = "usage_limit" if not seen else "completed"
            result = {"status": status, "audit": {"errors": []}}
            (directory / "stderr.txt").write_text('{"resets_at":1900000000}')
            if status == "completed":
                (directory / "events.jsonl").write_bytes(events())
                (directory / "stderr.txt").write_text("")
            result = runner.outcome(directory, result, policy, transient_failures)
            seen.append(clock[0])
            return result
        def sleep(seconds):
            # Jump to the explicit reset rather than actually waiting in tests.
            clock[0] = 1900000000.0
        with patch.object(runner, "executable", return_value="mock"), patch.object(runner, "check", return_value=self.verified), patch.object(runner, "attempt", fake), patch.object(runner.time, "time", side_effect=lambda: clock[0]), patch.object(runner.time, "sleep", side_effect=sleep):
            runner.run(self.root / "run", [prompt], self.args)
        self.assertEqual(seen, [100.0, 1900000000.0])

    def test_preflight_no_exec_and_context_rejection(self):
        inspected = [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "LOCAL_ONLY_ISOLATION_CHECK"}]}]
        commands = []
        def fake(cli, args, cwd, env):
            commands.append(args)
            if args == ["--version"]:
                return runner.VERIFIED_VERSION
            if args[:2] == ["exec", "--help"]:
                return "--ignore-user-config --ignore-rules --ephemeral --strict-config --json --output-last-message"
            if args[:2] == ["features", "list"]:
                return "\n".join(f"{n} stable true" for n in ("shell_tool", "code_mode", "plugins", "multi_agent", "skip_host_skill_discovery"))
            if args[:2] == ["debug", "models"]:
                return json.dumps({"models": [{"slug": runner.MODEL, "supported_reasoning_levels": [{"effort": "high"}]}]})
            return json.dumps(inspected)
        with patch.object(runner, "local", fake), patch.dict(os.environ, {"CODEX_HOME": str(self.root)}):
            result = runner.check("mock", self.root)
            self.assertEqual(result["settings"]["model"], runner.MODEL)
            inspected.append({"type": "additional_tools", "tools": [{"name": "shell"}]})
            with self.assertRaisesRegex(runner.ProtocolError, "Isolation blocked"):
                runner.check("mock", self.root)
        self.assertFalse(any(c[0] == "exec" and c != ["exec", "--help"] for c in commands))

    def test_unsupported_cli_stops_preflight(self):
        with patch.object(runner, "local", return_value="codex-cli 99.0"):
            with self.assertRaisesRegex(runner.ProtocolError, "Revalidate"):
                runner.check("mock", self.root)

    def test_recover_incomplete_and_live_process_guard(self):
        folder = self.root / "id/attempt-001"
        folder.mkdir(parents=True)
        runner.save(folder / "attempt.json", {"status": "running"})
        state = {"inputs": [{"id": "id", "status": "pending"}]}
        runner.recover(self.root, state)
        self.assertEqual(runner.load(folder / "attempt.json")["status"], "interrupted")
        runner.save(folder / "attempt.json", {"status": "running", "pid": os.getpid()})
        with self.assertRaisesRegex(runner.ProtocolError, "may still be running"):
            runner.recover(self.root, state)

    def test_truncated_tail_is_recoverable_not_tool_violation(self):
        path = self.root / "events.jsonl"
        path.write_bytes(events(complete=False) + b'{"type":"turn.compl')
        parsed = runner.audit(path)
        self.assertIsNotNone(parsed["truncated_tail"])
        self.assertEqual(parsed["violations"], [])
        self.assertFalse(parsed["audit_complete"])
        self.assertFalse(parsed["tool_free"])

    def test_truncated_utf8_and_valid_unterminated_json(self):
        path = self.root / "events.jsonl"
        path.write_bytes(b'{"type":"item.updated","item":{"type":"agent_message","text":"\xce')
        self.assertIsNotNone(runner.audit(path)["truncated_tail"])
        path.write_bytes(events().rstrip(b"\n"))
        self.assertTrue(runner.audit(path)["tool_free"])

    def test_truncated_attempt_retries_but_tool_plus_truncation_is_invalid(self):
        for index, tool in enumerate((None, "command_execution")):
            class Truncated(MockCodex):
                def communicate(self, input, timeout):
                    self.kwargs["stdout"].write(events(tool=tool, complete=False) + b'{"type":')
                    self.returncode = 1
            with patch.object(runner.subprocess, "Popen", Truncated):
                result = runner.attempt("mock", {}, b"p", self.root / str(index), 1)
            self.assertEqual(result["status"], "transient" if tool is None else "invalid")

    def test_completed_answer_with_truncated_audit_is_terminal(self):
        directory = self.root / "a"
        directory.mkdir()
        (directory / "events.jsonl").write_bytes(events() + b'{"type":')
        result = runner.outcome(directory, {}, runner.retry_policy(self.args))
        self.assertEqual(result["status"], "invalid")
        self.assertFalse(result["retry_allowed"])

    def failure_fixture(self, status="usage_limit", meta=None, raw=None, stderr="usage limit"):
        directory = self.root / "id/attempt-001"
        directory.mkdir(parents=True)
        record = {"status": status, "finished_at_epoch": 100.0, **(meta or {})}
        runner.save(directory / "attempt.json", record)
        (directory / "stderr.txt").write_text(stderr)
        if raw is not None:
            (directory / "events.jsonl").write_bytes(raw)
        state = {"inputs": [{"id": "id", "status": "pending"}], "recovery_policy": runner.retry_policy(self.args)}
        return directory, state

    def test_quota_outcome_saves_deadline_before_ledger_update(self):
        class Quota(MockCodex):
            def communicate(self, input, timeout):
                self.kwargs["stderr"].write(b'usage limit {"resets_at":1900000000}')
                self.returncode = 1
        with patch.object(runner.subprocess, "Popen", Quota):
            result = runner.attempt("mock", {}, b"p", self.root / "a", 1)
        saved = runner.load(self.root / "a/attempt.json")
        self.assertEqual(saved["status"], "usage_limit")
        self.assertEqual(saved["next_attempt_at"], 1900000000)
        self.assertEqual(saved, result)

    def test_recover_quota_status_and_deadline_after_ledger_crash(self):
        directory, state = self.failure_fixture(meta={"next_attempt_at": 1900000000})
        runner.recover(self.root, state)
        item = state["inputs"][0]
        self.assertEqual(item["status"], "usage_limit")
        self.assertEqual(item["next_attempt_at"], 1900000000)
        self.assertEqual(item["last_attempt"], 1)
        self.assertTrue(item["retry_allowed"])
        runner.recover(self.root, state)
        self.assertEqual(state["inputs"][0]["next_attempt_at"], 1900000000)
        # Unlimited quota waiting can exceed 999 attempts; sort numerically.
        for number in (999, 1000):
            later = directory.parent / f"attempt-{number:03d}"
            later.mkdir()
            runner.save(later / "attempt.json", {"status": "usage_limit", "next_attempt_at": 1900000000 + number})
        runner.recover(self.root, state)
        self.assertEqual(state["inputs"][0]["last_attempt"], 1000)
        self.assertEqual(state["inputs"][0]["next_attempt_at"], 1900001000)

    def test_recover_raw_quota_before_outcome_commit(self):
        raw = json.dumps({"type": "turn.failed", "error": {"message": 'usage limit; {"resets_at":1900000000}'}}).encode() + b"\n"
        directory, state = self.failure_fixture(status="running", raw=raw, stderr="")
        runner.recover(self.root, state)
        self.assertEqual(state["inputs"][0]["status"], "usage_limit")
        self.assertEqual(state["inputs"][0]["next_attempt_at"], 1900000000)
        self.assertEqual(runner.load(directory / "attempt.json")["next_attempt_at"], 1900000000)

    def test_recover_unknown_reset_from_original_failure_time(self):
        directory, state = self.failure_fixture()
        with patch.object(runner.time, "time", return_value=999):
            runner.recover(self.root, state)
        self.assertEqual(state["inputs"][0]["next_attempt_at"], 3700)

    def test_legacy_invalid_truncated_record_is_recovered(self):
        directory, state = self.failure_fixture(status="invalid", raw=b'{"type":"item.compl', stderr="")
        state["inputs"][0]["status"] = "invalid"
        runner.recover(self.root, state)
        self.assertEqual(state["inputs"][0]["status"], "transient")
        self.assertEqual(runner.load(directory / "attempt.json")["superseded_status"], "invalid")

    def test_transient_budget_survives_restart_and_can_be_extended(self):
        self.args.retries = 1
        with self.assertRaises(runner.ProtocolError):
            self.run_with(["transient", "transient"])
        with self.assertRaisesRegex(runner.ProtocolError, "Increase --retries"):
            self.run_with([], prompts=False)
        self.args.retries = 2
        self.assertEqual(len(self.run_with(["completed"], prompts=False)), 1)

    def test_ctrl_c_with_quota_preserves_reset_for_resume(self):
        directory, state = self.failure_fixture(status="running", meta={"interrupted": True, "process_exit_status": 1},
                                                stderr='usage limit {"resets_at":1900000000}')
        runner.recover(self.root, state)
        self.assertEqual(state["inputs"][0]["status"], "usage_limit")
        self.assertEqual(state["inputs"][0]["next_attempt_at"], 1900000000)

    def test_cli_diagnostic_items_are_not_tools(self):
        path = self.root / "events.jsonl"
        diagnostic = {"type": "item.completed", "item": {"type": "error", "message": "Code Mode is unavailable because code-mode host is disabled."}}
        path.write_bytes(json.dumps(diagnostic).encode() + b"\n" + events())
        parsed = runner.audit(path)
        self.assertTrue(parsed["tool_free"])
        self.assertEqual(parsed["errors"], [diagnostic])

    def test_reaudit_preserves_original_audit(self):
        directory, state = self.failure_fixture(status="invalid", raw=events(), stderr="")
        runner.save(directory / "attempt.json", {"status": "invalid", "audit": {"violations": ["old diagnostic parser"]}})
        runner.recover(self.root, state)
        record = runner.load(directory / "attempt.json")
        self.assertEqual(record["status"], "completed")
        self.assertEqual(record["audit_history"][0]["audit"]["violations"], ["old diagnostic parser"])

    def test_resume_does_not_submit_before_recovered_quota_deadline(self):
        directory, state = self.failure_fixture(meta={"next_attempt_at": 1900000000})
        raw = b"saved prompt\r\n"
        (directory.parent / "prompt.txt").write_bytes(raw)
        state["inputs"][0]["sha256"] = runner.digest(raw)
        runner.save(self.root / "state.json", state)
        clock = [100.0]
        seen = []
        def fake(cli, settings, prompt, target, timeout, policy=None, transient_failures=0):
            seen.append(clock[0])
            target.mkdir()
            (target / "events.jsonl").write_bytes(events())
            return runner.outcome(target, {}, policy, transient_failures)
        with patch.object(runner, "executable", return_value="mock"), patch.object(runner, "check", return_value=self.verified), patch.object(runner, "attempt", fake), patch.object(runner.time, "time", side_effect=lambda: clock[0]), patch.object(runner.time, "sleep", side_effect=lambda seconds: clock.__setitem__(0, 1900000000.0)):
            runner.run(self.root, [], self.args)
        self.assertEqual(seen, [1900000000.0])
        self.assertEqual(len(list(directory.parent.glob("attempt-*"))), 2)

    def test_stale_reported_reset_uses_conservative_wait(self):
        self.assertEqual(runner.reset_time('{"resets_at":1900000000}', 1900000001, 3600), 1900003601)


if __name__ == "__main__":
    unittest.main()
