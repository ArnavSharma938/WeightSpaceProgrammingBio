"""Initial pure-reasoning experiments: one immutable message, one fresh codex exec.

Python 3.11+ standard library only. No model/data/checkpoint dependencies.
The retry loop handles failed transports, never answer quality or follow-up turns.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import tomllib

HERE = Path(__file__).resolve().parent
VERIFIED_VERSION = "codex-cli 0.164.0"
MODEL = "gpt-6.1-sol"
TERMINAL = {"completed", "invalid"}


class ProtocolError(RuntimeError):
    pass


def stamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as out:
        json.dump(value, out, indent=2, ensure_ascii=False)
        out.write("\n")
        out.flush()
        os.fsync(out.fileno())
    os.replace(temporary, path)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def environment():
    # Preserve existing auth in place; never copy/read auth.json or log env values.
    allowed_codex = {"CODEX_HOME", "CODEX_API_KEY"}
    return {k: v for k, v in os.environ.items()
            if (not k.upper().startswith("CODEX_") or k.upper() in allowed_codex)
            and (not k.upper().startswith("OPENAI_") or k.upper() == "OPENAI_API_KEY")
            and k.upper() not in {"RUST_LOG", "RUST_BACKTRACE"}}


def executable(value=None):
    path = Path(value or shutil.which("codex.exe") or shutil.which("codex.cmd")
                or shutil.which("codex") or "")
    if path.suffix.lower() == ".cmd":
        # Avoid cmd.exe quoting/expansion of TOML and paths on Windows.
        native = path.parent / "node_modules/@openai/codex/node_modules/@openai/codex-win32-x64/vendor/x86_64-pc-windows-msvc/bin/codex.exe"
        if native.is_file():
            path = native
        else:
            raise ProtocolError("Pass --codex with the native codex.exe path; .cmd wrappers are not used.")
    if not path.is_file():
        raise ProtocolError("Codex executable not found; install the official Codex CLI or pass --codex.")
    return str(path.resolve())


def flatten(settings, prefix=""):
    for key, value in settings.items():
        if not re.fullmatch(r"[A-Za-z0-9_-]+", key):
            raise ProtocolError(f"Unsupported config key spelling: {key}")
        name = prefix + key
        if isinstance(value, dict):
            yield from flatten(value, name + ".")
        else:
            yield name + "=" + json.dumps(value, ensure_ascii=False)


def overrides(settings):
    return [part for entry in flatten(settings) for part in ("-c", entry)]


def local(cli, args, cwd, env):
    result = subprocess.run([cli, *args], cwd=cwd, env=env, capture_output=True,
                            encoding="utf-8", errors="strict", timeout=60)
    if result.returncode:
        raise ProtocolError(f"Local Codex configuration check failed ({args[0]}): {result.stderr.strip()}")
    return result.stdout


def check(cli, cwd):
    """Offline checks only: no exec, no model inference, no credential copying."""
    env = environment()
    version = local(cli, ["--version"], cwd, env).strip()
    if version != VERIFIED_VERSION:
        raise ProtocolError(f"Isolation is verified only for {VERIFIED_VERSION}; found {version}. Revalidate before upgrading.")
    help_text = local(cli, ["exec", "--help"], cwd, env)
    for flag in ("--ignore-user-config", "--ignore-rules", "--ephemeral", "--strict-config", "--json", "--output-last-message"):
        if flag not in help_text:
            raise ProtocolError(f"Installed CLI lacks required isolation flag {flag}")
    settings = tomllib.loads((HERE / "pure_reasoning.toml").read_text(encoding="utf-8"))
    if settings.get("model") != MODEL or settings.get("model_reasoning_effort") != "high":
        raise ProtocolError("Protocol requires gpt-6.1-sol and high reasoning effort.")
    settings["model_instructions_file"] = str(HERE / "neutral.md")
    if (HERE / "neutral.md").read_bytes() != b"Answer the user's request.\n":
        raise ProtocolError("Neutral instructions changed; protocol must be revalidated.")
    feature_text = local(cli, ["features", "list"], cwd, env)
    names = [line.split()[0] for line in feature_text.splitlines()
             if line.split() and "removed" not in line.split()]
    if not {"shell_tool", "code_mode", "plugins", "multi_agent", "skip_host_skill_discovery"} <= set(names):
        raise ProtocolError("Required feature switches are unavailable.")
    # Deny every native feature, including experimental ones, rather than keep
    # an incomplete capability blacklist. Host discovery is disabled explicitly.
    settings["features"] = {name: False for name in names}
    settings["features"]["skip_host_skill_discovery"] = True
    catalog = json.loads(local(cli, ["debug", "models", "--bundled"], cwd, env))
    model = next((m for m in catalog["models"] if m["slug"] == MODEL), None)
    if not model or "high" not in {x["effort"] for x in model["supported_reasoning_levels"]}:
        raise ProtocolError("Requested model/effort absent from installed catalog. No substitution allowed.")
    # debug prompt-input lacks --ignore-user-config. Disable each configured MCP
    # server for this LOCAL probe; real exec ignores user config altogether.
    probe = dict(settings)
    auth_home = Path(env.get("CODEX_HOME", str(Path.home() / ".codex")))
    config_path = auth_home / "config.toml"
    user_config = tomllib.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    probe["mcp_servers"] = {name: {"enabled": False} for name in user_config.get("mcp_servers", {})}
    sentinel = "LOCAL_ONLY_ISOLATION_CHECK"
    inspected = json.loads(local(cli, ["debug", "prompt-input", *overrides(probe), sentinel], cwd, env))
    # Fail closed on any context or tool catalog beyond the exact synthetic user
    # message. The probe does not send that message to an inference endpoint.
    expected = [{"type": "input_text", "text": sentinel}]
    if len(inspected) != 1 or inspected[0].get("type") != "message" or inspected[0].get("role") != "user" or inspected[0].get("content") != expected:
        raise ProtocolError("Isolation blocked: local prompt inspection contains tools or unrelated instructions/context.")
    return {"cli_version": version, "cli_executable": cli,
            "settings": settings, "neutral_instructions": "Answer the user's request.\n",
            "checked_at": stamp(), "local_prompt_audit": "only synthetic user message; no tools/context",
            "authentication": "existing CODEX_HOME, credentials neither copied nor logged"}


@contextlib.contextmanager
def locked(root):
    """OS-backed lock, automatically released after crashes; no stale PID logic."""
    path = root / ".runner.lock"
    handle = path.open("a+b")
    try:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            if path.stat().st_size == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise ProtocolError("Another runner owns this run directory.") from error
        else:
            import fcntl
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise ProtocolError("Another runner owns this run directory.") from error
        yield
    finally:
        handle.close()


def audit(path):
    answer = []
    usage = []
    errors = []
    violations = []
    completed = False
    counts = {}
    truncated = None
    # CLI 0.164 also emits startup/config diagnostics as item.type="error".
    # They carry messages, not native tool execution. Keep them for failure
    # classification without confusing them with command/MCP/browser items.
    allowed_items = {"agent_message", "reasoning", "error"}
    allowed_events = {"thread.started", "turn.started", "turn.completed", "turn.failed", "error",
                      "item.started", "item.updated", "item.completed"}
    if path.exists():
        raw = path.read_bytes()
        lines = raw.splitlines(keepends=True)
        for number, line in enumerate(lines, 1):
            try:
                event = json.loads(line)
                if not isinstance(event, dict):
                    raise ValueError("non-object event")
            except (ValueError, UnicodeDecodeError):
                if number == len(lines) and not line.endswith(b"\n"):
                    truncated = {"line": number, "bytes": len(line)}
                else:
                    violations.append(f"Malformed JSONL line {number}")
                continue
            kind = event.get("type")
            counts[kind] = counts.get(kind, 0) + 1
            if kind in {"thread.started", "turn.started", "turn.completed"} and counts[kind] > 1:
                violations.append(f"Multiple conversations/turns: {kind}")
            if event.get("model") and event["model"] != MODEL:
                violations.append(f"Unexpected reported model: {event['model']}")
            if kind not in allowed_events:
                violations.append(f"Unknown event type: {kind}")
            if kind == "turn.completed":
                completed = True
                usage.append(event.get("usage", {}))
            if kind in {"error", "turn.failed"}:
                errors.append(event)
            if "item" in event:
                item = event["item"]
                if not isinstance(item, dict) or item.get("type") not in allowed_items:
                    violations.append(f"Native tool or unknown item: {item.get('type') if isinstance(item, dict) else item}")
                elif item["type"] == "error":
                    errors.append(event)
                elif kind == "item.completed" and item["type"] == "agent_message":
                    answer.append(item.get("text", ""))
            # Additional wire formats are also checked, including nested calls.
            serialized = json.dumps(event)
            if re.search(r'"(?:function_call|custom_tool_call|tool_call|tool_use|recipient_name)"\s*:', serialized):
                violations.append(f"Native call metadata at line {number}")
    return {"turn_completed": completed, "answer": "\n\n".join(answer),
            "usage": usage, "errors": errors, "violations": violations,
            "truncated_tail": truncated, "audit_complete": not violations and truncated is None,
            "tool_free": not violations and truncated is None}


def classify(text, timed_out=False):
    low = text.lower()
    if any(s in low for s in ("usage_limit", "usage limit", "rate_limit", "rate limit", "429", "quota exceeded")):
        return "usage_limit"
    if any(s in low for s in ("unauthorized", "401", "authentication", "login required", "token expired", "invalid api key")):
        return "authentication"
    if any(s in low for s in ("unknown variant", "unknown field", "unrecognized", "unexpected argument", "invalid config", "not supported", "model_not_found", "model not found", "403", "permission denied")):
        return "unsupported_configuration"
    if timed_out or any(s in low for s in ("timeout", "timed out", "connection", "502", "503", "504", "500", "stream disconnected", "temporarily", "overloaded", "transport")):
        return "transient"
    return "incomplete"


def reset_time(text, now, fallback):
    """Machine reset epochs/ISO or explicit relative waits; never guess timezone."""
    match = re.search(r'"?(?:resets_at|reset_at|reset_timestamp)"?\s*[:=]\s*"?(\d{10,13})', text)
    if match:
        value = float(match[1])
        deadline = value / 1000 if value > 1e12 else value
        return deadline if deadline > now else now + fallback
    match = re.search(r'(\d{4}-\d\d-\d\d[T ][\d:.]+(?:Z|[+-]\d\d:\d\d))', text)
    if match:
        try:
            deadline = dt.datetime.fromisoformat(match[1].replace("Z", "+00:00")).timestamp()
            return deadline if deadline > now else now + fallback
        except ValueError:
            pass
    match = re.search(r'(?:retry.after|try again in)\D{0,8}(\d+)\s*(seconds?|minutes?|hours?|s\b|m\b|h\b)', text, re.I)
    if match:
        units = match[2].lower()[0]
        return now + int(match[1]) * {"s": 1, "m": 60, "h": 3600}[units]
    return now + fallback


def stop_process(process):
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True)
    else:
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def retry_policy(args):
    return {"transient_retries": args.retries, "backoff": args.backoff,
            "usage_recheck": args.usage_recheck}


def outcome(directory, result, policy, transient_failures=0):
    """One durable outcome/schedule calculation, shared by execution and recovery.

    attempt.json is authoritative; state.json is a reconstructable batch index.
    Quota waits are unlimited and never spend the transient retry allowance.
    """
    parsed = audit(directory / "events.jsonl")
    final = directory / "answer.txt"
    final_exists = final.exists() and bool(final.read_bytes())
    completed = final_exists or (parsed["turn_completed"] and bool(parsed["answer"]))
    stderr_path = directory / "stderr.txt"
    stderr = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.exists() else ""
    def strings(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for child in value.values():
                yield from strings(child)
        elif isinstance(value, list):
            for child in value:
                yield from strings(child)
    failure_text = stderr + json.dumps(parsed["errors"]) + "\n" + "\n".join(strings(parsed["errors"]))
    failure = classify(failure_text, result.get("timed_out", False))
    # Older attempts with no error records still retain their known failure class.
    if failure == "incomplete" and result.get("status") in {"usage_limit", "transient", "authentication", "unsupported_configuration"}:
        failure = result["status"]
    status = ("invalid" if not parsed["audit_complete"] else "completed") if completed else (
        "invalid" if parsed["violations"] else failure if failure in {"usage_limit", "authentication", "unsupported_configuration"} else
        "interrupted" if result.get("interrupted") or result.get("status") == "interrupted" else
        "transient" if parsed["truncated_tail"] else failure)
    if not completed and result.get("status") == "running" and result.get("process_exit_status") is None and status == "incomplete":
        status = "interrupted"
    if not final_exists and parsed["answer"]:
        target = final if completed else directory / "partial_answer.txt"
        target.write_text(parsed["answer"], encoding="utf-8", newline="")
    failures = transient_failures + (status == "transient")
    if "audit" in result and (result["audit"] != parsed or result.get("status") != status):
        result.setdefault("audit_history", []).append({"status": result.get("status"),
                                                       "audit": result["audit"], "superseded_at": stamp()})
    result.update(status=status, audit=parsed, recovery_policy=policy,
                  transient_failures=failures)
    if "finished_at_epoch" not in result:
        try:
            finished = dt.datetime.fromisoformat(result["ended_at"]).timestamp()
        except (KeyError, ValueError):
            finished = time.time()
        result["finished_at_epoch"] = finished
    result.setdefault("ended_at", dt.datetime.fromtimestamp(result["finished_at_epoch"], dt.timezone.utc).isoformat())
    if status == "usage_limit":
        result.setdefault("next_attempt_at", reset_time(failure_text, result["finished_at_epoch"], policy["usage_recheck"]))
        result["retry_allowed"] = True
    elif status == "transient":
        result.setdefault("next_attempt_at", result["finished_at_epoch"] + min(policy["backoff"] * 2 ** min(failures - 1, 20), 300))
        result["retry_allowed"] = failures <= policy["transient_retries"]
    else:
        result.pop("next_attempt_at", None)
        result["retry_allowed"] = False
    save(directory / "attempt.json", result)
    return result


def index_outcome(item, result, number):
    item.update(status=result["status"], last_attempt=number,
                transient_failures=result.get("transient_failures", 0),
                retry_allowed=result.get("retry_allowed", False))
    item.pop("next_attempt_at", None)
    if "next_attempt_at" in result:
        item["next_attempt_at"] = result["next_attempt_at"]


def attempt(cli, settings, prompt, directory, timeout, policy=None, transient_failures=0):
    directory.mkdir()
    (directory / "prompt.txt").write_bytes(prompt)
    answer_path = directory / "answer.txt"
    result = {"status": "running", "started_at": stamp(), "prompt_sha256": digest(prompt),
              "settings": settings, "process_exit_status": None,
              "recovery_policy": policy or {"transient_retries": 3, "backoff": 10, "usage_recheck": 3600},
              "transient_failures_before": transient_failures}
    command = [cli, "exec", "--ignore-user-config", "--ignore-rules", "--ephemeral",
               "--strict-config", "--skip-git-repo-check", "--json", "--color", "never",
               "--output-last-message", str(answer_path), *overrides(settings), "-"]
    result["command"] = command
    save(directory / "attempt.json", result)
    interrupted = timed_out = False
    process = None
    with tempfile.TemporaryDirectory(prefix="weight_space_pure_") as workspace:
        Path(workspace, ".git").mkdir()
        with (directory / "events.jsonl").open("wb") as out, (directory / "stderr.txt").open("wb") as err:
            try:
                process = subprocess.Popen(command, cwd=workspace, env=environment(), stdin=subprocess.PIPE,
                                           stdout=out, stderr=err,
                                           creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
                                           start_new_session=os.name != "nt")
                result["pid"] = process.pid
                save(directory / "attempt.json", result)
                process.communicate(input=prompt, timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                stop_process(process)
            except KeyboardInterrupt:
                interrupted = True
                if process:
                    stop_process(process)
            except OSError as error:
                err.write(str(error).encode("utf-8"))
    result.update(ended_at=stamp(), finished_at_epoch=time.time(),
                  process_exit_status=process.returncode if process else None,
                  timed_out=timed_out, interrupted=interrupted)
    # Persist process completion before parsing; a kill here can be recovered
    # from raw streams and the original failure timestamp/policy.
    save(directory / "attempt.json", result)
    return outcome(directory, result, result["recovery_policy"], transient_failures)


def recover(root, state):
    """Reconcile disk artifacts after a kill between answer and state update."""
    policy = state.get("recovery_policy", {"transient_retries": 3, "backoff": 10, "usage_recheck": 3600})
    for item in state["inputs"]:
        folder = root / item["id"]
        failures = 0
        for directory in sorted(folder.glob("attempt-*"), key=lambda path: int(path.name.split("-")[-1])) if folder.exists() else []:
            meta_path = directory / "attempt.json"
            if not meta_path.exists():
                continue
            meta = load(meta_path)
            if meta["status"] == "running":
                pid = meta.get("pid")
                if pid:
                    if os.name == "nt":
                        import ctypes
                        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
                        alive = bool(handle)
                        if handle:
                            ctypes.windll.kernel32.CloseHandle(handle)
                    else:
                        try:
                            os.kill(pid, 0)
                            alive = True
                        except ProcessLookupError:
                            alive = False
                    if alive:
                        raise ProtocolError(f"Previous Codex process PID {pid} may still be running. Wait for it to exit before resume.")
                meta["recovered_after_interruption"] = True
            # Re-audit legacy invalid records too: a truncated final JSON object
            # was formerly mislabeled as permanent protocol invalidity.
            if (directory / "events.jsonl").exists():
                previous_status = meta["status"]
                meta = outcome(directory, meta, policy, failures)
                if previous_status == "invalid" and meta["status"] != "invalid":
                    meta["superseded_status"] = previous_status
                    save(meta_path, meta)
            else:
                # Old records may have no stdout; their saved failure/schedule
                # is still authoritative and must reach the batch ledger.
                meta["recovery_policy"] = policy
                meta.setdefault("transient_failures", failures + (meta["status"] == "transient"))
                if meta["status"] in {"usage_limit", "transient", "running"}:
                    meta = outcome(directory, meta, policy, failures)
            failures = meta.get("transient_failures", failures)
            index_outcome(item, meta, int(directory.name.split("-")[-1]))
            if item["status"] in TERMINAL:
                break
    save(root / "state.json", state)


def run(root, prompts, args):
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with locked(root):
        if (root / "state.json").exists():
            state = load(root / "state.json")
            if prompts:
                raise ProtocolError("This run already exists. Use resume; create a different run directory for new inputs.")
        else:
            if not prompts:
                raise ProtocolError("No existing state to resume.")
            inputs = []
            for index, path in enumerate(prompts):
                raw = Path(path).read_bytes()
                raw.decode("utf-8", errors="strict")
                if not raw:
                    raise ProtocolError(f"Empty prompt file: {path}")
                identity = f"{index + 1:03d}-{digest(raw)[:12]}"
                folder = root / identity
                folder.mkdir()
                (folder / "prompt.txt").write_bytes(raw)
                inputs.append({"id": identity, "source": str(Path(path).resolve()),
                               "sha256": digest(raw), "status": "pending"})
            state = {"created_at": stamp(), "protocol": "initial-pure-reasoning-v1", "inputs": inputs}
            save(root / "state.json", state)
        state["recovery_policy"] = retry_policy(args)
        save(root / "state.json", state)
        recover(root, state)
        if all(item["status"] in TERMINAL for item in state["inputs"]):
            print("All inputs already finished; no Codex request made.")
            return
        cli = executable(args.codex)
        with tempfile.TemporaryDirectory(prefix="pure_check_") as cwd:
            Path(cwd, ".git").mkdir()
            verified = check(cli, cwd)
        if (root / "execution.json").exists():
            previous = load(root / "execution.json")
            if previous["settings"] != verified["settings"] or previous["cli_version"] != verified["cli_version"]:
                raise ProtocolError("Execution protocol changed since this run began; refusing to mix settings.")
        else:
            save(root / "execution.json", verified)
        for item in state["inputs"]:
            if item["status"] in TERMINAL:
                continue
            folder = root / item["id"]
            raw = (folder / "prompt.txt").read_bytes()
            if digest(raw) != item["sha256"]:
                raise ProtocolError("Saved prompt changed; refusing to submit different bytes.")
            prior = list(folder.glob("attempt-*"))
            if item["status"] == "transient" and item.get("transient_failures", 0) > args.retries:
                raise ProtocolError("Transient retry limit reached. Increase --retries to extend the saved budget.")
            while True:
                wait_until = item.get("next_attempt_at", 0)
                while time.time() < wait_until:
                    time.sleep(max(0, min(30, wait_until - time.time())))
                number = len(prior) + 1
                directory = folder / f"attempt-{number:03d}"
                print(f"{item['id']}: attempt {number}", flush=True)
                result = attempt(cli, verified["settings"], raw, directory, args.timeout,
                                 policy=state["recovery_policy"],
                                 transient_failures=item.get("transient_failures", 0))
                prior.append(directory)
                index_outcome(item, result, number)
                save(root / "state.json", state)
                print(f"{item['id']}: {result['status']}", flush=True)
                if result["status"] in TERMINAL:
                    break
                if result["status"] == "interrupted" or result.get("interrupted"):
                    raise KeyboardInterrupt
                if result["status"] in {"authentication", "unsupported_configuration", "incomplete"}:
                    raise ProtocolError(f"Stopped: {result['status']}. Inspect {directory / 'stderr.txt'} and attempt.json.")
                if not result.get("retry_allowed"):
                    raise ProtocolError("Retry limit reached. Progress and next permitted attempt time saved; resume when ready.")
                print(f"Waiting until {dt.datetime.fromtimestamp(result['next_attempt_at'], dt.timezone.utc).isoformat()} before a fresh attempt.", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codex", help="native Codex executable; auto-detected by default")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="offline CLI/isolation preflight; no inference")
    for name in ("run", "resume"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--run-dir", type=Path, required=True)
        cmd.add_argument("--timeout", type=float, default=1800, help="seconds per attempt")
        cmd.add_argument("--retries", type=int, default=3, help="additional transient retries per input across restarts; quota waits are unlimited")
        cmd.add_argument("--backoff", type=float, default=10)
        cmd.add_argument("--usage-recheck", type=float, default=3600, help="seconds if reset unknown (default one hour)")
        if name == "run":
            cmd.add_argument("prompts", nargs="+", type=Path)
    inspect = commands.add_parser("status")
    inspect.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "check":
            with tempfile.TemporaryDirectory(prefix="pure_check_") as cwd:
                Path(cwd, ".git").mkdir()
                verified = check(executable(args.codex), cwd)
            print(json.dumps(verified, indent=2))
        elif args.command == "status":
            print(json.dumps(load(args.run_dir / "state.json"), indent=2))
        else:
            if args.timeout <= 0 or args.retries < 0 or args.backoff < 0 or args.usage_recheck <= 0:
                raise ProtocolError("Timeout/recheck must be positive; retries/backoff must be nonnegative.")
            run(args.run_dir, getattr(args, "prompts", []), args)
        return 0
    except KeyboardInterrupt:
        print("Interrupted. Attempts and progress saved; use resume.", file=sys.stderr)
        return 130
    except (ProtocolError, OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"Stopped: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
