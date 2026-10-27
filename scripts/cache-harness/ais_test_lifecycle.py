#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Linux integration fault injections for ais_harness (requires built binaries)."""

import argparse
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import ais_harness as ais


def classify_fault(kind, result, calls, error, elapsed):
    """Require the intended HTTP fault, smoke stage and expected rejection."""
    methods = [c["method"] for c in calls]
    delivered = any(c.get("fault_sent") == kind for c in calls)
    if elapsed >= 35 or not methods or result.get("exit") not in (0, 1):
        return "FAIL"
    payload = result.get("result") or {}
    if kind in ("healthy", "footer-first"):
        if result["status"] != "PASS":
            return "FAIL"
        if kind == "footer-first":
            return (
                "PASS"
                if calls[0]["method"] == "GET" and calls[0].get("range") == "bytes=-512"
                else "FAIL"
            )
        return "PASS" if methods[0] == "HEAD" and "GET" in methods else "FAIL"
    if not delivered:
        return "FAIL"
    if kind in ("redirect", "timeout"):
        reached = all(m == "HEAD" for m in methods)
        stage = "open"
    else:
        reached = methods[0] == "HEAD" and "GET" in methods
        stage = "read"
    if not reached:
        return "FAIL"
    if (
        kind == "changed-validator"
        and result["status"] == "PASS"
        and result["exit"] == 0
    ):
        return "UNSUPPORTED"
    if (
        result["status"] != "FAIL"
        or result["exit"] != 1
        or payload.get("stage") != stage
    ):
        return "FAIL"
    expected = {
        "redirect": ("rest_reactor::head_object: HTTP 307",),
        "timeout": (
            "rest_reactor::head_object:",
            "exhausted retries",
            "Timeout was reached",
        ),
        "truncated": ("rest_reactor: exhausted retries",),
        "malformed-range": ("206 Content-Range mismatch",),
        "changed-validator": ("ETag", "mismatch"),
        "wrong-block": ("read bytes mismatch",),
    }[kind]
    return "PASS" if all(token in error for token in expected) else "FAIL"


def library_faults(args, root):
    """Actual HTTP delivery plus semantic rejection, with positive controls."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading

    rows = []
    for kind in (
        "healthy",
        "footer-first",
        "redirect",
        "truncated",
        "timeout",
        "changed-validator",
        "malformed-range",
        "wrong-block",
    ):
        data = ais.origin.fixture_bytes(
            "fault-object", 131103 if kind == "wrong-block" else 4096
        )
        fixture_path = root / (kind + "-fixture.bin")
        fixture_path.write_bytes(data)
        fixture = dict(key="object.bin", file=str(fixture_path), etag='"v1"')
        calls = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_HEAD(self):
                event = dict(method="HEAD", path=self.path)
                calls.append(event)
                if kind == "timeout":
                    event["fault_sent"] = kind  # Request reached the deliberate stall.
                    time.sleep(6)
                self.send_response(307 if kind == "redirect" else 200)
                if kind == "redirect":
                    self.send_header(
                        "Location", f"http://127.0.0.1:{server.server_port}/delegated"
                    )
                self.send_header("Content-Length", str(len(data)))
                self.send_header("ETag", '"v1"')
                try:
                    self.end_headers()
                    if kind == "redirect":
                        event["fault_sent"] = kind
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def do_GET(self):
                range_header = self.headers.get("Range", f"bytes=0-{len(data)-1}")
                event = dict(method="GET", path=self.path, range=range_header)
                calls.append(event)
                first, last = range_header.removeprefix("bytes=").split("-", 1)
                start = int(first) if first else max(0, len(data) - int(last))
                end = min(int(last), len(data) - 1) if first and last else len(data) - 1
                if start >= len(data):
                    self.send_response(416)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                payload = data[start : end + 1]
                wrong_block = kind == "wrong-block" and start >= 65536 and bool(first)
                if wrong_block:
                    payload = data[: len(payload)]
                self.send_response(206)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header(
                    "Content-Range",
                    (
                        "invalid"
                        if kind == "malformed-range"
                        else f"bytes {start}-{end}/{len(data)}"
                    ),
                )
                self.send_header(
                    "ETag", '"v2"' if kind == "changed-validator" and first else '"v1"'
                )
                try:
                    self.end_headers()
                    self.wfile.write(payload[:-1] if kind == "truncated" else payload)
                    self.wfile.flush()
                    if (
                        kind in ("truncated", "malformed-range")
                        or (kind == "changed-validator" and first)
                        or wrong_block
                    ):
                        event["fault_sent"] = kind
                except (BrokenPipeError, ConnectionResetError):
                    pass
                self.close_connection = True

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        # Join all handlers before the loop changes their captured fixture state.
        server.daemon_threads = False
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        started = time.monotonic()
        try:
            result = ais.origin.probe(
                args.binary,
                dict(
                    endpoint=f"http://127.0.0.1:{server.server_port}",
                    bucket="fault-bucket",
                ),
                fixture,
                root,
                "library-" + kind,
                mode="footer-first" if kind == "footer-first" else "read",
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
        elapsed = round(time.monotonic() - started, 3)
        error_file = root / f"library-{kind}.stderr"
        error = error_file.read_text() if error_file.exists() else ""
        status = classify_fault(kind, result, calls, error, elapsed)
        rows.append(
            dict(
                case=kind,
                status=status,
                probe=result,
                elapsed_seconds=elapsed,
                requests=calls,
                error=error[:2000],
            )
        )
        ais.origin.save(root / "library-faults.json", rows)
    if any(row["status"] == "FAIL" for row in rows):
        raise AssertionError("protocol regression failed; inspect library-faults.json")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--ais-source", type=Path, required=True)
    parser.add_argument("--aisnode", type=Path, required=True)
    parser.add_argument("--aisnode-sha256", required=True)
    parser.add_argument("--weed", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    checks = []
    script = Path(ais.__file__).resolve()
    base = [
        sys.executable,
        str(script),
        "all",
        "--ais-source",
        str(args.ais_source),
        "--aisnode",
        str(args.aisnode),
        "--aisnode-sha256",
        args.aisnode_sha256,
        "--weed",
        str(args.weed),
        "--binary",
        str(args.binary),
    ]

    def record(name):
        checks.append(dict(case=name, status="PASS"))
        ais.origin.save(root / "checks.json", checks)

    def stopped(run):
        state = ais.state_for(run)
        assert state["stopped"] and not state["cleanup_errors"]
        for node in state["nodes"].values():
            assert ais.origin.process_identity(node["pid"]) is None
        if (run / "origin/state.json").exists():
            origin = ais.origin.state_for(run / "origin")
            if "pid" in origin:
                assert (
                    origin["stopped"]
                    and ais.origin.process_identity(origin["pid"]) is None
                )

    def invoke(name, extras=(), expected=1, terminate=False):
        run = root / name
        label = name + ("-repeat" if (root / f"{name}.stdout").exists() else "")
        with (root / f"{label}.stdout").open("w") as out, (
            root / f"{label}.stderr"
        ).open("w") as err:
            proc = subprocess.Popen(
                base + ["--run-dir", str(run)] + list(extras), stdout=out, stderr=err
            )
            try:
                if terminate:
                    deadline = time.monotonic() + 60
                    while time.monotonic() < deadline:
                        if proc.poll() is not None:
                            raise RuntimeError("run exited before interruption")
                        if (run / "ais-state.json").exists() and ais.state_for(run)[
                            "nodes"
                        ]:
                            proc.send_signal(signal.SIGTERM)
                            break
                        time.sleep(0.05)
                    else:
                        raise TimeoutError("no node to interrupt")
                assert proc.wait(timeout=330) == expected
                if (run / "ais-state.json").exists():
                    stopped(run)  # Assert before emergency cleanup; never mask a leak.
            finally:
                if proc.poll() is None:
                    proc.send_signal(signal.SIGTERM)
                    proc.wait(timeout=45)
                if (run / "ais-state.json").exists():
                    ais.down(run)
        return run

    try:
        library_faults(args, root)
        record("library-adversarial-protocol-probes")
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", 19430))
            listener.listen(1)
            run = invoke("port-collision")
            assert not run.exists()
            with socket.create_connection(("127.0.0.1", 19430), timeout=1):
                connection, _ = listener.accept()
                connection.close()
        record("port-conflict-preserves-occupant")
        run = invoke("not-ready", ["--ready-timeout", "0.001"])
        stopped(run)
        record("readiness-deadline-cleanup")
        run = invoke("user-term", terminate=True)
        stopped(run)
        assert ais.origin.load(run / "error.json")["type"] == "RunInterrupted"
        record("sigterm-cleanup")
        run = invoke("whole-deadline", ["--deadline", "1"])
        stopped(run)
        assert ais.origin.load(run / "error.json")["type"] == "RunInterrupted"
        record("whole-run-deadline-cleanup")
        # Fault injection replaces the seed function only inside this test process.
        seed_run = root / "seed-failure"
        cfg = argparse.Namespace(
            **vars(args),
            run_dir=seed_run,
            port_base=19330,
            ais_port_base=19430,
            ready_timeout=75,
            mode="root",
            weed_sha256=ais.origin.WEED_SHA,
        )
        saved = ais.origin.seed

        def fail_seed(run):
            raise RuntimeError("injected seed failure")

        ais.origin.seed = fail_seed
        try:
            try:
                ais.up(cfg)
                raise AssertionError("seed failure was hidden")
            except RuntimeError as exc:
                assert str(exc) == "injected seed failure"
                stopped(seed_run)
        finally:
            ais.origin.seed = saved
            if (seed_run / "ais-state.json").exists():
                ais.down(seed_run)
        stopped(seed_run)
        record("seed-failure-cleanup")
        run = invoke("probe-failure", ["--binary", "/bin/false"])
        stopped(run)
        rows = ais.origin.load(run / "results.json")
        assert next(r for r in rows if r["case"] == "A2-cucascade")["status"] == "FAIL"
        assert ais.origin.load(run / "summary.json")["status"] == "FAIL"
        record("probe-failure-stays-failure-and-cleans-up")
        ais.down(run)
        ais.down(run)
        stopped(run)
        record("idempotent-down")
        before = ais.origin.sha(run / "ais-state.json")
        invoke("probe-failure")
        assert before == ais.origin.sha(run / "ais-state.json")
        record("existing-run-refused-unmodified")
        guard = root / "pid-guard"
        guard.mkdir()
        ais.origin.save(
            guard / "ais-state.json",
            dict(
                schema=1,
                run_dir=str(guard),
                nodes={"proxy": dict(pid=os.getpid(), start="wrong")},
                stopped=False,
            ),
        )
        try:
            ais.stop_node(guard, "proxy")
            raise AssertionError("PID reuse accepted")
        except RuntimeError as exc:
            assert "PID reused" in str(exc)
        record("pid-reuse-guard-preserves-test-process")
        # A network redirect to anything outside the exact owned ports is refused.
        foreign = dict(endpoint="http://127.0.0.1:9", ports=[19430])
        try:
            ais.request(foreign, "GET", "/")
            raise AssertionError("foreign authority accepted")
        except RuntimeError as exc:
            assert "outside owned" in str(exc)
        record("reference-authority-allowlist")
    except BaseException as exc:
        checks.append(dict(case="acceptance", status="FAIL", error=repr(exc)))
        raise
    finally:
        ais.origin.save(root / "checks.json", checks)
        print(ais.json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
