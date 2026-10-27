#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Linux process/CLI controls; exercise the actual cleanup implementation."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
import harness as h
import ais_harness as ais


class Reliability(unittest.TestCase):
    def test_report_cli(self):
        for statuses, complete, failure, expected in (
            (["PASS"], True, False, 0),
            (["UNSUPPORTED"], True, False, 0),
            (["NOT_TESTED"], True, False, 0),
            (["FAIL"], True, False, 1),
            (["UNKNOWN"], True, False, 1),
            (["invented"], True, False, 1),
            ([], True, False, 1),
            (["PASS"], False, False, 1),
            (["PASS"], True, True, 1),
        ):
            with self.subTest(
                statuses=statuses, complete=complete
            ), tempfile.TemporaryDirectory() as tmp:
                run = Path(tmp)
                h.save(
                    run / "state.json", dict(schema=1, run_dir=str(run), stopped=True)
                )
                h.save(
                    run / "results.json",
                    [
                        dict(case=str(i), path="test", status=s)
                        for i, s in enumerate(statuses)
                    ],
                )
                if complete:
                    h.save(run / "test-status.json", dict(complete=True))
                if failure:
                    h.save(run / "failure.json", dict(error="injected"))
                p = subprocess.run(
                    [sys.executable, h.__file__, "report", "--run-dir", tmp],
                    capture_output=True,
                    timeout=10,
                )
                self.assertEqual(p.returncode, expected, p.stderr)
        for contents in (None, "{", "{}"):
            with tempfile.TemporaryDirectory() as tmp:
                if contents is not None:
                    Path(tmp, "state.json").write_text(contents)
                p = subprocess.run(
                    [sys.executable, h.__file__, "report", "--run-dir", tmp],
                    capture_output=True,
                    timeout=10,
                )
                self.assertNotEqual(p.returncode, 0)

    def spawn(self, marker, stubborn=False):
        code = "import signal,time; "
        if stubborn:
            code += "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        code += "print('ready',flush=True); time.sleep(60)"
        p = subprocess.Popen(
            [sys.executable, "-c", code, marker], stdout=subprocess.PIPE, text=True
        )
        self.assertEqual(p.stdout.readline().strip(), "ready")
        self.addCleanup(self.reap, p)
        return p

    @staticmethod
    def reap(p):
        if p.poll() is None:
            p.kill()
        p.wait(timeout=3)
        p.stdout.close()

    def test_stale_and_forged_identity_preserve_sentinel(self):
        p = self.spawn("owned-marker")
        with self.assertRaisesRegex(RuntimeError, "PID reused"):
            h.stop_owned(p.pid, "wrong", "owned-marker")
        with self.assertRaisesRegex(RuntimeError, "command mismatch"):
            h.stop_owned(p.pid, h.process_identity(p.pid), "foreign-marker")
        self.assertIsNone(p.poll())
        with self.assertRaisesRegex(
            RuntimeError, "ownership mismatch"
        ), tempfile.TemporaryDirectory() as tmp:
            h.save(Path(tmp) / "state.json", dict(schema=1, run_dir="other", pid=p.pid))
            h.down(Path(tmp))
        self.assertIsNone(p.poll())

    def test_pidfd_retains_original_across_exit_before_signal(self):
        p = self.spawn("owned-marker")
        sentinel = self.spawn("sentinel")
        send = signal.pidfd_send_signal
        calls = []

        def exit_then_send(fd, sig):
            # A deterministic schedule at the validation/signal boundary.
            # Actual PID recycling is not forced; no numeric-PID kill is allowed.
            send(fd, signal.SIGKILL)
            p.wait(timeout=3)
            calls.append(fd)
            return send(fd, sig)

        with patch.object(
            signal, "pidfd_send_signal", side_effect=exit_then_send
        ), patch.object(
            h.os, "kill", side_effect=AssertionError("numeric kill forbidden")
        ):
            self.assertEqual(
                h.stop_owned(p.pid, h.process_identity(p.pid), "owned-marker"), []
            )
        self.assertEqual(len(calls), 1)
        self.assertIsNone(sentinel.poll())

    def test_escalation_and_idempotent_down(self):
        p = self.spawn("owned-marker", stubborn=True)
        sent = h.stop_owned(p.pid, h.process_identity(p.pid), "owned-marker", 0.05, 1)
        self.assertEqual(sent, ["SIGTERM", "SIGKILL"])
        p.wait(timeout=3)
        self.assertEqual(h.stop_owned(p.pid, "old", "owned-marker"), [])

    def test_repeated_signals_inside_direct_and_ais_cleanup(self):
        # Signals are delivered to this process while stop is in progress;
        # save/remaining-node cleanup must still complete, then handlers restore.
        for role in ("direct", "ais"):
            with self.subTest(role=role), tempfile.TemporaryDirectory() as tmp:
                run = Path(tmp)
                marker = (
                    f"-dir={run / 'data'}"
                    if role == "direct"
                    else f"-local_config={run / 'target/local.json'}"
                )
                p = self.spawn(marker)
                start = h.process_identity(p.pid)
                if role == "direct":
                    h.save(
                        run / "state.json",
                        dict(
                            schema=1,
                            run_dir=str(run),
                            pid=p.pid,
                            process_start=start,
                            stopped=False,
                        ),
                    )
                else:
                    h.save(
                        run / "ais-state.json",
                        dict(
                            schema=1,
                            run_dir=str(run),
                            nodes={"target": dict(pid=p.pid, start=start)},
                            stopped=False,
                        ),
                    )
                real = signal.pidfd_send_signal
                previous = {
                    sig: signal.getsignal(sig)
                    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM)
                }

                def send(fd, sig):
                    for _ in range(3):
                        for incoming in previous:
                            signal.raise_signal(incoming)
                    return real(fd, sig)

                with patch.object(signal, "pidfd_send_signal", side_effect=send):
                    (h.down if role == "direct" else ais.down)(run)
                p.wait(timeout=3)
                (h.down if role == "direct" else ais.down)(run)
                state = (h.state_for if role == "direct" else ais.state_for)(run)
                self.assertTrue(state["stopped"])
                for sig, handler in previous.items():
                    self.assertEqual(signal.getsignal(sig), handler)

    def test_missing_pidfd_fails_closed(self):
        p = self.spawn("owned-marker")
        with patch.object(
            h.os, "pidfd_open", side_effect=OSError("unavailable")
        ), self.assertRaises(OSError):
            h.stop_owned(p.pid, h.process_identity(p.pid), "owned-marker")
        self.assertIsNone(p.poll())


if __name__ == "__main__":
    unittest.main(verbosity=2)
