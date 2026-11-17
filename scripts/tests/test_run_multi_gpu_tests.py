# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import os
import signal
import subprocess
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "run_multi_gpu_tests.py"
SPEC = importlib.util.spec_from_file_location("run_multi_gpu_tests", SCRIPT)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)

PASS_REPORT = """<Catch2TestRun>
<TestCase name="selected"><OverallResult success="true" skips="0"/></TestCase>
<OverallResults successes="1" failures="0" expectedFailures="0" skips="0"/>
<OverallResultsCases successes="1" failures="0" expectedFailures="0" skips="0"/>
</Catch2TestRun>"""


class MultiGpuRunnerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.report = self.root / "report.xml"

    def test_complete_report(self):
        self.report.write_text(PASS_REPORT)
        self.assertEqual(RUNNER.check_report(self.report), 1)

    def test_incomplete_reports_are_rejected(self):
        reports = {
            "empty": "<Catch2TestRun/>",
            "truncated": PASS_REPORT[:-10],
            "other format": PASS_REPORT.replace("Catch2TestRun", "testsuites"),
            "missing totals": PASS_REPORT.replace("OverallResultsCases", "OtherTotals"),
            "all skipped": PASS_REPORT.replace('successes="1"', 'successes="0"'),
            "partial skip": PASS_REPORT.replace('skips="0"', 'skips="1"'),
            "case-only skip": PASS_REPORT.replace('skips="0"', 'skips="1"', 1),
            "skip in section": PASS_REPORT.replace(
                "<OverallResult", "<Skip/> <OverallResult", 1
            ),
            "failure": PASS_REPORT.replace('failures="0"', 'failures="1"'),
            "expected failure": PASS_REPORT.replace(
                'expectedFailures="0"', 'expectedFailures="1"'
            ),
            "invalid counts": PASS_REPORT.replace('successes="1"', 'successes="many"'),
            "missing case result": PASS_REPORT.replace(
                '<OverallResult success="true" skips="0"/>', ""
            ),
            "failed case result": PASS_REPORT.replace(
                'success="true"', 'success="false"'
            ),
            "mismatched case count": PASS_REPORT.replace(
                '<OverallResultsCases successes="1"',
                '<OverallResultsCases successes="2"',
            ),
        }
        for name, report in reports.items():
            with self.subTest(name=name):
                self.report.write_text(report)
                with self.assertRaises((ValueError, RUNNER.ET.ParseError)):
                    RUNNER.check_report(self.report)

    def make_binary(self, body):
        binary = self.root / "test_binary"
        binary.write_text(f"#!{sys.executable}\n" + body)
        binary.chmod(0o755)
        return binary

    def test_successful_process(self):
        binary = self.make_binary(
            "import pathlib, sys\n"
            "assert sys.argv[1:-1] == ['[multi-device]', '--reporter', 'xml', '--warn', 'NoAssertions', '--out']\n"
            f"pathlib.Path(sys.argv[-1]).write_text({PASS_REPORT!r})\n"
            "print('test diagnostics')\n"
        )
        self.assertTrue(RUNNER.run_tests(binary, self.root, 5))
        self.assertIn("test diagnostics", (self.root / "test_binary.log").read_text())

    def test_process_failure_rejects_even_a_passing_report(self):
        binary = self.make_binary(
            "import pathlib, sys\n"
            f"pathlib.Path(sys.argv[-1]).write_text({PASS_REPORT!r})\n"
            "sys.exit(1)\n"
        )
        self.assertFalse(RUNNER.run_tests(binary, self.root, 5))

    def test_previous_report_cannot_mask_missing_output(self):
        binary = self.make_binary("pass\n")
        report = self.root / "test_binary.xml"
        report.write_text(PASS_REPORT)
        self.assertFalse(RUNNER.run_tests(binary, self.root, 5))
        self.assertFalse(report.exists())

    def test_report_cleanup_failure_is_reported(self):
        binary = self.make_binary("pass\n")
        (self.root / "test_binary.xml").mkdir()
        log = self.root / "test_binary.log"
        log.write_text("stale diagnostics")
        self.assertFalse(RUNNER.run_tests(binary, self.root, 5))
        self.assertEqual(log.read_text(), "")

    def test_reap_after_timeout_is_bounded(self):
        with patch.object(RUNNER.subprocess, "Popen") as popen:
            with patch.object(RUNNER.os, "killpg") as kill:
                process = popen.return_value
                process.pid = 123
                process.wait.side_effect = subprocess.TimeoutExpired("test", 5)
                with patch.object(RUNNER.os, "waitid", return_value=None) as observe:
                    with patch.object(RUNNER.time, "monotonic", side_effect=[0, 0, 2]):
                        self.assertFalse(
                            RUNNER.run_tests(self.root / "stuck", self.root, 1)
                        )
                self.assertTrue(observe.call_args.args[2] & os.WNOWAIT)
                kill.assert_called_once_with(123, signal.SIGKILL)
                process.wait.assert_called_once_with(timeout=5)

    def test_exit_during_final_sleep_is_not_a_timeout(self):
        with patch.object(RUNNER.subprocess, "Popen") as popen:
            popen.return_value.wait.return_value = 0
            with patch.object(RUNNER.os, "killpg"):
                with patch.object(RUNNER.os, "waitid", side_effect=[None, object()]):
                    with patch.object(RUNNER.time, "monotonic", side_effect=[0, 0, 2]):
                        with patch.object(RUNNER.time, "sleep"):
                            with patch.object(RUNNER, "check_report", return_value=1):
                                self.assertTrue(
                                    RUNNER.run_tests(
                                        self.root / "finished", self.root, 1
                                    )
                                )

    def test_timeout_kills_child_processes(self):
        child = self.root / "child.py"
        child.write_text(
            "import os, pathlib, sys, time\n"
            "while True:\n"
            "    try: os.kill(int(sys.argv[1]), 0)\n"
            "    except ProcessLookupError: break\n"
            "    time.sleep(0.02)\n"
            f"pathlib.Path({str(self.root / 'orphan')!r}).touch()\n"
        )
        binary = self.make_binary(
            "import os, subprocess, sys, time\n"
            f"subprocess.Popen([sys.executable, {str(child)!r}, str(os.getpid())])\n"
            "time.sleep(30)\n"
        )
        self.assertFalse(RUNNER.run_tests(binary, self.root, 1))
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline and not (self.root / "orphan").exists():
            time.sleep(0.02)
        self.assertFalse((self.root / "orphan").exists())

    def test_normal_exit_still_cleans_up_child_processes(self):
        with patch.object(RUNNER.subprocess, "Popen") as popen:
            with patch.object(RUNNER.os, "killpg") as kill:
                popen.return_value.pid = 123
                popen.return_value.wait.return_value = 1
                with patch.object(RUNNER.os, "waitid", return_value=object()):
                    self.assertFalse(
                        RUNNER.run_tests(self.root / "failed", self.root, 1)
                    )
                kill.assert_called_once_with(123, signal.SIGKILL)

    def test_interrupt_still_cleans_up_child_processes(self):
        with patch.object(RUNNER.subprocess, "Popen") as popen:
            with patch.object(RUNNER.os, "killpg") as kill:
                popen.return_value.pid = 123
                popen.return_value.wait.return_value = 0
                with patch.object(RUNNER.os, "waitid", side_effect=KeyboardInterrupt):
                    with self.assertRaises(KeyboardInterrupt):
                        RUNNER.run_tests(self.root / "interrupted", self.root, 1)
                kill.assert_called_once_with(123, signal.SIGKILL)

    def test_interrupt_survives_reap_timeout(self):
        with patch.object(RUNNER.subprocess, "Popen") as popen:
            with patch.object(RUNNER.os, "killpg"):
                popen.return_value.wait.side_effect = subprocess.TimeoutExpired(
                    "test", 5
                )
                with patch.object(RUNNER.os, "waitid", side_effect=KeyboardInterrupt):
                    with self.assertRaises(KeyboardInterrupt):
                        RUNNER.run_tests(self.root / "interrupted", self.root, 1)

    def test_sigterm_during_launch_cleans_up_and_restores_handler(self):
        binary = self.make_binary("import time\ntime.sleep(30)\n")
        test_dir = self.root / "test"
        test_dir.mkdir()
        (test_dir / "cucascade_tests").symlink_to(binary)
        original_popen = subprocess.Popen
        original_handler = signal.getsignal(signal.SIGTERM)
        children = []

        def launch(*args, **kwargs):
            child = original_popen(*args, **kwargs)
            children.append(child)
            os.kill(os.getpid(), signal.SIGTERM)
            return child

        try:
            with patch.object(RUNNER.subprocess, "Popen", side_effect=launch):
                with patch.object(
                    sys, "argv", [str(SCRIPT), "--build-dir", str(self.root)]
                ):
                    with self.assertRaises(SystemExit) as exit:
                        RUNNER.main()
            self.assertEqual(exit.exception.code, 143)
            self.assertIs(signal.getsignal(signal.SIGTERM), original_handler)
            self.assertIsNotNone(children[0].poll())
        finally:
            for child in children:
                if child.poll() is None:
                    child.kill()
                    child.wait(timeout=5)

    def test_sigterm_during_cleanup_does_not_interrupt_group_kill(self):
        completed = []

        def kill(*args):
            os.kill(os.getpid(), signal.SIGTERM)
            completed.append(True)

        with patch.object(RUNNER.subprocess, "Popen") as popen:
            popen.return_value.wait.return_value = 1
            with patch.object(RUNNER.os, "killpg", side_effect=kill):
                with patch.object(RUNNER.os, "waitid", return_value=object()):
                    with patch.object(
                        sys, "argv", [str(SCRIPT), "--output-dir", str(self.root)]
                    ):
                        with self.assertRaises(SystemExit) as exit:
                            RUNNER.main()
            popen.return_value.wait.assert_called_once_with(timeout=5)
        self.assertEqual(exit.exception.code, 143)
        self.assertEqual(completed, [True])

    def test_reap_timeout_does_not_inherit_callers_exception(self):
        with patch.object(RUNNER.subprocess, "Popen") as popen:
            popen.return_value.wait.side_effect = subprocess.TimeoutExpired("test", 5)
            with patch.object(RUNNER.os, "killpg"):
                with patch.object(RUNNER.os, "waitid", return_value=object()):
                    try:
                        raise ValueError("caller exception")
                    except ValueError:
                        self.assertFalse(
                            RUNNER.run_tests(self.root / "stuck", self.root, 1)
                        )

    def test_main_sets_and_restores_sigchld(self):
        original = signal.signal(signal.SIGCHLD, signal.SIG_IGN)

        def run(*args):
            self.assertEqual(signal.getsignal(signal.SIGCHLD), signal.SIG_DFL)
            return True

        try:
            with patch.object(RUNNER, "run_tests", side_effect=run):
                with patch.object(
                    sys, "argv", [str(SCRIPT), "--output-dir", str(self.root)]
                ):
                    self.assertEqual(RUNNER.main(), 0)
            self.assertEqual(signal.getsignal(signal.SIGCHLD), signal.SIG_IGN)
        finally:
            signal.signal(signal.SIGCHLD, original)

    def test_sigint_during_launch_cleans_up(self):
        binary = self.make_binary("import time\ntime.sleep(30)\n")
        test_dir = self.root / "test"
        test_dir.mkdir()
        (test_dir / "cucascade_tests").symlink_to(binary)
        original_popen = subprocess.Popen
        original_handler = signal.signal(signal.SIGINT, signal.default_int_handler)
        children = []

        def launch(*args, **kwargs):
            child = original_popen(*args, **kwargs)
            children.append(child)
            os.kill(os.getpid(), signal.SIGINT)
            return child

        try:
            with patch.object(RUNNER.subprocess, "Popen", side_effect=launch):
                with patch.object(
                    sys, "argv", [str(SCRIPT), "--build-dir", str(self.root)]
                ):
                    with self.assertRaises(KeyboardInterrupt):
                        RUNNER.main()
            self.assertIsNotNone(children[0].poll())
        finally:
            signal.signal(signal.SIGINT, original_handler)
            for child in children:
                if child.poll() is None:
                    child.kill()
                    child.wait(timeout=5)

    def test_main_runs_every_binary_after_a_failure(self):
        with patch.object(RUNNER, "run_tests", side_effect=[False, True, True]) as run:
            with patch.object(
                sys, "argv", [str(SCRIPT), "--output-dir", str(self.root)]
            ):
                self.assertEqual(RUNNER.main(), 1)
        self.assertEqual(
            [call.args[0].name for call in run.call_args_list],
            list(RUNNER.TEST_BINARIES),
        )

    def test_missing_executable(self):
        self.assertFalse(RUNNER.run_tests(self.root / "missing", self.root, 1))


if __name__ == "__main__":
    unittest.main()
