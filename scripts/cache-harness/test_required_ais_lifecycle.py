#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Whole H entrypoint lifecycle: real ownership helpers, isolated harmless services."""
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest

SCRIPTS = Path(__file__).parent.resolve()

STUB = """from pathlib import Path
import json,os,subprocess,sys,time
import harness
process = None
root = Path(os.environ["H_TEST_ROOT"])
phase = os.environ["H_TEST_PHASE"]

def pause(stage):
    (root / "stage").write_text(stage)
    end = time.monotonic() + 8
    while not (root / "release").exists():
        if time.monotonic() > end:
            raise TimeoutError("test release gate")
        time.sleep(.01)

def up(args):
    global process
    run = args.run_dir
    run.mkdir()
    state = {"nodes": {}}
    harness.save(run / "ais-state.json", state)
    if phase == "before_launch":
        pause(phase)
    def persist(child):
        (root / "child.pid").write_text(str(child.pid))
        end = time.monotonic() + 3
        while not (root / "port").exists():
            if time.monotonic() > end: raise TimeoutError("service readiness")
            time.sleep(.01)
        if phase == "before_persist":
            pause(phase)
        state["nodes"]["service"] = {"pid": child.pid, "start": harness.process_identity(child.pid)}
        harness.save(run / "ais-state.json", state)
    process = harness.launch_owned([sys.executable, "-B", str(root / "service.py")], persist, start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if phase == "after_persist":
        pause(phase)

def qualify(args):
    harness.save(args.run_dir / "summary.json", {"status": "PASS"})

@harness.protect_cleanup
def down(run):
    (root / "cleaning").write_text("entered")
    time.sleep(.3)
    state = harness.load(run / "ais-state.json")
    for node in state["nodes"].values():
        harness.stop_owned(node["pid"], node["start"], str(root / "service.py"), term_seconds=.5, kill_seconds=2)
    if process is not None:
        process.wait(timeout=3)
    (root / "cleaned").write_text("finished")
"""


@unittest.skipUnless(
    sys.platform == "linux", "Linux real pidfd ownership helpers required"
)
class WholeEntrypoint(unittest.TestCase):
    def run_case(self, phase, trigger):
        with tempfile.TemporaryDirectory(prefix="h-entry-") as temporary:
            root = Path(temporary)
            for name in ("required_ais.py", "required_fixture.py", "harness.py"):
                shutil.copyfile(SCRIPTS / name, root / name)
            (root / "ais_harness.py").write_text(STUB)
            (root / "service.py").write_text(
                'import socket,time,os,signal\nfrom pathlib import Path\nr=Path(os.environ["H_TEST_ROOT"])\ns=socket.socket();s.bind(("127.0.0.1",0));s.listen(1)\n(r/"port").write_text(str(s.getsockname()[1]))\nif os.environ.get("H_TEST_IGNORE_TERM"): signal.signal(signal.SIGTERM,signal.SIG_IGN)\ntime.sleep(60)\n'
            )
            # Keep actual main/parser/signal/up/down flow; only LIST business work
            # is replaced in this lifecycle fixture. Real AIS is checked separately.
            (root / "entry.py").write_text(
                "import required_ais as h\nh.observe_list=lambda *a: None\nraise SystemExit(h.main())\n"
            )
            env = dict(
                os.environ,
                H_TEST_ROOT=str(root),
                H_TEST_PHASE=phase,
                PYTHONDONTWRITEBYTECODE="1",
            )
            if trigger == "escalation":
                env["H_TEST_IGNORE_TERM"] = "1"
            command = [
                sys.executable,
                "-B",
                str(root / "entry.py"),
                "--run-dir",
                str(root / "run"),
                "--probe",
                "unused",
                "--mode",
                "root",
                "--ais-source",
                "unused",
                "--aisnode",
                "unused",
                "--aisnode-sha256",
                "unused",
                "--weed",
                "unused",
                "--binary",
                "unused",
                "--deadline",
                "1" if trigger == "deadline" else "20",
            ]
            with (root / "out.log").open("w") as log:
                parent = subprocess.Popen(command, env=env, stdout=log, stderr=log)
                child = None
                child_fd = None
                try:
                    if phase != "normal":
                        limit = time.monotonic() + 5
                        while (
                            not (root / "stage").exists()
                            and parent.poll() is None
                            and time.monotonic() < limit
                        ):
                            time.sleep(0.01)
                        self.assertTrue(
                            (root / "stage").exists(), (root / "out.log").read_text()
                        )
                    if phase == "normal":
                        limit = time.monotonic() + 3
                        while (
                            not (root / "child.pid").exists()
                            and parent.poll() is None
                            and time.monotonic() < limit
                        ):
                            time.sleep(0.01)
                    if (root / "child.pid").exists():
                        child = int((root / "child.pid").read_text())
                        try:
                            child_fd = os.pidfd_open(child)
                        except ProcessLookupError:
                            pass
                        else:
                            command_line = (
                                Path(f"/proc/{child}/cmdline").read_bytes().split(b"\0")
                            )
                            self.assertIn(
                                str(root / "service.py").encode(), command_line
                            )
                    if trigger == "deadline":
                        time.sleep(1.2)
                    elif trigger != "normal":
                        sig = signal.SIGINT if "INT" in trigger else signal.SIGTERM
                        parent.send_signal(sig)
                        if "repeat" in trigger:
                            time.sleep(0.03)
                            if parent.poll() is None:
                                parent.send_signal(sig)
                    (root / "release").touch()
                    if "repeat" in trigger:
                        limit = time.monotonic() + 4
                        while (
                            not (root / "cleaning").exists()
                            and parent.poll() is None
                            and time.monotonic() < limit
                        ):
                            time.sleep(0.01)
                        if parent.poll() is None:
                            parent.send_signal(signal.SIGTERM)
                    code = parent.wait(timeout=12)
                    self.assertEqual(
                        code == 0, trigger == "normal", (root / "out.log").read_text()
                    )
                    self.assertTrue(
                        (root / "cleaned").exists(), (root / "out.log").read_text()
                    )
                    if (root / "child.pid").exists():
                        child = int((root / "child.pid").read_text())
                        with self.assertRaises(ProcessLookupError):
                            os.kill(child, 0)
                        with socket.socket() as sock:
                            sock.bind(("127.0.0.1", int((root / "port").read_text())))
                    print(
                        f"{phase}/{trigger}: exit={code}, no owned service/port remains",
                        flush=True,
                    )
                finally:
                    if parent.poll() is None:
                        parent.kill()
                        parent.wait()
                    # Fixture rescue only, never the pass condition. Use the
                    # pidfd if the failed test left its own known child alive.
                    if child_fd is not None:
                        try:
                            signal.pidfd_send_signal(child_fd, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        finally:
                            os.close(child_fd)

    def test_whole_entrypoint(self):
        for phase in ("before_launch", "before_persist", "after_persist"):
            for trigger in (
                "SIGINT",
                "SIGTERM",
                "repeat-SIGINT",
                "repeat-SIGTERM",
                "deadline",
            ):
                with self.subTest(phase=phase, trigger=trigger):
                    self.run_case(phase, trigger)
        self.run_case("normal", "normal")
        self.run_case("after_persist", "escalation")


if __name__ == "__main__":
    unittest.main(verbosity=2)
