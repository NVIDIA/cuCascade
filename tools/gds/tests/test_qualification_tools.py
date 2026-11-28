# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]


class QualificationToolsTest(unittest.TestCase):
    def test_sanitizer_presets_preserve_environment(self):
        presets = json.loads((ROOT / "CMakePresets.json").read_text())[
            "configurePresets"
        ]
        for name in ("asan", "tsan"):
            flags = next(p["cacheVariables"] for p in presets if p["name"] == name)
            self.assertIn("$env{CXXFLAGS}", flags["CMAKE_CXX_FLAGS"])
            self.assertIn("$env{LDFLAGS}", flags["CMAKE_EXE_LINKER_FLAGS"])
            self.assertIn("$env{LDFLAGS}", flags["CMAKE_SHARED_LINKER_FLAGS"])

    def run_runner(self, dry=False, mount_override=None):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        path = Path(temp.name)
        mount = subprocess.check_output(
            ["findmnt", "-n", "-o", "TARGET", "-T", str(path)], text=True
        ).strip()
        conf = path / "config.conf"
        conf.write_text(
            f'MOUNTS=("test|compat|{mount_override or mount}|{path}/scratch")\n'
            f"RESULTS_ROOT={path}/results\nQD_BIN=/bin/false\n"
            "ENGINE_BIN=/bin/false\nGDSCHECK_BIN=/bin/false\n"
        )
        args = [
            "bash",
            "tools/gds/qualify.sh",
            "--config",
            str(conf),
            "--compat-smoke",
            "--only",
            "B-1",
        ]
        if dry:
            args.append("--dry-run")
        result = subprocess.run(
            args, cwd=ROOT, text=True, capture_output=True, timeout=30
        )
        return path, result

    def test_dry_run_is_non_mutating(self):
        path, result = self.run_runner(dry=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse((path / "results").exists())
        self.assertFalse((path / "scratch").exists())

    def test_failed_case_propagates_exit_status(self):
        path, result = self.run_runner()
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(list((path / "results").glob("*/failures.log")))

    def test_wrong_mount_rejected_before_writes(self):
        path, result = self.run_runner(mount_override="/not-the-mount")
        self.assertEqual(result.returncode, 2)
        self.assertFalse((path / "results").exists())


if __name__ == "__main__":
    unittest.main()
