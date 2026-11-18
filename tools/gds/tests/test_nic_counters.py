# SPDX-License-Identifier: Apache-2.0
import importlib.util
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "nic_counters", Path(__file__).resolve().parents[1] / "nic_counters.py"
)
counters = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(counters)


class NicCounterTests(unittest.TestCase):
    def test_empty_sysfs(self):
        with tempfile.TemporaryDirectory() as root:
            result = counters.snapshot(Path(root))
        self.assertEqual(result["counters"], {})
        self.assertIn("no readable RDMA port counters", result["unavailable"])

    def test_read_only_snapshot_and_exact_delta(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "class/infiniband/mlx5_0/ports/1/counters"
            path.mkdir(parents=True)
            rx, tx = path / "port_rcv_data", path / "port_xmit_data"
            rx.write_text("100\n")
            tx.write_text("200\n")
            before = counters.snapshot(Path(root))
            rx.write_text("110\n")
            tx.write_text("201\n")
            after = counters.snapshot(Path(root))
            self.assertEqual(rx.read_text(), "110\n")
            self.assertEqual(tx.read_text(), "201\n")
            result = counters.delta(before, after)
            self.assertEqual(result["delta_bytes"]["mlx5_0/1/port_rcv_data"], 40)
            self.assertEqual(result["delta_bytes"]["mlx5_0/1/port_xmit_data"], 4)
            rx.write_text("unavailable")
            self.assertTrue(counters.snapshot(Path(root))["unavailable"])

    def test_reset_and_new_counter_are_not_negative_deltas(self):
        result = counters.delta(
            {"counters": {"reset": 10}, "unavailable": []},
            {"counters": {"reset": 1, "new": 100}, "unavailable": []},
        )
        self.assertEqual(result["delta_bytes"], {"reset": None, "new": None})


if __name__ == "__main__":
    unittest.main()
