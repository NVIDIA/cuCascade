# SPDX-License-Identifier: Apache-2.0
import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "make_cufile_json", Path(__file__).resolve().parents[1] / "make_cufile_json.py"
)
generator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = generator
SPEC.loader.exec_module(generator)


class MakeCuFileJsonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.out = self.root / "selected gpu.json"
        self.gpu0 = self.pci("0000:11:00.0", 0, "0000:10:00.0")
        self.gpu1 = self.pci("0000:81:00.0", 1, "0000:80:00.0")

    def put(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)

    def pci(self, name, node, bridge):
        path = self.root / "devices/pci0000:00" / bridge / name
        self.put(path / "numa_node", str(node))
        link = self.root / "bus/pci/devices" / name
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(path, target_is_directory=True)
        return path

    def nic(self, name, pci, node, bridge, addr):
        device = self.pci(pci, node, bridge)
        ib = self.root / "class/infiniband" / name
        ib.mkdir(parents=True)
        (ib / "device").symlink_to(device, target_is_directory=True)
        net = "eth-" + name
        (device / "net" / net).mkdir(parents=True)
        (self.root / "class/net" / net).mkdir(parents=True)
        self.put(ib / "ports/1/gid_attrs/ndevs/0", net)
        self.put(ib / "ports/1/gids/0", addr)
        return ib

    def run_main(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = generator.main(
                [
                    "--gpu",
                    "0000:11:00.0",
                    "--sysfs-root",
                    str(self.root),
                    "--out",
                    str(self.out),
                    *args,
                ]
            )
        return code, out.getvalue(), err.getvalue()

    def test_shared_bridge_beats_numa_and_preserves_base(self):
        self.nic("mlx5_0", "0000:12:00.0", 1, "0000:10:00.0", "::ffff:10.0.0.1")
        self.nic("mlx5_1", "0000:22:00.0", 0, "0000:20:00.0", "2001:db8::2")
        base = self.root / "base.json"
        base.write_text(
            '{// comment\n"url":"https://example//x", "properties":{"other":1}, "fs":{}}'
        )
        code, out, err = self.run_main("--base", str(base))
        self.assertEqual((code, err), (0, ""))
        value = json.loads(self.out.read_text())
        self.assertEqual(value["properties"]["rdma_dev_addr_list"], ["10.0.0.1"])
        self.assertEqual(
            value["properties"]["rdma_load_balancing_policy"], "RoundRobin"
        )
        self.assertEqual(value["properties"]["other"], 1)
        self.assertEqual(value["url"], "https://example//x")
        self.assertEqual(value["fs"], {})
        self.assertIn("shared upstream PCIe bridge", out)
        self.assertIn(f"export CUFILE_ENV_PATH_JSON='{self.out}'", out)
        self.assertIn("[verify]", out)
        for hint in ("verbsPorts", "net=", "localports="):
            self.assertIn(hint, out)

    def test_same_numa_ipv6_excluded_by_default_and_explicitly_opted_in(self):
        self.nic("mlx5_1", "0000:22:00.0", 0, "0000:20:00.0", "2001:db8::2")
        self.nic("mlx5_2", "0000:32:00.0", 0, "0000:30:00.0", "::ffff:10.0.0.3")
        self.nic("mlx5_3", "0000:82:00.0", 1, "0000:80:00.0", "::ffff:10.0.0.4")
        code, out, _ = self.run_main()
        self.assertEqual(code, 0)
        self.assertIn("IPv6 candidates excluded", out)
        self.assertEqual(
            json.loads(self.out.read_text())["properties"]["rdma_dev_addr_list"],
            ["10.0.0.3"],
        )
        self.assertEqual(self.run_main("--allow-ipv6")[0], 0)
        self.assertEqual(
            json.loads(self.out.read_text())["properties"]["rdma_dev_addr_list"],
            ["10.0.0.3", "2001:db8::2"],
        )

    def test_multi_gpu_union_is_deduplicated(self):
        self.nic("mlx5_0", "0000:12:00.0", 0, "0000:10:00.0", "::ffff:10.0.0.1")
        self.nic("mlx5_1", "0000:82:00.0", 1, "0000:80:00.0", "::ffff:10.0.0.2")
        code, _, _ = self.run_main(
            "--gpu",
            "0000:81:00.0",
            "--gpu",
            "0000:11:00.0",
            "--policy",
            "RoundRobinMaxMin",
        )
        self.assertEqual(code, 0)
        props = json.loads(self.out.read_text())["properties"]
        self.assertEqual(props["rdma_dev_addr_list"], ["10.0.0.1", "10.0.0.2"])
        self.assertEqual(props["rdma_load_balancing_policy"], "RoundRobinMaxMin")

    def test_no_nics_is_graceful_and_does_not_overwrite(self):
        self.out.write_text("keep")
        code, out, err = self.run_main()
        self.assertEqual((code, err), (0, ""))
        self.assertIn("no JSON written", out)
        self.assertNotIn("export CUFILE_ENV_PATH_JSON=", out)
        self.assertEqual(self.out.read_text(), "keep")

    def test_no_addresses_or_link_local_does_not_invent_ip(self):
        self.nic("mlx5_0", "0000:12:00.0", 0, "0000:10:00.0", "fe80::1")
        code, out, _ = self.run_main()
        self.assertEqual(code, 0)
        self.assertIn("(no usable IP)", out)
        self.assertFalse(self.out.exists())

    def test_broken_symlink_is_skipped(self):
        broken = self.root / "class/infiniband/broken"
        broken.mkdir(parents=True)
        (broken / "device").symlink_to(self.root / "gone")
        self.assertEqual(self.run_main()[0], 0)

    def test_unknown_gpu_fails(self):
        code, _, err = self.run_main("--gpu", "99")
        self.assertEqual(code, 2)
        self.assertIn("use its PCI BDF", err)

    def test_numeric_gpu_uses_nvidia_inventory_not_sysfs_order(self):
        with patch.object(generator, "command", return_value="3, 00000000:11:00.0\n"):
            gpus = generator.discover_gpus(["3"], self.root, True)
        self.assertEqual(gpus[0].bdf, "0000:11:00.0")
        self.assertEqual(gpus[0].index, "3")

    def test_ip_and_rdma_fallback(self):
        ib = self.nic("mlx5_0", "0000:12:00.0", 0, "0000:10:00.0", "::")
        (ib / "ports/1/gid_attrs/ndevs/0").unlink()

        def run(args):
            if args[0] == "ip":
                return json.dumps(
                    [{"ifname": "eth-mlx5_0", "addr_info": [{"local": "10.2.3.4"}]}]
                )
            return json.dumps([{"ifname": "mlx5_0/1", "netdev": "eth-mlx5_0"}])

        with patch.object(generator, "command", side_effect=run):
            nics = generator.discover_nics(self.root, True)
        self.assertEqual(nics[0].addresses, ("10.2.3.4",))

    def test_topo_matrix_fallback(self):
        topo = self.root / "topo.txt"
        topo.write_text(
            "    GPU0 GPU1 NIC0 CPU Affinity NUMA Affinity\nGPU0 X SYS PIX 0-7 0\n"
            "GPU1 SYS X SYS 8-15 1\nNIC Legend:\n NIC0: mlx5_0\n"
        )
        self.assertEqual(generator.topo_links(topo)[("0", "mlx5_0")], "PIX")
        gpu = generator.Device("0", "0000:11:00.0", -1, self.gpu0, index="0")
        nic = generator.Device("mlx5_0", "0000:22:00.0", -1, self.root / "other")
        score, reason = generator.rank(gpu, nic, generator.topo_links(topo))
        self.assertEqual(score, (0, 0))
        self.assertIn("[verify]", reason)

    def test_two_level_switch_prefers_nearest_common_bridge(self):
        root = "0000:a0:01.0/0000:a1:00.0/"
        leaf = root + "0000:a2:00.0"
        sibling = root + "0000:a2:01.0"
        (self.root / "bus/pci/devices/0000:11:00.0").unlink()
        self.pci("0000:11:00.0", 0, leaf)
        self.nic("mlx5_0", "0000:a4:00.0", 0, leaf, "::ffff:10.0.0.1")
        self.nic("mlx5_1", "0000:a5:00.0", 0, sibling, "::ffff:10.0.0.2")
        self.assertEqual(self.run_main()[0], 0)
        self.assertEqual(
            json.loads(self.out.read_text())["properties"]["rdma_dev_addr_list"],
            ["10.0.0.1"],
        )

    def test_topo_pix_beats_pxb_and_strips_ansi(self):
        topo = self.root / "topo.txt"
        topo.write_text(
            "\x1b[4mGPU0 NIC0 NIC1\x1b[0m\nGPU0 X PXB PIX\n"
            "NIC Legend:\n NIC0: mlx5_0\n NIC1: mlx5_1\n"
        )
        links = generator.topo_links(topo)
        gpu = generator.Device("0", "0000:11:00.0", 0, self.gpu0, index="0")
        nics = [
            generator.Device(
                f"mlx5_{i}",
                f"0000:{i+32:02x}:00.0",
                0,
                self.root / f"other{i}",
                (f"10.0.0.{i+1}",),
            )
            for i in range(2)
        ]
        self.assertEqual(generator.select([gpu], nics, links)[0], ["10.0.0.2"])

    def test_base_policy_survives_and_explicit_override_works(self):
        self.nic("mlx5_0", "0000:12:00.0", 0, "0000:10:00.0", "::ffff:10.0.0.1")
        base = self.root / "base.json"
        base.write_text(
            json.dumps(
                {"properties": {"rdma_load_balancing_policy": "RoundRobinMaxMin"}}
            )
        )
        self.assertEqual(self.run_main("--base", str(base))[0], 0)
        self.assertEqual(
            json.loads(self.out.read_text())["properties"][
                "rdma_load_balancing_policy"
            ],
            "RoundRobinMaxMin",
        )
        self.assertEqual(
            self.run_main("--base", str(base), "--policy", "RoundRobin")[0], 0
        )
        self.assertEqual(
            json.loads(self.out.read_text())["properties"][
                "rdma_load_balancing_policy"
            ],
            "RoundRobin",
        )
        with self.assertRaises(SystemExit):
            self.run_main("--policy", "MaxMinFit")

    def test_filesystem_lists_and_dynamic_routing_warn_without_modification(self):
        self.nic("mlx5_0", "0000:12:00.0", 0, "0000:10:00.0", "::ffff:10.0.0.1")
        base = self.root / "base.json"
        fs = {
            "nfs": {"rdma_dev_addr_list": ["10.8.0.1"], "mount_table": []},
            "lustre": {"mount_table": [{"rdma_dev_addr_list": ["10.9.0.1"]}]},
        }
        base.write_text(
            json.dumps({"fs": fs, "properties": {"rdma_dynamic_routing": True}})
        )
        code, out, _ = self.run_main("--base", str(base))
        self.assertEqual(code, 0)
        for key in (
            "fs.nfs.rdma_dev_addr_list",
            "fs.nfs.mount_table",
            "fs.lustre.mount_table",
            "properties.rdma_dynamic_routing",
        ):
            self.assertIn("WARNING: base " + key, out)
        self.assertEqual(json.loads(self.out.read_text())["fs"], fs)

    def test_refuses_overwriting_base_and_atomic_replace_failure_preserves_output(self):
        self.out.write_text("{}")
        self.assertEqual(self.run_main("--base", str(self.out))[0], 2)
        self.nic("mlx5_0", "0000:12:00.0", 0, "0000:10:00.0", "::ffff:10.0.0.1")
        before = set(self.root.iterdir())
        with patch.object(
            generator.os, "replace", side_effect=OSError("injected failure")
        ):
            self.assertEqual(self.run_main()[0], 2)
        self.assertEqual(self.out.read_text(), "{}")
        self.assertEqual(set(self.root.iterdir()), before)

    def test_auxiliary_device_is_reported_as_skipped(self):
        device = self.root / "devices/mlx5_core.sf.1"
        device.mkdir(parents=True)
        ib = self.root / "class/infiniband/mlx5_aux"
        ib.mkdir(parents=True)
        (ib / "device").symlink_to(device, target_is_directory=True)
        self.assertIn("Skipped RDMA device mlx5_aux", self.run_main()[1])

    def test_comments_preserve_escaped_quotes(self):
        data = '{"text":"escaped \\" // inside", // outside\n"number":3}'
        self.assertEqual(json.loads(generator.uncomment(data))["number"], 3)

    def test_bad_base_fails_without_output(self):
        base = self.root / "bad.json"
        base.write_text('{"properties": []}')
        self.assertEqual(self.run_main("--base", str(base))[0], 2)
        self.assertFalse(self.out.exists())


if __name__ == "__main__":
    unittest.main()
