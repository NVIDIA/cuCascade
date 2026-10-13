#!/usr/bin/env python3
"""Loader-only data-symbol parity; no resource, CUDA or extension entrypoint calls."""
import ctypes
import json
import os
import sys

arm = sys.argv[1]
assert arm in ("baseline", "candidate")
assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
prefix = "/home/kkristensen/Code/sirius/.pixi/envs/default"
extension_path = "/tmp/cucascade-pinned-pr-20261012/sirius-build-corrected/" + arm + "/sirius.duckdb_extension"
libcudf = ctypes.CDLL(prefix + "/lib/libcudf.so", mode=os.RTLD_NOW | os.RTLD_GLOBAL)
extension = ctypes.CDLL(extension_path, mode=os.RTLD_NOW | os.RTLD_LOCAL)
libdl = ctypes.CDLL("libdl.so.2")
libdl.dlsym.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
libdl.dlsym.restype = ctypes.c_void_p
names = [
    "_ZZN3rmm9_RMM_26_82mr6detail11get_ref_mapEvE21device_id_to_resource",
    "_ZGVZN3rmm9_RMM_26_82mr6detail11get_ref_mapEvE21device_id_to_resource",
    "_ZZN3rmm9_RMM_26_82mr6detail12ref_map_lockEvE12ref_map_lock",
]
result = {"arm": arm, "CUDA_VISIBLE_DEVICES": "", "method": "dlopen+dlsym only", "symbols": {}}
for name in names:
    library_address = libdl.dlsym(libcudf._handle, name.encode())
    extension_address = libdl.dlsym(extension._handle, name.encode())
    assert library_address and library_address == extension_address
    result["symbols"][name] = {"libcudf": hex(library_address), "extension": hex(extension_address), "equal": True}
result["verdict"] = "PASS"
print(json.dumps(result, indent=2))
