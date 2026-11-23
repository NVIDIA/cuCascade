# libcufile configuration examples (profile v1)

These are **commented JSON templates**, not qualification results. Filenames
version the example and minimum target release; no invented schema-version key
is passed to libcufile. Keys come from NVIDIA's
[configuration guide](https://docs.nvidia.com/gpudirect-storage/configuration-guide/index.html)
and the 1.17 parameter declarations in `cufile.h`.
A newer library/client can change applicability. Comments marked **[verify]**
require installed-release and target-host validation, not guessed additional keys.

| Example | Starting preference of the superseded async engine (not configured by this JSON) |
|---|---|
| `nvme-libcufile-1.17-v1.json` | QD 32, 4 MiB, 40 lanes |
| `gpfs-libcufile-1.17-v1.json` | QD 64, validated GPFS block clamped to 1–8 MiB, 80 lanes |
| `weka-libcufile-1.17-v1.json` | QD 64, 2 MiB, 80 lanes |
| `vast-libcufile-1.17-v1.json` | QD 64, 1 MiB, 80 lanes |
| `lustre-libcufile-1.17-v1.json` | QD 64, stripe size up to 8 MiB (1 MiB fallback), 80 lanes |

Strip **full-line** `//` comments and validate before use; do not assume every
libcufile release accepts comments. This is also how the dedicated pre-commit
validator checks these files (the strict JSON hooks exclude only this directory).

```sh
python - <<'PY'
import json
from pathlib import Path
p = Path("docs/io/cufile-profiles/nvme-libcufile-1.17-v1.json")
data = json.loads("\n".join(x for x in p.read_text().splitlines()
                            if not x.lstrip().startswith("//")))
Path("/tmp/cufile-nvme.json").write_text(json.dumps(data, indent=2) + "\n")
PY
# Set BEFORE any cuFile driver initialization, in a new process:
export CUFILE_ENV_PATH_JSON=/tmp/cufile-nvme.json
```

In `properties`, sizes are **KiB**. `allow_compat_mode=true` permits a fallback;
it neither forces compat nor proves GDS. The JSON's direct-I/O maximum is a
libcufile limit, independent of engine `op_bytes` / available arena classes.
Pinned limits are ceilings, not reservations. Review other applications'
BAR1 usage before copying the examples.

RDMA addresses/policy have **no typed setter IDs in the installed 1.17 header**.
Use the documented JSON mechanism and validate with vendor tooling instead.
The templates deliberately omit client-specific routing/write-enable keys and
GPU slab JSON names whose spelling/support is not established here.
They do not configure the cuCascade backend; see [cufile-gds.md](../cufile-gds.md).
