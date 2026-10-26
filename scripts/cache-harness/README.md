# Local direct S3 correctness harness

This opt-in harness starts a private SeaweedFS service and checks **cuCascade public
REST I/O**, using the built-in header SigV4 authorizer. It requires no AWS account.
Python/boto3 only seeds data and runs separately labelled protocol probes. It never
stands in for the C++ reads. This is host-byte correctness, not a benchmark or GPU,
GDS, RDMA, SQL, or AIStore qualification.

## Build and dependencies

Linux, Python 3.10+, boto3/botocore (validated with boto3 1.40.72), and SeaweedFS
4.48 are required. Supply an existing `weed` path; the runner does not install
packages or alter system services. The default SHA256 pins the tested conda-forge
linux-64 binary, `a299c1528d7b9838828d1b6147c98e4b38b2ce1ed3f683e7ec7850b2c29f7ca9`.
For another platform/build, verify its provenance and pass `--weed-sha256` explicitly;
a changed binary is a different service baseline.

From the cuCascade checkout, use a previously sealed `pixi.lock` and an installed
`cuda-13-stable` environment. The tested base is main
`1a9281093a0fe49634594d541d7ce327129f53d6` and lock SHA256
`58a8ee1b06c805938fc2942793a7bb1066d2a6addb87b839a4c7cad255110779`.
The lock is an external experiment receipt, not shipped by this source change;
retain and distribute it with run results. Resolving a new lock must be recorded as
a new dependency baseline.

```sh
pixi install --locked -e cuda-13-stable
pixi run --locked -e cuda-13-stable cmake --preset release -B build/p0-io-cudf \
  -DCMAKE_CUDA_ARCHITECTURES=86 -DCUCASCADE_BUILD_CUDF=ON \
  -DCUCASCADE_BUILD_IO=ON -DCUCASCADE_BUILD_TESTS=ON \
  -DCUCASCADE_BUILD_BENCHMARKS=OFF -DCUCASCADE_BUILD_S3_BENCHMARK=OFF \
  -DCUCASCADE_BUILD_S3_HARNESS=ON
pixi run --locked -e cuda-13-stable cmake --build build/p0-io-cudf \
  --target cucascade_direct_s3_smoke cucascade_io_tests -j2
```

Use the GPU architecture of your machine. cuDF ON is the accepted working
configuration: two existing registry tests on this base assume kvikIO is registered
and fail with cuDF OFF. This harness does not repair or skip those tests. The new
executable is not in default CTest because it requires a seeded external service.

## Lifecycle

All arguments are paths/ports, not a remote endpoint. The service binds loopback.
The eight consecutive ports starting at `--port-base` must all be free. A conflict
exits without stopping the occupant. All data lives in a new, private run directory;
no existing run is overwritten. Optional Iceberg/Lance services and telemetry are off.

```sh
python3 scripts/cache-harness/harness.py preflight \
  --run-dir /tmp/cucascade-run-1 --weed /path/to/weed
python3 scripts/cache-harness/harness.py all \
  --run-dir /tmp/cucascade-run-1 --weed /path/to/weed --port-base 19330
python3 scripts/cache-harness/harness.py report --run-dir /tmp/cucascade-run-1
python3 scripts/cache-harness/harness.py down --run-dir /tmp/cucascade-run-1
```

`all` runs `up`, `seed`, `test-direct`, and finally `down`/`report`, including on
ordinary failures and keyboard interruption. Each stage can also be run manually;
a manual `up` deliberately leaves the service running until `down`. `down` is
idempotent, verifies the saved Linux PID/start-time and command identity, and never
uses `pkill` or removes data. SIGKILL/power loss cannot execute a finally block:
use `down` against the saved run after reconnecting. Do not edit `state.json`.

Boto3 uses explicit public test credentials, region, endpoint, and empty proxy
configuration; it does not load ambient AWS credentials. C++ requests use those
same local test credentials and bypass proxies for localhost. Never deploy this
configuration on a public interface or replace its values with production keys.
For build downloads, set your existing HTTP proxy in the build shell only.

## What is checked

- Fixtures: 0, 1, 4096, 65553, 1048607 bytes, deterministic data and SHA256 manifests.
- Public HEAD size/ETag, full reads, 64 KiB async chunks, prefix/middle/tail,
  EOF and zero-length semantics, suffix GET metadata/stash bytes.
- Every read checks returned length, exact expected bytes and untouched guard bytes.
  Object/context and destination owners remain alive through future completion.
- Space, plus, percent and repeated slash keys; URI encoding and colliding key data
  are distinct. Reference reads occur after all writes to reveal service aliases.
- Library missing object and invalid secret must specifically report HTTP 404/403.
  An arbitrary exception is a failure, not an expected negative test.
- Separate boto3 probes check matching/stale If-Match and raw unsatisfiable Range.

The public library clips ranges at EOF and returns zero beyond EOF; this is different
from a raw S3 416 response. The ordinary read API does not expose If-Match: ETag
capture is not proof of version-pinned reads. The report explicitly marks that
capability unavailable. Service key aliasing or missing condition support is also
reported as `UNSUPPORTED`, not silently normalized or counted as library success.
A core read failure or incomplete test run returns failure; a fully executed run
with documented limitations reports `PASS_WITH_LIMITATIONS`. Review all limitations
before treating a run as phase acceptance.

## Evidence and repeatability

Keep `state.json` (binary hash/version), `fixtures.json`, `results.json`, per-case
stdout/stderr, `service.log`, `service-command.json`, and `report.json`/`report.md`.
`failure.json` records orchestrator failures. Saved service credentials are public
local-only constants; no real credential material is needed.

Run `all` again with a **new** directory and the same port base after cleanup.
Run the existing regression separately:

```sh
pixi run --locked -e cuda-13-stable ctest --test-dir build/p0-io-cudf \
  --output-on-failure --no-tests=error -R '^cucascade_io_tests$'
```

To exercise failure cleanup, use `--binary /bin/false` with a fresh run. The command
must return nonzero and the report show stopped=true. To exercise port protection,
occupy one configured port with a disposable local listener and verify startup
fails while that listener remains alive. Do not run these checks against someone
else's service. No production transport or dependency manifests are changed.

Validated service caveat: with SeaweedFS 4.48 and boto3 1.40.72, the PUT after
an empty PUT on a reused connection timed out. The seed tool closes/recreates
its reference client after the empty PUT. This workaround is limited to seeding;
the C++ read path is unchanged.
