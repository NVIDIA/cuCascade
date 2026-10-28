# Local S3 and AIStore functional harness

Start isolated SeaweedFS/AIStore services, seed known objects, read them through
cuCascade's public C++ REST I/O, and retain results while stopping owned services.
No AWS account is needed. This tests host-byte correctness and compatibility;
it does not implement cache policy or measure performance.

## Prerequisites and build

Use Linux, Python with boto3 (tested 1.40.72), SeaweedFS 4.48, and the existing
cuCascade `cuda-13-stable` Pixi environment. The tested production base is
`1a9281093a0fe49634594d541d7ce327129f53d6`. Use the separately supplied sealed
`pixi.lock` (SHA256 `58a8ee1b06c805938fc2942793a7bb1066d2a6addb87b839a4c7cad255110779`);
this patch does not ship or resolve a new dependency lock. Clean-machine
installation has not been validated; the recorded runs reuse installed tools.

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

Choose your GPU architecture. The pinned base has two registry tests that fail
with cuDF OFF; this harness does not modify or skip them. The smoke target is
opt-in and is not part of default CTest because it needs a seeded service.

Set `WEED` to an existing binary. The default SHA256 pins the tested conda-forge
linux-64 build: `a299c1528d7b9838828d1b6147c98e4b38b2ce1ed3f683e7ec7850b2c29f7ca9`.
For another verified build use `--weed-sha256`; record it as a different baseline.

## Direct S3

```sh
python3 scripts/cache-harness/harness.py all \
  --run-dir /tmp/cucascade-direct-1 --weed "$WEED" --port-base 19330
python3 scripts/cache-harness/harness.py report --run-dir /tmp/cucascade-direct-1
```

`all` requires a new directory and runs startup, seeding, library checks, cleanup
and reporting. It reserves eight consecutive loopback ports. A conflict never
stops the occupant. Manual commands are `preflight`, `up`, `seed`, `test-direct`,
`report`, and `down`; manual `up` leaves services running until `down`.

Fixtures include empty, single-byte and larger objects, spaces/plus/percent and
repeated-slash keys. SHAKE256 streams keyed by object identity distinguish
aligned blocks; expected bytes and SHA256 manifests are saved. Reference reads
after all writes expose key aliases without attributing them prematurely to
the client or service. Seeding recreates the reference client after an empty PUT
to avoid a known timeout in the pinned SeaweedFS/boto3 combination.

The C++ probe checks HEAD size/tag, full and ranged reads, 64 KiB async chunks,
EOF/zero-length behavior, suffix stash and destination guards. Each async call
is immediately awaited: **this is not concurrent-outstanding-read coverage**.
Separate service probes test If-Match and raw HTTP 416. Library EOF clipping
returns zero; it is a different API contract from raw HTTP 416.

## AIStore

Use a clean AIStore checkout at `a3769a15d9a28ad69ec7f9f0785577b0fa2c7af9` and an
`aisnode` built with `aws mono` tags. The pinned source requires Go 1.27; recorded
runs use an isolated Go 1.27.3 installation. Set `AIS_SOURCE`, `AIS_TOOLS` and
`GO_ROOT` to your own paths, then build with the repository Makefile:

```sh
export PATH="$GO_ROOT/bin:$PATH"
export GOBIN="$AIS_TOOLS/bin" GOPATH="$AIS_TOOLS/gopath" GOCACHE="$AIS_TOOLS/gocache"
export GOTOOLCHAIN=local GOMAXPROCS=2
mkdir -p "$GOBIN"
(cd "$AIS_SOURCE" && make node TAGS=aws GOFLAGS='-trimpath -mod=readonly -p=2')
sha256sum "$GOBIN/aisnode"
```

Set `AIS_NODE` and `AIS_SHA` to that binary and its hash. From cuCascade:

```sh
python3 scripts/cache-harness/ais_harness.py all \
  --run-dir /tmp/cucascade-ais-1 --mode reverse \
  --ais-source "$AIS_SOURCE" --aisnode "$AIS_NODE" --aisnode-sha256 "$AIS_SHA" \
  --weed "$WEED" --binary "$PWD/build/p0-io-cudf/test/cucascade_direct_s3_smoke"
```

The deployment is one proxy and one target, with a local SeaweedFS origin and
public test credentials. No system service or administrator access is needed.
Default origin ports are 19330–19337 and AIStore ports 19430–19435; override with
`--port-base` and `--ais-port-base`. Startup defaults to 75 seconds and the whole
operation to 300 seconds, plus bounded cleanup. Data stays under the run path.

| Mode | Frontend | What it exercises |
| --- | --- | --- |
| `prefix` | `/s3/bucket/key` | Endpoint-path compatibility |
| `root` (default) | `/bucket/key` | Proxy-to-target redirect |
| `reverse` | `/bucket/key`, S3ReverseProxy enabled | Server forwarding; local compatibility only |

Library read correctness and cold-fill are separate rows. Cold-fill requires
presence **404 before / 200 after**, successful C++ reads, and an observed
increase in backend GET count. Already-resident input fails the cold assertion;
unobservable counters are UNKNOWN and fail qualification. A separate untouched
object uses `footer-first`: its initial library operation is the suffix probe,
followed by normal reads. The reference HTTP client never counts as a library test.

Further probes cover resident reads with origin offline, default warm v1 versus
explicit latest-version v2, service Range/If-Match/404, and target outage.
Authentication is disabled and reported NOT_TESTED. Root/reverse compatibility
does not select a production topology or establish multi-target scalability.

## Verdicts and diagnostics

- PASS: the stated assertion held; read and cold-fill assertions stay separate.
- FAIL: incorrect bytes/state, unexpected error, missing fault delivery, crash,
  startup failure, or an outer subprocess timeout; the command fails.
- UNSUPPORTED: an observed capability limitation, such as the pinned ordinary
  read accepting a changed validator. It is not a passing protection.
- UNKNOWN: required evidence could not be observed; qualification fails.
- NOT_TESTED: intentionally unconfigured capability, or cold-fill not attempted
  because its read mode is unsupported. It is not a product incompatibility claim.

Keep per-case stdout/stderr, results, fixture manifests, source/binary hashes,
node configuration/logs, topology, presence and backend-stat snapshots. A run
with explicit unsupported/not-tested rows is PASS_WITH_LIMITATIONS, not full
cache certification. Inspect `results.json`, `summary.json` (AIStore) or
`report.json` (direct), and `error.json`/`failure.json` if execution failed.

## Regression and fault checks

```sh
python3 -B scripts/cache-harness/test_harness.py
python3 scripts/cache-harness/ais_test_lifecycle.py \
  --run-root /tmp/cucascade-faults-1 --ais-source "$AIS_SOURCE" \
  --aisnode "$AIS_NODE" --aisnode-sha256 "$AIS_SHA" --weed "$WEED" \
  --binary "$PWD/build/p0-io-cudf/test/cucascade_direct_s3_smoke"
pixi run --locked -e cuda-13-stable ctest --test-dir build/p0-io-cudf \
  --output-on-failure --no-tests=error --timeout 300 -R '^cucascade_io_tests$'
```

The unit controls reject unrelated crashes, warm objects and missing counters,
and exercise real child cleanup before PID persistence. The HTTP suite records
requests/fault delivery and requires the expected C++ stage/error: 307, timeout,
truncation and malformed Content-Range must be rejected. Returning the first
block for a later range must fail byte comparison. Positive controls check normal
reads and suffix-first ordering. Changed-validator acceptance remains an explicit
limitation. Lifecycle checks cover port collisions, seed/probe failure, deadlines,
SIGTERM, repeated cleanup and process ownership.

`down --run-dir PATH` preserves data/logs and only stops matching owned PIDs.
Launch interruption is deferred through identity persistence; a failed save uses
the retained child handle for cleanup. SIGKILL/power loss cannot run cleanup;
inspect the saved run and use `down` after reconnecting. Do not edit state files
or concurrently operate on the same run. Never use this local-only configuration
with production credentials or on a public interface.

Not qualified: concurrent cancellation/connection retention, AIS LIST,
production authentication, arbitrary-key transparency, fresh-machine bootstrap,
GPU transfer/GDS/RDMA, memory/FD leak freedom, or query/performance gains.

## Reliability controls

Standalone `report` exits nonzero for failed, incomplete, empty, malformed or
unknown results. Characterization may report `PASS_WITH_LIMITATIONS` for recorded
unsupported/not-tested capabilities; this is not a required-feature qualification.
Cleanup requires Linux Python `os.pidfd_open` and `signal.pidfd_send_signal`. It
opens a process handle before checking start time and run-specific command
arguments and never falls back to signaling a numeric PID. Repeated INT/TERM/ALRM
are ignored only during bounded cleanup. This protects against stale state and
PID reuse; it is not a security boundary against an attacker running as this user.
Lifecycle accepts `--weed-sha256` and forwards it to every service launch.
Run `python scripts/cache-harness/test_reliability.py` on Linux in addition to
`test_harness.py` and the full `ais_test_lifecycle.py` suite.

## H: required-feature profile (fixed old cuCascade baseline)

H adds test tools, not an external-cache implementation. Its production baseline
is `1a9281093a0fe49634594d541d7ce327129f53d6`; the E1 harness fixes remain in
place. `required_features.py` is independent of the original characterization
report. All **17 mandatory rows** and all retained green controls must PASS;
missing, duplicate, malformed, changed evidence, timeout, UNSUPPORTED, UNKNOWN
or NOT_TESTED never qualify. It prints FAIL and exits **1** on the old library.

| Mandatory rows | Evidence / interpretation |
| --- | --- |
| root-read, root-full-cold, root-footer, root-footer-cold | Real AIStore, actual original C++ smoke, independent untouched objects, presence and backend GET counts. Reverse mode is only a separate green control. |
| admission-nonallowlisted/second-307/userinfo/fragment/downgrade | Actual requests to controlled redirect fixtures. Legal delegation must work before rejecting bad redirects can count as policy success. Downgrade starts with verified local TLS. |
| delegated-head/footer/range | Legitimate `/s3/` + `pid`/`utm` rewrite; method, Range and validator preservation, credential-header absence, exact bytes. A ranged read requires the strong fixture If-Match. |
| reuse-off | Two different ranges on one open object must rediscover the origin; the real policy control must be selectable. |
| retention-f0 | W1/C1, 32 serial ranged reads, one origin/one target: expanded <=2 GET connections plus separately counted HEAD warmup, legacy >=16. |
| stress-q64/q256 | All futures submitted before collection, buffers retained through every result; a server gate prevents early responses. Three targets are configured. This is not the F0 retention proof. |
| list-etag | Actual library pagination/key/size compared with captured service and backing LIST inventories. Raw ETag presence/empty/nonempty is observed separately. |

The old public API has no redirect-policy, reuse or retention switches. The two
retention runs explicitly say `policy_controls_available=false`; they do not
pretend different modes were selected. Old code rejects the first 307, so no
delegated requests reach the three stress targets and admission cannot be
qualified. Direct Q64/Q256 controls do establish queued reads and correct bytes.
The old data GET also omits If-Match: healthy direct reads are **not** validator
coverage. Future production-port work must supply its real configuration adapter
and revise the explicit source pin before using this rig for green qualification.
Do not set a receipt flag to stand in for an unimplemented API.

`list_entry` in this baseline exposes only key/size, not ETag. Empty or absent
service ETags do not fail an otherwise complete characterization, and do not
grant nonempty-validator capability. A two-page fixture covers empty, absent and
nonempty values; the actual AIS observation records whatever the service returns.
The backend inventory includes the readiness object and reflects the already
known duplicate-slash limitation; uploaded fixture names are not an inventory
oracle. No object GET is used to prepare LIST or prewarm the cold-read objects.

### Reproduce H

Use the existing Python environment with boto3 and the frozen build, without
updating dependencies. All run/output directories must be fresh. For example,
from this checkout, with `$FROZEN_REPO` set to the original P0 checkout:

```sh
pixi run --as-is --offline -e cuda-13-stable \
  --manifest-path "$FROZEN_REPO/pixi.toml" \
  python -B scripts/cache-harness/build_required_probe.py \
  --frozen-repo "$FROZEN_REPO" \
  --build-dir "$FROZEN_REPO/build/p0-io-cudf" --output-dir "$RUN/build"

python3 -B scripts/cache-harness/required_fixture.py \
  --binary "$RUN/build/required_feature_probe" --run-dir "$RUN/wire"

# Run each mode with fresh services and distinct run directories. The wrapper
# cleans up before returning; sequential runs may reuse the released port range.
for mode in root reverse; do
  python3 -B scripts/cache-harness/required_ais.py \
    --run-dir "$RUN/ais-$mode" --mode "$mode" \
    --probe "$RUN/build/required_feature_probe" \
    --binary "$FROZEN_REPO/build/p0-io-cudf/test/cucascade_direct_s3_smoke" \
    --ais-source "$AIS_SOURCE" --aisnode "$AIS_NODE" \
    --aisnode-sha256 "$AIS_SHA" --weed "$WEED" \
    --port-base 21130 --ais-port-base 21230 || break
done

python3 -B scripts/cache-harness/package_required_evidence.py \
  --root "$RUN" --wire wire --ais-root ais-root --ais-reverse ais-reverse --build build
python3 -B scripts/cache-harness/required_features.py "$RUN/packet.json" \
  --output "$RUN/required-report.json"
# Expected exit 1. Do not suppress it in a feature qualification pipeline.

# Separate implementation checks; a baseline-check PASS still means qualification FAIL.
python3 -B scripts/cache-harness/check_required_baseline.py "$RUN/packet.json"
python3 -B scripts/cache-harness/test_required_features.py "$RUN/packet.json"
```

The builder compiles a new test-only probe against the existing shared libraries,
checks both production source trees against the pin, and records compile/link
commands, source, lock, library and binary hashes. This is not a fresh library
build. Packet paths are relative for offline review; subprocess commands retain
the original execution paths. Hashes detect changed evidence, not malicious
rewriting of both evidence and its manifest. Keep the original receipts.

Use the original regression/lifecycle/I/O commands above as well. The H unit
suite includes complete required-status checks, 17 real-receipt mutation cases
through both CLIs, and synthetic delegated-wire oracle controls. Synthetic
successes are never reported as successful engine runs. No performance claim is
made by these tests.

### H review fixes: evidence and lifecycle

The strict evaluator also validates the concrete Location and owned authority for
all redirect cases. A first-hop-unsupported verdict requires one corresponding
307 for every failed operation and **no target or canary request**. Admission
labels alone are not evidence that the intended negative was delivered.

`required_ais_evidence.py` declares the raw receipts needed by every AIS control,
rejudges HTTP status/body hashes/presence/counters, and compares those results
with the original summary. An absent counter is zero only when the recorded
metric registry proves the counter exists. Removing a required file from both
the directory and manifest still fails. `target-stopped` currently requires the
recorded HTTP failure: a summary-only transport exception is insufficient.

LIST validation parses IsTruncated and NextContinuationToken from every raw XML
page, checks its own request/token chain, and checks library page boundaries,
entries and non-repeating tokens. The reference and library LIST requests are
independent AIS sessions, whose opaque token values need not be equal. This does
not change the empty/absent ETag characterization rule.

The H AIS wrapper now calls the original `ais_harness.up` and `qualify` functions
in its own process. It does not start a registrar child that Python can SIGKILL
on interruption. The original `defer_signals`/`launch_owned` critical section
therefore completes ownership recording before handling a signal; original
identity-checked, bounded TERM/KILL cleanup remains responsible for services.
`--deadline` (default 300 seconds) is a catchable whole-run alarm, deferred during
that critical section. Repeated termination signals cannot interrupt final
cleanup. No production library or original lifecycle helper was changed.

```sh
python3 -B scripts/cache-harness/test_required_review.py "$RUN/packet.json"
python3 -B scripts/cache-harness/test_required_ais_lifecycle.py
```

The first suite rejects the six independent review counterexamples and twenty
adjacent semantic mutations via both CLIs. The Linux lifecycle suite exercises
the whole H entrypoint with original ownership helpers and isolated harmless
services: before launch, before journal persistence and after persistence,
SIGINT/SIGTERM, repeated signals, deadline, normal completion and cleanup
escalation. Only its LIST business work is stubbed; actual AIS root/reverse runs
remain separate evidence. A successful expected-red check still never grants
full feature qualification.
