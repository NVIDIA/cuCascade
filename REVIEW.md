# Issue 199 Review

## Blocker

None.

## High

### Resolved: failed-probe teardown now targets the matching access direction

The prior High finding is resolved. A probe slot `(source, destination)` now consistently models
the destination device accessing memory owned by the source device:

- capability is queried as `destination -> source`;
- peer access is enabled with `destination` current and `source` as its peer; and
- failed-probe cleanup selects `destination` and disables access to `source`.

The deterministic asymmetric regression covers both `(0, 1)` and `(1, 0)` cleanup mappings. This
matches CUDA's unidirectional peer-access contract and prevents a failed direction from disabling
the working reverse direction.

No active High findings remain.

## Medium

None.

## Low

### Resolved: legacy helper documentation now matches its behavior

The prior Low finding is resolved. `include/cucascade/memory/common.hpp` now says
`disable_peer_access_where_broken(pools_by_device)` leaves CUDA pool permissions unchanged and
retains its ignored argument for API compatibility. This matches `src/memory/common.cpp:480-490`.

No active Low findings remain.

## Validation Notes

- Reviewed the exact issue #199 requirements, repository guidance, reviewer role guidance, the
  complete tracked diff, and both untracked new files.
- Confirmed the change remains scoped to per-pool peer access, borrowed pool-handle exposure,
  deterministic coverage, hardware-gated coverage, and the required documentation.
- Confirmed the public grant checks live capability and cached byte verification in both
  directions even for an existing read/write permission, while only the duplicate pool-access
  mutation is skipped.
- Confirmed caller-device restoration errors and CUDA query/grant/probe errors remain observable.
- No tests were run during this reviewer pass, as instructed.

## Current Retry-Fix Review Addendum

- Inspected the complete current implementation, detail test seam, new untracked files, public
  declarations, and changed memory-management documentation. The process-wide cache is justified
  by the synchronized probes and its mutex covers all cached entries; conclusive results remain
  cached while CUDA errors are retried.
- Confirmed transient CUDA errors do not trigger legacy peer teardown, byte mismatches target the
  matching access direction, and device restoration failures remain observable. Deterministic
  tests cover those outcomes and a subsequent successful retry.
- The parent reported a successful `cucascade_tests` build, `git diff --check`, and 77 assertions
  across 10 non-GPU peer-access cases. The two-GPU path could not run on this machine.
- Re-read the corrected legacy helper comment against its implementation; the prior Low finding
  is resolved. The subsequent `git diff --check` passes.
