# cuCascade Pinned Host Memory

## Language

**Large-buffer cache**:
The retained pinned host buffers available for reuse by streams and devices sharing one small
pinned host memory resource.
_Avoid_: Process-wide cache

**Retained large-buffer capacity**:
The capacity held in the large-buffer cache, excluding buffers in use, buffers being reclaimed,
and storage withheld after an unconfirmed completion.
_Avoid_: Total pinned memory, total resource capacity

**Quarantined slab backing**:
The backing storage withheld from reuse because completion of a slab's previous GPU use could not
be confirmed.
_Avoid_: Quarantined arena, when only one backing allocation is affected
