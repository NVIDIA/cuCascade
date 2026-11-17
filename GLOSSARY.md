# cuCascade

Tiered GPU/host/disk memory management and data movement for GPU data batches.

## Cross-device copy

**Copy route**:
The path bytes take when a GPU-tier representation moves from one GPU to another: either the peer route or the host-staged route.

**Peer route**:
A copy route that moves bytes directly between two GPUs over peer DMA.
_Avoid_: P2P path, direct copy

**Host-staged route**:
A copy route that moves bytes from the source GPU into pinned host memory and then into the destination GPU.
_Avoid_: bounce, bounce buffer path, host bounce

**Host-staged pipeline**:
The host-staged route run as overlapping source-to-host and host-to-destination legs over a table's leaf buffers treated as one byte stream.
_Avoid_: staged copy, pipelined bounce

**Leaf buffer**:
One contiguous device buffer of a column: its data, validity mask, offsets or chars.
_Avoid_: segment, span

**Chunk**:
The unit of bytes the host-staged pipeline moves through one staging slot in one pass; it may span leaf-buffer boundaries.
_Avoid_: block (reserved for host pool allocation blocks), segment

**Staging slot**:
A chunk-sized region of pinned host memory that one chunk occupies between its source-to-host and host-to-destination legs.
_Avoid_: staging buffer, bounce buffer

**Pipeline depth**:
The number of staging slots one host-staged pipeline owns, which bounds how many chunks can be in flight at once.
_Avoid_: queue depth, buffer count
