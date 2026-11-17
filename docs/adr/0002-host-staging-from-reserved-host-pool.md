# Staging slots come from the reserved HOST pool, per conversion

Each host-staged pipeline takes its staging slots (depth × chunk bytes) from the HOST memory space on the destination GPU's NUMA node, under a HOST reservation that the converter requests itself. The PIPELINE disk backend uses a dedicated pinned ring shared behind a mutex; we did not copy that here. Staging memory therefore stays visible to the memory budget, stays bounded no matter how large the table is, and concurrent conversions share no staging memory and take no global lock.

## Considered Options

- **Dedicated pinned ring per device pair** (as in the PIPELINE disk backend): rejected. It hides memory from the reservation system and serializes concurrent conversions.
- **HOST pool without a reservation** (the old behavior): rejected. Staging bytes went unaccounted.

## Consequences

- If no reservation can be obtained, the converter retries with a smaller depth, down to 1. If even that fails, it throws `cucascade_out_of_memory` rather than blocking, because blocking could deadlock a caller that already holds HOST memory, or falling back to an untracked pinned allocation. The caller's OOM policy decides what to do next.
- The only limit on how many conversions run at once is HOST reservation capacity. Conversions are not fully independent, though: each takes its source stream and its target stream in turn from the 16-stream pool of the GPU's memory_space. When more than 16 conversions read from or write to one GPU at once, some share a stream and their copies queue behind each other on it. That costs time, not correctness.
- A staging slot is made of the pool's fixed-size blocks and is not contiguous. Every copy stays inside one block, even when neighbouring blocks are adjacent in memory: the pool pins its memory in separate regions, and CUDA rejects a copy that crosses from one region into the next.
