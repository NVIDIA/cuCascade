# Preserve stream-ordered pinned host allocation

Pinned host allocation remains stream-ordered: CPU callers synchronize the allocation stream or use
`allocate_sync`, and returned contents are uninitialized. Making every cache hit CPU-ready would add
host waits to the normal allocation path and undermine the pooling benefit. Memory-pressure recovery
and eviction may block on prior GPU work, including work on another device, with completion waits and
buffer frees outside the cache mutex.

Allocation and deallocation inside CUDA host callbacks or during graph capture are unsupported.
On a capturing stream, allocation throws and non-throwing deallocation withholds storage without
recording, waiting, freeing, or evicting buffers. Slab-owner quarantine bookkeeping remains necessary;
rejection preserves capture validity and prevents cache publication or reuse. Supporting graph
capture would require a separate lifetime and reclamation design.
