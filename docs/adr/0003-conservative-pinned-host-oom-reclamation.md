# Reclaim pinned host cache capacity conservatively

After a rounded allocation fails with OOM, try the exact requested size before reclaiming cached
buffers. If the exact-size attempt also fails, reclaim only when the currently retained cache
capacity covers the entire exact allocation size; detach enough oldest entries to cover that size
and retry exact-size allocation once. This preserves useful retained buffers when an impossible or
disproportionately large request fails, and bounds the recovery work.

This is a deliberate policy rather than a test of physical satisfiability. For example, a failed
256 MiB request with 200 MiB available and 128 MiB cached skips reclamation even though freeing the
cached buffers might let it succeed. Prefer predictable conservative recovery to speculative waits
on unrelated producers. No additional configuration switch is introduced.
