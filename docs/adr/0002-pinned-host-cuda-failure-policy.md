# Contain pinned host cleanup failures

When GPU completion cannot be confirmed, withhold the affected storage and retire its suspect event
instead of reusing either. Slab quarantine retains only the owning backing allocation so a single
failure does not retain unrelated storage. Cleanup remains non-throwing and clears errors from its
own failed CUDA calls where the runtime permits recovery.

Event retirement relinquishes the wrapper's handle before attempting destruction and never retries
an ambiguously destroyed handle. A shared non-throwing reset operation returns the destruction
status and clears a failure; the destructor and move assignment use the same cleanup path.

Failure to restore the calling thread's previous CUDA device is fatal. Silently returning on another
device could redirect subsequent GPU work, and a non-throwing deallocation cannot report that failure
through an exception. This policy is specific to failed device restoration; ordinary cleanup errors
continue to use storage withholding where safe.
