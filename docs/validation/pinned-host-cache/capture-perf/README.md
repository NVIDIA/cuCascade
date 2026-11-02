# Bounded capture-check diagnostics

These are diagnostic artifacts, not acceptance results or a proposed allocator change. The `noquery` variant deliberately disables the capture-rejection contract. Never use it for application execution or integrate it. The production implementation was not edited.

`query.cpp` measures two runtime APIs with static CUDA runtime linkage. `query.jsonl` is the uninstrumented output, seven alternating trials of 10 million calls per API. The same nonblocking stream, status output, compiler, CPU, and warmed runtime are used for both APIs. Query-only timing does not predict complete allocator throughput.

`build_ablation.py` preserves the exact historical commands used to copy and recompile only the candidate allocator translation unit. Its `intact` and `noquery` libraries reuse the other existing candidate object files. `noquery.patch` is the entire diagnostic source change. `ablation-metadata.json` records complete compile/link commands and source/library hashes. The exact source hash for the intact translation unit is also recorded in `experiment-metadata.json`.

`run_ablation.py` preserves the final historical runner. It uses the existing controlled benchmark executables with matching headers, seven alternating trial orders, 300,000 deterministic iterations, 128 warmup iterations, and CPUs 15-18. Each arm explicitly preloads its exact core library. `experiment-metadata.json` records executable hashes, library hashes, commands, and `ldd` verification of loader selection. `ablation-raw.jsonl` and `ablation-summary.json` are the final results.

The first runner used only `LD_LIBRARY_PATH`. That failed because the existing executable's `DT_RPATH` took precedence, so the intended isolated libraries were not loaded. `invalid-rpath-ablation-*` preserves that invalid first run; it contributes no evidence about guard cost.

For reproduction on the original machine, the recorded commands use `/tmp/cucascade-spec-implementation/capture-perf`, `/tmp/cucascade-spec-builds/issue-11-default`, and `/tmp/cucascade-spec-worktrees/issue-11`. These are historical paths and may be removed during cleanup. On another machine, use fresh paths and rebuild baseline and candidate first with identical settings as described in the main validation report. Update the scripts' source/build/output/compiler paths to those fresh locations; retain the recorded files unchanged as evidence. Generate `ninja-commands.txt` for the build script with the candidate build's configured Ninja:

```bash
ninja -C <candidate-build> -t commands cucascade_shared > <diagnostic-output>/ninja-commands.txt
python3 <diagnostic-output>/build_ablation.py
flock <shared-gpu-lock> taskset --cpu-list <matching-cpus> \
  python3 <diagnostic-output>/run_ablation.py
```

Use the query compilation and execution commands in `experiment-metadata.json`, substituting your environment and paths. Verify the loaded libraries with `ldd` before collecting data. Preserve a new metadata file with the adjusted commands, source/configuration/compiler identities, unique library and executable hashes, CPU/GPU inventory, and all raw trials. Neither diagnostic compiles or installs a replacement into the candidate production build. The scripts are retained byte for byte as run, rather than maintained as portable product tooling.

CPU guidance read for this investigation: Agner Fog's profiler and threading sections (3.2 and 7.31), the C++ Core Guidelines lock/callback guidance (CP.22), the C++ Software Design RAII discussion, and the C++20 `jthread` stop-token discussion. Architecture-specific x86 advice was not assumed to apply to this ARM GB10 host. CUDA query semantics were checked against the installed CUDA 13.2 runtime header and the primary NVIDIA stream-management reference linked from the validation report.
