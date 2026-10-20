# I/O research notes: GDS / cuFile and TPC-H read patterns

Research and recommendations behind the `cucascade::io` disk reactors, in particular the cuFile / GPUDirect Storage reactor (`include/cucascade/io/cufile/`). Compiled 2026-10-14/15.

**Start with [gds-reactor-recommendations.md](gds-reactor-recommendations.md)**: the prioritised, concrete suggestions, a bring-up checklist for avoiding silent compatibility mode, the alignment rules, and the experiment plan.

| Document | Contents | Evidence |
|---|---|---|
| [gds-reactor-recommendations.md](gds-reactor-recommendations.md) | What to change and in what order; compat-mode and alignment checklists; experiment plan; known limitations of the current reactor | synthesis of everything below |
| [gds-cufile-best-practices.md](gds-cufile-best-practices.md) | NVIDIA's cuFile/GDS documentation (v1.19.1) distilled: which API when, buffer registration, alignment, I/O size and queue depth, `cufile.json` reference, filesystems, topology, Grace, pitfalls, measured results, contradictions with our approach, open questions | NVIDIA docs, headers, release notes, samples, issues; key quotes re-verified verbatim |
| [gds-field-experience.md](gds-field-experience.md) | How other systems use GDS for analytics (DALI, NIXL, cuDF, DeepSpeed, fastsafetensors, Theseus, PystachIO, ...), GDS versus host-bounce measurements, parquet request sizes across engines, the host-bounce alternative, and a short, flagged kvikIO section | papers, white papers, source code, forum threads |
| [tpch-io-patterns.md](tpch-io-patterns.md) | What a TPC-H scan reads: request sizes, gaps, alignment, repetition, I/O share of query time, throughput ceilings | synthesis of the two data documents and GB300 measurements |
| [tpch-sf1000-gb300-trace.md](tpch-sf1000-gb300-trace.md) | Statistics of the Sirius SF1000 range trace (22 queries) from the GB300 host | relayed from the host; not re-run here |
| [tpch-parquet-layout-sf10-sf100.md](tpch-parquet-layout-sf10-sf100.md) | Physical layout of the local DuckDB-written SF10/SF100 parquet files and a simulated column-pruned scan of Q1-Q22 | measured from footers; scan is a model |

## Caveats that apply to all of it

- **Nothing was measured on a working GDS stack.** Both known hosts run cuFile in compatibility mode (no `nvidia-fs`). Performance statements are NVIDIA's, other systems', or inferences, and are labelled as such.
- The NVIDIA documentation read is v1.19.1; the installed libcufile is 1.17.1 (CUDA 13.2). Version-dependent behaviour is marked in the documents.
- kvikIO is treated as a low-confidence source for GDS (project guidance); its defaults are not recommendations.
- The GB300 figures were gathered by a remote agent reading existing result files and running small statistics scripts there; they were checked for internal consistency (listed in the trace document) but not reproduced.
- Local-layout and SF1000-layout request sizes differ by roughly 8-9x because of row-group size; do not mix their conclusions.
