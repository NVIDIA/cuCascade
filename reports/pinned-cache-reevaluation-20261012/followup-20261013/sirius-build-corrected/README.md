# Corrected isolated consumer artifacts

The earlier allocator-only relink moved Sirius object inputs into `libsirius_core.a` while retaining `--exclude-libs,ALL`. This changed RMM's device-resource map and lock from GNU-unique dynamic exports to local symbols. Those earlier timings and traces describe that isolated registry regime and cannot establish the benefit in the normal consumer integration.

`build.py` repairs only the placement of the frozen Sirius inputs. It reads every GNU ar member, including both distinct members named `type.pb.cc.o`, and directly links all 334 objects in their archive order. The existing CUDA device-link member remains present. It reuses the previously rebuilt baseline/candidate owner and allocator objects and common wrapper object, with no source compilation. The same 333 other Sirius members and 21 other cuCascade members match the original donor byte for byte.

The other link flags and archives retain the previous frozen configuration, including `--exclude-libs,ALL`, which now applies only to the same supporting archives as in the preserved original direct-object extension. The link dependency file and output paths are redirected into this directory. The normal extensions and both corrected arms export the RMM map, map guard, and lock as `UNIQUE DEFAULT`; their process-exit helper remains a dynamic undefined import. No `initial_resource` symbol appears in any of those four dynamic symbol tables.

The source snapshot remains current Sirius `45253bf` source-equivalent to the preserved donor, with fixed consumer cuCascade accounting and allocator-only main/candidate overlays. This is a static direct-object extension using the verified existing DuckDB objects, not a full fresh build of the current wrapper's default shared-library configuration. Current wrapper glue, current donor objects, and cached DuckDB link inputs remain as recorded in the earlier build metadata. Restored registry visibility is a necessary parity check; root must validate the loaded runtime and results before using timings.

Artifacts:

- Baseline: `baseline/sirius.duckdb_extension`, SHA256 `5cc828e67b6e4d1007a257569c5383b301037dcadcf32b4ea2aee2b3cf18690a`.
- Candidate: `candidate/sirius.duckdb_extension`, SHA256 `2673bb70648cf4d1f749b44b3bf88813db0b947d87ed311c12e3b33e88dc20ad`.
- `build-metadata.json` records commands, each direct member's path and hash, byte comparisons against the donor, dynamic registry symbols, and before/after hashes of protected original and production artifacts.
- `build.log` contains successful link and DuckDB metadata commands with no warnings.

Compiler-only command used:

```sh
pixi run --as-is --manifest-path /home/kkristensen/Code/sirius/pixi.toml python /tmp/cucascade-pinned-pr-20261012/sirius-build-corrected/build.py
```

No CUDA execution occurred here. Root owns all GPU validation and serialized benchmark execution; reviewer independently audits these artifacts before that work.
