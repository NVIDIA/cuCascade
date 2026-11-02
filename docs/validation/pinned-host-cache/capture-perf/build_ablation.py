from pathlib import Path
import json, shlex, subprocess, hashlib

out = Path("/tmp/cucascade-spec-implementation/capture-perf")
build = Path("/tmp/cucascade-spec-builds/issue-11-default")
source = Path(
    "/tmp/cucascade-spec-worktrees/issue-11/src/memory/small_pinned_host_memory_resource.cpp"
)
commands = (out / "ninja-commands.txt").read_text().splitlines()
compile_line = next(line for line in commands if " -c " in line and str(source) in line)
parts = shlex.split(compile_line)
flags = []
i = 1
while i < len(parts):
    p = parts[i]
    if p in ("-isystem", "-I"):
        flags.extend(parts[i : i + 2])
        i += 2
        continue
    if p.startswith(("-D", "-I")):
        flags.append(p)
    i += 1
link = shlex.split(commands[-1])[2:-2]
meta = {}
for mode in ("intact", "noquery"):
    target = out / mode
    (target / "src/memory").mkdir(parents=True, exist_ok=True)
    (target / "src/pinned_host_test_hooks.hpp").write_bytes(
        source.parent.parent.joinpath("pinned_host_test_hooks.hpp").read_bytes()
    )
    text = source.read_text()
    if mode == "noquery":
        start = text.index(
            "  return CUCASCADE_PINNED_HOST_CUDA_CALL(stream_capture_status,"
        )
        end = text.index("\n}", start)
        text = (
            text[:start]
            + "  std::ignore = stream;\n  status = cudaStreamCaptureStatusNone;\n  return cudaSuccess;"
            + text[end:]
        )
    local = target / "src/memory/small_pinned_host_memory_resource.cpp"
    local.write_text(text)
    obj = target / "allocator.o"
    cmd = [
        parts[0],
        *flags,
        "-O3",
        "-DNDEBUG",
        "-std=c++20",
        "-fPIC",
        "-pthread",
        "-Wall",
        "-Wextra",
        "-Wpedantic",
        "-Werror",
        "-c",
        str(local),
        "-o",
        str(obj),
    ]
    subprocess.run(cmd, check=True)
    result = []
    skip = False
    for p in link:
        if skip:
            skip = False
            continue
        if p == "-o":
            result.extend(["-o", str(target / "libcucascade.so")])
            skip = True
            continue
        if p.startswith("-Wl,--dependency-file="):
            continue
        if p.endswith("/small_pinned_host_memory_resource.cpp.o"):
            result.append(str(obj))
            continue
        result.append(p)
    subprocess.run(result, cwd=build, check=True)
    meta[mode] = {
        "source_sha256": hashlib.sha256(local.read_bytes()).hexdigest(),
        "library_sha256": hashlib.sha256(
            (target / "libcucascade.so").read_bytes()
        ).hexdigest(),
        "compile": cmd,
        "link": result,
    }
(out / "ablation-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
subprocess.run(
    [
        "diff",
        "-u",
        str(out / "intact/src/memory/small_pinned_host_memory_resource.cpp"),
        str(out / "noquery/src/memory/small_pinned_host_memory_resource.cpp"),
    ],
    stdout=(out / "noquery.patch").open("w"),
    check=False,
)
