#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build only the H probe against an existing, source-matched frozen build."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

BASE = "1a9281093a0fe49634594d541d7ce327129f53d6"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frozen-repo", type=Path, required=True)
    p.add_argument("--build-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    repo = Path(__file__).resolve().parents[2]
    frozen, build, out = (
        x.resolve() for x in (args.frozen_repo, args.build_dir, args.output_dir)
    )
    out.mkdir(parents=True, exist_ok=False)
    for root in (repo, frozen):
        subprocess.run(
            ["git", "diff", "--exit-code", BASE, "--", "src", "include"],
            cwd=root,
            check=True,
            stdout=subprocess.PIPE,
        )
    source = repo / "test/io/integration/required_feature_probe.cpp"
    original = next(
        c
        for c in json.loads((build / "compile_commands.json").read_text())
        if c["file"].endswith("/direct_s3_smoke.cpp")
    )
    compile_cmd = shlex.split(original["command"])
    compile_cmd = [
        c
        for c in compile_cmd
        if c not in ("-fmodules-ts", "-MD")
        and not c.startswith(("-fmodule-mapper=", "-fdeps-format="))
    ]
    old_object = compile_cmd[compile_cmd.index("-o") + 1]
    obj = out / "required_feature_probe.o"
    compile_cmd[compile_cmd.index("-o") + 1] = str(obj)
    compile_cmd[compile_cmd.index("-c") + 1] = str(source)
    link_text = subprocess.check_output(
        ["ninja", "-t", "commands", "cucascade_direct_s3_smoke"], cwd=build, text=True
    ).splitlines()[-1]
    link_cmd = shlex.split(link_text)
    if link_cmd[:2] != [":", "&&"] or link_cmd[-2:] != ["&&", ":"]:
        raise RuntimeError("unexpected frozen Ninja link format")
    link_cmd = [c for c in link_cmd[2:-2] if not c.startswith("-Wl,--dependency-file=")]
    link_cmd[link_cmd.index(old_object)] = str(obj)
    binary = out / "required_feature_probe"
    link_cmd[link_cmd.index("-o") + 1] = str(binary)
    production = {}
    for relative in subprocess.check_output(
        ["git", "ls-files", "src", "include"], cwd=frozen, text=True
    ).splitlines():
        a, b = frozen / relative, repo / relative
        if sha(a) != sha(b):
            raise RuntimeError("source mismatch: " + relative)
        production[relative] = sha(a)
    libraries = {}
    for token in link_cmd:
        path = Path(token) if token.startswith("/") else build / token
        if path.is_file() and (".so" in path.name or path.suffix == ".a"):
            libraries[str(path.resolve())] = sha(path)
    receipt = dict(
        schema=1,
        production_base=BASE,
        source_head=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        source_sha256=sha(source),
        production_files=production,
        libraries=libraries,
        pixi_lock_sha256=sha(frozen / "pixi.lock"),
        compile=compile_cmd,
        link=link_cmd,
        cwd=str(build),
    )
    for label, command in (("compile", compile_cmd), ("link", link_cmd)):
        with (out / (label + ".log")).open("w") as log:
            result = subprocess.run(
                command, cwd=build, stdout=log, stderr=subprocess.STDOUT
            )
        receipt[label + "_exit"] = result.returncode
        (out / "build.json").write_text(json.dumps(receipt, indent=2) + "\n")
        result.check_returncode()
    receipt["binary"] = str(binary)
    receipt["binary_sha256"] = sha(binary)
    (out / "build.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(binary)


if __name__ == "__main__":
    main()
