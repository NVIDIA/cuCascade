from pathlib import Path
import json, subprocess, os, statistics

root = Path("/tmp/cucascade-spec-implementation/capture-perf")
bins = Path("/tmp/cucascade-spec-implementation/issue-11-controlled-benchmark-7pairs")
envpath = "/home/kkristensen/Code/cuCascade/.pixi/envs/default/lib"
variants = {
    "baseline": (
        "/tmp/cucascade-spec-builds/issue-11-baseline-default",
        bins / "baseline",
    ),
    "intact": (str(root / "intact"), bins / "candidate"),
    "noquery": (str(root / "noquery"), bins / "candidate"),
}
rows = []
with (root / "ablation-raw.jsonl").open("w") as raw:
    for trial in range(7):
        order = list(variants)
        if trial % 2:
            order.reverse()
        for variant in order:
            lib, binary = variants[variant]
            env = os.environ.copy()
            env["LD_LIBRARY_PATH"] = lib + ":" + envpath
            env["LD_PRELOAD"] = lib + "/libcucascade.so"
            value = json.loads(
                subprocess.check_output(
                    [str(binary), "full", "1", "300000", "128"], env=env, text=True
                )
            )
            value.update(trial=trial, variant=variant, library_directory=lib)
            rows.append(value)
            raw.write(json.dumps(value) + "\n")
            raw.flush()
summary = {}
for variant in variants:
    values = [x["operations_per_second"] for x in rows if x["variant"] == variant]
    summary[variant] = {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "cv": statistics.stdev(values) / statistics.mean(values),
    }
(root / "ablation-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
