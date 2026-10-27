#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Isolated local SeaweedFS lifecycle; reference probes never count as library tests."""

import argparse
from contextlib import contextmanager
import hashlib
from functools import wraps
import select
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import uuid

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

# Deliberately public, local-only test credentials. Never resolve ambient AWS credentials.
ACCESS = "cucascade-local-test"
SECRET = "cucascade-local-test-secret"
WEED_SHA = "a299c1528d7b9838828d1b6147c98e4b38b2ce1ed3f683e7ec7850b2c29f7ca9"


def save(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def load(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def client(state, bad=False):
    return boto3.client(
        "s3",
        endpoint_url=state["endpoint"],
        aws_access_key_id=ACCESS,
        aws_secret_access_key="invalid-local-secret" if bad else SECRET,
        region_name="us-east-1",
        config=Config(
            signature_version="s3v4",
            s3={"addressing_style": "path"},
            proxies={},
            connect_timeout=2,
            read_timeout=5,
            retries={"total_max_attempts": 1},
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
        ),
    )


def read_body(response):
    body = response["Body"]
    try:
        return body.read()
    finally:
        body.close()


def process_identity(pid):
    try:
        # comm may contain spaces: fields after the final ')' start with state.
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return fields[19]  # starttime, field22
    except FileNotFoundError:
        return None


def state_for(run):
    state = load(run / "state.json")
    if state["run_dir"] != str(run) or state["schema"] != 1:
        raise RuntimeError("run directory ownership mismatch")
    return state


def protect_cleanup(function):
    """Bounded cleanup must finish even when termination is requested repeatedly."""

    @wraps(function)
    def guarded(*args, **kwargs):
        signals = (signal.SIGINT, signal.SIGTERM, signal.SIGALRM)
        previous = {sig: signal.getsignal(sig) for sig in signals}
        try:
            for sig in signals:
                signal.signal(sig, signal.SIG_IGN)
            return function(*args, **kwargs)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)

    return guarded


@protect_cleanup
def stop_owned(pid, start, marker, term_seconds=8, kill_seconds=3):
    """Bind identity checks and every signal to one Linux process, never its PID reuse."""
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 1:
        raise RuntimeError("invalid owned PID")
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        raise RuntimeError(
            "Linux pidfd support required; refusing numeric-PID fallback"
        )
    try:
        fd = os.pidfd_open(pid, 0)
    except ProcessLookupError:
        return []
    sent = []
    try:
        poller = select.poll()
        poller.register(fd, select.POLLIN)
        if poller.poll(0):
            return sent
        if process_identity(pid) != start:
            raise RuntimeError("PID reused; refusing to stop unrelated process")
        try:
            command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        except FileNotFoundError:
            if poller.poll(0):
                return sent
            raise
        if marker.encode() not in command:
            raise RuntimeError("process command mismatch; refusing to stop")
        # If the original process exits after validation, the fd cannot signal a
        # replacement even if /proc now exposes the same numeric PID again.
        for sig, seconds in (
            (signal.SIGTERM, term_seconds),
            (signal.SIGKILL, kill_seconds),
        ):
            try:
                signal.pidfd_send_signal(fd, sig)
            except ProcessLookupError:
                return sent
            sent.append(sig.name)
            if poller.poll(int(seconds * 1000)):
                return sent
        raise RuntimeError("owned service failed to stop")
    finally:
        os.close(fd)


@protect_cleanup
def down(run):
    state = state_for(run)
    if state.get("pid"):
        stop_owned(state["pid"], state["process_start"], f"-dir={run / 'data'}")
    state["stopped"] = True
    save(run / "state.json", state)
    # Preserve all data/logs for inspection. No recursive deletion, no pkill.
    return state


def preflight(args):
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        raise RuntimeError("Linux pidfd support required before starting services")
    fd = os.pidfd_open(os.getpid(), 0)
    os.close(fd)
    weed = args.weed.resolve(strict=True)
    binary = args.binary.resolve(strict=True)
    digest = sha(weed)
    if digest != args.weed_sha256:
        raise RuntimeError("SeaweedFS hash mismatch")
    version = subprocess.run(
        [str(weed), "version"], capture_output=True, text=True, check=True, timeout=10
    ).stdout.strip()
    return {
        "weed": str(weed),
        "weed_sha256": digest,
        "weed_version": version,
        "binary": str(binary),
        "binary_sha256": sha(binary),
        "boto3_version": boto3.__version__,
        "python_version": sys.version.split()[0],
    }


@contextmanager
def defer_signals():
    """Defer Python handlers, not OS delivery (children inherit no blocked mask)."""
    pending = []
    signals = (signal.SIGINT, signal.SIGTERM, signal.SIGALRM)
    previous = {sig: signal.getsignal(sig) for sig in signals}

    def remember(signum, frame):
        pending.append((signum, frame))

    for sig in signals:
        signal.signal(sig, remember)
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if pending:
            signum, frame = pending[0]
            handler = previous[signum]
            if callable(handler):
                handler(signum, frame)
            elif handler != signal.SIG_IGN:
                raise KeyboardInterrupt(f"interrupted by signal {signum}")


def launch_owned(command, persist, **kwargs):
    """Retain the child handle until ownership is durable, including save failure."""
    proc = None
    try:
        with defer_signals():
            proc = subprocess.Popen(command, **kwargs)
            persist(proc)
        return proc
    except BaseException:
        # A direct child handle is authoritative even if no PID was saved.
        with defer_signals():
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=3)
        raise


def fixture_bytes(identity, size):
    """Stable SHAKE256 stream; offset-aligned blocks are not periodic copies."""
    return hashlib.shake_256(b"cucascade-fixture-v2\0" + identity.encode()).digest(size)


def up(args):
    info = preflight(args)
    run = args.run_dir
    run.mkdir(parents=True, exist_ok=False)
    run.chmod(0o700)
    # Explicit consecutive ports, including all component gRPC ports.
    ports = list(range(args.port_base, args.port_base + 8))
    if not 1024 <= ports[0] <= ports[-1] <= 65535:
        raise ValueError("port base outside unprivileged range")
    checks = []
    try:
        for port in ports:
            sock = socket.socket()
            checks.append(sock)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", port))
            sock.listen(1)
    finally:
        for sock in checks:
            sock.close()
    for name in ("data", "filer", "fixtures"):
        (run / name).mkdir()
    (run / "filer.toml").write_text(
        '[leveldb2]\nenabled = true\ndir = "' + str(run / "filer") + '"\n'
    )
    save(
        run / "s3.json",
        {
            "identities": [
                {
                    "name": "local",
                    "credentials": [{"accessKey": ACCESS, "secretKey": SECRET}],
                    "actions": ["Admin", "Read", "Write", "List", "Tagging"],
                }
            ]
        },
    )
    state = dict(
        info,
        schema=1,
        run_dir=str(run),
        bucket="cucascade-" + uuid.uuid4().hex[:16],
        endpoint=f"http://127.0.0.1:{ports[6]}",
        ports=ports,
        stopped=False,
    )
    save(run / "state.json", state)
    cmd = [
        info["weed"],
        "server",
        "-ip=127.0.0.1",
        "-ip.bind=127.0.0.1",
        f"-dir={run / 'data'}",
        "-master.telemetry=false",
        "-volume.max=8",
        "-master.volumeSizeLimitMB=64",
        f"-master.port={ports[0]}",
        f"-master.port.grpc={ports[1]}",
        f"-volume.port={ports[2]}",
        f"-volume.port.grpc={ports[3]}",
        "-filer",
        f"-filer.port={ports[4]}",
        f"-filer.port.grpc={ports[5]}",
        "-s3",
        f"-s3.port={ports[6]}",
        f"-s3.port.grpc={ports[7]}",
        f"-s3.config={run / 's3.json'}",
        "-s3.port.iceberg=0",
        "-s3.port.lance=0",
        "-s3.iam=false",
    ]
    save(run / "service-command.json", cmd)
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("AWS_", "WEED_", "S3_"))
    }

    def persist(proc):
        state.update(pid=proc.pid, process_start=process_identity(proc.pid))
        if state["process_start"] is None:
            raise RuntimeError("service exited before ownership could be recorded")
        save(run / "state.json", state)

    try:
        with (run / "service.log").open("wb") as log:
            proc = launch_owned(
                cmd,
                persist,
                cwd=run,
                env=env,
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
        s3 = client(state)
        deadline = time.monotonic() + 40
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError("service exited; inspect service.log")
            try:
                s3.list_buckets()
                s3.create_bucket(Bucket=state["bucket"])
                # Wait for a writable volume, not merely a listening HTTP socket.
                s3.put_object(Bucket=state["bucket"], Key="ready", Body=b"ready")
                return state
            except Exception:
                time.sleep(0.5)
        raise TimeoutError("service readiness exceeded 40 seconds")
    except BaseException:
        down(run)
        raise


def seed(run):
    state = state_for(run)
    s3 = client(state)
    sizes = [0, 1, 4096, 65553, 1048607]
    specs = [(f"ordinary/{n}.bin", n) for n in sizes]
    specs += [
        (key, 4096)
        for key in [
            "keys/space key",
            "keys/plus+key",
            "keys/percent%key",
            "keys/a//b",
            "keys/a/b",
        ]
    ]
    specs.append(("qualification/footer-first.bin", 1048607))
    fixtures = []
    for i, (key, size) in enumerate(specs):
        data = fixture_bytes(key, size)
        file = run / "fixtures" / f"{i}.bin"
        file.write_bytes(data)
        reply = s3.put_object(Bucket=state["bucket"], Key=key, Body=data)
        fixtures.append(
            dict(
                key=key,
                size=size,
                file=str(file),
                sha256=sha(file),
                etag=reply["ETag"],
                generator="shake256-v2",
            )
        )
        if size == 0:
            # SeaweedFS 4.48/boto3: next PUT on this keepalive connection times out.
            # Isolate reference seeding; this does not alter the cuCascade read path.
            s3.close()
            s3 = client(state)
    # Read after ALL puts, so aliases cannot pass by being tested before their collision.
    for fixture in fixtures:
        data = read_body(s3.get_object(Bucket=state["bucket"], Key=fixture["key"]))
        fixture["reference_sha256"] = hashlib.sha256(data).hexdigest()
        fixture["reference_exact"] = fixture["reference_sha256"] == fixture["sha256"]
    save(run / "fixtures.json", fixtures)
    return fixtures


def probe(binary, state, fixture, run, label, mode="read", bad=False):
    env = os.environ.copy()
    env.update(
        CUCASCADE_TEST_ACCESS_KEY=ACCESS,
        CUCASCADE_TEST_SECRET_KEY="invalid-local-secret" if bad else SECRET,
    )
    env.update(no_proxy="localhost,127.0.0.1,::1", NO_PROXY="localhost,127.0.0.1,::1")
    uri = "s3://" + state["bucket"] + "/" + fixture["key"]
    cmd = [str(binary), state["endpoint"], uri, fixture["file"], fixture["etag"], mode]
    try:
        result = subprocess.run(
            cmd, env=env, capture_output=True, text=True, timeout=45
        )
        (run / f"{label}.stdout").write_text(result.stdout)
        (run / f"{label}.stderr").write_text(result.stderr)
        try:
            payload = json.loads(result.stdout)
        except (ValueError, TypeError):
            payload = None
        return dict(
            case=label,
            path="cuCascade",
            status=(
                "PASS"
                if result.returncode == 0
                and isinstance(payload, dict)
                and payload.get("status") == "PASS"
                else "FAIL"
            ),
            exit=result.returncode,
            result=payload,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        return dict(
            case=label,
            path="cuCascade",
            status="FAIL",
            error=(
                "45 second subprocess timeout"
                if isinstance(exc, subprocess.TimeoutExpired)
                else str(exc)
            ),
        )


def protocol_probes(state):
    s3 = client(state)
    bucket = state["bucket"]
    key = "conditions/versioned.bin"
    tag = s3.put_object(Bucket=bucket, Key=key, Body=b"version-one")["ETag"]
    positive = (
        read_body(s3.get_object(Bucket=bucket, Key=key, IfMatch=tag)) == b"version-one"
    )
    newtag = s3.put_object(Bucket=bucket, Key=key, Body=b"version-two")["ETag"]
    cases = [
        dict(
            case="matching-if-match",
            path="boto3-service-probe",
            status="PASS" if positive and tag != newtag else "FAIL",
        )
    ]
    for name, kwargs, expected in [
        ("stale-if-match", dict(IfMatch=tag), 412),
        ("unsatisfiable-range", dict(Range="bytes=999-1000"), 416),
    ]:
        try:
            response = s3.get_object(Bucket=bucket, Key=key, **kwargs)
            read_body(response)
            actual = response["ResponseMetadata"]["HTTPStatusCode"]
        except ClientError as error:
            actual = error.response["ResponseMetadata"]["HTTPStatusCode"]
        cases.append(
            dict(
                case=name,
                path="boto3-service-probe",
                expected=expected,
                actual=actual,
                status=(
                    "PASS"
                    if actual == expected
                    else "UNSUPPORTED" if actual in (200, 206) else "FAIL"
                ),
            )
        )
    cases.append(
        dict(
            case="client-if-match",
            path="cuCascade",
            status="UNSUPPORTED",
            reason="public ordinary read API has no If-Match control; capturing ETag does not enforce it",
        )
    )
    return cases


def test_direct(args):
    run = args.run_dir
    state = state_for(run)
    fixtures = load(run / "fixtures.json")
    cases = []
    test_status = {
        "complete": False,
        "binary": str(args.binary),
        "binary_sha256": sha(args.binary),
    }
    save(run / "test-status.json", test_status)
    save(run / "results.json", cases)
    for i, fixture in enumerate(fixtures):
        if not fixture["reference_exact"]:
            # Client/service combination changed identity; attribution is unverified.
            cases.append(
                dict(
                    case=f"object-{i}",
                    key=fixture["key"],
                    path="boto3-service-probe",
                    status="UNSUPPORTED" if fixture["key"] == "keys/a//b" else "FAIL",
                    reason="read-after-all-puts differs; client/service key identity mismatch (attribution unverified)",
                )
            )
        else:
            cases.append(probe(args.binary, state, fixture, run, f"object-{i}"))
        save(run / "results.json", cases)
    missing = dict(fixtures[2], key="does-not-exist")
    cases.append(probe(args.binary, state, missing, run, "missing", mode="404"))
    cases.append(
        probe(args.binary, state, fixtures[2], run, "bad-secret", mode="403", bad=True)
    )
    save(run / "results.json", cases)
    cases.extend(protocol_probes(state))
    save(run / "results.json", cases)
    test_status["complete"] = True
    save(run / "test-status.json", test_status)
    if any(c["status"] == "FAIL" for c in cases):
        raise RuntimeError(
            "direct checks failed; inspect results.json and per-case logs"
        )
    return cases


def report(run):
    state = state_for(run)
    cases = load(run / "results.json") if (run / "results.json").exists() else []
    complete = (run / "test-status.json").exists() and load(run / "test-status.json")[
        "complete"
    ]
    success = (
        complete is True
        and bool(cases)
        and not (run / "failure.json").exists()
        and all(c["status"] in ("PASS", "UNSUPPORTED", "NOT_TESTED") for c in cases)
    )
    result = dict(
        run_dir=str(run),
        status=(
            "PASS_WITH_LIMITATIONS"
            if success and any(c["status"] != "PASS" for c in cases)
            else "PASS" if success else "FAIL"
        ),
        stopped=state["stopped"],
        cases=cases,
    )
    save(run / "report.json", result)
    lines = [
        "# Direct S3 run",
        "",
        f"Status: {result['status']}; service stopped: {state['stopped']}",
        "",
        "| Case | Path | Status |",
        "|---|---|---|",
    ]
    lines.extend(f"| {c['case']} | {c['path']} | {c['status']} |" for c in cases)
    lines += [
        "",
        "See results.json for exact errors and capability limits. This is host-byte correctness, not GPU/GDS/performance or AIStore qualification.",
    ]
    (run / "report.md").write_text("\n".join(lines) + "\n")
    return result


@protect_cleanup
def cleanup_report(run):
    down(run)
    report(run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["preflight", "up", "seed", "test-direct", "report", "down", "all"],
    )
    parser.add_argument("--run-dir", type=lambda x: Path(x).resolve(), required=True)
    parser.add_argument("--weed", type=Path, default=Path("weed"))
    parser.add_argument("--weed-sha256", default=WEED_SHA)
    parser.add_argument(
        "--binary",
        type=lambda x: Path(x).resolve(),
        default=Path("build/p0-io-cudf/test/cucascade_direct_s3_smoke").resolve(),
    )
    parser.add_argument("--port-base", type=int, default=19330)
    args = parser.parse_args()

    def interrupted(_signum, _frame):
        raise KeyboardInterrupt("terminated; cleaning owned service")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        if args.command == "all":
            if args.run_dir.exists():
                raise FileExistsError(
                    "all requires a new run directory; refusing to touch existing run"
                )
            try:
                up(args)
                seed(args.run_dir)
                test_direct(args)
            except BaseException as error:
                if (args.run_dir / "state.json").exists():
                    save(
                        args.run_dir / "failure.json",
                        dict(error=type(error).__name__, message=str(error)),
                    )
                raise
            finally:
                if (args.run_dir / "state.json").exists():
                    cleanup_report(args.run_dir)
        elif args.command == "up":
            up(args)
        elif args.command == "preflight":
            print(json.dumps(preflight(args), indent=2))
        elif args.command == "seed":
            seed(args.run_dir)
        elif args.command == "test-direct":
            test_direct(args)
        elif args.command == "report":
            result = report(args.run_dir)
            print(json.dumps(result, indent=2))
            return int(result["status"] == "FAIL")
        else:
            down(args.run_dir)
        if args.command == "all":
            return int(load(args.run_dir / "report.json")["status"] == "FAIL")
        return 0
    except Exception as error:
        print(f"harness: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
