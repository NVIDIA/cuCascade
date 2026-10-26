#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Qualify a pinned, loopback-only AIStore cluster without changing cuCascade."""

import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.parse import quote, urljoin, urlsplit

sys.dont_write_bytecode = True
import harness as origin

AIS_COMMIT = "a3769a15d9a28ad69ec7f9f0785577b0fa2c7af9"
MODES = {"prefix": 16384, "root": 16392, "reverse": 24584}
_LAUNCHING = False
_PENDING_SIGNAL = None


class RunInterrupted(BaseException):
    """Whole-run deadline or user termination; never an expected HTTP error."""


def state_for(run):
    state = origin.load(run / "ais-state.json")
    if state["schema"] != 1 or state["run_dir"] != str(run):
        raise RuntimeError("AIStore run ownership mismatch")
    return state


def stop_node(run, role):
    state = state_for(run)
    node = state["nodes"].get(role)
    if not node:
        return
    pid, start = node["pid"], node["start"]
    if origin.process_identity(pid) is not None:
        if origin.process_identity(pid) != start:
            raise RuntimeError("PID reused; refusing to stop")
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        if f"-local_config={run / role / 'local.json'}".encode() not in cmdline:
            raise RuntimeError("node command does not belong to this run")
        for sig, seconds in ((signal.SIGTERM, 10), (signal.SIGKILL, 3)):
            if origin.process_identity(pid) != start:
                break
            os.kill(pid, sig)
            node.setdefault("signals_sent", []).append(sig.name)
            deadline = time.monotonic() + seconds
            while origin.process_identity(pid) == start and time.monotonic() < deadline:
                time.sleep(0.1)
        if origin.process_identity(pid) == start:
            raise RuntimeError("owned node failed to stop")
    node["stopped"] = True
    origin.save(run / "ais-state.json", state)


def down(run):
    errors = []
    for role in ("target", "proxy"):
        try:
            stop_node(run, role)
        except Exception as exc:
            errors.append(f"{role}: {exc}")
    if (run / "origin/state.json").exists():
        try:
            origin.down(run / "origin")
        except Exception as exc:
            errors.append(f"origin: {exc}")
    state = state_for(run)
    state["cleanup_errors"] = errors
    state["stopped"] = not errors
    origin.save(run / "ais-state.json", state)
    if errors:
        raise RuntimeError("; ".join(errors))
    return state


def request(state, method, path, *, headers=None, body=None, follow=True, target=False):
    url = state["target_endpoint" if target else "endpoint"] + path
    chain = []
    started = time.monotonic()
    for _ in range(4):
        loc = urlsplit(url)
        if (
            loc.scheme != "http"
            or loc.hostname != "127.0.0.1"
            or loc.port not in state["ports"]
            or loc.username
            or loc.password
        ):
            raise RuntimeError("redirect outside owned loopback cluster")
        conn = http.client.HTTPConnection(loc.hostname, loc.port, timeout=5)
        try:
            conn.request(
                method,
                loc.path + ("?" + loc.query if loc.query else ""),
                body=body,
                headers=headers or {},
            )
            response = conn.getresponse()
            data = response.read(4 * 1024 * 1024 + 1)
            if len(data) > 4 * 1024 * 1024:
                raise RuntimeError("response exceeds fixture bound")
            item = dict(
                url=url,
                status=response.status,
                headers=dict(response.getheaders()),
                length=len(data),
                sha256=hashlib.sha256(data).hexdigest(),
            )
            chain.append(item)
            location = response.getheader("Location")
            if (
                follow
                and response.status in (301, 302, 307, 308)
                and location
                and method in ("GET", "HEAD")
            ):
                url = urljoin(url, location)
                continue
            return (
                dict(
                    status=response.status,
                    headers=item["headers"],
                    chain=chain,
                    elapsed_seconds=round(time.monotonic() - started, 3),
                ),
                data,
            )
        finally:
            conn.close()
    raise RuntimeError("redirect hop limit exceeded")


def snapshot(run, label, method, path, **kwargs):
    result, body = request(state_for(run), method, path, **kwargs)
    origin.save(run / f"{label}.json", result)
    if body and result["status"] >= 400:
        (run / f"{label}.body").write_bytes(body)
    return result, body


def stats(run, label):
    result, body = snapshot(
        run, label, "GET", "/v1/daemon?what=node_stats", target=True
    )
    try:
        payload = json.loads(body)
        origin.save(run / f"{label}-payload.json", payload)
        tracker = payload["tracker"]
        if "aws.get.n" not in tracker:
            # Pinned stats/common_prom.go omits zero-valued counters. Require
            # the metric registry to confirm the counter exists before using 0.
            names_result, names_body = snapshot(
                run, label + "-names", "GET", "/v1/daemon?what=metrics", target=True
            )
            names = json.loads(names_body)
            origin.save(run / f"{label}-names-payload.json", names)
            if names_result["status"] == 200 and names.get("aws.get.n") == "counter":
                return 0
            return None
        value = tracker["aws.get.n"]
        return value["value"] if isinstance(value, dict) else value
    except (ValueError, KeyError, TypeError):
        return None


def configure(args, role, ports):
    run = args.run_dir
    directory = run / role
    directory.mkdir()
    (directory / "logs").mkdir()
    # Clean environment: never resolve real AWS credentials, subscriptions or AIS auth.
    env = {"PATH": "/usr/bin:/bin", "HOME": str(directory)}
    env.update(
        AIS_CONF_FILE=str(directory / "ais.json"),
        AIS_LOCAL_CONF_FILE=str(directory / "local.json"),
        AIS_CONF_DIR=str(directory),
        AIS_LOG_DIR=str(directory / "logs"),
        AIS_PRIMARY_URL=f"http://127.0.0.1:{args.ais_port_base}",
        AIS_DISCOVERY_URL=f"http://127.0.0.1:{args.ais_port_base}",
        AIS_FEATURES=str(MODES[args.mode]),
        AIS_STARTUP_TIME="30s",
        AIS_JOIN_STARTUP_TIME="60s",
        HOSTNAME_LIST="127.0.0.1",
        HOSTNAME_LIST_INTRA_CONTROL="127.0.0.1",
        HOSTNAME_LIST_INTRA_DATA="127.0.0.1",
        PORT=str(ports[0]),
        PORT_INTRA_CONTROL=str(ports[1]),
        PORT_INTRA_DATA=str(ports[2]),
        TEST_FSPATH_ROOT=str(run / "mounts"),
        TEST_FSPATH_COUNT="1" if role == "target" else "0",
        INSTANCE="0",
    )
    subprocess.run(
        [
            "bash",
            "-c",
            "make_backend_conf() { printf '%s' '{\"aws\":{}}'; }; "
            'make_tracing_conf() { :; }; source "$1"',
            "config",
            str(args.ais_source / "deploy/dev/local/aisnode_config.sh"),
        ],
        env=env,
        check=True,
        timeout=10,
    )
    return directory


def up(args):
    global _LAUNCHING
    run = args.run_dir
    source = args.ais_source.resolve(strict=True)
    head = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain"], text=True
    )
    if head != AIS_COMMIT or dirty:
        raise RuntimeError("expected clean, pinned AIStore source")
    if origin.sha(args.aisnode) != args.aisnode_sha256:
        raise RuntimeError("aisnode hash mismatch")
    ports = list(range(args.ais_port_base, args.ais_port_base + 6))
    if not 1024 <= ports[0] <= ports[-1] <= 65535:
        raise ValueError("AIS ports outside unprivileged range")
    if set(ports) & set(range(args.port_base, args.port_base + 8)):
        raise ValueError("AIS and origin ports overlap")
    sockets = []
    try:
        for port in ports:
            sock = socket.socket()
            sockets.append(sock)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", port))
            sock.listen(1)
    finally:
        for sock in sockets:
            sock.close()
    run.mkdir(parents=True, exist_ok=False)
    run.chmod(0o700)
    state = dict(
        schema=1,
        run_dir=str(run),
        source=head,
        aisnode=str(args.aisnode),
        aisnode_sha256=args.aisnode_sha256,
        runner_sha256=origin.sha(Path(__file__)),
        origin_runner_sha256=origin.sha(Path(origin.__file__)),
        config_generator_sha256=origin.sha(
            source / "deploy/dev/local/aisnode_config.sh"
        ),
        mode=args.mode,
        features=MODES[args.mode],
        ports=ports,
        endpoint=f"http://127.0.0.1:{ports[0]}",
        target_endpoint=f"http://127.0.0.1:{ports[3]}",
        nodes={},
        stopped=False,
    )
    origin.save(run / "ais-state.json", state)
    try:
        origin_args = argparse.Namespace(**vars(args))
        origin_args.run_dir = run / "origin"
        backing = origin.up(origin_args)
        origin.seed(run / "origin")
        state["bucket"] = backing["bucket"]
        origin.save(run / "ais-state.json", state)
        for role, offset in (("proxy", 0), ("target", 3)):
            directory = configure(args, role, ports[offset : offset + 3])
            env = {
                "PATH": "/usr/bin:/bin",
                "HOME": str(directory),
                "GOMAXPROCS": "2",
                "GOMEMLIMIT": "1GiB",
                "AWS_REGION": "us-east-1",
                "AWS_ACCESS_KEY_ID": origin.ACCESS,
                "AWS_SECRET_ACCESS_KEY": origin.SECRET,
                "AWS_EC2_METADATA_DISABLED": "true",
                "S3_ENDPOINT": backing["endpoint"],
                "AWS_SHARED_CREDENTIALS_FILE": str(directory / "empty-credentials"),
                "AWS_CONFIG_FILE": str(directory / "empty-config"),
            }
            (directory / "empty-credentials").touch()
            (directory / "empty-config").touch()
            cmd = [
                str(args.aisnode),
                f"-config={directory / 'ais.json'}",
                f"-local_config={directory / 'local.json'}",
                f"-role={role}",
                f"-daemon_id=p0{role}001",
            ]
            if role == "proxy":
                cmd.append("-ntargets=1")
            _LAUNCHING = True
            try:
                with (directory / "console.log").open("wb") as log:
                    proc = subprocess.Popen(
                        cmd,
                        env=env,
                        cwd=directory,
                        stdout=log,
                        stderr=log,
                        start_new_session=True,
                    )
                state["nodes"][role] = dict(
                    pid=proc.pid, start=origin.process_identity(proc.pid), command=cmd
                )
                origin.save(run / "ais-state.json", state)
            finally:
                _LAUNCHING = False
                if _PENDING_SIGNAL is not None:
                    raise RunInterrupted(f"interrupted by signal {_PENDING_SIGNAL}")
        deadline = time.monotonic() + args.ready_timeout
        while time.monotonic() < deadline:
            for node in state["nodes"].values():
                if origin.process_identity(node["pid"]) != node["start"]:
                    raise RuntimeError("AIS node exited; inspect console.log")
            try:
                result, body = request(state, "GET", "/v1/daemon?what=smap")
                smap = json.loads(body)
                healthy, _ = request(state, "GET", "/v1/health?readiness=true")
                if (
                    result["status"] == healthy["status"] == 200
                    and len(smap["tmap"]) == len(smap["pmap"]) == 1
                ):
                    origin.save(run / "smap.json", smap)
                    state["ready"] = True
                    origin.save(run / "ais-state.json", state)
                    return state
            except (OSError, ValueError, KeyError, http.client.HTTPException):
                pass
            time.sleep(0.5)
        raise TimeoutError("AIS cluster readiness deadline exceeded")
    except BaseException:
        down(run)
        raise


def qualify(args):
    run = args.run_dir
    state = state_for(run)
    bucket = state["bucket"]
    fixtures = origin.load(run / "origin/fixtures.json")
    fixture = next(f for f in fixtures if f["key"] == "ordinary/65553.bin")
    expected = Path(fixture["file"]).read_bytes()
    native = f"/v1/objects/{bucket}/{quote(fixture['key'], safe='/')}?provider=aws"
    s3path = (
        "/s3" if state["mode"] == "prefix" else ""
    ) + f"/{bucket}/{fixture['key']}"
    rows = []

    def row(name, status, **evidence):
        rows.append(dict(case=name, status=status, **evidence))
        origin.save(run / "results.json", rows)

    registered, _ = snapshot(
        run, "bucket-head", "HEAD", f"/v1/buckets/{bucket}?provider=aws"
    )
    if registered["status"] != 200:
        raise RuntimeError("backing bucket registration failed")
    before, _ = snapshot(
        run, "presence-before", "HEAD", native + "&presence=3&silent=true"
    )
    count0 = stats(run, "stats-before")
    cold, body = snapshot(run, "native-cold", "GET", native)
    after, _ = snapshot(
        run, "presence-after", "HEAD", native + "&presence=3&silent=true"
    )
    count1 = stats(run, "stats-cold")
    row(
        "A1-native-cold",
        "PASS" if cold["status"] == 200 and body == expected else "FAIL",
        presence_before=before["status"],
        presence_after=after["status"],
    )
    warm, body = snapshot(run, "native-warm", "GET", native)
    count2 = stats(run, "stats-warm")
    row(
        "A3-native-presence-and-warm",
        (
            "PASS"
            if before["status"] == 404
            and after["status"] == 200
            and warm["status"] == 200
            and body == expected
            else "FAIL"
        ),
    )
    known = all(isinstance(n, int) for n in (count0, count1, count2))
    row(
        "A3-origin-get-count",
        (
            ("PASS" if count1 > count0 and count2 == count1 else "FAIL")
            if known
            else "UNKNOWN"
        ),
        before=count0,
        after_cold=count1,
        after_warm=count2,
    )
    front, _ = snapshot(run, "s3-head-unfollowed", "HEAD", s3path, follow=False)
    ref, body = snapshot(run, "s3-get-followed", "GET", s3path)
    row(
        "A2-s3-reference",
        "PASS" if ref["status"] == 200 and body == expected else "FAIL",
        initial_head_status=front["status"],
        redirects=ref["chain"],
    )
    ctx = dict(
        origin.state_for(run / "origin"),
        endpoint=state["endpoint"] + ("/s3" if state["mode"] == "prefix" else ""),
    )
    # A separate fixture stays cold until the actual C++ probe starts.
    library_fixture = next(f for f in fixtures if f["key"] == "ordinary/1048607.bin")
    library_path = f"/v1/objects/{bucket}/{library_fixture['key']}?provider=aws&presence=3&silent=true"
    library_before, _ = snapshot(run, "library-presence-before", "HEAD", library_path)
    library_count0 = stats(run, "library-stats-before")
    result = origin.probe(args.binary, ctx, library_fixture, run, "cucascade-ais")
    library_after, _ = snapshot(run, "library-presence-after", "HEAD", library_path)
    library_count1 = stats(run, "library-stats-after")
    # Only known, directly evidenced compatibility failures are UNSUPPORTED.
    error = (
        (run / "cucascade-ais.stderr").read_text()
        if (run / "cucascade-ais.stderr").exists()
        else ""
    )
    unsupported = (
        state["mode"] == "prefix"
        and "endpoint must be scheme://host[:port] only" in error
    ) or (front["status"] == 307 and "307" in error)
    row(
        "A2-cucascade",
        (
            "UNSUPPORTED"
            if result["status"] == "FAIL" and unsupported
            else result["status"]
        ),
        probe=result,
        error=error[:2000],
        presence_before=library_before["status"],
        presence_after=library_after["status"],
        origin_get_before=library_count0,
        origin_get_after=library_count1,
    )
    for label, path, headers, expected_status in (
        ("missing", s3path + ".missing", {}, 404),
        ("invalid-range", s3path, {"Range": "bytes=999999999-"}, 416),
        ("stale-if-match", s3path, {"If-Match": '"definitely-stale"'}, 412),
        ("valid-range", s3path, {"Range": "bytes=7-38"}, 206),
    ):
        result, body = snapshot(run, "s3-" + label, "GET", path, headers=headers)
        ok = result["status"] == expected_status
        if label == "valid-range":
            ok = ok and body == expected[7:39]
        status = (
            "PASS"
            if ok
            else (
                "UNSUPPORTED"
                if label == "stale-if-match" and result["status"] == 200
                else "FAIL"
            )
        )
        row("F-s3-" + label, status, actual=result["status"], expected=expected_status)
    # Authentication is disabled on this isolated AIS deployment. Do not claim IAM coverage.
    row(
        "F-ais-client-auth",
        "UNSUPPORTED",
        reason="client_auth_required=false; origin 403 covered separately",
    )
    backing = origin.state_for(run / "origin")
    s3 = origin.client(backing)
    version_key = "qualification/version.bin"
    v1, v2 = b"p0-version-one", b"p0-version-two-longer"
    tag1 = s3.put_object(Bucket=bucket, Key=version_key, Body=v1)["ETag"]
    version_path = f"/v1/objects/{bucket}/{version_key}?provider=aws"
    first, first_body = snapshot(run, "fresh-v1", "GET", version_path)
    tag2 = s3.put_object(Bucket=bucket, Key=version_key, Body=v2)["ETag"]
    ordinary, ordinary_body = snapshot(run, "fresh-ordinary", "GET", version_path)
    latest, latest_body = snapshot(
        run, "fresh-latest", "GET", version_path + "&latest-ver=true"
    )
    s3.close()
    row(
        "A4-freshness",
        (
            "PASS"
            if first["status"] == ordinary["status"] == latest["status"] == 200
            and first_body == v1
            and ordinary_body == v1
            and latest_body == v2
            and tag1 != tag2
            else "FAIL"
        ),
        policy="validate_warm_get=false; explicit latest-ver=true refresh",
        etag_v1=tag1,
        etag_v2=tag2,
        ordinary_version="v1" if ordinary_body == v1 else "other",
        latest_version="v2" if latest_body == v2 else "other",
    )
    # Demonstrate a resident read still succeeds with the origin stopped.
    origin.down(run / "origin")
    offline, offline_body = snapshot(run, "origin-offline-warm", "GET", native)
    row(
        "F-origin-offline-warm",
        "PASS" if offline["status"] == 200 and offline_body == expected else "FAIL",
    )
    stop_node(run, "target")
    started = time.monotonic()
    try:
        failed, _ = snapshot(run, "target-stopped", "GET", native)
        row(
            "F-target-stopped",
            "PASS" if failed["status"] >= 400 else "FAIL",
            actual=failed["status"],
        )
    except (OSError, http.client.HTTPException) as exc:
        row(
            "F-target-stopped",
            "PASS",
            error=type(exc).__name__,
            elapsed_seconds=round(time.monotonic() - started, 3),
        )
    origin.save(
        run / "summary.json",
        dict(
            status=(
                "FAIL"
                if any(r["status"] == "FAIL" for r in rows)
                else "PASS_WITH_LIMITATIONS"
            ),
            counts={
                s: sum(r["status"] == s for r in rows)
                for s in ("PASS", "FAIL", "UNSUPPORTED", "UNKNOWN")
            },
        ),
    )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("up", "down", "qualify", "all"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--ais-source", type=Path)
    parser.add_argument("--aisnode", type=Path)
    parser.add_argument("--aisnode-sha256")
    parser.add_argument("--weed", type=Path)
    parser.add_argument("--weed-sha256", default=origin.WEED_SHA)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--port-base", type=int, default=19330)
    parser.add_argument("--ais-port-base", type=int, default=19430)
    parser.add_argument("--mode", choices=MODES, default="root")
    parser.add_argument("--ready-timeout", type=float, default=75)
    parser.add_argument("--deadline", type=int, default=300)
    args = parser.parse_args()
    args.run_dir = args.run_dir.resolve()
    if args.command in ("up", "all") and not all(
        (args.ais_source, args.aisnode, args.aisnode_sha256, args.weed, args.binary)
    ):
        parser.error(
            "up/all require ais-source, aisnode, aisnode-sha256, weed and binary"
        )
    if args.command == "qualify" and not args.binary:
        parser.error("qualify requires binary")
    if args.deadline <= 0 or args.ready_timeout <= 0:
        parser.error("deadlines must be positive")

    def interrupted(signum, frame):
        global _PENDING_SIGNAL
        if _LAUNCHING:
            _PENDING_SIGNAL = signum
            return
        raise RunInterrupted(f"interrupted by signal {signum}")

    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGALRM, interrupted)
    signal.alarm(args.deadline)
    owns_run = not args.run_dir.exists()
    try:
        if args.command == "down":
            down(args.run_dir)
        elif args.command == "up":
            up(args)
        elif args.command == "qualify":
            qualify(args)
        else:
            up(args)
            qualify(args)
        if (args.run_dir / "summary.json").exists():
            summary = origin.load(args.run_dir / "summary.json")
            print(json.dumps(summary))
            return int(summary["status"] == "FAIL")
        return 0
    except (Exception, KeyboardInterrupt, RunInterrupted) as exc:
        if owns_run and (args.run_dir / "ais-state.json").exists():
            origin.save(
                args.run_dir / "error.json",
                dict(type=type(exc).__name__, message=str(exc)),
            )
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        signal.alarm(0)
        # Cleanup has its own per-process bounds; repeated termination cannot skip it.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if (
            args.command == "all"
            and owns_run
            and (args.run_dir / "ais-state.json").exists()
        ):
            down(args.run_dir)


if __name__ == "__main__":
    sys.exit(main())
