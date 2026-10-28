#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Local wire observations for the actual cuCascade H probe (not a client emulator)."""
import argparse
import contextlib
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import ssl
import signal
import subprocess
import threading
import time
from urllib.parse import parse_qs, urlsplit
from xml.sax.saxutils import escape

DATA = bytes((i * 19 + 7) % 251 for i in range(16384))
ETAG = '"h-fixture-version-1"'


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Server(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False

    def get_request(self):
        sock, address = super().get_request()
        with self.owner.lock:
            self.owner.accepts += 1
            self.connections[sock] = self.owner.accepts
        return sock, address

    def close_request(self, request):
        with self.owner.lock:
            self.connections.pop(request, None)
        super().close_request(request)


class Fixture:
    def __init__(self, case, mode, tls=None):
        self.case, self.mode = case, mode
        self.lock = threading.Lock()
        self.wire, self.accepts, self.ordinal = [], 0, 0
        self.gate = threading.Event()
        self.gate_observation = None
        self.servers, self.threads, self.endpoints = [], [], []
        for role in ("origin", "target-0", "target-1", "target-2", "canary"):
            server = Server(("127.0.0.1", 0), Handler)
            server.owner, server.role, server.connections = self, role, {}
            scheme = "http"
            if role == "origin" and tls:
                ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
                ctx.load_cert_chain(str(tls[0]), str(tls[1]))
                server.socket = ctx.wrap_socket(server.socket, server_side=True)
                scheme = "https"
            self.endpoints.append(f"{scheme}://127.0.0.1:{server.server_port}")
            self.servers.append(server)
        for server in self.servers:
            thread = threading.Thread(
                target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
            )
            thread.start()
            self.threads.append(thread)

    def close(self):
        self.gate.set()
        for server in self.servers:
            server.shutdown()
            with self.lock:
                sockets = list(server.connections)
            for sock in sockets:
                with contextlib.suppress(OSError):
                    sock.shutdown(2)
                    sock.close()
            server.server_close()
        for thread in self.threads:
            thread.join(2)

    def release(self, event):
        with self.lock:
            # Requests may have reached the handler; no response is sent before this.
            self.gate_observation = dict(
                submitted=event["count"],
                responses_before_release=sum(
                    "response" in w for w in self.wire if w["method"] == "GET"
                ),
                time=time.monotonic(),
            )
        self.gate.set()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_HEAD(self):
        self.handle_request()

    def do_GET(self):
        self.handle_request()

    def handle_request(self):
        owner, role = self.server.owner, self.server.role
        with owner.lock:
            row = dict(
                sequence=len(owner.wire),
                role=role,
                connection=self.server.connections[self.connection],
                method=self.command,
                path=self.path,
                range=self.headers.get("Range"),
                if_match=self.headers.get("If-Match"),
                credential_headers=[
                    k.lower()
                    for k in self.headers
                    if k.lower()
                    in (
                        "authorization",
                        "x-amz-date",
                        "x-amz-security-token",
                        "x-amz-content-sha256",
                    )
                ],
                time=time.monotonic(),
            )
            owner.wire.append(row)
        if owner.mode == "queue" and self.command == "GET":
            if not owner.gate.wait(10):
                self.respond(row, 504, b"queue submission gate timeout")
                return
        query = parse_qs(urlsplit(self.path).query)
        if "list-type" in query:
            second = query.get("continuation-token") == ["page2"]
            items = (
                [("empty", ""), ("missing", None)]
                if not second
                else [("strong", '"v1"')]
            )
            content = "".join(
                "<Contents><Key>"
                + name
                + "</Key><Size>3</Size>"
                + ("" if tag is None else "<ETag>" + escape(tag) + "</ETag>")
                + "</Contents>"
                for name, tag in items
            )
            xml = (
                '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/"><IsTruncated>'
                + ("false" if second else "true")
                + "</IsTruncated>"
                + (
                    ""
                    if second
                    else "<NextContinuationToken>page2</NextContinuationToken>"
                )
                + content
                + "</ListBucketResult>"
            )
            row["list_body"] = xml
            self.respond(row, 200, xml.encode())
            return
        redirect = (
            role == "origin"
            and owner.case != "direct"
            and (self.command == "GET" or owner.mode == "head")
        )
        if role.startswith("target") and owner.case == "second-307":
            self.respond(row, 307, b"", {"Location": owner.endpoints[4] + "/h/object"})
            return
        if redirect:
            with owner.lock:
                n = owner.ordinal
                owner.ordinal += 1
            target = owner.endpoints[1 + (n % 3 if owner.mode == "queue" else 0)]
            location = target + "/s3/h/object?pid=fixture&utm=fixture"
            if owner.case == "nonallowlisted":
                location = owner.endpoints[4] + "/h/object"
            elif owner.case == "userinfo":
                location = target.replace("://", "://fixture-user@") + "/h/object"
            elif owner.case == "fragment":
                location += "#fragment"
            self.respond(row, 307, b"", {"Location": location})
            return
        if role == "canary":
            self.respond(row, 500, b"forbidden endpoint reached")
            return
        data, status, headers = DATA, 200, {"ETag": ETAG, "Accept-Ranges": "bytes"}
        match = self.headers.get("If-Match")
        if match and match != ETAG:
            self.respond(row, 412, b"")
            return
        value = self.headers.get("Range")
        if value:
            bounds = value.removeprefix("bytes=").split("-")
            lo = len(DATA) - int(bounds[1]) if not bounds[0] else int(bounds[0])
            hi = len(DATA) - 1 if not bounds[0] or not bounds[1] else int(bounds[1])
            data, status = DATA[lo : hi + 1], 206
            headers["Content-Range"] = f"bytes {lo}-{hi}/{len(DATA)}"
        self.respond(row, status, data, headers)

    def respond(self, row, status, body, headers=None):
        with self.server.owner.lock:
            row.update(response=status, response_headers=headers or {})
        try:
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            row["peer_closed"] = True


def invoke(
    binary, endpoint, uri, datafile, tag, mode, count, directory, release=None, ca="-"
):
    command = [
        str(binary),
        endpoint,
        uri,
        str(datafile),
        tag,
        mode,
        str(count),
        str(ca),
    ]
    env = os.environ.copy()
    env.update(
        CUCASCADE_TEST_ACCESS_KEY="h-local-key",
        CUCASCADE_TEST_SECRET_KEY="h-local-secret",
        NO_PROXY="127.0.0.1,localhost",
        no_proxy="127.0.0.1,localhost",
    )
    started = time.monotonic()
    process = subprocess.Popen(
        command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    events, lines, parse_errors = [], [], []

    def read_stdout():
        for line in process.stdout:
            lines.append(line)
            try:
                event = json.loads(line)
                events.append(event)
                if release and event.get("event") == "submitted":
                    release(event)
            except (ValueError, KeyError, TypeError) as exc:
                parse_errors.append(str(exc))

    reader = threading.Thread(target=read_stdout)
    reader.start()
    errors = []
    error_reader = threading.Thread(target=lambda: errors.append(process.stderr.read()))
    error_reader.start()
    timeout = False
    try:
        process.wait(timeout=45)
    except subprocess.TimeoutExpired:
        timeout = True
        process.kill()
        process.wait()
    except BaseException:
        process.kill()
        process.wait(timeout=5)
        raise
    finally:
        reader.join(5)
        error_reader.join(5)
        process.stdout.close()
        process.stderr.close()
    (directory / "stdout.jsonl").write_text("".join(lines))
    (directory / "stderr.log").write_text("".join(errors))
    return dict(
        command=command,
        binary_sha256=digest(binary),
        exit=process.returncode,
        timeout=timeout,
        elapsed_seconds=time.monotonic() - started,
        events=events,
        parse_errors=parse_errors,
    )


def run_case(binary, root, name, case, mode, count, tls=None):
    directory = root / name
    directory.mkdir()
    fixture = Fixture(case, mode, tls)
    try:
        result = invoke(
            binary,
            fixture.endpoints[0],
            "h" if mode == "list" else "s3://h/object",
            root / "object.bin",
            ETAG,
            mode,
            count,
            directory,
            fixture.release,
            tls[0] if tls else "-",
        )
    finally:
        fixture.close()
    observation = dict(
        schema=1,
        kind="wire",
        name=name,
        fixture_case=case,
        mode=mode,
        count=count,
        endpoints=fixture.endpoints,
        allowed_targets=fixture.endpoints[1:4],
        worker_count=1,
        max_connections=4 if mode == "queue" else 1,
        policy_controls_available=False,
        probe=result,
        gate=fixture.gate_observation,
        wire=fixture.wire,
        cleanup=True,
    )
    save(directory / "observation.json", observation)
    return observation


def install_signals():
    def interrupted(signum, frame):
        # Repeated termination cannot interrupt child/service cleanup.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)


def main():
    install_signals()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / "object.bin").write_bytes(DATA)
    cert, key = root / "fixture-ca.pem", root / "fixture-key.pem"
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            str(key),
            "-out",
            str(cert),
            "-days",
            "2",
            "-subj",
            "/CN=127.0.0.1",
            "-addext",
            "subjectAltName=IP:127.0.0.1",
            "-addext",
            "basicConstraints=critical,CA:TRUE",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    cases = [
        ("direct-head", "direct", "head", 1),
        ("direct-footer", "direct", "footer", 1),
        ("direct-serial", "direct", "serial", 32),
        ("direct-q64", "direct", "queue", 64),
        ("direct-q256", "direct", "queue", 256),
        ("list-fixture", "direct", "list", 1),
        ("rewrite-head", "rewrite", "head", 1),
        ("rewrite-footer", "rewrite", "footer", 1),
        ("rewrite-range", "rewrite", "serial", 2),
        ("reuse-off", "rewrite", "serial", 2),
        ("retention-expanded", "rewrite", "serial", 32),
        ("retention-legacy", "rewrite", "serial", 32),
        ("stress-q64", "rewrite", "queue", 64),
        ("stress-q256", "rewrite", "queue", 256),
    ]
    cases += [
        (name, name, "serial", 1)
        for name in ("nonallowlisted", "second-307", "userinfo", "fragment")
    ]
    cases += [
        ("tls-direct", "direct", "serial", 1),
        ("downgrade", "downgrade", "serial", 1),
    ]
    for name, case, mode, count in cases:
        row = run_case(
            args.binary.resolve(),
            root,
            name,
            case,
            mode,
            count,
            (cert, key) if name in ("tls-direct", "downgrade") else None,
        )
        print(
            json.dumps(
                dict(case=name, exit=row["probe"]["exit"], requests=len(row["wire"]))
            ),
            flush=True,
        )
    key.unlink()  # Ephemeral fixture server key is not part of the review evidence.
    save(
        root / "manifest.json",
        {
            str(p.relative_to(root)): digest(p)
            for p in sorted(root.rglob("*"))
            if p.is_file() and p.name != "manifest.json"
        },
    )


if __name__ == "__main__":
    main()
