# SPDX-License-Identifier: Apache-2.0
"""Rejudge AIS controls from required HTTP, fixture and counter receipts."""
import hashlib
from urllib.parse import urlsplit


def require(value, message):
    if not value:
        raise ValueError(message)


def http_response(value, state):
    chain = value["chain"]
    require(bool(chain), "empty HTTP response chain")
    require(
        value["status"] == chain[-1]["status"]
        and value["headers"] == chain[-1]["headers"],
        "HTTP response summary/terminal mismatch",
    )
    for index, response in enumerate(chain):
        url = urlsplit(response["url"])
        require(
            url.scheme == "http"
            and url.hostname == "127.0.0.1"
            and url.port in state["ports"]
            and not url.username
            and not url.password
            and not url.fragment,
            "HTTP response outside owned AIS endpoints",
        )
        if index < len(chain) - 1:
            require(
                response["status"] in (301, 302, 307, 308),
                "invalid HTTP redirect chain",
            )
    return chain[-1]


def ais_controls(load, prefix, summaries, state):
    """Return raw-derived statuses/count reader; missing required files raise."""
    bucket = state["bucket"]
    fixtures = load(prefix + "origin/fixtures.json")
    fixture = next(f for f in fixtures if f["key"] == "ordinary/65553.bin")
    require(
        fixture["generator"] == "shake256-v2" and fixture["size"] == 65553,
        "unexpected native fixture",
    )
    expected = hashlib.shake_256(
        b"cucascade-fixture-v2\0" + fixture["key"].encode()
    ).digest(fixture["size"])
    require(
        hashlib.sha256(expected).hexdigest() == fixture["sha256"],
        "native fixture hash mismatch",
    )
    native = f"/v1/objects/{bucket}/{fixture['key']}?provider=aws"
    s3path = f"/{bucket}/{fixture['key']}"

    def response(label, path=None, status=None, body=None):
        value = load(prefix + label + ".json")
        terminal = http_response(value, state)
        if path:
            require(
                value["chain"][0]["url"] == state["endpoint"] + path
                or value["chain"][0]["url"] == state["target_endpoint"] + path,
                label + ": request path mismatch",
            )
        if status is not None:
            require(terminal["status"] == status, label + ": HTTP status mismatch")
        if body is not None:
            require(
                terminal["length"] == len(body)
                and terminal["sha256"] == hashlib.sha256(body).hexdigest(),
                label + ": body length/hash mismatch",
            )
        return terminal

    def count(label):
        response(label, "/v1/daemon?what=node_stats", 200)
        tracker = load(prefix + label + "-payload.json")["tracker"]
        if "aws.get.n" not in tracker:
            response(label + "-names", "/v1/daemon?what=metrics", 200)
            registry = load(prefix + label + "-names-payload.json")
            require(
                registry.get("aws.get.n") == "counter",
                "absent counter lacks metric registry",
            )
            return 0
        value = tracker["aws.get.n"]
        if isinstance(value, dict):
            value = value["value"]
        require(type(value) is int and value >= 0, "invalid aws.get.n counter")
        return value

    response("bucket-head", f"/v1/buckets/{bucket}?provider=aws", 200)
    before = response("presence-before", native + "&presence=3&silent=true")["status"]
    after = response("presence-after", native + "&presence=3&silent=true")["status"]
    response("native-cold", native, 200, expected)
    response("native-warm", native, 200, expected)
    require(before == 404 and after == 200, "native cold/warm presence mismatch")
    a, b, c = count("stats-before"), count("stats-cold"), count("stats-warm")
    require(b > a and b == c, "native backend GET counters do not prove cold/warm")
    require(
        summaries["A1-native-cold"]["presence_before"] == before
        and summaries["A1-native-cold"]["presence_after"] == after,
        "native presence summary mismatch",
    )
    require(
        (
            summaries["A3-origin-get-count"]["before"],
            summaries["A3-origin-get-count"]["after_cold"],
            summaries["A3-origin-get-count"]["after_warm"],
        )
        == (a, b, c),
        "native counter summary mismatch",
    )
    front = response("s3-head-unfollowed", s3path)
    response("s3-get-followed", s3path, 200, expected)
    require(
        summaries["A2-s3-reference"]["initial_head_status"] == front["status"]
        and summaries["A2-s3-reference"]["redirects"]
        == load(prefix + "s3-get-followed.json")["chain"],
        "S3 reference summary mismatch",
    )
    statuses = {
        key: "PASS"
        for key in (
            "A1-native-cold",
            "A3-native-presence-and-warm",
            "A3-origin-get-count",
            "A2-s3-reference",
        )
    }
    for label, path, status, body in (
        ("missing", s3path + ".missing", 404, None),
        ("invalid-range", s3path, 416, None),
        ("valid-range", s3path, 206, expected[7:39]),
        ("stale-if-match", s3path, None, None),
    ):
        raw = response("s3-" + label, path, status, body)
        key = "F-s3-" + label
        require(
            summaries[key]["actual"] == raw["status"], key + ": status summary mismatch"
        )
        statuses[key] = "PASS"
        if label == "stale-if-match":
            require(raw["status"] in (200, 412), "unexpected stale validator response")
            statuses[key] = "UNSUPPORTED" if raw["status"] == 200 else "PASS"
    version_path = f"/v1/objects/{bucket}/qualification/version.bin?provider=aws"
    response("fresh-v1", version_path, 200, b"p0-version-one")
    response("fresh-ordinary", version_path, 200, b"p0-version-one")
    response(
        "fresh-latest", version_path + "&latest-ver=true", 200, b"p0-version-two-longer"
    )
    fresh = summaries["A4-freshness"]
    require(
        fresh["ordinary_version"] == "v1"
        and fresh["latest_version"] == "v2"
        and fresh["etag_v1"] != fresh["etag_v2"],
        "freshness summary mismatch",
    )
    statuses["A4-freshness"] = "PASS"
    response("origin-offline-warm", native, 200, expected)
    origin_state = load(prefix + "origin/state.json")
    require(origin_state["stopped"], "origin was not stopped")
    statuses["F-origin-offline-warm"] = "PASS"
    stopped = response("target-stopped", native)
    require(
        stopped["status"] >= 400 and state["nodes"]["target"]["stopped"],
        "target-stopped control did not fail after shutdown",
    )
    require(
        summaries["F-target-stopped"].get("actual") == stopped["status"],
        "target-stopped summary mismatch",
    )
    statuses["F-target-stopped"] = "PASS"
    # Disabled client authentication is a limit, not an HTTP success control.
    require(
        summaries["F-ais-client-auth"]["status"] == "NOT_TESTED",
        "unproven client authentication",
    )
    statuses["F-ais-client-auth"] = "NOT_TESTED"
    for key, status in statuses.items():
        require(
            summaries[key]["status"] == status,
            key + ": raw verdict disagrees with summary",
        )
    return statuses, count, response
