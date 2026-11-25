#!/usr/bin/env python3
"""Run one managed sgl-router per model and a single sgl-global-router.

JSON schema (unknown keys are rejected):
{"global": {"host": "0.0.0.0", "port": 30000, "default_model": "model-a"},
 "models": [{"name": "model-a", "port": 30001,
             "args": ["--worker-urls", "http://127.0.0.1:31000"]}]}
The global object and all its fields are optional; host/port default as above.
models is required and nonempty; name/port are required, args defaults to [].
Names cannot contain '=' or controls or surrounding whitespace. Ports are unique.
args contains raw argv strings, never shell text. Managed host, port, model-id,
help and version arguments are reserved. Workers must already be running.
"""

import argparse
import ipaddress
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import unicodedata
import urllib.error
import urllib.request


def object_keys(value, allowed, label):
    if not isinstance(value, dict) or set(value) - allowed:
        raise ValueError("invalid fields in " + label)


def port(value):
    if type(value) is not int or not 1 <= value <= 65535:
        raise ValueError("ports must be integers from 1 to 65535")
    return value


def read_config(path):
    with open(path, encoding="utf-8") as source:
        config = json.load(source)
    object_keys(config, {"global", "models"}, "config")
    gateway = config.get("global", {})
    object_keys(gateway, {"host", "port", "default_model"}, "global")
    host = gateway.get("host", "0.0.0.0")
    if not isinstance(host, str):
        raise ValueError("global host must be an IP address")
    ipaddress.ip_address(host)
    gateway = {**gateway, "host": host, "port": port(gateway.get("port", 30000))}
    models = config.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("models must be a nonempty array")
    names, ports = set(), {gateway["port"]}
    for model in models:
        object_keys(model, {"name", "port", "args"}, "model")
        name = model.get("name")
        if (
            not isinstance(name, str)
            or not name
            or name.strip() != name
            or "=" in name
            or any(unicodedata.category(c) == "Cc" for c in name)
        ):
            raise ValueError("invalid model name")
        number = port(model.get("port"))
        if name in names or number in ports:
            raise ValueError("duplicate model name or port")
        names.add(name)
        ports.add(number)
        args = model.setdefault("args", [])
        if not isinstance(args, list) or any(
            not isinstance(arg, str) or "\0" in arg for arg in args
        ):
            raise ValueError("model args must be an array of argv strings")
        for arg in args:
            option = arg.split("=", 1)[0]
            if option in {
                "--host",
                "--port",
                "--model-id",
                "--help",
                "--version",
                "--",
            } or arg.startswith(("-h", "-V")):
                raise ValueError("model args contain a reserved argument")
    if "default_model" in gateway and (
        not isinstance(gateway["default_model"], str)
        or gateway["default_model"] not in names
    ):
        raise ValueError("default_model must name a configured model")
    return gateway, models


def check_ports(gateway, models):
    bindings = [(gateway["host"], gateway["port"])] + [
        ("127.0.0.1", model["port"]) for model in models
    ]
    sockets = []
    try:
        for host, number in bindings:
            sock = socket.socket(socket.AF_INET6 if ":" in host else socket.AF_INET)
            sockets.append(sock)
            # Match server bind semantics so TIME_WAIT permits a quick restart.
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, number))
    finally:
        for sock in sockets:
            sock.close()


def positive(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("timeout must be positive and finite")
    return result


def stop_children(children, timeout):
    # Groups may outlive their leaders; always signal the entire managed group.
    for child in reversed(children):
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except PermissionError:
            # macOS can report EPERM for an already reaped process group.
            if child.poll() is None:
                raise
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and any(
        child.poll() is None for child in children
    ):
        time.sleep(0.02)
    for child in children:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            if child.poll() is None:
                raise
    for child in children:
        child.wait()


def run(args, gateway, models):
    children = []
    stopping = False

    def request_stop(signum, frame):
        nonlocal stopping
        stopping = True

    previous = {
        sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)
    }
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def check_children():
        if any(child.poll() is not None for child in children):
            raise RuntimeError("managed router exited unexpectedly")

    def spawn(binary, argv):
        if stopping:
            return
        children.append(
            subprocess.Popen(
                [str(args.bin_dir / binary), *argv], start_new_session=True
            )
        )

    def await_ready(addresses):
        deadline = time.monotonic() + args.startup_timeout
        pending = set(addresses)
        while pending and not stopping:
            check_children()
            if time.monotonic() >= deadline:
                raise RuntimeError("router readiness deadline exceeded")
            for address in list(pending):
                try:
                    with opener.open(
                        address + "/readyz",
                        timeout=min(0.2, max(0.001, deadline - time.monotonic())),
                    ) as response:
                        if response.status == 200:
                            pending.remove(address)
                except (OSError, urllib.error.URLError):
                    pass
            time.sleep(0.02)
        check_children()

    try:
        for model in models:
            spawn(
                "sgl-router",
                [
                    *model["args"],
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(model["port"]),
                    "--model-id",
                    model["name"],
                ],
            )
        await_ready(["http://127.0.0.1:" + str(model["port"]) for model in models])
        argv = ["--host", gateway["host"], "--port", str(gateway["port"])]
        for model in models:
            argv += [
                "--model-route",
                model["name"] + "=http://127.0.0.1:" + str(model["port"]),
            ]
        if "default_model" in gateway:
            argv += ["--default-model", gateway["default_model"]]
        spawn("sgl-global-router", argv)
        address = ipaddress.ip_address(gateway["host"])
        host = (
            ("::1" if address.version == 6 else "127.0.0.1")
            if address.is_unspecified
            else str(address)
        )
        if address.version == 6:
            host = "[" + host + "]"
        await_ready(["http://" + host + ":" + str(gateway["port"])])
        while not stopping:
            check_children()
            time.sleep(0.05)
        return 0
    finally:
        stop_children(children, args.shutdown_timeout)
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--bin-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "target/release",
    )
    parser.add_argument("--startup-timeout", type=positive, default=120)
    parser.add_argument("--shutdown-timeout", type=positive, default=10)
    args = parser.parse_args()
    try:
        gateway, models = read_config(args.config)
        check_ports(gateway, models)
        return run(args, gateway, models)
    except (ValueError, OSError, RuntimeError) as error:
        # Do not echo argv or config values, which can contain credentials.
        detail = (
            str(error)
            if isinstance(error, (ValueError, RuntimeError))
            else type(error).__name__
        )
        print("launcher failed: " + detail, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
