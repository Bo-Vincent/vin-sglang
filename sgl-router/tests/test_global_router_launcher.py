import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/launch_global_router.py"
FAKE = '''#!/usr/bin/env python3
import http.server, os, signal, subprocess, sys, time
from pathlib import Path
args = sys.argv[1:]
with open(os.environ["SPAWNS"], "a") as f:
    f.write(str(os.getpid()) + " " + Path(sys.argv[0]).name + "\\n")
if "--exit" in args: sys.exit(0)
if "--hang" in args: time.sleep(60)
if "--ignore-term" in args: signal.signal(signal.SIGTERM, signal.SIG_IGN)
signal.signal(signal.SIGUSR1, lambda *args: sys.exit(0))
if "--grandchild" in args:
    subprocess.Popen([sys.executable, "-c", r"""
import os, signal, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
with open(os.environ["SPAWNS"], "a") as f:
    f.write(str(os.getpid()) + " grandchild\\n")
time.sleep(60)
"""])
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b"ok")
    def log_message(self, *args): pass
class Server(http.server.HTTPServer):
    allow_reuse_address = True
Server(("127.0.0.1", int(args[args.index("--port") + 1])), Handler).serve_forever()
'''


class LauncherTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=SCRIPT.parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.spawns = self.root / "spawns"
        for name in ("sgl-router", "sgl-global-router"):
            path = self.root / name
            path.write_text(FAKE)
            path.chmod(0o755)
        # Reserve all case ports together; releasing individual ephemeral binds
        # can select the same port twice or let another case claim it.
        self.reservations = [socket.socket() for _ in range(3)]
        for sock in self.reservations:
            sock.bind(("127.0.0.1", 0))
            self.addCleanup(sock.close)
        self.ports = [sock.getsockname()[1] for sock in self.reservations]
        self.config = {
            "global": {"host": "127.0.0.1", "port": self.ports[0]},
            "models": [{"name": "example/model", "port": self.ports[1], "args": []}],
        }

    def launch(self, timeout="10", release_ports=True):
        if release_ports:
            for sock in self.reservations:
                sock.close()
        config = self.root / "config.json"
        config.write_text(json.dumps(self.config))
        proc = subprocess.Popen(
            [
                sys.executable,
                str(SCRIPT),
                "--config",
                str(config),
                "--bin-dir",
                str(self.root),
                "--startup-timeout",
                timeout,
                "--shutdown-timeout",
                "0.2",
            ],
            env={**os.environ, "SPAWNS": str(self.spawns)},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        def cleanup():
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
            proc.communicate(timeout=15)

        self.addCleanup(cleanup)
        return proc

    def wait_ready(self, proc):
        deadline = time.monotonic() + 15
        address = "http://127.0.0.1:" + str(self.config["global"]["port"]) + "/readyz"
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                self.fail(
                    "fleet exited before readiness: " + proc.communicate()[1].decode()
                )
            try:
                with urllib.request.urlopen(address, timeout=0.2) as response:
                    self.assertEqual(response.read(), b"ok")
                return
            except (OSError, urllib.error.URLError):
                pass
            time.sleep(0.02)
        proc.send_signal(signal.SIGTERM)
        self.fail(
            "fleet readiness timed out: " + proc.communicate(timeout=15)[1].decode()
        )

    def assert_failed(self, expected=None, timeout="10", release_ports=True):
        proc = self.launch(timeout, release_ports)
        _, err = proc.communicate(timeout=15)
        self.assertNotEqual(proc.returncode, 0, err.decode())
        if expected is not None:
            self.assertIn(expected, err.decode())
        self.last_error = err.decode()

    def test_invalid_config_never_spawns(self):
        for arg in (
            "--host=0.0.0.0",
            "--model-id",
            "--port",
            "-h",
            "-V",
            "--",
            "--help",
        ):
            with self.subTest(arg=arg):
                self.config["models"][0]["args"] = [arg]
                self.assert_failed()
                self.assertFalse(self.spawns.exists())

    def test_duplicate_and_unknown_fields(self):
        self.config["models"].append(dict(self.config["models"][0]))
        self.assert_failed()
        self.assertFalse(self.spawns.exists())
        self.config["models"].pop()
        self.config["surprise"] = True
        self.assert_failed()
        self.assertFalse(self.spawns.exists())

    def test_exit_zero_is_failure(self):
        self.config["models"][0]["args"] = ["--exit"]
        self.assert_failed("managed router exited unexpectedly")
        self.assertTrue(self.spawns.exists(), self.last_error)
        self.assertNotIn("sgl-global-router", self.spawns.read_text())

    def test_startup_timeout(self):
        self.config["models"][0]["args"] = ["--hang"]
        self.assert_failed("router readiness deadline exceeded", timeout="1")

    def test_spawn_failure_cleans_started_children(self):
        (self.root / "sgl-global-router").unlink()
        self.assert_failed()
        self.assert_children_dead()

    def assert_children_dead(self):
        self.assertTrue(
            self.spawns.exists(), getattr(self, "last_error", "no child spawn records")
        )
        for line in self.spawns.read_text().splitlines():
            with self.assertRaises(ProcessLookupError):
                os.kill(int(line.split()[0]), 0)

    def test_ready_then_sigterm_cleans_fleet(self):
        self.config["models"][0]["args"] = ["--ignore-term"]
        proc = self.launch()
        self.wait_ready(proc)
        proc.send_signal(signal.SIGTERM)
        _, err = proc.communicate(timeout=15)
        self.assertEqual(proc.returncode, 0, err.decode())
        self.assert_children_dead()

    def test_shutdown_kills_term_ignoring_grandchild(self):
        self.config["models"][0]["args"] = ["--grandchild", "--ignore-term"]
        proc = self.launch()
        self.wait_ready(proc)
        deadline = time.monotonic() + 10
        while "grandchild" not in self.spawns.read_text():
            self.assertLess(time.monotonic(), deadline, "grandchild did not initialize")
            time.sleep(0.02)
        proc.send_signal(signal.SIGTERM)
        _, err = proc.communicate(timeout=15)
        self.assertEqual(proc.returncode, 0, err.decode())
        # Orphan reaping can lag process-group SIGKILL briefly.
        deadline = time.monotonic() + 5
        while True:
            try:
                self.assert_children_dead()
                break
            except AssertionError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.02)

    def test_occupied_port_rejects_without_spawn(self):
        self.reservations[1].listen()
        self.assert_failed("OSError", release_ports=False)
        self.assertFalse(self.spawns.exists())

    def test_two_models_and_sigint(self):
        self.config["models"].append({"name": "second", "port": self.ports[2]})
        proc = self.launch()
        self.wait_ready(proc)
        self.assertEqual(len(self.spawns.read_text().splitlines()), 3)
        proc.send_signal(signal.SIGINT)
        _, err = proc.communicate(timeout=15)
        self.assertEqual(proc.returncode, 0, err.decode())
        self.assert_children_dead()

    def check_runtime_child_exit(self, exit_signal):
        self.config["models"].append({"name": "second", "port": self.ports[2]})
        proc = self.launch()
        self.wait_ready(proc)
        os.kill(int(self.spawns.read_text().splitlines()[0].split()[0]), exit_signal)
        proc.communicate(timeout=15)
        self.assertNotEqual(proc.returncode, 0)
        self.assert_children_dead()

    def test_runtime_child_exit_zero_cleans_fleet(self):
        self.check_runtime_child_exit(signal.SIGUSR1)

    def test_runtime_child_exit_nonzero_cleans_fleet(self):
        self.check_runtime_child_exit(signal.SIGTERM)

    def test_immediate_restart_same_ports(self):
        for _ in range(2):
            proc = self.launch()
            self.wait_ready(proc)
            proc.send_signal(signal.SIGTERM)
            _, err = proc.communicate(timeout=15)
            self.assertEqual(proc.returncode, 0, err.decode())
            self.assert_children_dead()
            self.spawns.unlink()


if __name__ == "__main__":
    unittest.main()
