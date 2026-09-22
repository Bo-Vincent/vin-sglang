"""GPU serving contracts for persistent input pins and output KV retention.

This fixture caps the actual KV allocator, so cache-pressure assertions exercise
eviction rather than relying on a particular GPU's free memory. Run locally with
``python test/registered/radix_cache/test_input_pin_output_policy.py``.
Set SGLANG_KV_CONTROL_PAGE_SIZE=16 or SGLANG_KV_CONTROL_TP_SIZE=2 to exercise
paged allocation or tensor parallelism; SGLANG_KV_CONTROL_MODEL selects weights.
"""

import json
import os
import time
import unittest

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=240, stage="base-b", runner_config="1-gpu-small")


class CacheControlServer(CustomTestCase):
    feature_args = []
    capacity = 4096
    page_size = int(os.environ.get("SGLANG_KV_CONTROL_PAGE_SIZE", "1"))
    tp_size = int(os.environ.get("SGLANG_KV_CONTROL_TP_SIZE", "1"))

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            os.environ.get(
                "SGLANG_KV_CONTROL_MODEL", DEFAULT_SMALL_MODEL_NAME_FOR_TEST
            ),
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--max-total-tokens",
                str(cls.capacity),
                "--context-length",
                "4096",
                "--page-size",
                str(cls.page_size),
                "--tp-size",
                str(cls.tp_size),
                "--chunked-prefill-size",
                "128",
                "--disable-cuda-graph",
                "--enable-metrics",
                "--decode-log-interval",
                "1",
            ]
            + cls.feature_args,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            terminate_and_kill_process_tree(cls.process)

    def post(self, path, body):
        response = requests.post(self.base_url + path, json=body, timeout=120)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def flush(self):
        response = requests.post(
            self.base_url + "/flush_cache", params={"timeout": 10}, timeout=20
        )
        self.assertEqual(response.status_code, 200, response.text)

    def policy(self, disabled):
        result = self.post(
            "/set_internal_state",
            {"server_args": {"disable_output_kv_cache": disabled}},
        )
        self.assertEqual(result, [True], result)

    def aligned(self, count):
        return count // self.page_size * self.page_size

    def retained_tokens(self, prompt_length, output_length, disabled):
        if disabled:
            return self.aligned(prompt_length)
        response = requests.get(self.base_url + "/server_info", timeout=10)
        self.assertEqual(response.status_code, 200, response.text)
        # Overlap launches the next decode before consuming the result that
        # finishes this request, committing KV for the final returned token.
        # A serial scheduler has KV only through the penultimate output token.
        serial = response.json()["disable_overlap_schedule"]
        return self.aligned(prompt_length + output_length - int(serial))

    def load(self):
        response = requests.get(self.base_url + "/v1/loads", timeout=10)
        self.assertEqual(response.status_code, 200, response.text)
        loads = response.json()["loads"]
        self.assertEqual(len(loads), 1)
        return loads[0]

    def generate(self, ids, count=32, pin=False):
        result = self.post(
            "/generate",
            {
                "input_ids": ids,
                "pin_input_kv": pin,
                "sampling_params": {
                    "max_new_tokens": count,
                    "temperature": 0,
                    "ignore_eos": True,
                },
            },
        )
        self.assertEqual(result["meta_info"]["completion_tokens"], count)
        self.assertEqual(len(result["output_ids"]), count)
        return result

    def assert_free_slots(self, expected):
        # Idle metrics refresh at most every 30 seconds, including after flush.
        deadline = time.monotonic() + 40
        observed = None
        while time.monotonic() < deadline:
            response = requests.get(self.base_url + "/metrics", timeout=10)
            self.assertEqual(response.status_code, 200, response.text)
            values = [
                float(line.rsplit(" ", 1)[1])
                for line in response.text.splitlines()
                if line.startswith("sglang:kv_available_tokens{")
            ]
            if values:
                self.assertEqual(len(values), 1, "Fixture must have one KV pool")
                observed = values[0]
                if observed == expected:
                    return
            time.sleep(0.1)
        self.fail(f"Expected {expected} free KV slots, observed {observed}")


class TestInputPinOutputPolicy(CacheControlServer):
    feature_args = ["--enable-input-kv-pin", "--disable-output-kv-cache"]

    def setUp(self):
        self.policy(True)
        self.flush()

    def test_pin_survives_eviction_and_flush_reclaims_it(self):
        """Dropping persistent ownership on finish/split loses the cold prefix."""
        pinned = [101] * 257
        ordinary = [102] * 257
        self.generate(pinned, pin=True)
        self.generate(ordinary)
        # Re-pin and split an existing node without growing its permanent hold.
        self.generate(pinned[:128], pin=True)
        for _ in range(3):
            self.generate(pinned, pin=True)
        pinned_slots = self.aligned(len(pinned))
        if len(pinned) % self.page_size:
            pinned_slots += self.page_size
        self.assert_free_slots(
            self.capacity - pinned_slots - self.aligned(len(ordinary))
        )

        # Disjoint prefixes exceed the entire pool twice. The ordinary prefix
        # is a positive control proving that actual eviction has taken place.
        for token in range(200, 212):
            self.generate([token] * 769, count=1)
        cold = self.generate(ordinary, count=1)
        self.assertEqual(cold["meta_info"]["cached_tokens"], 0)
        # Non-evictable allocation includes the private input tail page even
        # though radix prefix matching only returns complete pages.
        self.assertEqual(self.load()["num_used_tokens"], pinned_slots)
        hot = self.generate(pinned + [103], count=1)
        self.assertEqual(hot["meta_info"]["cached_tokens"], self.aligned(len(pinned)))

        self.flush()
        self.assert_free_slots(self.capacity)
        cold_again = self.generate(pinned, count=1)
        self.assertEqual(cold_again["meta_info"]["cached_tokens"], 0)

    def test_output_slots_and_cache_hits_follow_policy(self):
        """Restricting insertion without freeing output leaks allocator slots."""
        prompt = [301] * 257
        for disabled in (True, False, True):
            with self.subTest(disable_output_kv_cache=disabled):
                self.policy(disabled)
                self.flush()
                generated = self.generate(prompt, count=64)
                retained = self.retained_tokens(len(prompt), 64, disabled)
                self.assert_free_slots(self.capacity - retained)
                probe = prompt + generated["output_ids"] + [302]
                reused = self.generate(probe, count=1)
                self.assertEqual(reused["meta_info"]["cached_tokens"], retained)

    def test_dynamic_update_preserves_inflight_and_historical_policy(self):
        """Reading a mutable default at finish changes already accepted requests."""
        for old_disabled in (False, True):
            with self.subTest(old_disable_output_kv_cache=old_disabled):
                self.flush()
                self.policy(old_disabled)
                prompt = [401] * 257
                with requests.post(
                    self.base_url + "/generate",
                    json={
                        "input_ids": prompt,
                        "stream": True,
                        "sampling_params": {
                            "max_new_tokens": 768,
                            "temperature": 0,
                            "ignore_eos": True,
                        },
                    },
                    stream=True,
                    timeout=120,
                ) as response:
                    self.assertEqual(response.status_code, 200)
                    chunks = response.iter_lines(chunk_size=1)
                    for line in chunks:
                        if line.startswith(b"data: "):
                            first = json.loads(line[6:])
                            break
                    else:
                        self.fail("No generation event received")
                    self.assertLess(first["meta_info"]["completion_tokens"], 768)
                    self.post("/pause_generation", {"mode": "in_place"})
                    try:
                        self.policy(not old_disabled)
                        # The control reply orders behind the pause in the
                        # scheduler. Require a live request while it is paused
                        # so a fast completion cannot yield a vacuous pass.
                        self.assertEqual(self.load()["num_running_reqs"], 1)
                    finally:
                        self.post("/continue_generation", {})
                    final = first
                    for line in chunks:
                        if line.startswith(b"data: ") and line[6:] != b"[DONE]":
                            final = json.loads(line[6:])
                self.assertEqual(final["meta_info"]["completion_tokens"], 768)
                retained = self.retained_tokens(len(prompt), 768, old_disabled)
                self.assert_free_slots(self.capacity - retained)
                # New requests must use the new default without deleting the
                # previous request's cached output when the policy turns on.
                new_prompt = [402] * 129
                new = self.generate(new_prompt, count=32)
                new_retained = self.retained_tokens(
                    len(new_prompt), 32, not old_disabled
                )
                self.assert_free_slots(self.capacity - retained - new_retained)
                old_probe = self.generate(prompt + final["output_ids"] + [403], 1)
                self.assertEqual(old_probe["meta_info"]["cached_tokens"], retained)
                new_probe = self.generate(new_prompt + new["output_ids"] + [404], 1)
                self.assertEqual(new_probe["meta_info"]["cached_tokens"], new_retained)


class TestCacheControlDefaults(CacheControlServer):
    def test_disabled_pin_rejects_and_default_retains_output(self):
        """Request opt-in cannot silently succeed when the server disallows pin."""
        self.flush()
        prompt = [501] * 129
        response = requests.post(
            self.base_url + "/generate",
            json={
                "input_ids": prompt,
                "pin_input_kv": True,
                "sampling_params": {"max_new_tokens": 1},
            },
            timeout=30,
        )
        self.assertEqual(response.status_code, 400, response.text)
        generated = self.generate(prompt, count=32)
        retained = self.retained_tokens(len(prompt), 32, False)
        self.assert_free_slots(self.capacity - retained)
        reused = self.generate(prompt + generated["output_ids"] + [502], count=1)
        self.assertEqual(reused["meta_info"]["cached_tokens"], retained)


if __name__ == "__main__":
    unittest.main()
