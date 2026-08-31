# SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "swebench_pro_replay.py"


def load_replay():
    spec = importlib.util.spec_from_file_location("swebench_pro_replay", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import replay module from {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def row(*, instance_id: str, repo: str = "acme/project"):
    return {
        "instance_id": instance_id,
        "repo": repo,
        "base_commit": "abc123",
        "problem_statement": "Fix the parser crash.",
        "requirements": "Keep the public API stable.",
        "interface": "src/parser.py",
        "patch": "SECRET_GOLD_PATCH",
        "test_patch": "SECRET_TEST_PATCH",
        "fail_to_pass": "SECRET_FAILURES",
        "pass_to_pass": "SECRET_REGRESSIONS",
    }


class SWEbenchProReplayTest(unittest.TestCase):
    def test_parse_and_prompt_use_public_request_fields_only(self):
        replay = load_replay()

        task = replay.parse_task(row(instance_id="task-1"), source_index=0)
        prompt = replay.build_task_prompt(task)

        self.assertIn("acme/project", prompt)
        self.assertIn("Fix the parser crash.", prompt)
        self.assertNotIn("SECRET_GOLD_PATCH", prompt)
        self.assertNotIn("SECRET_TEST_PATCH", prompt)
        self.assertNotIn("SECRET_FAILURES", prompt)
        self.assertNotIn("SECRET_REGRESSIONS", prompt)

    def test_selection_preserves_dataset_order_and_has_no_shared_commit_assumption(self):
        replay = load_replay()

        tasks = replay.select_tasks(
            [
                row(instance_id="task-2", repo="acme/second"),
                row(instance_id="task-1", repo="acme/first"),
            ]
        )

        self.assertEqual([task.instance_id for task in tasks], ["task-2", "task-1"])
        self.assertEqual(len({(task.repo, task.base_commit) for task in tasks}), 2)

    def test_manifest_records_prompt_contract_without_solution_fields(self):
        replay = load_replay()
        tasks = replay.select_tasks([row(instance_id="task-1")])
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "dataset.jsonl"
            replay.write_dataset_cache(cache, [row(instance_id="task-1")])
            manifest = replay.write_replay_manifest(Path(directory) / "manifest.json", tasks, cache)
            self.assertNotIn("SECRET_GOLD_PATCH", cache.read_text())

        encoded = json.dumps(manifest, sort_keys=True)
        self.assertEqual(manifest["task_count"], 1)
        self.assertIn("problem_statement", manifest["prompt_fields"])
        self.assertNotIn("patch", manifest["prompt_fields"])
        self.assertNotIn("SECRET_GOLD_PATCH", encoded)

    def test_page_fetch_stops_at_requested_limit(self):
        replay = load_replay()
        calls = []

        def fetch_page(offset, length):
            calls.append((offset, length))
            return [row(instance_id=f"task-{index}") for index in range(offset, offset + length)]

        rows = replay.fetch_rows(fetch_page, task_limit=3, page_size=2)

        self.assertEqual([item["instance_id"] for item in rows], ["task-0", "task-1", "task-2"])
        self.assertEqual(calls, [(0, 2), (2, 1)])


if __name__ == "__main__":
    unittest.main()
