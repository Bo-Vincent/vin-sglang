#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
# SPDX-License-Identifier: Apache-2.0

"""下载并重建 SWE-bench Pro 的公开任务 prompt-shape replay。

该 replay 只使用公开 task 的仓库、基线版本、issue、requirements 和 interface。
它不读取 patch、test_patch 或测试结果，因此衡量的是 Router 对公开 agent 请求形状的
相对调度趋势，而不是 agent 的解题能力或真实 trajectory。
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Callable, Iterable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen


DATASET_ID = "ScaleAI/SWE-bench_Pro"
DATASET_CONFIG = "default"
DATASET_SPLIT = "test"
DATASET_SERVER_ROWS_URL = "https://datasets-server.huggingface.co/rows"
PROMPT_FIELDS = (
    "repo",
    "base_commit",
    "problem_statement",
    "requirements",
    "interface",
)
DEFAULT_AGENT_CONTEXT_REPETITIONS = 20

_AGENT_INSTRUCTIONS = """You are a software-engineering agent. Inspect the repository before editing.
Use the terminal to search, read files, modify source, and run focused tests. Preserve public interfaces
unless the task explicitly changes them. Explain the diagnosis, implement the smallest complete fix, and
validate the affected behavior. Do not invent files, commands, test results, or dependencies."""
_TOOL_SCHEMA = """Tool: terminal(command, timeout). Tool: search(query, path). Tool: read(path, start, end).
Tool: edit(path, patch). Tool: test(command). Each tool call must use repository-local paths and report
the observed result before deciding the next action."""


@dataclass(frozen=True)
class SWEbenchProTask:
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    requirements: str
    interface: str
    source_index: int


def _required_text(row: Mapping[str, object], field: str, source_index: int) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"row {source_index}: {field} must be a non-empty string")
    return value


def _optional_text(row: Mapping[str, object], field: str) -> str:
    value = row.get(field)
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def parse_task(row: Mapping[str, object], *, source_index: int) -> SWEbenchProTask:
    """从原始公开 dataset row 提取可以进入请求的字段。"""
    return SWEbenchProTask(
        instance_id=_required_text(row, "instance_id", source_index),
        repo=_required_text(row, "repo", source_index),
        base_commit=_required_text(row, "base_commit", source_index),
        problem_statement=_required_text(row, "problem_statement", source_index),
        requirements=_optional_text(row, "requirements"),
        interface=_optional_text(row, "interface"),
        source_index=source_index,
    )


def select_tasks(
    rows: Iterable[Mapping[str, object]], *, task_limit: int | None = None
) -> tuple[SWEbenchProTask, ...]:
    """保留官方 test split 的原始顺序，不假设 repo/base_commit 可共享。"""
    if task_limit is not None and task_limit <= 0:
        raise ValueError("task_limit must be positive when set")
    tasks: list[SWEbenchProTask] = []
    seen_ids: set[str] = set()
    for source_index, row in enumerate(rows):
        task = parse_task(row, source_index=source_index)
        if task.instance_id in seen_ids:
            raise ValueError(f"duplicate instance_id: {task.instance_id}")
        seen_ids.add(task.instance_id)
        tasks.append(task)
        if task_limit is not None and len(tasks) >= task_limit:
            break
    if not tasks:
        raise ValueError("SWE-bench Pro selection is empty")
    return tuple(tasks)


def agent_prefix(repo: str, *, context_repetitions: int = DEFAULT_AGENT_CONTEXT_REPETITIONS) -> str:
    if context_repetitions <= 0:
        raise ValueError("context_repetitions must be positive")
    return "\n\n".join(
        (
            _AGENT_INSTRUCTIONS,
            f"Repository: {repo}",
            *(_TOOL_SCHEMA for _ in range(context_repetitions)),
        )
    )


def build_task_prompt(
    task: SWEbenchProTask, *, context_repetitions: int = DEFAULT_AGENT_CONTEXT_REPETITIONS
) -> str:
    """构造共享 agent/tool 前缀 + task 私有 suffix 的单请求 replay prompt。"""
    return "\n\n".join(
        (
            agent_prefix(task.repo, context_repetitions=context_repetitions),
            f"Base commit: {task.base_commit}",
            "Issue:\n" + task.problem_statement,
            "Requirements:\n" + task.requirements,
            "Interface:\n" + task.interface,
            "Return the implementation plan and the patch summary.",
        )
    )


def fetch_rows(
    fetch_page: Callable[[int, int], Sequence[Mapping[str, object]]],
    *,
    task_limit: int | None = None,
    page_size: int = 100,
) -> list[Mapping[str, object]]:
    """通过可注入分页函数拉取连续 task rows，便于离线测试。"""
    if task_limit is not None and task_limit <= 0:
        raise ValueError("task_limit must be positive when set")
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    rows: list[Mapping[str, object]] = []
    offset = 0
    while task_limit is None or len(rows) < task_limit:
        length = page_size if task_limit is None else min(page_size, task_limit - len(rows))
        page = list(fetch_page(offset, length))
        if len(page) > length:
            raise RuntimeError("dataset server returned more rows than requested")
        rows.extend(page)
        offset += len(page)
        if len(page) < length:
            break
    return rows


def _dataset_page(offset: int, length: int, *, timeout_seconds: float) -> list[Mapping[str, object]]:
    query = urlencode(
        {
            "dataset": DATASET_ID,
            "config": DATASET_CONFIG,
            "split": DATASET_SPLIT,
            "offset": offset,
            "length": length,
        }
    )
    url = f"{DATASET_SERVER_ROWS_URL}?{query}"
    for attempt in range(2):
        try:
            with urlopen(url, timeout=timeout_seconds) as response:
                payload = json.loads(response.read())
            raw_rows = payload.get("rows") if isinstance(payload, dict) else None
            if not isinstance(raw_rows, list):
                raise RuntimeError("dataset server response has no rows array")
            rows: list[Mapping[str, object]] = []
            for item in raw_rows:
                row = item.get("row") if isinstance(item, dict) else None
                if not isinstance(row, dict):
                    raise RuntimeError("dataset server row is not an object")
                rows.append(row)
            return rows
        except HTTPError as error:
            if error.code == 429 and attempt == 0:
                time.sleep(20)
                continue
            if 500 <= error.code < 600 and attempt == 0:
                time.sleep(2)
                continue
            raise RuntimeError(f"SWE-bench Pro dataset request failed: HTTP {error.code}") from error
        except URLError as error:
            if attempt == 0:
                time.sleep(2)
                continue
            raise RuntimeError(f"SWE-bench Pro dataset request failed: {error.reason}") from error
    raise AssertionError("unreachable")


def fetch_public_test_rows(
    *, task_limit: int | None = None, page_size: int = 100, timeout_seconds: float = 30.0
) -> list[Mapping[str, object]]:
    return fetch_rows(
        lambda offset, length: _dataset_page(offset, length, timeout_seconds=timeout_seconds),
        task_limit=task_limit,
        page_size=page_size,
    )


def write_dataset_cache(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    """只持久化 replay 所需的公开输入字段，不落盘 gold patch 或测试答案。"""
    records = []
    for source_index, row in enumerate(rows):
        task = parse_task(row, source_index=source_index)
        records.append(
            {
                "instance_id": task.instance_id,
                "repo": task.repo,
                "base_commit": task.base_commit,
                "problem_statement": task.problem_statement,
                "requirements": task.requirements,
                "interface": task.interface,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records))
    temporary.replace(path)


def load_dataset_cache(path: Path) -> list[Mapping[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"SWE-bench Pro dataset cache does not exist: {path}")
    rows: list[Mapping[str, object]] = []
    for line_number, raw in enumerate(path.read_text().splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError(f"dataset cache line {line_number}: invalid JSON") from error
        if not isinstance(row, dict):
            raise ValueError(f"dataset cache line {line_number}: row must be an object")
        rows.append(row)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def build_replay_manifest(
    tasks: Sequence[SWEbenchProTask],
    dataset_cache: Path,
    *,
    context_repetitions: int = DEFAULT_AGENT_CONTEXT_REPETITIONS,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "workload": "swebench_pro_prompt_shape_replay",
        "measurement_kind": "simulator_predicted_relative",
        "dataset": {"id": DATASET_ID, "config": DATASET_CONFIG, "split": DATASET_SPLIT},
        "dataset_cache_sha256": sha256_file(dataset_cache),
        "prompt_fields": list(PROMPT_FIELDS),
        "agent_context_repetitions": context_repetitions,
        "task_count": len(tasks),
        "tasks": [asdict(task) for task in tasks],
    }


def write_replay_manifest(
    path: Path,
    tasks: Sequence[SWEbenchProTask],
    dataset_cache: Path,
    *,
    context_repetitions: int = DEFAULT_AGENT_CONTEXT_REPETITIONS,
) -> dict[str, object]:
    payload = build_replay_manifest(
        tasks, dataset_cache, context_repetitions=context_repetitions
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-cache", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--task-limit", type=int)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--agent-context-repetitions", type=int, default=DEFAULT_AGENT_CONTEXT_REPETITIONS)
    args = parser.parse_args(argv)
    if args.timeout_seconds <= 0.0:
        parser.error("--timeout-seconds must be positive")
    try:
        if args.refresh or not args.dataset_cache.exists():
            rows = fetch_public_test_rows(
                task_limit=args.task_limit,
                page_size=args.page_size,
                timeout_seconds=args.timeout_seconds,
            )
            write_dataset_cache(args.dataset_cache, rows)
        tasks = select_tasks(load_dataset_cache(args.dataset_cache), task_limit=args.task_limit)
        payload = write_replay_manifest(
            args.manifest,
            tasks,
            args.dataset_cache,
            context_repetitions=args.agent_context_repetitions,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({"manifest": str(args.manifest), "task_count": payload["task_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
