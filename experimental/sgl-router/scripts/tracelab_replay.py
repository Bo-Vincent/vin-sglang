#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
# SPDX-License-Identifier: Apache-2.0

"""TraceLab 的可重放 session slice 选择。

公开 TraceLab 不包含 prompt 文本或 KV block hash。此模块只处理其公开 token
分解和 session 顺序，后续 driver 据此重建确定性的 session-local prompt 链。
"""

from __future__ import annotations

import argparse
import hashlib
import gzip
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence


@dataclass(frozen=True)
class TraceRound:
    session_id: str
    round_index: int
    input_tokens: int
    prefix_tokens: int
    newly_append_tokens: int
    output_tokens: int
    source_line: int


@dataclass(frozen=True)
class ReplayRound:
    session_id: str
    round_index: int
    input_tokens: int
    prefix_tokens: int
    newly_append_tokens: int
    output_tokens: int
    source_line: int
    is_warmup: bool


@dataclass(frozen=True)
class ReplaySelectionConfig:
    session_count: int
    rounds_per_session: int
    seed: int
    min_input_tokens: int
    max_input_tokens: int
    min_prefix_tokens: int
    max_append_tokens: int
    provider: str = "codex"

    def __post_init__(self) -> None:
        if self.session_count <= 0:
            raise ValueError("session_count must be positive")
        if self.rounds_per_session < 2:
            raise ValueError("rounds_per_session must be at least two")
        if self.min_input_tokens < 0 or self.max_input_tokens < self.min_input_tokens:
            raise ValueError("invalid input token range")
        if self.min_prefix_tokens < 0 or self.max_append_tokens < 0:
            raise ValueError("invalid prefix or append token limit")
        if not self.provider:
            raise ValueError("provider must be non-empty")


@dataclass(frozen=True)
class VirtualTokenPlan:
    """一次 replay slice 内无冲突的 session/round 虚拟 token 分配。"""

    prefix_token_ids: Mapping[str, int]
    append_token_ids: Mapping[tuple[str, int, int], int]


def load_trace_rows(path: Path) -> Iterator[Mapping[str, object]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as source:
        for source_line, raw in enumerate(source, start=1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as error:
                raise ValueError(f"line {source_line}: invalid JSON") from error
            if not isinstance(row, dict):
                raise ValueError(f"line {source_line}: TraceLab row must be an object")
            yield row


def _required_int(row: Mapping[str, object], name: str, source_line: int) -> int:
    value = row.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"line {source_line}: {name} must be a non-negative integer")
    return value


def parse_trace_row(row: Mapping[str, object], source_line: int) -> TraceRound:
    session_id = row.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise ValueError(f"line {source_line}: session_id must be a non-empty string")
    round_index = _required_int(row, "round_index", source_line)
    input_tokens = _required_int(row, "input_tokens_total", source_line)
    prefix_tokens = _required_int(row, "prefix_tokens", source_line)
    newly_append_tokens = _required_int(row, "newly_append_tokens", source_line)
    output_tokens = _required_int(row, "output_tokens", source_line)
    if input_tokens != prefix_tokens + newly_append_tokens:
        raise ValueError(
            f"line {source_line}: input_tokens_total must equal "
            "prefix_tokens + newly_append_tokens"
        )
    return TraceRound(
        session_id=session_id,
        round_index=round_index,
        input_tokens=input_tokens,
        prefix_tokens=prefix_tokens,
        newly_append_tokens=newly_append_tokens,
        output_tokens=output_tokens,
        source_line=source_line,
    )


def _input_eligible(round_: TraceRound, config: ReplaySelectionConfig) -> bool:
    return (
        config.min_input_tokens <= round_.input_tokens <= config.max_input_tokens
        and round_.newly_append_tokens <= config.max_append_tokens
    )


def _first_contiguous_window(
    rounds: Sequence[TraceRound], config: ReplaySelectionConfig
) -> tuple[TraceRound, ...] | None:
    if len(rounds) < config.rounds_per_session:
        return None
    ordered = sorted(rounds, key=lambda round_: round_.round_index)
    for start in range(len(ordered) - config.rounds_per_session + 1):
        window = tuple(ordered[start : start + config.rounds_per_session])
        contiguous = all(
            later.round_index == earlier.round_index + 1
            for earlier, later in zip(window, window[1:])
        )
        measurements_have_prefix = all(
            round_.prefix_tokens >= config.min_prefix_tokens for round_ in window[1:]
        )
        if contiguous and measurements_have_prefix:
            return window
    return None


def _session_order(session_id: str, seed: int) -> bytes:
    return hashlib.sha256(f"{seed}:{session_id}".encode()).digest()


def select_replay_slice(
    rows: Iterable[Mapping[str, object]], config: ReplaySelectionConfig
) -> tuple[ReplayRound, ...]:
    by_session: dict[str, list[TraceRound]] = {}
    for source_line, row in enumerate(rows, start=1):
        provider = row.get("provider")
        if provider != config.provider:
            continue
        round_ = parse_trace_row(row, source_line)
        if _input_eligible(round_, config):
            by_session.setdefault(round_.session_id, []).append(round_)

    candidates: list[tuple[str, tuple[TraceRound, ...]]] = []
    for session_id, rounds in by_session.items():
        window = _first_contiguous_window(rounds, config)
        if window is not None:
            candidates.append((session_id, window))
    candidates.sort(key=lambda item: (_session_order(item[0], config.seed), item[0]))
    if len(candidates) < config.session_count:
        raise ValueError(
            f"requested {config.session_count} sessions but only {len(candidates)} qualify"
        )

    selected: list[ReplayRound] = []
    for _, window in candidates[: config.session_count]:
        selected.extend(
            ReplayRound(
                session_id=round_.session_id,
                round_index=round_.round_index,
                input_tokens=round_.input_tokens,
                prefix_tokens=round_.prefix_tokens,
                newly_append_tokens=round_.newly_append_tokens,
                output_tokens=round_.output_tokens,
                source_line=round_.source_line,
                is_warmup=index == 0,
            )
            for index, round_ in enumerate(window)
        )
    return tuple(selected)


def find_reversible_token_ids(tokenizer: object, minimum: int) -> tuple[int, ...]:
    """寻找重复后仍可逐 token 往返的普通文本 token。

    不能直接截断任意 BPE token stream：prefix 与 append 在 decode 后相接时，
    边界可能重新 merge。可逆 token 必须有前导空白、包含可见内容，且单个和
    两个连续 token 都能精确重新编码。Qwen tokenizer 上这类 token 很充足。
    """
    if minimum <= 0:
        raise ValueError("minimum must be positive")
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if not isinstance(vocab_size, int) or vocab_size <= 0:
        raise ValueError("tokenizer must expose a positive vocab_size")

    selected: list[int] = []
    for token_id in range(vocab_size):
        text = tokenizer.decode(
            [token_id],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if not isinstance(text, str) or not text or not text[0].isspace() or not text.strip():
            continue
        if list(tokenizer.encode(text, add_special_tokens=False)) != [token_id]:
            continue
        repeated = tokenizer.decode(
            [token_id, token_id],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if list(tokenizer.encode(repeated, add_special_tokens=False)) != [
            token_id,
            token_id,
        ]:
            continue
        selected.append(token_id)
        if len(selected) >= minimum:
            return tuple(selected)
    raise ValueError(
        f"tokenizer only has {len(selected)} reversible tokens; need {minimum}"
    )


def _stable_token_id(namespace: str, safe_token_ids: Sequence[int]) -> int:
    if len(safe_token_ids) < 2:
        raise ValueError("at least two reversible token ids are required")
    digest = hashlib.sha256(namespace.encode()).digest()
    return safe_token_ids[int.from_bytes(digest[:8], "big") % len(safe_token_ids)]


def build_virtual_token_plan(
    rounds: Sequence[ReplayRound], *, safe_token_ids: Sequence[int]
) -> VirtualTokenPlan:
    """为固定 replay slice 预分配不同的 session prefix 和 round append token。

    不能只把 session hash 到有限 token pool：128 个 session 即使 pool 有 1024
    个元素也可能碰撞，进而制造公开 TraceLab 中不存在的跨 session cache hit。
    固定 slice 已知时，直接按排序后的 key 注入分配，消除这类碰撞。
    """
    session_ids = sorted({round_.session_id for round_ in rounds})
    append_keys = sorted(
        {
            (round_.session_id, round_.round_index, round_.source_line)
            for round_ in rounds
        }
    )
    required = len(session_ids) + len(append_keys)
    if len(safe_token_ids) < required:
        raise ValueError(
            f"need {required} reversible token ids for this replay slice; "
            f"only have {len(safe_token_ids)}"
        )
    prefix_token_ids = {
        session_id: safe_token_ids[index]
        for index, session_id in enumerate(session_ids)
    }
    append_token_ids = {
        key: safe_token_ids[len(session_ids) + index]
        for index, key in enumerate(append_keys)
    }
    return VirtualTokenPlan(prefix_token_ids, append_token_ids)


def reconstruct_virtual_prompt(
    round_: ReplayRound,
    tokenizer: object,
    *,
    safe_token_ids: Sequence[int] | None = None,
    token_plan: VirtualTokenPlan | None = None,
) -> str:
    """把一个公开 token geometry 重建为明确标注的虚拟 prompt。

    返回值的裸 tokenizer token 数严格等于 ``input_tokens``。实际 chat
    request 还会有 role/template framing，因此 driver 必须另行记录 Router
    实测 token 数，不能把此处数字伪报为线上请求 token 数。
    """
    if round_.input_tokens != round_.prefix_tokens + round_.newly_append_tokens:
        raise ValueError("round token decomposition is invalid")

    if token_plan is not None:
        try:
            prefix_id = token_plan.prefix_token_ids[round_.session_id]
            append_id = token_plan.append_token_ids[
                (round_.session_id, round_.round_index, round_.source_line)
            ]
        except KeyError as error:
            raise ValueError("round is not present in the virtual token plan") from error
    else:
        if safe_token_ids is None:
            safe_token_ids = find_reversible_token_ids(tokenizer, minimum=1024)
        prefix_id = _stable_token_id(
            f"tracelab-prefix:{round_.session_id}", safe_token_ids
        )
        append_id = _stable_token_id(
            f"tracelab-append:{round_.session_id}:{round_.round_index}:{round_.source_line}",
            safe_token_ids,
        )
        if append_id == prefix_id:
            append_id = safe_token_ids[
                (safe_token_ids.index(append_id) + 1) % len(safe_token_ids)
            ]
    token_ids = [prefix_id] * round_.prefix_tokens + [append_id] * round_.newly_append_tokens
    prompt = tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    roundtrip = list(tokenizer.encode(prompt, add_special_tokens=False))
    if roundtrip != token_ids:
        raise ValueError(
            "tokenizer cannot round-trip the generated virtual prompt exactly"
        )
    return prompt


def write_replay_manifest(
    path: Path,
    rounds: Sequence[ReplayRound],
    config: ReplaySelectionConfig,
    *,
    trace_sha256: str,
) -> dict[str, object]:
    if not trace_sha256:
        raise ValueError("trace_sha256 must be non-empty")
    payload: dict[str, object] = {
        "schema_version": 1,
        "trace_sha256": trace_sha256,
        "selection": asdict(config),
        "rounds": [asdict(round_) for round_ in rounds],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Select a deterministic session-local TraceLab replay slice."
    )
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sessions", type=int, required=True)
    parser.add_argument("--rounds-per-session", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--min-input-tokens", type=int, required=True)
    parser.add_argument("--max-input-tokens", type=int, required=True)
    parser.add_argument("--min-prefix-tokens", type=int, required=True)
    parser.add_argument("--max-append-tokens", type=int, required=True)
    parser.add_argument("--provider", default="codex")
    args = parser.parse_args(argv)
    if not args.trace.is_file():
        parser.error(f"trace does not exist: {args.trace}")

    config = ReplaySelectionConfig(
        session_count=args.sessions,
        rounds_per_session=args.rounds_per_session,
        seed=args.seed,
        min_input_tokens=args.min_input_tokens,
        max_input_tokens=args.max_input_tokens,
        min_prefix_tokens=args.min_prefix_tokens,
        max_append_tokens=args.max_append_tokens,
        provider=args.provider,
    )
    rounds = select_replay_slice(load_trace_rows(args.trace), config)
    payload = write_replay_manifest(
        args.output,
        rounds,
        config,
        trace_sha256=sha256_file(args.trace),
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sessions": args.sessions,
                "rounds": len(rounds),
                "trace_sha256": payload["trace_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
