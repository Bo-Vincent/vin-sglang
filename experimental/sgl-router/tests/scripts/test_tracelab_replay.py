# SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import gzip
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "tracelab_replay.py"


def load_adapter():
    if not MODULE_PATH.exists():
        return None
    spec = importlib.util.spec_from_file_location("tracelab_replay", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load TraceLab adapter from {MODULE_PATH}")
    adapter = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = adapter
    spec.loader.exec_module(adapter)
    return adapter


def round_row(
    session_id: str,
    round_index: int,
    prefix_tokens: int,
    newly_append_tokens: int,
    *,
    provider: str = "codex",
    output_tokens: int = 64,
) -> dict:
    return {
        "provider": provider,
        "session_id": session_id,
        "round_index": round_index,
        "input_tokens_total": prefix_tokens + newly_append_tokens,
        "prefix_tokens": prefix_tokens,
        "newly_append_tokens": newly_append_tokens,
        "output_tokens": output_tokens,
        "timing_events": [],
        "tools": [],
    }


class TraceLabReplaySelectionTests(unittest.TestCase):
    def setUp(self):
        self.adapter = load_adapter()
        self.assertIsNotNone(
            self.adapter,
            "TraceLab adapter is missing; add scripts/tracelab_replay.py",
        )

    def test_selects_contiguous_codex_rounds_and_marks_first_as_warmup(self):
        rows = [
            round_row("codex:alpha", 0, 0, 2048),
            round_row("codex:alpha", 1, 2048, 1024),
            round_row("codex:alpha", 2, 3072, 1024),
            # This deliberately models a compaction: the next prefix shrinks.
            round_row("codex:alpha", 3, 1536, 1024),
            round_row("claude:ignored", 0, 0, 2048, provider="claude"),
            round_row("claude:ignored", 1, 2048, 1024, provider="claude"),
            round_row("claude:ignored", 2, 3072, 1024, provider="claude"),
            round_row("claude:ignored", 3, 4096, 1024, provider="claude"),
        ]

        selected = self.adapter.select_replay_slice(
            rows,
            self.adapter.ReplaySelectionConfig(
                session_count=1,
                rounds_per_session=4,
                seed=7,
                min_input_tokens=1024,
                max_input_tokens=8192,
                min_prefix_tokens=0,
                max_append_tokens=4096,
            ),
        )

        self.assertEqual([row.session_id for row in selected], ["codex:alpha"] * 4)
        self.assertEqual([row.round_index for row in selected], [0, 1, 2, 3])
        self.assertEqual([row.is_warmup for row in selected], [True, False, False, False])
        self.assertEqual(
            [row.prefix_tokens for row in selected], [0, 2048, 3072, 1536]
        )

    def test_seeded_selection_is_stable_and_never_selects_noncontiguous_window(self):
        rows = []
        for session_id in ("codex:alpha", "codex:bravo", "codex:charlie"):
            rows.extend(
                [
                    round_row(session_id, 0, 0, 2048),
                    round_row(session_id, 1, 2048, 1024),
                    round_row(session_id, 2, 3072, 1024),
                    round_row(session_id, 3, 4096, 1024),
                ]
            )
        rows.extend(
            [
                round_row("codex:gap", 0, 0, 2048),
                round_row("codex:gap", 2, 2048, 1024),
                round_row("codex:gap", 3, 3072, 1024),
                round_row("codex:gap", 4, 4096, 1024),
            ]
        )
        config = self.adapter.ReplaySelectionConfig(
            session_count=2,
            rounds_per_session=4,
            seed=11,
            min_input_tokens=1024,
            max_input_tokens=8192,
            min_prefix_tokens=0,
            max_append_tokens=4096,
        )

        first = self.adapter.select_replay_slice(rows, config)
        second = self.adapter.select_replay_slice(rows, config)

        self.assertEqual(first, second)
        selected_ids = {row.session_id for row in first}
        self.assertEqual(len(selected_ids), 2)
        self.assertNotIn("codex:gap", selected_ids)
        for session_id in selected_ids:
            self.assertEqual(
                [row.round_index for row in first if row.session_id == session_id],
                [0, 1, 2, 3],
            )

    def test_allows_zero_prefix_warmup_but_requires_measurement_prefix(self):
        rows = [
            round_row("codex:alpha", 0, 0, 2048),
            round_row("codex:alpha", 1, 2048, 1024),
            round_row("codex:alpha", 2, 3072, 1024),
            round_row("codex:alpha", 3, 4096, 1024),
            round_row("codex:weak", 0, 0, 2048),
            round_row("codex:weak", 1, 512, 1024),
            round_row("codex:weak", 2, 1024, 1024),
            round_row("codex:weak", 3, 2048, 1024),
        ]

        selected = self.adapter.select_replay_slice(
            rows,
            self.adapter.ReplaySelectionConfig(
                session_count=1,
                rounds_per_session=4,
                seed=7,
                min_input_tokens=1024,
                max_input_tokens=8192,
                min_prefix_tokens=1024,
                max_append_tokens=4096,
            ),
        )

        self.assertEqual({row.session_id for row in selected}, {"codex:alpha"})
        self.assertTrue(selected[0].is_warmup)
        self.assertEqual(selected[0].prefix_tokens, 0)

    def test_rejects_invalid_token_decomposition(self):
        invalid = round_row("codex:broken", 0, 0, 2048)
        invalid["input_tokens_total"] = 2047

        with self.assertRaisesRegex(ValueError, "input_tokens_total"):
            self.adapter.parse_trace_row(invalid, source_line=1)

    def test_gzip_trace_loader_and_manifest_preserve_replay_contract(self):
        rows = [
            round_row("codex:alpha", 0, 0, 2048),
            round_row("codex:alpha", 1, 2048, 1024),
            round_row("codex:alpha", 2, 3072, 1024),
            round_row("codex:alpha", 3, 4096, 1024),
        ]
        config = self.adapter.ReplaySelectionConfig(
            session_count=1,
            rounds_per_session=4,
            seed=7,
            min_input_tokens=1024,
            max_input_tokens=8192,
            min_prefix_tokens=0,
            max_append_tokens=4096,
        )
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            trace = root / "trace.jsonl.gz"
            with gzip.open(trace, "wt") as output:
                for row in rows:
                    output.write(json.dumps(row) + "\n")
            manifest = root / "replay.json"

            loaded = list(self.adapter.load_trace_rows(trace))
            selected = self.adapter.select_replay_slice(loaded, config)
            payload = self.adapter.write_replay_manifest(
                manifest,
                selected,
                config,
                trace_sha256="trace-sha",
            )

            self.assertEqual(payload, json.loads(manifest.read_text()))
            self.assertEqual(payload["schema_version"], 1)
            self.assertEqual(payload["trace_sha256"], "trace-sha")
            self.assertEqual(payload["selection"]["seed"], 7)
            self.assertEqual(
                [row["is_warmup"] for row in payload["rounds"]],
                [True, False, False, False],
            )

    def test_cli_writes_manifest_with_actual_trace_digest(self):
        rows = [
            round_row("codex:alpha", 0, 0, 2048),
            round_row("codex:alpha", 1, 2048, 1024),
            round_row("codex:alpha", 2, 3072, 1024),
            round_row("codex:alpha", 3, 4096, 1024),
        ]
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            trace = root / "trace.jsonl.gz"
            with gzip.open(trace, "wt") as output:
                for row in rows:
                    output.write(json.dumps(row) + "\n")
            manifest = root / "manifest.json"

            exit_code = self.adapter.main(
                [
                    "--trace",
                    str(trace),
                    "--output",
                    str(manifest),
                    "--sessions",
                    "1",
                    "--rounds-per-session",
                    "4",
                    "--seed",
                    "7",
                    "--min-input-tokens",
                    "1024",
                    "--max-input-tokens",
                    "8192",
                    "--min-prefix-tokens",
                    "1024",
                    "--max-append-tokens",
                    "4096",
                ]
            )

            self.assertEqual(exit_code, 0)
            payload = json.loads(manifest.read_text())
            self.assertEqual(
                payload["trace_sha256"], hashlib.sha256(trace.read_bytes()).hexdigest()
            )
            self.assertEqual(len(payload["rounds"]), 4)


class WhitespaceTokenCodec:
    """供重建测试使用的可逆空格 tokenizer。"""

    def __init__(self):
        self._token_to_id = {}
        self._id_to_token = {}
        self.vocab_size = 1024

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        token_ids = []
        for token in text.split():
            if token.startswith("v") and token[1:].isdigit():
                token_id = int(token[1:])
            else:
                token_id = self._token_to_id.setdefault(
                    token, len(self._token_to_id) + 1
                )
            self._id_to_token[token_id] = token
            token_ids.append(token_id)
        return token_ids

    def decode(
        self,
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ):
        del skip_special_tokens, clean_up_tokenization_spaces
        return " ".join(
            self._id_to_token.get(token_id, f"v{token_id}")
            for token_id in token_ids
        )


class TraceLabPromptReconstructionTests(unittest.TestCase):
    def setUp(self):
        self.adapter = load_adapter()
        self.assertIsNotNone(self.adapter)
        self.codec = WhitespaceTokenCodec()

    def test_virtual_prompt_preserves_session_local_prefix_and_token_geometry(self):
        warmup = self.adapter.ReplayRound(
            session_id="codex:alpha",
            round_index=0,
            input_tokens=10,
            prefix_tokens=6,
            newly_append_tokens=4,
            output_tokens=64,
            source_line=1,
            is_warmup=True,
        )
        measured = self.adapter.ReplayRound(
            session_id="codex:alpha",
            round_index=1,
            input_tokens=11,
            prefix_tokens=6,
            newly_append_tokens=5,
            output_tokens=64,
            source_line=2,
            is_warmup=False,
        )

        safe_token_ids = tuple(range(101, 113))
        warmup_tokens = self.codec.encode(
            self.adapter.reconstruct_virtual_prompt(
                warmup, self.codec, safe_token_ids=safe_token_ids
            )
        )
        measured_tokens = self.codec.encode(
            self.adapter.reconstruct_virtual_prompt(
                measured, self.codec, safe_token_ids=safe_token_ids
            )
        )

        self.assertEqual(len(warmup_tokens), warmup.input_tokens)
        self.assertEqual(len(measured_tokens), measured.input_tokens)
        self.assertEqual(warmup_tokens[:6], measured_tokens[:6])
        self.assertNotEqual(warmup_tokens[6:], measured_tokens[6:])

    def test_virtual_prompt_does_not_share_session_prefix_between_sessions(self):
        alpha = self.adapter.ReplayRound(
            session_id="codex:alpha",
            round_index=1,
            input_tokens=10,
            prefix_tokens=6,
            newly_append_tokens=4,
            output_tokens=64,
            source_line=1,
            is_warmup=False,
        )
        bravo = self.adapter.ReplayRound(
            session_id="codex:bravo",
            round_index=1,
            input_tokens=10,
            prefix_tokens=6,
            newly_append_tokens=4,
            output_tokens=64,
            source_line=2,
            is_warmup=False,
        )

        safe_token_ids = tuple(range(101, 113))
        alpha_tokens = self.codec.encode(
            self.adapter.reconstruct_virtual_prompt(
                alpha, self.codec, safe_token_ids=safe_token_ids
            )
        )
        bravo_tokens = self.codec.encode(
            self.adapter.reconstruct_virtual_prompt(
                bravo, self.codec, safe_token_ids=safe_token_ids
            )
        )

        self.assertNotEqual(alpha_tokens[: alpha.prefix_tokens], bravo_tokens[: bravo.prefix_tokens])

    def test_virtual_prompt_uses_reversible_tokens_across_prefix_append_boundary(self):
        round_ = self.adapter.ReplayRound(
            session_id="codex:alpha",
            round_index=3,
            input_tokens=13,
            prefix_tokens=8,
            newly_append_tokens=5,
            output_tokens=64,
            source_line=17,
            is_warmup=False,
        )

        prompt = self.adapter.reconstruct_virtual_prompt(
            round_, self.codec, safe_token_ids=tuple(range(101, 113))
        )

        self.assertEqual(len(self.codec.encode(prompt)), round_.input_tokens)

    def test_token_plan_assigns_distinct_prefixes_to_each_session(self):
        alpha = self.adapter.ReplayRound(
            session_id="codex:alpha",
            round_index=1,
            input_tokens=10,
            prefix_tokens=6,
            newly_append_tokens=4,
            output_tokens=64,
            source_line=1,
            is_warmup=False,
        )
        bravo = self.adapter.ReplayRound(
            session_id="codex:bravo",
            round_index=1,
            input_tokens=10,
            prefix_tokens=6,
            newly_append_tokens=4,
            output_tokens=64,
            source_line=2,
            is_warmup=False,
        )

        plan = self.adapter.build_virtual_token_plan(
            [alpha, bravo], safe_token_ids=tuple(range(101, 105))
        )
        alpha_tokens = self.codec.encode(
            self.adapter.reconstruct_virtual_prompt(alpha, self.codec, token_plan=plan)
        )
        bravo_tokens = self.codec.encode(
            self.adapter.reconstruct_virtual_prompt(bravo, self.codec, token_plan=plan)
        )

        self.assertNotEqual(alpha_tokens[: alpha.prefix_tokens], bravo_tokens[: bravo.prefix_tokens])
        self.assertNotEqual(alpha_tokens[alpha.prefix_tokens :], bravo_tokens[bravo.prefix_tokens :])


if __name__ == "__main__":
    unittest.main()
