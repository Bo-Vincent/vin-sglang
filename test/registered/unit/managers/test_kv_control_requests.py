import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from pydantic import TypeAdapter

from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
)
from sglang.srt.managers.io_struct import (
    GenerateReqInput,
    SetInternalStateReq,
    SetInternalStateReqOutput,
)
from sglang.srt.managers.tokenizer_control_mixin import TokenizerControlMixin
from sglang.srt.runtime_context import get_context, get_memory, publish, reset_context
from sglang.srt.server_args import ServerArgs


class TestKVControlRequests(unittest.TestCase):
    def test_batch_keeps_pin_and_received_output_policy(self):
        request = GenerateReqInput(
            input_ids=[[1, 2], [3, 4]],
            sampling_params={"max_new_tokens": 1},
            pin_input_kv=True,
        )
        request._disable_output_kv_cache = True
        request.normalize_batch_and_arguments()
        for item in (request[0], request[1]):
            self.assertTrue(item.pin_input_kv)
            self.assertTrue(item._disable_output_kv_cache)

    def test_openai_requests_keep_pin_extension(self):
        for request in (
            CompletionRequest(model="test", prompt="prefix", pin_input_kv=True),
            ChatCompletionRequest(
                model="test",
                messages=[{"role": "user", "content": "prefix"}],
                pin_input_kv=True,
            ),
        ):
            self.assertTrue(request.pin_input_kv)


class TestOutputPolicyUpdate(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        reset_context()
        publish(ServerArgs(model_path="dummy"), role="test")

    def tearDown(self):
        reset_context()

    def manager(self, *, updated):
        return SimpleNamespace(
            auto_create_handle_loop=lambda: None,
            set_internal_state_communicator=AsyncMock(
                return_value=[SetInternalStateReqOutput(updated=updated)]
            ),
            record_config_updates=get_context().override,
        )

    async def test_only_successful_update_changes_future_request_default(self):
        old_request = GenerateReqInput(input_ids=[1, 2])
        old_request._disable_output_kv_cache = get_memory().disable_output_kv_cache
        update = TypeAdapter(SetInternalStateReq).validate_json(
            '{"server_args": {"disable_output_kv_cache": true}}'
        )
        self.assertEqual(
            await TokenizerControlMixin.set_internal_state(
                self.manager(updated=False), update
            ),
            [False],
        )
        self.assertFalse(get_memory().disable_output_kv_cache)
        self.assertEqual(
            await TokenizerControlMixin.set_internal_state(
                self.manager(updated=True), update
            ),
            [True],
        )
        self.assertTrue(get_memory().disable_output_kv_cache)
        self.assertFalse(old_request._disable_output_kv_cache)

        update = TypeAdapter(SetInternalStateReq).validate_json(
            '{"server_args": {"disable_output_kv_cache": false}}'
        )
        await TokenizerControlMixin.set_internal_state(
            self.manager(updated=True), update
        )
        self.assertFalse(get_memory().disable_output_kv_cache)

    def test_json_preserves_numeric_scheduler_knobs(self):
        update = TypeAdapter(SetInternalStateReq).validate_json(
            '{"server_args": {"max_running_requests": 1, "new_token_ratio": 0.5}}'
        )
        self.assertIs(type(update.server_args["max_running_requests"]), int)
        self.assertIs(type(update.server_args["new_token_ratio"]), float)

    async def test_non_boolean_update_does_not_mutate_default(self):
        with self.assertRaisesRegex(ValueError, "boolean"):
            await TokenizerControlMixin.set_internal_state(
                self.manager(updated=True),
                SetInternalStateReq(server_args={"disable_output_kv_cache": "false"}),
            )
        self.assertFalse(get_memory().disable_output_kv_cache)


if __name__ == "__main__":
    unittest.main()
