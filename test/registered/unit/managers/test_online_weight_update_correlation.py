import asyncio
import contextvars
import pickle
from types import SimpleNamespace

import pytest
import zmq

from sglang.srt.managers import io_struct as io_struct_module
from sglang.srt.managers.communicator import FanOutCommunicator
from sglang.srt.managers.io_struct import (
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromDistributedReqOutput,
    msgpack_decode,
    msgpack_encode,
    weight_update_request_context,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.request_receiver import (
    SchedulerRequestReceiver,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _PayloadSocket:
    def __init__(self, payloads):
        self.payloads = list(payloads)

    def recv(self, flags=0):
        assert flags == zmq.NOBLOCK
        if not self.payloads:
            raise zmq.Again()
        return self.payloads.pop(0)


def _request_receiver(payloads):
    return SchedulerRequestReceiver(
        recv_from_tokenizer=_PayloadSocket(payloads),
        recv_from_rpc=_PayloadSocket([]),
        recv_skipper=None,
        input_blocker=None,
        mm_receiver=None,
        ps=SimpleNamespace(pp_rank=0, attn_tp_rank=0, attn_cp_rank=0),
        tp_group=None,
        tp_cpu_group=None,
        attn_tp_group=None,
        attn_tp_cpu_group=None,
        attn_cp_group=None,
        attn_cp_cpu_group=None,
        world_group=None,
        server_args=SimpleNamespace(),
        model_config=SimpleNamespace(),
        max_recv_per_poll=-1,
        stream_output=lambda *_args, **_kwargs: None,
        get_last_batch=lambda: None,
    )


def test_request_without_id_masks_outer_dispatch_context() -> None:
    outer = UpdateWeightsFromDistributedReqInput(
        names=["weight"],
        dtypes=["float16"],
        shapes=[[1]],
        request_id="request-a",
    )
    inner = UpdateWeightsFromDistributedReqInput(
        names=["weight"],
        dtypes=["float16"],
        shapes=[[1]],
    )

    with weight_update_request_context(outer):
        with weight_update_request_context(inner):
            uncorrelated = UpdateWeightsFromDistributedReqOutput(
                success=True,
                message="updated",
            )
        restored = UpdateWeightsFromDistributedReqOutput(
            success=True,
            message="updated",
        )

    assert uncorrelated.request_id is None
    assert uncorrelated.responder_id is None
    assert restored.request_id == "request-a"
    assert restored.responder_id


def test_dispatch_context_resets_after_handler_error() -> None:
    request = UpdateWeightsFromDistributedReqInput(
        names=["weight"],
        dtypes=["float16"],
        shapes=[[1]],
        request_id="request-a",
    )

    with pytest.raises(RuntimeError, match="update failed"):
        with weight_update_request_context(request):
            raise RuntimeError("update failed")

    output = UpdateWeightsFromDistributedReqOutput(
        success=False,
        message="failed",
    )
    assert output.request_id is None
    assert output.responder_id is None


def test_pickle_round_trip_does_not_set_dispatch_context() -> None:
    request = UpdateWeightsFromDistributedReqInput(
        names=["weight"],
        dtypes=["float16"],
        shapes=[[1]],
        request_id="request-a",
    )
    decoded = pickle.loads(pickle.dumps(request))

    outside_dispatch = UpdateWeightsFromDistributedReqOutput(
        success=True,
        message="updated",
    )
    with weight_update_request_context(decoded):
        inside_dispatch = UpdateWeightsFromDistributedReqOutput(
            success=True,
            message="updated",
        )

    assert outside_dispatch.request_id is None
    assert inside_dispatch.request_id == "request-a"
    assert inside_dispatch.responder_id


@pytest.mark.parametrize(
    ("first_success", "retry_success"),
    [(True, False), (False, True)],
)
def test_cancelled_update_cannot_complete_same_type_retry(
    first_success,
    retry_success,
    monkeypatch,
) -> None:
    monkeypatch.setattr(io_struct_module, "_USE_PICKLE_IPC", False)

    async def scenario():
        payloads = []
        communicator = FanOutCommunicator(
            send=lambda request: payloads.append(msgpack_encode(request)),
            fan_out=1,
            mode="queueing",
            correlation_attr="request_id",
            responder_attr="responder_id",
        )
        first = UpdateWeightsFromDistributedReqInput(
            names=["weight"],
            dtypes=["float16"],
            shapes=[[1]],
            weight_version="weights-a",
            request_id="request-a",
        )
        retry = UpdateWeightsFromDistributedReqInput(
            names=["weight"],
            dtypes=["float16"],
            shapes=[[1]],
            weight_version="weights-b",
            request_id="request-b",
        )

        first_call = asyncio.create_task(communicator(first))
        await asyncio.sleep(0)
        assert len(payloads) == 1
        first_call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first_call

        retry_call = asyncio.create_task(communicator(retry))
        await asyncio.sleep(0)
        assert len(payloads) == 2

        requests = _request_receiver(payloads)._pull_raw_reqs()
        assert [request.request_id for request in requests] == [
            "request-a",
            "request-b",
        ]
        wire_outputs = []

        def handle(request):
            success = (
                first_success if request.request_id == "request-a" else retry_success
            )
            return UpdateWeightsFromDistributedReqOutput(
                success=success,
                message="updated" if success else "failed",
            )

        def send_output(output, _request):
            wire_output = msgpack_decode(msgpack_encode(output))
            wire_outputs.append(wire_output)
            communicator.handle_recv(wire_output)

        scheduler = SimpleNamespace(
            session_controller=SimpleNamespace(maybe_reap=lambda _now: None),
            _request_dispatcher=handle,
            ipc_channels=SimpleNamespace(
                send_to_tokenizer=SimpleNamespace(send_output=send_output)
            ),
            weight_updater=SimpleNamespace(
                check_pending_remote_instance_weight_transfers=lambda: []
            ),
            flush_wrapper=SimpleNamespace(check_pending=lambda: None),
            external_corpus_manager=None,
        )
        Scheduler.process_input_requests(scheduler, requests)

        retry_results = await asyncio.wait_for(retry_call, timeout=1)
        published = ["weights-b"] if retry_results[0].success else []

        assert [
            (output.request_id, output.success, output.message)
            for output in wire_outputs
        ] == [
            ("request-a", first_success, "updated" if first_success else "failed"),
            ("request-b", retry_success, "updated" if retry_success else "failed"),
        ]
        assert [
            (output.request_id, output.success, output.message)
            for output in retry_results
        ] == [("request-b", retry_success, "updated" if retry_success else "failed")]
        assert published == (["weights-b"] if retry_success else [])

    contextvars.Context().run(lambda: asyncio.run(scenario()))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
