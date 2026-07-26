from __future__ import annotations

import pytest

import sglang.srt.weight_transfer as weight_transfer
from sglang.srt.model_executor.weight_runtime_manifest import (
    RuntimeWeightBinding,
    WeightParallelRank,
    WeightPlacementManifest,
    WeightPlacementTensor,
    WeightRuntimeBindingManifest,
    WeightSnapshotCoordinator,
    compute_weight_placement_id,
)
from sglang.srt.weight_transfer.api import load_weights
from sglang.srt.weight_transfer.provider import (
    WeightLoadReceipt,
    WeightProviderCapabilities,
    WeightTargetLoadMode,
    WeightTargetLoadSession,
    WeightTransferCompletionUnknownError,
    WeightTransferError,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.mark.parametrize(
    "name",
    [
        "WeightLoadRequest",
        "WeightMaterializeRequest",
        "WeightProviderReceipt",
        "WeightProviderRequest",
        "WeightTransferAttestor",
        "WeightTransferProvider",
    ],
)
def test_public_facade_exports_provider_contract(name: str) -> None:
    assert name in weight_transfer.__all__
    assert getattr(weight_transfer, name) is not None


@pytest.mark.parametrize(
    "name",
    ["LocalWeightBufferRegistry", "LocalWeightTransferProvider"],
)
def test_public_facade_does_not_export_reference_provider(name: str) -> None:
    assert name not in weight_transfer.__all__


def placement(side: str) -> WeightPlacementManifest:
    tensor = WeightPlacementTensor(
        placement_fragment_id=f"{side}:fragment",
        tensor_id="weight",
        runtime_name="weight",
        aliases=("weight",),
        global_shape=(8,),
        global_offset=(0,),
        local_shape=(8,),
        dtype="bfloat16",
        itemsize=2,
        partition_dim=None,
        shard_dims=(),
        layer_id=0,
        expert_id=None,
        layout_fingerprint="layout:v1",
        nbytes=16,
        byte_offset=0,
        rank=WeightParallelRank(),
    )
    return WeightPlacementManifest(
        model_id="model",
        revision="revision",
        placement_id=compute_weight_placement_id((tensor,)),
        tensors=(tensor,),
    )


def binding(
    manifest: WeightPlacementManifest,
    address: int,
    *,
    nbytes_delta: int = 0,
) -> WeightRuntimeBindingManifest:
    tensor = manifest.tensors[0]
    return WeightRuntimeBindingManifest(
        model_id=manifest.model_id,
        revision=manifest.revision,
        placement_id=manifest.placement_id,
        instance_id=f"instance:{manifest.placement_id}",
        generation=1,
        lease_id=f"lease:{manifest.placement_id}",
        fragments=(
            RuntimeWeightBinding(
                placement_fragment_id=tensor.placement_fragment_id,
                fragment_id=f"runtime:{tensor.placement_fragment_id}",
                address=address,
                nbytes=tensor.nbytes + nbytes_delta,
                storage_offset=0,
                device="cuda:0",
                is_contiguous=True,
                worker_id=manifest.placement_id,
                endpoint=f"{manifest.placement_id}:12345",
            ),
        ),
    )


class RecordingProvider:
    name = "recording"
    requires_runtime_attestation = False

    def __init__(
        self,
        *,
        fail_known: bool = False,
        fail_unknown: bool = False,
        fail_unknown_base: bool = False,
        interrupt_wait: bool = False,
        fail_cancel: bool = False,
        max_total_bytes: int | None = None,
        max_total_operations: int | None = None,
        events: list[str] | None = None,
    ) -> None:
        self.fail_known = fail_known
        self.fail_unknown = fail_unknown
        self.fail_unknown_base = fail_unknown_base
        self.interrupt_wait = interrupt_wait
        self.fail_cancel = fail_cancel
        self.max_total_bytes = max_total_bytes
        self.max_total_operations = max_total_operations
        self.events = [] if events is None else events

    def probe(self, request):
        self.events.append("probe")
        return WeightProviderCapabilities(
            provider=self.name,
            load_profiles=frozenset({"runtime_to_runtime"}),
            materialize_profiles=frozenset(),
            supports_nd_regions=True,
            supports_strided_regions=True,
            supports_safe_cancel=True,
            supports_completion_ticket=True,
            supports_transactional_publish=False,
            max_regions=1024,
            max_segments_per_region=1_000_000,
            max_total_operations=self.max_total_operations,
            max_total_bytes=self.max_total_bytes,
        )

    def prepare(self, request):
        self.events.append("prepare")
        return request

    def submit(self, prepared):
        self.events.append("submit")
        return prepared

    def wait(self, submission):
        self.events.append("wait")
        if self.fail_unknown:
            raise WeightTransferCompletionUnknownError(
                "completion is unknown",
                provider=self.name,
                phase="wait",
                operation_id="operation",
            )
        if self.fail_unknown_base:
            raise WeightTransferError(
                "completion is unknown",
                code="BACKEND_FAILURE",
                provider=self.name,
                phase="wait",
                operation_id="operation",
                retryable=True,
                completion_known=False,
                cleanup_required=True,
            )
        if self.interrupt_wait:
            raise KeyboardInterrupt("interrupted while waiting")
        if self.fail_known:
            raise WeightTransferError(
                "known failure",
                code="BACKEND_FAILURE",
                provider=self.name,
                phase="wait",
                operation_id="operation",
                retryable=True,
                completion_known=True,
                cleanup_required=True,
            )
        return WeightLoadReceipt(
            operation_id="operation",
            provider=self.name,
            plan_digest=submission.plan.digest,
            total_bytes=submission.plan.total_bytes,
            region_count=len(submission.plan.regions),
        )

    def cancel(self, submission):
        self.events.append("cancel")
        if self.fail_cancel:
            raise TimeoutError("cancel did not reach a terminal state")

    def synchronize(self, receipt):
        self.events.append("synchronize")

    def release(self, prepared, receipt):
        self.events.append("release")


def test_load_weights_runs_provider_lifecycle_in_order() -> None:
    source = placement("source")
    target = placement("target")
    provider = RecordingProvider()

    receipt = load_weights(
        source_placements=(source,),
        source_bindings=(binding(source, 0x10000),),
        target_placements=(target,),
        target_bindings=(binding(target, 0x20000),),
        provider=provider,
        target_mode=WeightTargetLoadMode.COLD_START,
    )

    assert receipt.total_bytes == 16
    assert [name for name, _ in receipt.provider_phase_seconds] == [
        "probe",
        "prepare",
        "submit",
        "wait",
        "synchronize",
        "release",
    ]
    assert all(seconds >= 0 for _, seconds in receipt.provider_phase_seconds)
    assert provider.events == [
        "probe",
        "prepare",
        "submit",
        "wait",
        "synchronize",
        "release",
    ]


def test_load_requires_an_explicit_cold_start_or_live_update_mode() -> None:
    source = placement("source")
    target = placement("target")
    source_binding = binding(source, 0x10000)
    target_binding = binding(target, 0x20000)
    provider = RecordingProvider()

    with pytest.raises(TypeError, match="target_mode"):
        load_weights(
            source_placements=(source,),
            source_bindings=(source_binding,),
            target_placements=(target,),
            target_bindings=(target_binding,),
            provider=provider,
        )
    with pytest.raises(ValueError, match="requires a target load session"):
        load_weights(
            source_placements=(source,),
            source_bindings=(source_binding,),
            target_placements=(target,),
            target_bindings=(target_binding,),
            provider=provider,
            target_mode=WeightTargetLoadMode.LIVE_UPDATE,
        )
    with pytest.raises(ValueError, match="must not use a live target session"):
        load_weights(
            source_placements=(source,),
            source_bindings=(source_binding,),
            target_placements=(target,),
            target_bindings=(target_binding,),
            provider=provider,
            target_mode=WeightTargetLoadMode.COLD_START,
            target_session=WeightTargetLoadSession(
                target_bindings=(target_binding,),
                owners=(object(),),
                coordinator=WeightSnapshotCoordinator(),
            ),
        )

    assert provider.events == []


def test_preflight_binding_failure_never_calls_provider() -> None:
    source = placement("source")
    target = placement("target")
    provider = RecordingProvider()

    with pytest.raises(ValueError, match="byte size differs"):
        load_weights(
            source_placements=(source,),
            source_bindings=(binding(source, 0x10000, nbytes_delta=-2),),
            target_placements=(target,),
            target_bindings=(binding(target, 0x20000),),
            provider=provider,
            target_mode=WeightTargetLoadMode.COLD_START,
        )

    assert provider.events == []


def test_known_failure_cancels_and_releases() -> None:
    source = placement("source")
    target = placement("target")
    provider = RecordingProvider(fail_known=True)

    with pytest.raises(WeightTransferError) as raised:
        load_weights(
            source_placements=(source,),
            source_bindings=(binding(source, 0x10000),),
            target_placements=(target,),
            target_bindings=(binding(target, 0x20000),),
            provider=provider,
            target_mode=WeightTargetLoadMode.COLD_START,
        )

    assert raised.value.completion_known is True
    assert provider.events == [
        "probe",
        "prepare",
        "submit",
        "wait",
        "cancel",
        "release",
    ]


def test_completion_unknown_retains_provider_resources() -> None:
    source = placement("source")
    target = placement("target")
    provider = RecordingProvider(fail_unknown=True)

    with pytest.raises(WeightTransferCompletionUnknownError) as raised:
        load_weights(
            source_placements=(source,),
            source_bindings=(binding(source, 0x10000),),
            target_placements=(target,),
            target_bindings=(binding(target, 0x20000),),
            provider=provider,
            target_mode=WeightTargetLoadMode.COLD_START,
        )

    assert raised.value.completion_known is False
    assert provider.events == ["probe", "prepare", "submit", "wait"]


def test_base_completion_unknown_error_retains_provider_resources() -> None:
    source = placement("source")
    target = placement("target")
    provider = RecordingProvider(fail_unknown_base=True)

    with pytest.raises(WeightTransferError) as raised:
        load_weights(
            source_placements=(source,),
            source_bindings=(binding(source, 0x10000),),
            target_placements=(target,),
            target_bindings=(binding(target, 0x20000),),
            provider=provider,
            target_mode=WeightTargetLoadMode.COLD_START,
        )

    assert raised.value.completion_known is False
    assert provider.events == ["probe", "prepare", "submit", "wait"]


def test_cancel_failure_after_interrupt_becomes_completion_unknown() -> None:
    source = placement("source")
    target = placement("target")
    provider = RecordingProvider(interrupt_wait=True, fail_cancel=True)

    with pytest.raises(WeightTransferCompletionUnknownError) as raised:
        load_weights(
            source_placements=(source,),
            source_bindings=(binding(source, 0x10000),),
            target_placements=(target,),
            target_bindings=(binding(target, 0x20000),),
            provider=provider,
            target_mode=WeightTargetLoadMode.COLD_START,
        )

    assert raised.value.completion_known is False
    assert raised.value.__cause__.__class__ is KeyboardInterrupt
    assert provider.events == [
        "probe",
        "prepare",
        "submit",
        "wait",
        "cancel",
    ]


def test_capability_limit_fails_before_prepare() -> None:
    source = placement("source")
    target = placement("target")
    provider = RecordingProvider(max_total_bytes=8)

    with pytest.raises(WeightTransferError, match="byte limit"):
        load_weights(
            source_placements=(source,),
            source_bindings=(binding(source, 0x10000),),
            target_placements=(target,),
            target_bindings=(binding(target, 0x20000),),
            provider=provider,
            target_mode=WeightTargetLoadMode.COLD_START,
        )

    assert provider.events == ["probe"]


class RecordingAttestor:
    def __init__(
        self,
        events: list[str],
        *,
        error: BaseException | None = None,
    ) -> None:
        self.events = events
        self.error = error

    def attest(self, request) -> None:
        assert request.profile == "runtime_to_runtime"
        self.events.append("attest")
        if self.error is not None:
            raise self.error


def test_attestor_runs_before_provider_probe() -> None:
    source = placement("source")
    target = placement("target")
    events = []
    provider = RecordingProvider(events=events)

    receipt = load_weights(
        source_placements=(source,),
        source_bindings=(binding(source, 0x10000),),
        target_placements=(target,),
        target_bindings=(binding(target, 0x20000),),
        provider=provider,
        target_mode=WeightTargetLoadMode.COLD_START,
        attestor=RecordingAttestor(events),
    )

    assert events[:2] == ["attest", "probe"]
    assert receipt.provider_phase_seconds[0][0] == "attest"


def test_attestation_failure_never_probes_provider() -> None:
    source = placement("source")
    target = placement("target")
    events = []
    provider = RecordingProvider(events=events)

    with pytest.raises(ValueError, match="source lease was revoked"):
        load_weights(
            source_placements=(source,),
            source_bindings=(binding(source, 0x10000),),
            target_placements=(target,),
            target_bindings=(binding(target, 0x20000),),
            provider=provider,
            target_mode=WeightTargetLoadMode.COLD_START,
            attestor=RecordingAttestor(
                events,
                error=ValueError("source lease was revoked"),
            ),
        )

    assert events == ["attest"]


def test_required_attestation_fails_before_provider_probe() -> None:
    source = placement("source")
    target = placement("target")
    provider = RecordingProvider()
    provider.requires_runtime_attestation = True

    with pytest.raises(WeightTransferError, match="attestor is required") as raised:
        load_weights(
            source_placements=(source,),
            source_bindings=(binding(source, 0x10000),),
            target_placements=(target,),
            target_bindings=(binding(target, 0x20000),),
            provider=provider,
            target_mode=WeightTargetLoadMode.COLD_START,
        )

    assert raised.value.code == "ATTESTATION_REQUIRED"
    assert raised.value.phase == "attest"
    assert provider.events == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
