import importlib.util
import sys
import threading
import time
from types import ModuleType, SimpleNamespace

import pytest

if importlib.util.find_spec("requests") is None:
    requests = ModuleType("requests")
    requests.post = None
    requests.get = None
    requests.delete = None
    sys.modules["requests"] = requests

from sglang.srt.model_loader import remote_instance_weight_loader_utils as utils
from sglang.srt.weight_transfer.provider import (
    WeightTransferError,
    WeightTransferExecutionContext,
)
from sglang.srt.weight_transfer.remote_protocol import ARTIFACT_WEIGHT_VERSION_V1
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.fixture(autouse=True)
def deterministic_transfer_id(monkeypatch) -> None:
    monkeypatch.setattr(
        utils.uuid,
        "uuid4",
        lambda: SimpleNamespace(hex="transfer-1"),
    )


def _set_transfer_capabilities(
    monkeypatch,
    *,
    native_executor: bool,
    legacy_planner: bool = False,
) -> None:
    monkeypatch.setattr(
        utils,
        "probe_remote_instance_weight_transfer_capabilities",
        lambda **_kwargs: utils.RemoteInstanceWeightTransferCapabilities(
            native_executor=native_executor,
            canonical_adapter=True,
            legacy_planner=legacy_planner,
        ),
    )


def test_begin_preserves_server_lease_timeout(monkeypatch) -> None:
    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {
                "transfer_id": "transfer-1",
                "weight_runtime_manifests": [{"model_id": "model"}],
                "lease_timeout_sec": 90,
            }

    _set_transfer_capabilities(monkeypatch, native_executor=True)
    monkeypatch.setattr(utils.requests, "post", lambda *args, **kwargs: Response())

    session = utils.begin_remote_instance_weight_transfer(
        "http://source", lease_timeout_sec=90
    )

    assert session.lease_timeout_sec == 90


def test_begin_reuses_one_client_fence_across_http_retries(monkeypatch) -> None:
    calls = []

    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {
                "transfer_id": "transfer-1",
                "weight_runtime_manifests": [{"model_id": "model"}],
                "lease_timeout_sec": 90,
                "manifest_revision_semantics": "hf_revision_v1",
                "lease_fence": "target-fence",
                "generation": 7,
            }

    def post(*args, **kwargs):
        calls.append((args, kwargs))
        if len(calls) == 1:
            raise TimeoutError("response lost")
        return Response()

    _set_transfer_capabilities(monkeypatch, native_executor=True)
    monkeypatch.setattr(utils.requests, "post", post)

    session = utils.begin_remote_instance_weight_transfer(
        "http://source",
        lease_timeout_sec=90,
        lease_fence="target-fence",
    )

    assert session.lease_fence == "target-fence"
    assert session.generation == 7
    assert [call[1]["params"]["lease_fence"] for call in calls] == [
        "target-fence",
        "target-fence",
    ]


def test_first_heartbeat_renew_uses_authoritative_begin_identity(monkeypatch) -> None:
    calls = []

    class BeginResponse:
        status_code = 200

        @staticmethod
        def json():
            return {
                "transfer_id": "transfer-1",
                "weight_runtime_manifests": [{"model_id": "model"}],
                "lease_timeout_sec": 90,
                "manifest_revision_semantics": "hf_revision_v1",
                "lease_fence": "source-fence",
                "generation": 7,
            }

    class RenewResponse:
        status_code = 200

        @staticmethod
        def json():
            return {}

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return RenewResponse() if url.endswith("/renew") else BeginResponse()

    _set_transfer_capabilities(monkeypatch, native_executor=True)
    monkeypatch.setattr(utils.requests, "post", post)

    session = utils.begin_remote_instance_weight_transfer(
        "http://source",
        lease_timeout_sec=90,
        lease_fence="target-fence",
    )
    heartbeat = utils.RemoteInstanceWeightTransferHeartbeat(
        "http://source",
        session.transfer_id,
        lease_timeout_sec=session.lease_timeout_sec,
        lease_fence=session.lease_fence,
        generation=session.generation,
    )

    assert heartbeat._renew()
    assert calls[1][0].endswith("/remote_instance_weight_transfer/transfer-1/renew")
    assert calls[1][1]["params"]["lease_fence"] == "source-fence"
    assert calls[1][1]["params"]["generation"] == 7


def test_begin_marks_missing_revision_semantics_unattested(monkeypatch) -> None:
    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {
                "transfer_id": "transfer-1",
                "weight_runtime_manifests": [{"model_id": "model"}],
                "lease_timeout_sec": 90,
            }

    _set_transfer_capabilities(
        monkeypatch,
        native_executor=False,
        legacy_planner=True,
    )
    monkeypatch.setattr(utils.requests, "post", lambda *args, **kwargs: Response())

    session = utils.begin_remote_instance_weight_transfer("http://source")

    assert session.manifest_revision_semantics == utils.LEGACY_HF_UNATTESTED


def test_begin_preserves_explicit_hf_revision_attestation(monkeypatch) -> None:
    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {
                "transfer_id": "transfer-1",
                "weight_runtime_manifests": [{"model_id": "model"}],
                "lease_timeout_sec": 90,
                "manifest_revision_semantics": "hf_revision_v1",
            }

    _set_transfer_capabilities(
        monkeypatch,
        native_executor=False,
        legacy_planner=True,
    )
    monkeypatch.setattr(utils.requests, "post", lambda *args, **kwargs: Response())

    session = utils.begin_remote_instance_weight_transfer("http://source")

    assert session.manifest_revision_semantics == "hf_revision_v1"


@pytest.mark.parametrize(
    ("native_executor", "legacy_planner"),
    ((False, True), (True, False)),
)
def test_begin_rejects_unsupported_revision_semantics_without_fallback(
    monkeypatch,
    native_executor,
    legacy_planner,
) -> None:
    calls = []

    class UnsupportedResponse:
        status_code = 422
        text = ""

        @staticmethod
        def json():
            return {
                "detail": [
                    {
                        "type": "extra_forbidden",
                        "loc": ["query", "manifest_revision_semantics"],
                        "msg": "Extra inputs are not permitted",
                    }
                ]
            }

    class LegacyResponse:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {
                "transfer_id": "transfer-1",
                "weight_runtime_manifests": [{"model_id": "model"}],
                "lease_timeout_sec": 90,
            }

    def post(*args, **kwargs):
        calls.append((args, kwargs))
        return UnsupportedResponse() if len(calls) == 1 else LegacyResponse()

    _set_transfer_capabilities(
        monkeypatch,
        native_executor=native_executor,
        legacy_planner=legacy_planner,
    )
    monkeypatch.setattr(utils.requests, "post", post)

    session = utils.begin_remote_instance_weight_transfer("http://source")

    assert "manifest_revision_semantics" in calls[0][1]["params"]
    assert len(calls) == 1
    assert session is None


@pytest.mark.parametrize(
    ("native_executor", "legacy_planner"),
    ((False, True), (True, False)),
)
def test_begin_retries_without_revision_semantics_when_fallback_is_explicit(
    monkeypatch,
    native_executor,
    legacy_planner,
) -> None:
    calls = []

    class UnsupportedResponse:
        status_code = 422
        text = ""

        @staticmethod
        def json():
            return {
                "detail": [
                    {
                        "type": "extra_forbidden",
                        "loc": ["query", "manifest_revision_semantics"],
                        "msg": "Extra inputs are not permitted",
                    }
                ]
            }

    class LegacyResponse:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {
                "transfer_id": "transfer-1",
                "weight_runtime_manifests": [{"model_id": "model"}],
                "lease_timeout_sec": 90,
            }

    def post(*args, **kwargs):
        calls.append((args, kwargs))
        return UnsupportedResponse() if len(calls) == 1 else LegacyResponse()

    _set_transfer_capabilities(
        monkeypatch,
        native_executor=native_executor,
        legacy_planner=legacy_planner,
    )
    monkeypatch.setattr(utils.requests, "post", post)

    session = utils.begin_remote_instance_weight_transfer(
        "http://source",
        allow_legacy_hf_fallback=True,
    )

    assert "manifest_revision_semantics" in calls[0][1]["params"]
    assert "manifest_revision_semantics" not in calls[1][1]["params"]
    assert session.manifest_revision_semantics == utils.LEGACY_HF_UNATTESTED
    assert session.allow_legacy_hf_fallback is True


def test_begin_negotiates_three_legacy_downgrades_in_four_states(
    monkeypatch,
) -> None:
    calls = []

    class UnsupportedResponse:
        status_code = 422
        text = ""

        def __init__(self, field):
            self.field = field

        def json(self):
            return {
                "detail": [
                    {
                        "type": "extra_forbidden",
                        "loc": ["query", self.field],
                        "msg": "Extra inputs are not permitted",
                    }
                ]
            }

    class LegacyResponse:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {
                "transfer_id": "transfer-1",
                "weight_runtime_manifests": [{"model_id": "model"}],
                "lease_timeout_sec": 90,
            }

    responses = [
        UnsupportedResponse("manifest_revision_semantics"),
        UnsupportedResponse("lease_fence"),
        UnsupportedResponse("manifest_format"),
        LegacyResponse(),
    ]

    def post(*args, **kwargs):
        calls.append((args, kwargs))
        return responses[len(calls) - 1]

    _set_transfer_capabilities(
        monkeypatch,
        native_executor=True,
        legacy_planner=True,
    )
    monkeypatch.setattr(utils.requests, "post", post)

    session = utils.begin_remote_instance_weight_transfer(
        "http://source",
        lease_timeout_sec=90,
        allow_legacy_hf_fallback=True,
    )

    assert [
        (
            call[1]["params"]["manifest_format"],
            "manifest_revision_semantics" in call[1]["params"],
            "lease_fence" in call[1]["params"],
        )
        for call in calls
    ] == [
        ("placement_binding_v1", True, True),
        ("placement_binding_v1", False, True),
        ("placement_binding_v1", False, False),
        ("runtime_v1", False, False),
    ]
    assert session.transfer_id == "transfer-1"
    assert session.manifest_format == "runtime_v1"
    assert session.manifest_revision_semantics == utils.LEGACY_HF_UNATTESTED


def _native_provider(*, validate_environment=lambda: None):
    return SimpleNamespace(
        name="test-provider",
        bounded_execution_contract_version=1,
        validate_environment=validate_environment,
        probe=lambda request: request,
        prepare=lambda request, **kwargs: (request, kwargs),
        submit=lambda prepared: prepared,
        wait=lambda submission, **kwargs: (submission, kwargs),
        cancel=lambda submission: None,
        synchronize=lambda receipt, **kwargs: None,
        release=lambda prepared, receipt, **kwargs: None,
    )


def test_capability_probe_accepts_backend_neutral_provider() -> None:
    capabilities = utils.probe_remote_instance_weight_transfer_capabilities(
        provider=_native_provider(),
    )

    assert capabilities.native_executor is True
    assert capabilities.supports_placement_binding_v1 is True


@pytest.mark.parametrize("contract_version", [None, 0, 2, True])
def test_capability_probe_rejects_unnegotiated_native_contract(
    contract_version,
) -> None:
    provider = _native_provider()
    if contract_version is None:
        del provider.bounded_execution_contract_version
    else:
        provider.bounded_execution_contract_version = contract_version

    capabilities = utils.probe_remote_instance_weight_transfer_capabilities(
        provider=provider,
    )

    assert capabilities.native_executor is False
    assert capabilities.native_contract_error == (
        "native provider requires bounded execution contract version 1"
    )


def test_capability_probe_rejects_provider_environment_failure() -> None:
    def unavailable():
        raise WeightTransferError(
            "missing executor adapter",
            code="UNAVAILABLE_PROVIDER",
            provider="test-provider",
            phase="probe",
            operation_id="unbound",
            retryable=False,
            completion_known=True,
            cleanup_required=False,
        )

    capabilities = utils.probe_remote_instance_weight_transfer_capabilities(
        provider=_native_provider(validate_environment=unavailable),
    )

    assert capabilities.native_executor is False


def test_capability_probe_rejects_incomplete_provider_contract() -> None:
    provider = _native_provider()
    del provider.release

    capabilities = utils.probe_remote_instance_weight_transfer_capabilities(
        provider=provider,
    )

    assert capabilities.native_executor is False


def test_capability_probe_reports_executor_adapter_and_legacy_independently(
    monkeypatch,
) -> None:
    backend = ModuleType("mooncake.weight_transfer")

    class MemoryRegistrationLease:
        from_fragment = staticmethod(lambda fragment, **kwargs: (fragment, kwargs))

    class RuntimeManifest:
        from_runtime_inventory = staticmethod(lambda inventory: inventory)

    backend.MemoryRegistrationLease = MemoryRegistrationLease
    backend.MooncakeTransferEngineReader = object
    backend.RuntimeManifest = RuntimeManifest
    backend.TransferCompletionUnknownError = RuntimeError
    backend.TransferEngineError = RuntimeError
    backend.bounded_execution_contract_version = 1
    backend.plan_runtime_transfer_to_local_target = lambda sources, target: (
        sources,
        target,
    )
    monkeypatch.setattr(
        utils,
        "runtime_manifest_to_parts",
        None,
        raising=False,
    )

    unbounded_capabilities = utils.probe_remote_instance_weight_transfer_capabilities(
        provider=_native_provider(),
        legacy_backend=backend,
    )
    assert unbounded_capabilities.legacy_planner is False
    assert unbounded_capabilities.legacy_contract_error == (
        "legacy backend requires supports_bounded_execution=true "
        "for bounded execution contract version 1"
    )

    backend.supports_bounded_execution = True
    capabilities = utils.probe_remote_instance_weight_transfer_capabilities(
        provider=_native_provider(),
        legacy_backend=backend,
    )

    assert capabilities.native_executor is True
    assert capabilities.canonical_adapter is False
    assert capabilities.legacy_planner is True
    assert capabilities.supports_placement_binding_v1 is True
    assert capabilities.supports_runtime_v1 is True


def test_capability_probe_does_not_claim_unexecutable_runtime_v1(
    monkeypatch,
) -> None:
    backend = ModuleType("mooncake.weight_transfer")
    for name in (
        "MemoryRegistrationLease",
        "MooncakeTransferEngineReader",
        "RuntimeManifest",
        "TransferCompletionUnknownError",
        "TransferEngineError",
    ):
        setattr(backend, name, object)
    monkeypatch.setattr(
        utils,
        "runtime_manifest_to_parts",
        lambda manifest: manifest,
        raising=False,
    )

    capabilities = utils.probe_remote_instance_weight_transfer_capabilities(
        legacy_backend=backend,
    )

    assert capabilities.native_executor is False
    assert capabilities.canonical_adapter is True
    assert capabilities.legacy_planner is False
    assert capabilities.supports_placement_binding_v1 is False
    assert capabilities.supports_runtime_v1 is False


def test_begin_does_not_request_an_unexecutable_manifest_format(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        utils,
        "probe_remote_instance_weight_transfer_capabilities",
        lambda **_kwargs: utils.RemoteInstanceWeightTransferCapabilities(
            native_executor=False,
            canonical_adapter=True,
            legacy_planner=False,
        ),
    )
    monkeypatch.setattr(
        utils.requests,
        "post",
        lambda *args, **kwargs: pytest.fail("HTTP request must not be sent"),
    )

    assert utils.begin_remote_instance_weight_transfer("http://source") is None


def test_begin_retries_and_cleanup_share_target_absolute_deadline(
    monkeypatch,
) -> None:
    post_timeouts = []
    cleanup_timeouts = []

    class SessionResponse:
        status_code = 200

        @staticmethod
        def json():
            return {
                "transfer_id": "transfer-1",
                "lease_fence": "fence-1",
                "generation": 1,
            }

    class DeleteResponse:
        status_code = 200
        text = ""

    def post(*_args, **kwargs):
        post_timeouts.append(kwargs["timeout"])
        raise TimeoutError("response lost")

    def get(*_args, **kwargs):
        cleanup_timeouts.append(kwargs["timeout"])
        return SessionResponse()

    def delete(*_args, **kwargs):
        cleanup_timeouts.append(kwargs["timeout"])
        return DeleteResponse()

    monkeypatch.setattr(utils.requests, "post", post)
    monkeypatch.setattr(utils.requests, "get", get)
    monkeypatch.setattr(utils.requests, "delete", delete)
    context = WeightTransferExecutionContext(
        deadline_unix_sec=time.time() + 0.5,
    )

    session = utils.begin_remote_instance_weight_transfer(
        "http://source",
        capabilities=utils.RemoteInstanceWeightTransferCapabilities(
            native_executor=True,
            canonical_adapter=True,
            legacy_planner=False,
        ),
        execution_context=context,
    )

    assert session is None
    assert len(post_timeouts) == 3
    assert cleanup_timeouts
    assert all(0 < timeout <= 0.25 for timeout in post_timeouts)
    assert all(0 < timeout <= 0.5 for timeout in cleanup_timeouts)


def test_begin_requests_and_parses_split_source_manifest(monkeypatch) -> None:
    calls = []

    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {
                "transfer_id": "transfer-1",
                "source_weight_placements": [{"placement_id": "source-placement"}],
                "source_weight_runtime_bindings": [
                    {
                        "placement_id": "source-placement",
                        "lease_id": "source-lease",
                    }
                ],
                "lease_timeout_sec": 90,
            }

    def post(*args, **kwargs):
        calls.append((args, kwargs))
        return Response()

    _set_transfer_capabilities(monkeypatch, native_executor=True)
    monkeypatch.setattr(utils.requests, "post", post)

    session = utils.begin_remote_instance_weight_transfer(
        "http://source",
        lease_timeout_sec=90,
        manifest_revision_semantics=ARTIFACT_WEIGHT_VERSION_V1,
    )

    assert calls[0][1]["params"] == {
        "lease_timeout_sec": 90,
        "lease_fence": "transfer-1",
        "manifest_format": "placement_binding_v1",
        "manifest_revision_semantics": ARTIFACT_WEIGHT_VERSION_V1,
        "transfer_id": "transfer-1",
    }
    assert session.manifests == []
    assert session.source_placements == [{"placement_id": "source-placement"}]
    assert session.source_bindings == [
        {"placement_id": "source-placement", "lease_id": "source-lease"}
    ]
    assert session.manifest_format == "placement_binding_v1"


def test_begin_prefers_valid_split_manifest_when_response_also_has_runtime_v1(
    monkeypatch,
) -> None:
    runtime_manifests = [{"model_id": "model", "lease_id": "runtime-lease"}]
    source_placements = [{"placement_id": "source-placement"}]
    source_bindings = [
        {
            "placement_id": "source-placement",
            "lease_id": "source-lease",
        }
    ]

    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {
                "transfer_id": "transfer-1",
                "weight_runtime_manifests": runtime_manifests,
                "source_weight_placements": source_placements,
                "source_weight_runtime_bindings": source_bindings,
                "lease_timeout_sec": 90,
            }

    _set_transfer_capabilities(monkeypatch, native_executor=True)
    monkeypatch.setattr(utils.requests, "post", lambda *args, **kwargs: Response())

    session = utils.begin_remote_instance_weight_transfer(
        "http://source", lease_timeout_sec=90
    )

    assert session.manifest_format == "placement_binding_v1"
    assert session.source_placements == source_placements
    assert session.source_bindings == source_bindings
    assert session.manifests == runtime_manifests


def test_begin_falls_back_to_runtime_manifest_when_capability_is_missing(
    monkeypatch,
) -> None:
    calls = []

    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {
                "transfer_id": "transfer-1",
                "weight_runtime_manifests": [{"model_id": "model"}],
                "lease_timeout_sec": 90,
            }

    def post(*args, **kwargs):
        calls.append((args, kwargs))
        return Response()

    _set_transfer_capabilities(
        monkeypatch,
        native_executor=False,
        legacy_planner=True,
    )
    monkeypatch.setattr(utils.requests, "post", post)

    session = utils.begin_remote_instance_weight_transfer(
        "http://source", lease_timeout_sec=90
    )

    assert calls[0][1]["params"] == {
        "lease_timeout_sec": 90,
        "lease_fence": "transfer-1",
        "manifest_format": "runtime_v1",
        "manifest_revision_semantics": "hf_revision_v1",
        "transfer_id": "transfer-1",
    }
    assert session.manifests == [{"model_id": "model"}]
    assert session.source_placements is None
    assert session.source_bindings is None
    assert session.manifest_format == "runtime_v1"


def test_begin_retries_runtime_manifest_once_for_unsupported_split_format(
    monkeypatch,
) -> None:
    calls = []

    class UnsupportedResponse:
        status_code = 422
        text = ""

        @staticmethod
        def json():
            return {
                "detail": [
                    {
                        "type": "literal_error",
                        "loc": ["query", "manifest_format"],
                        "msg": "Input should be 'runtime_v1'",
                    }
                ],
            }

    class RuntimeResponse:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {
                "transfer_id": "transfer-1",
                "weight_runtime_manifests": [{"model_id": "model"}],
                "lease_timeout_sec": 90,
                "manifest_revision_semantics": ARTIFACT_WEIGHT_VERSION_V1,
            }

    def post(*args, **kwargs):
        calls.append((args, kwargs))
        return UnsupportedResponse() if len(calls) == 1 else RuntimeResponse()

    _set_transfer_capabilities(monkeypatch, native_executor=True)
    monkeypatch.setattr(utils.requests, "post", post)

    session = utils.begin_remote_instance_weight_transfer(
        "http://source",
        lease_timeout_sec=90,
        manifest_revision_semantics=ARTIFACT_WEIGHT_VERSION_V1,
    )

    assert [call[1]["params"]["manifest_format"] for call in calls] == [
        "placement_binding_v1",
        "runtime_v1",
    ]
    assert [call[1]["params"]["manifest_revision_semantics"] for call in calls] == [
        ARTIFACT_WEIGHT_VERSION_V1,
        ARTIFACT_WEIGHT_VERSION_V1,
    ]
    assert session.transfer_id == "transfer-1"
    assert session.manifest_format == "runtime_v1"
    assert session.manifest_revision_semantics == ARTIFACT_WEIGHT_VERSION_V1
    assert session.allow_legacy_hf_fallback is False


def test_begin_records_explicit_legacy_hf_policy(monkeypatch) -> None:
    class Response:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {
                "transfer_id": "transfer-1",
                "weight_runtime_manifests": [{"model_id": "model"}],
                "lease_timeout_sec": 90,
            }

    _set_transfer_capabilities(
        monkeypatch,
        native_executor=False,
        legacy_planner=True,
    )
    monkeypatch.setattr(utils.requests, "post", lambda *args, **kwargs: Response())

    session = utils.begin_remote_instance_weight_transfer(
        "http://source",
        allow_legacy_hf_fallback=True,
    )

    assert session.manifest_revision_semantics == utils.LEGACY_HF_UNATTESTED
    assert session.allow_legacy_hf_fallback is True


def test_begin_retries_runtime_manifest_once_for_explicit_conflict_format(
    monkeypatch,
) -> None:
    calls = []

    class UnsupportedResponse:
        status_code = 409
        text = "unsupported manifest_format=placement_binding_v1"

        @staticmethod
        def json():
            return {"detail": "unsupported manifest_format=placement_binding_v1"}

    class RuntimeResponse:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {
                "transfer_id": "transfer-1",
                "weight_runtime_manifests": [{"model_id": "model"}],
                "lease_timeout_sec": 90,
            }

    def post(*args, **kwargs):
        calls.append((args, kwargs))
        return UnsupportedResponse() if len(calls) == 1 else RuntimeResponse()

    _set_transfer_capabilities(monkeypatch, native_executor=True)
    monkeypatch.setattr(utils.requests, "post", post)

    session = utils.begin_remote_instance_weight_transfer(
        "http://source", lease_timeout_sec=90
    )

    assert [call[1]["params"]["manifest_format"] for call in calls] == [
        "placement_binding_v1",
        "runtime_v1",
    ]
    assert session.transfer_id == "transfer-1"


def test_begin_does_not_retry_unrelated_conflict(monkeypatch) -> None:
    calls = []
    released = []

    class Response:
        status_code = 409
        text = "a weight snapshot lease is active"

        @staticmethod
        def json():
            return {"detail": "a weight snapshot lease is active"}

    _set_transfer_capabilities(monkeypatch, native_executor=True)
    monkeypatch.setattr(
        utils.requests,
        "post",
        lambda *args, **kwargs: calls.append((args, kwargs)) or Response(),
    )
    monkeypatch.setattr(
        utils,
        "release_remote_instance_weight_transfer",
        lambda seed_url, transfer_id: released.append((seed_url, transfer_id)) or True,
    )

    assert (
        utils.begin_remote_instance_weight_transfer(
            "http://source", lease_timeout_sec=90
        )
        is None
    )
    assert len(calls) == 1
    assert released == []


def test_begin_retries_release_for_structured_cleanup_pending_response(
    monkeypatch,
) -> None:
    release_attempts = []

    class Response:
        status_code = 409
        text = "snapshot cleanup remains pending"

        @staticmethod
        def json():
            return {
                "transfer_id": "transfer-1",
                "session_state": "cleanup_pending",
                "message": "snapshot cleanup remains pending",
            }

    def release(seed_url, transfer_id):
        release_attempts.append((seed_url, transfer_id))
        return len(release_attempts) >= 2

    _set_transfer_capabilities(
        monkeypatch,
        native_executor=False,
        legacy_planner=True,
    )
    monkeypatch.setattr(utils.requests, "post", lambda *args, **kwargs: Response())
    monkeypatch.setattr(utils, "release_remote_instance_weight_transfer", release)

    assert (
        utils.begin_remote_instance_weight_transfer(
            "http://source",
            lease_timeout_sec=90,
        )
        is None
    )
    assert release_attempts == [
        ("http://source", "transfer-1"),
        ("http://source", "transfer-1"),
    ]


def test_begin_does_not_release_structured_conflict_response(monkeypatch) -> None:
    released = []

    class Response:
        status_code = 409
        text = "transfer ID conflict"

        @staticmethod
        def json():
            return {
                "transfer_id": "transfer-1",
                "session_state": "conflict",
                "message": "transfer ID conflict",
            }

    _set_transfer_capabilities(
        monkeypatch,
        native_executor=False,
        legacy_planner=True,
    )
    monkeypatch.setattr(utils.requests, "post", lambda *args, **kwargs: Response())
    monkeypatch.setattr(
        utils,
        "release_remote_instance_weight_transfer",
        lambda seed_url, transfer_id: released.append((seed_url, transfer_id)) or True,
    )

    assert (
        utils.begin_remote_instance_weight_transfer(
            "http://source",
            lease_timeout_sec=90,
        )
        is None
    )
    assert released == []


def test_begin_does_not_retry_split_request_after_server_error(monkeypatch) -> None:
    calls = []

    class Response:
        status_code = 503
        text = "temporarily unavailable"

        @staticmethod
        def json():
            return {"detail": "temporarily unavailable"}

    _set_transfer_capabilities(monkeypatch, native_executor=True)
    monkeypatch.setattr(
        utils.requests,
        "post",
        lambda *args, **kwargs: calls.append((args, kwargs)) or Response(),
    )

    assert (
        utils.begin_remote_instance_weight_transfer(
            "http://source", lease_timeout_sec=90
        )
        is None
    )
    assert len(calls) == 1


def test_begin_does_not_release_unsupported_response_without_ownership_state(
    monkeypatch,
) -> None:
    calls = []
    released = []

    class Response:
        status_code = 422
        text = "unsupported manifest_format=placement_binding_v1"

        @staticmethod
        def json():
            return {
                "detail": "unsupported manifest_format=placement_binding_v1",
                "transfer_id": "transfer-rejected",
            }

    _set_transfer_capabilities(monkeypatch, native_executor=True)
    monkeypatch.setattr(
        utils.requests,
        "post",
        lambda *args, **kwargs: calls.append((args, kwargs)) or Response(),
    )
    monkeypatch.setattr(
        utils,
        "release_remote_instance_weight_transfer",
        lambda seed_url, transfer_id: released.append((seed_url, transfer_id)) or True,
    )

    assert (
        utils.begin_remote_instance_weight_transfer(
            "http://source", lease_timeout_sec=90
        )
        is None
    )
    assert len(calls) == 1
    assert released == []


def test_begin_reuses_legacy_runtime_session_returned_for_split_request(
    monkeypatch,
) -> None:
    calls = []

    class Response:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {
                "transfer_id": "transfer-1",
                "weight_runtime_manifests": [{"model_id": "model"}],
                "lease_timeout_sec": 90,
            }

    _set_transfer_capabilities(monkeypatch, native_executor=True)
    monkeypatch.setattr(
        utils.requests,
        "post",
        lambda *args, **kwargs: calls.append((args, kwargs)) or Response(),
    )

    session = utils.begin_remote_instance_weight_transfer(
        "http://source", lease_timeout_sec=90
    )

    assert len(calls) == 1
    assert session.transfer_id == "transfer-1"
    assert session.manifests == [{"model_id": "model"}]
    assert session.manifest_format == "runtime_v1"


def test_begin_releases_invalid_split_session_and_fails_closed(
    monkeypatch,
) -> None:
    calls = []
    released = []

    class InvalidSplitResponse:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {
                "transfer_id": "transfer-1",
                "source_weight_placements": [{"placement_id": "source-placement"}],
                "source_weight_runtime_bindings": [],
                "lease_timeout_sec": 90,
            }

    def post(*args, **kwargs):
        calls.append((args, kwargs))
        return InvalidSplitResponse()

    _set_transfer_capabilities(monkeypatch, native_executor=True)
    monkeypatch.setattr(utils.requests, "post", post)
    monkeypatch.setattr(
        utils,
        "release_remote_instance_weight_transfer",
        lambda seed_url, transfer_id: released.append((seed_url, transfer_id)) or False,
    )

    session = utils.begin_remote_instance_weight_transfer(
        "http://source", lease_timeout_sec=90
    )

    assert released == [
        ("http://source", "transfer-1"),
        ("http://source", "transfer-1"),
        ("http://source", "transfer-1"),
    ]
    assert len(calls) == 1
    assert calls[0][1]["params"]["manifest_format"] == "placement_binding_v1"
    assert session is None


def test_begin_releases_transfer_id_when_payload_validation_fails(
    monkeypatch,
) -> None:
    released = []

    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {
                "transfer_id": "transfer-1",
                "weight_runtime_manifests": [],
                "lease_timeout_sec": 90,
            }

    _set_transfer_capabilities(
        monkeypatch,
        native_executor=False,
        legacy_planner=True,
    )
    monkeypatch.setattr(utils.requests, "post", lambda *args, **kwargs: Response())
    monkeypatch.setattr(
        utils,
        "release_remote_instance_weight_transfer",
        lambda seed_url, transfer_id: released.append((seed_url, transfer_id)) or True,
    )

    session = utils.begin_remote_instance_weight_transfer(
        "http://source", lease_timeout_sec=90
    )

    assert session is None
    assert released == [("http://source", "transfer-1")]


def test_begin_retries_response_loss_with_the_same_target_generated_id(
    monkeypatch,
) -> None:
    calls = []
    released = []

    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {
                "transfer_id": "transfer-1",
                "weight_runtime_manifests": [{"model_id": "model"}],
                "lease_timeout_sec": 90,
            }

    def post(*args, **kwargs):
        calls.append((args, kwargs))
        if len(calls) == 1:
            raise RuntimeError("response lost")
        return Response()

    _set_transfer_capabilities(
        monkeypatch,
        native_executor=False,
        legacy_planner=True,
    )
    monkeypatch.setattr(utils.requests, "post", post)
    monkeypatch.setattr(
        utils,
        "release_remote_instance_weight_transfer",
        lambda seed_url, transfer_id: released.append((seed_url, transfer_id)) or True,
    )

    session = utils.begin_remote_instance_weight_transfer(
        "http://source",
        lease_timeout_sec=90,
    )

    assert session.transfer_id == "transfer-1"
    assert len(calls) == 2
    assert {call[1]["params"]["transfer_id"] for call in calls} == {"transfer-1"}
    assert released == []


def test_begin_releases_known_id_after_repeated_response_loss(monkeypatch) -> None:
    released = []
    _set_transfer_capabilities(
        monkeypatch,
        native_executor=False,
        legacy_planner=True,
    )
    monkeypatch.setattr(
        utils.requests,
        "post",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("response lost")),
    )

    def release(seed_url, transfer_id):
        released.append((seed_url, transfer_id))
        return len(released) >= 2

    monkeypatch.setattr(utils, "release_remote_instance_weight_transfer", release)

    assert (
        utils.begin_remote_instance_weight_transfer(
            "http://source",
            lease_timeout_sec=90,
        )
        is None
    )
    assert released == [
        ("http://source", "transfer-1"),
        ("http://source", "transfer-1"),
    ]


def test_begin_rejects_explicit_empty_transfer_id(monkeypatch) -> None:
    monkeypatch.setattr(
        utils.requests,
        "post",
        lambda *args, **kwargs: pytest.fail("invalid ID must fail before HTTP"),
    )

    with pytest.raises(ValueError, match="non-empty string"):
        utils.begin_remote_instance_weight_transfer(
            "http://source",
            transfer_id="",
        )


def test_renew_timeout_is_strictly_inside_remaining_lease_window(
    monkeypatch,
) -> None:
    calls = []

    class Response:
        status_code = 200

    def post(*args, **kwargs):
        calls.append((args, kwargs))
        return Response()

    monkeypatch.setattr(utils.requests, "post", post)

    assert (
        utils.renew_remote_instance_weight_transfer(
            "http://source",
            "transfer-1",
            lease_timeout_sec=30,
            remaining_lease_sec=2.0,
        )
        is True
    )
    assert 0 < calls[0][1]["timeout"] < 2.0


def test_renew_returns_server_authoritative_deadline(monkeypatch) -> None:
    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {"success": True, "deadline_unix_sec": 1060.0}

    monkeypatch.setattr(utils.requests, "post", lambda *args, **kwargs: Response())

    renewal = utils.renew_remote_instance_weight_transfer_lease(
        "http://source",
        "transfer-1",
        lease_timeout_sec=90,
    )

    assert renewal is not None
    assert renewal.deadline_unix_sec == 1060.0


def test_control_requests_keep_transfer_id_and_fenced_identity(monkeypatch) -> None:
    calls = []

    class Response:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {"success": True, "deadline_unix_sec": 1060.0}

    def delete(*args, **kwargs):
        calls.append(("delete", args, kwargs))
        return Response()

    def post(*args, **kwargs):
        calls.append(("post", args, kwargs))
        return Response()

    monkeypatch.setattr(utils.requests, "delete", delete)
    monkeypatch.setattr(utils.requests, "post", post)

    assert utils.release_remote_instance_weight_transfer(
        "http://source",
        "transfer-1",
        lease_fence="fence-1",
        generation=7,
    )
    renewal = utils.renew_remote_instance_weight_transfer_lease(
        "http://source",
        "transfer-1",
        lease_timeout_sec=90,
        lease_fence="fence-1",
        generation=7,
    )

    assert renewal is not None
    assert calls[0][1][0].endswith("/remote_instance_weight_transfer/transfer-1")
    assert calls[0][2]["params"] == {
        "lease_fence": "fence-1",
        "generation": 7,
    }
    assert calls[1][1][0].endswith("/remote_instance_weight_transfer/transfer-1/renew")
    assert calls[1][2]["params"] == {
        "lease_timeout_sec": 90,
        "lease_fence": "fence-1",
        "generation": 7,
    }


def test_heartbeat_uses_server_authoritative_deadline(monkeypatch) -> None:
    monotonic_now = [100.0]
    unix_now = [1000.0]
    monkeypatch.setattr(utils.time, "monotonic", lambda: monotonic_now[0])
    monkeypatch.setattr(utils.time, "time", lambda: unix_now[0])
    monkeypatch.setattr(
        utils,
        "renew_remote_instance_weight_transfer_lease",
        lambda *args, **kwargs: utils.RemoteInstanceWeightLeaseRenewal(
            deadline_unix_sec=1010.0
        ),
    )
    heartbeat = utils.RemoteInstanceWeightTransferHeartbeat(
        "http://source",
        "transfer-1",
        lease_timeout_sec=90,
        renew_interval_sec=1,
    )

    assert heartbeat._renew()
    assert heartbeat._lease_deadline == 110.0


def test_heartbeat_caps_renewal_at_session_absolute_deadline(monkeypatch) -> None:
    monotonic_now = [100.0]
    unix_now = [1000.0]
    requests = []
    monkeypatch.setattr(utils.time, "monotonic", lambda: monotonic_now[0])
    monkeypatch.setattr(utils.time, "time", lambda: unix_now[0])

    def renew(seed_url, transfer_id, lease_timeout_sec, **kwargs):
        requests.append((seed_url, transfer_id, lease_timeout_sec, kwargs))
        return utils.RemoteInstanceWeightLeaseRenewal(deadline_unix_sec=None)

    monkeypatch.setattr(
        utils,
        "renew_remote_instance_weight_transfer_lease",
        renew,
    )
    execution_context = WeightTransferExecutionContext(
        deadline_unix_sec=1100.0,
    )
    heartbeat = utils.RemoteInstanceWeightTransferHeartbeat(
        "http://source",
        "transfer-1",
        lease_timeout_sec=300,
        execution_context=execution_context,
    )

    assert heartbeat._renew()
    assert requests == [
        (
            "http://source",
            "transfer-1",
            100,
            {
                "remaining_lease_sec": 100.0,
                "lease_fence": None,
                "generation": None,
                "execution_context": execution_context,
            },
        )
    ]
    assert heartbeat._lease_deadline == 200.0


def test_heartbeat_renews_in_background(monkeypatch) -> None:
    background_renewed = threading.Event()
    attempts = []

    def renew(seed_url, transfer_id, lease_timeout_sec, **kwargs):
        del kwargs
        attempts.append((seed_url, transfer_id, lease_timeout_sec))
        if len(attempts) >= 2:
            background_renewed.set()
        return utils.RemoteInstanceWeightLeaseRenewal(deadline_unix_sec=None)

    monkeypatch.setattr(
        utils,
        "renew_remote_instance_weight_transfer_lease",
        renew,
    )
    heartbeat = utils.RemoteInstanceWeightTransferHeartbeat(
        "http://source",
        "transfer-1",
        lease_timeout_sec=30,
        renew_interval_sec=0.01,
    )

    heartbeat.start()
    try:
        assert attempts[0] == ("http://source", "transfer-1", 30)
        assert background_renewed.wait(timeout=1)
        assert len(attempts) >= 2
        heartbeat.raise_if_failed()
    finally:
        heartbeat.stop()


def test_heartbeat_rejects_an_expired_session_before_loader_uses_it(
    monkeypatch,
) -> None:
    attempted = threading.Event()

    def renew(seed_url, transfer_id, lease_timeout_sec, **kwargs):
        del seed_url, transfer_id, lease_timeout_sec, kwargs
        attempted.set()
        return None

    monkeypatch.setattr(
        utils,
        "renew_remote_instance_weight_transfer_lease",
        renew,
    )
    heartbeat = utils.RemoteInstanceWeightTransferHeartbeat(
        "http://source",
        "transfer-1",
        lease_timeout_sec=30,
        renew_interval_sec=0.01,
    )

    try:
        heartbeat.start()
    except RuntimeError as error:
        assert "renew" in str(error).lower()
    else:
        raise AssertionError("initial lease renewal failure must fail closed")
    assert attempted.wait(timeout=1)
    heartbeat.stop()


class _FakeWorldGroup:
    def __init__(
        self,
        *,
        rank: int,
        broadcast_session=None,
        gathered_readiness=None,
        gathered_outcomes=None,
        broadcast_outcome=None,
        readiness_error=None,
    ) -> None:
        self.rank_in_group = rank
        self.world_size = 4
        self.broadcast_session = broadcast_session
        self.gathered_readiness = (
            gathered_readiness
            if gathered_readiness is not None
            else [True] * self.world_size
        )
        self.gathered_outcomes = gathered_outcomes or [(True, True)] * self.world_size
        self.broadcast_outcome = broadcast_outcome
        self.readiness_error = readiness_error
        self.broadcasts = []
        self.gathers = []

    def broadcast_object(self, obj=None, src=0):
        self.broadcasts.append((obj, src))
        if len(self.broadcasts) == 1:
            return obj if self.rank_in_group == src else self.broadcast_session
        return obj if self.rank_in_group == src else self.broadcast_outcome

    def all_gather_object(self, obj):
        self.gathers.append(obj)
        if isinstance(obj, bool):
            if self.readiness_error is not None:
                raise self.readiness_error
            return self.gathered_readiness
        return self.gathered_outcomes


class _NoopHeartbeat:
    def __init__(self, *_args, **_kwargs):
        self.started = False

    def start(self):
        self.started = True

    def raise_if_failed(self):
        return None

    def stop(self):
        self.started = False


class _RecordingBoundedCollectives:
    def __init__(self, *, rank: int = 0, world_size: int = 4) -> None:
        self.rank = rank
        self.world_size = world_size
        self.calls = []
        self.deadline_controls = []

    def _record(self, phase, execution_context):
        assert isinstance(execution_context, WeightTransferExecutionContext)
        self.calls.append((phase, execution_context))

    def synchronize_object_collective_deadline(
        self,
        *,
        phase,
        execution_context,
    ):
        assert phase == "remote_instance.acquire.deadline_control"
        assert isinstance(execution_context, WeightTransferExecutionContext)
        self.deadline_controls.append((phase, execution_context))
        return execution_context.deadline_unix_sec

    def broadcast_object(
        self,
        obj,
        *,
        src,
        phase,
        execution_context,
    ):
        assert src == 0
        self._record(phase, execution_context)
        return obj

    def all_gather_object(
        self,
        obj,
        *,
        phase,
        execution_context,
    ):
        self._record(phase, execution_context)
        return [obj] * self.world_size

    def gather_object(
        self,
        obj,
        *,
        dst,
        phase,
        execution_context,
    ):
        assert dst == 0
        self._record(phase, execution_context)
        return [obj] * self.world_size if self.rank == dst else None

    def scatter_object(
        self,
        objects,
        *,
        src,
        phase,
        execution_context,
    ):
        assert src == 0
        self._record(phase, execution_context)
        return objects[self.rank]


def test_world_transfer_collectives_share_one_absolute_deadline(
    monkeypatch,
) -> None:
    session = _session()
    collectives = _RecordingBoundedCollectives()
    group = _FakeWorldGroup(rank=0)

    monkeypatch.setattr(
        utils,
        "begin_remote_instance_weight_transfer",
        lambda seed_url, **_kwargs: session,
    )
    monkeypatch.setattr(
        utils,
        "RemoteInstanceWeightTransferHeartbeat",
        _NoopHeartbeat,
    )
    monkeypatch.setattr(
        utils,
        "release_remote_instance_weight_transfer",
        lambda seed_url, transfer_id: True,
    )
    coordinator = utils.RemoteInstanceWeightTransferWorldCoordinator(
        "http://source",
        group,
        collective_coordinator=collectives,
    )

    assert coordinator.acquire() is session
    assert group.gather_object("target", dst=0) == ["target"] * group.world_size
    assert group.scatter_object(list(range(group.world_size)), src=0) == 0
    assert coordinator.ready_for_transfer(True) is True
    assert coordinator.finish(local_success=True) == (True, True)

    assert [phase for phase, _ in collectives.calls] == [
        "remote_instance.acquire.broadcast",
        "remote_instance.central_plan.gather",
        "remote_instance.central_plan.scatter",
        "remote_instance.readiness.gather",
        "remote_instance.finish.gather",
        "remote_instance.finish.terminal_broadcast",
    ]
    contexts = [context for _, context in collectives.calls]
    assert contexts
    assert all(context is contexts[0] for context in contexts)
    assert coordinator.execution_context is contexts[0]
    assert len(collectives.deadline_controls) == 1
    assert not any(
        name in vars(group)
        for name in (
            "broadcast_object",
            "all_gather_object",
            "gather_object",
            "scatter_object",
        )
    )


def test_world_transfer_collective_timeout_stops_heartbeat_and_poison_fails_closed(
    monkeypatch,
) -> None:
    calls = []
    session = _session()

    class StuckRankCollectives(_RecordingBoundedCollectives):
        def __init__(self):
            super().__init__()
            self.poisoned = False

        def all_gather_object(
            self,
            obj,
            *,
            phase,
            execution_context,
        ):
            self._record(phase, execution_context)
            self.poisoned = True
            raise RuntimeError(
                "deadline exceeded; target-world process group is poisoned"
            )

        def broadcast_object(self, *args, **kwargs):
            if self.poisoned:
                raise AssertionError("poisoned group must not broadcast terminal state")
            return super().broadcast_object(*args, **kwargs)

    class RecordingHeartbeat(_NoopHeartbeat):
        def start(self):
            calls.append("heartbeat-started")
            super().start()

        def stop(self):
            calls.append("heartbeat-stopped")
            super().stop()

    collectives = StuckRankCollectives()
    monkeypatch.setattr(
        utils,
        "begin_remote_instance_weight_transfer",
        lambda seed_url, **_kwargs: session,
    )
    monkeypatch.setattr(
        utils,
        "RemoteInstanceWeightTransferHeartbeat",
        RecordingHeartbeat,
    )
    monkeypatch.setattr(
        utils,
        "release_remote_instance_weight_transfer",
        lambda seed_url, transfer_id: calls.append("release") or True,
    )
    coordinator = utils.RemoteInstanceWeightTransferWorldCoordinator(
        "http://source",
        _FakeWorldGroup(rank=0),
        collective_coordinator=collectives,
    )

    assert coordinator.acquire() is session
    assert coordinator.ready_for_transfer(True) is False
    assert coordinator.finish(local_success=False) == (False, False)
    assert (
        coordinator.release_after_terminal_recovery(
            completion_ticket="stuck-rank-0",
            local_terminal_status="NO_SUBMISSION",
        )
        is False
    )

    assert calls == ["heartbeat-started", "heartbeat-stopped"]
    assert coordinator.heartbeat is None
    assert coordinator.world_release_safe is False
    assert [phase for phase, _ in collectives.calls] == [
        "remote_instance.acquire.broadcast",
        "remote_instance.readiness.gather",
        "remote_instance.finish.gather",
    ]


def test_world_transfer_session_releases_source_when_broadcast_fails(
    monkeypatch,
) -> None:
    calls = []
    session = _session()

    class FailingWorldGroup(_FakeWorldGroup):
        def broadcast_object(self, obj=None, src=0):
            del obj, src
            raise RuntimeError("broadcast failed")

    class FakeHeartbeat:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def start(self):
            calls.append("heartbeat-started")

        def raise_if_failed(self):
            return None

        def stop(self):
            calls.append("heartbeat-stopped")

    monkeypatch.setattr(
        utils,
        "begin_remote_instance_weight_transfer",
        lambda seed_url, **_kwargs: session,
    )
    monkeypatch.setattr(
        utils,
        "release_remote_instance_weight_transfer",
        lambda seed_url, transfer_id: (
            calls.append(("release", seed_url, transfer_id)) or True
        ),
    )
    monkeypatch.setattr(utils, "RemoteInstanceWeightTransferHeartbeat", FakeHeartbeat)
    coordinator = utils.RemoteInstanceWeightTransferWorldCoordinator(
        "http://source", FailingWorldGroup(rank=0)
    )

    try:
        coordinator.acquire()
    except RuntimeError as error:
        assert "broadcast failed" in str(error)
    else:
        raise AssertionError("broadcast failure must escape acquire")

    assert calls == [
        "heartbeat-started",
        "heartbeat-stopped",
        ("release", "http://source", "transfer-1"),
    ]


def _session():
    return SimpleNamespace(
        transfer_id="transfer-1",
        manifests=[{"model_id": "model"}],
        lease_timeout_sec=90,
    )


def _placement_binding_session():
    return utils.RemoteInstanceWeightTransferSession(
        transfer_id="transfer-1",
        manifests=[],
        lease_timeout_sec=90,
        source_placements=[{"placement_id": "source-placement"}],
        source_bindings=[{"placement_id": "source-placement"}],
        manifest_format="placement_binding_v1",
        manifest_revision_semantics="artifact_weight_version_v1",
    )


def test_world_transfer_placement_binding_broadcasts_handle_only(monkeypatch) -> None:
    session = _placement_binding_session()
    monkeypatch.setattr(
        utils,
        "begin_remote_instance_weight_transfer",
        lambda seed_url, **_kwargs: session,
    )
    monkeypatch.setattr(
        utils,
        "RemoteInstanceWeightTransferHeartbeat",
        _NoopHeartbeat,
    )
    group = _FakeWorldGroup(rank=0)
    coordinator = utils.RemoteInstanceWeightTransferWorldCoordinator(
        "http://source",
        group,
    )

    acquired = coordinator.acquire()

    assert acquired is not session
    assert acquired.transfer_id == session.transfer_id
    assert 0 < acquired.lease_timeout_sec <= session.lease_timeout_sec
    assert acquired.deadline_unix_sec == coordinator.execution_context.deadline_unix_sec
    assert acquired.manifest_format == session.manifest_format
    assert not hasattr(acquired, "manifests")
    assert not hasattr(acquired, "source_placements")
    assert not hasattr(acquired, "source_bindings")
    assert coordinator.owner_source_session is session
    assert group.broadcasts == [(acquired, 0)]


def test_world_transfer_placement_binding_follower_has_no_source_payload(
    monkeypatch,
) -> None:
    session = _placement_binding_session()
    handle = utils.RemoteInstanceWeightTransferSessionHandle(
        transfer_id=session.transfer_id,
        lease_timeout_sec=session.lease_timeout_sec,
        manifest_format=session.manifest_format,
        manifest_revision_semantics=session.manifest_revision_semantics,
        allow_legacy_hf_fallback=session.allow_legacy_hf_fallback,
    )
    monkeypatch.setattr(
        utils,
        "begin_remote_instance_weight_transfer",
        lambda seed_url, **_kwargs: pytest.fail("only target rank zero may acquire"),
    )
    group = _FakeWorldGroup(rank=2, broadcast_session=handle)
    coordinator = utils.RemoteInstanceWeightTransferWorldCoordinator(
        "http://source",
        group,
    )

    assert coordinator.acquire() == handle
    assert coordinator.owner_source_session is None
    assert not hasattr(coordinator.session, "source_placements")


def test_world_transfer_finish_drops_placement_binding_owner_payload(
    monkeypatch,
) -> None:
    session = _placement_binding_session()
    monkeypatch.setattr(
        utils,
        "begin_remote_instance_weight_transfer",
        lambda seed_url, **_kwargs: session,
    )
    monkeypatch.setattr(
        utils,
        "RemoteInstanceWeightTransferHeartbeat",
        _NoopHeartbeat,
    )
    monkeypatch.setattr(
        utils,
        "release_remote_instance_weight_transfer",
        lambda seed_url, transfer_id: True,
    )
    coordinator = utils.RemoteInstanceWeightTransferWorldCoordinator(
        "http://source",
        _FakeWorldGroup(rank=0),
    )

    coordinator.acquire()
    assert coordinator.owner_source_session is session

    assert coordinator.finish(local_success=True) == (True, True)
    assert coordinator.owner_source_session is None


def test_world_transfer_session_owner_acquires_and_releases_once(monkeypatch) -> None:
    calls = []
    session = _session()

    class FakeHeartbeat:
        def __init__(self, *args, **kwargs):
            calls.append("heartbeat-created")

        def start(self):
            calls.append("heartbeat-started")

        def raise_if_failed(self):
            calls.append("heartbeat-checked")

        def stop(self):
            calls.append("heartbeat-stopped")

    monkeypatch.setattr(
        utils,
        "begin_remote_instance_weight_transfer",
        lambda seed_url, **_kwargs: calls.append("acquire") or session,
    )
    monkeypatch.setattr(
        utils,
        "release_remote_instance_weight_transfer",
        lambda seed_url, transfer_id: calls.append("release") or True,
    )
    monkeypatch.setattr(utils, "RemoteInstanceWeightTransferHeartbeat", FakeHeartbeat)
    coordinator = utils.RemoteInstanceWeightTransferWorldCoordinator(
        "http://source", _FakeWorldGroup(rank=0)
    )

    assert coordinator.acquire() is session
    coordinator.raise_if_failed()
    assert coordinator.ready_for_transfer(True) is True
    world_success, release_success = coordinator.finish(local_success=True)

    assert world_success is True
    assert release_success is True
    assert calls.count("acquire") == 1
    assert calls.count("release") == 1
    assert "heartbeat-started" in calls
    assert "heartbeat-stopped" in calls


def test_world_transfer_acquire_broadcasts_failure_when_cleanup_raises(
    monkeypatch,
) -> None:
    session = _session()

    class FailingHeartbeat:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            raise RuntimeError("heartbeat start failed")

    monkeypatch.setattr(
        utils,
        "begin_remote_instance_weight_transfer",
        lambda seed_url, **_kwargs: session,
    )
    monkeypatch.setattr(
        utils,
        "release_remote_instance_weight_transfer",
        lambda *args: (_ for _ in ()).throw(RuntimeError("release failed")),
    )
    monkeypatch.setattr(
        utils, "RemoteInstanceWeightTransferHeartbeat", FailingHeartbeat
    )
    group = _FakeWorldGroup(rank=0)
    coordinator = utils.RemoteInstanceWeightTransferWorldCoordinator(
        "http://source", group
    )

    assert coordinator.acquire() is None
    assert group.broadcasts == [(None, 0)]


def test_world_transfer_finish_broadcasts_when_release_raises(monkeypatch) -> None:
    session = _session()

    class FakeHeartbeat:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

        def raise_if_failed(self):
            pass

        def stop(self):
            pass

    monkeypatch.setattr(
        utils,
        "begin_remote_instance_weight_transfer",
        lambda seed_url, **_kwargs: session,
    )
    monkeypatch.setattr(
        utils,
        "release_remote_instance_weight_transfer",
        lambda *args: (_ for _ in ()).throw(RuntimeError("release failed")),
    )
    monkeypatch.setattr(utils, "RemoteInstanceWeightTransferHeartbeat", FakeHeartbeat)
    group = _FakeWorldGroup(rank=0)
    coordinator = utils.RemoteInstanceWeightTransferWorldCoordinator(
        "http://source", group
    )

    assert coordinator.acquire() is session
    assert coordinator.ready_for_transfer(True) is True
    assert coordinator.finish(local_success=True) == (True, False)
    assert group.broadcasts[-1] == ((True, False), 0)


def test_world_transfer_releases_terminal_snapshot_after_heartbeat_failure(
    monkeypatch,
) -> None:
    calls = []
    session = _session()

    class FailedHeartbeat:
        def __init__(self, *args, **kwargs):
            del args, kwargs
            self.checks = 0

        def start(self):
            calls.append("heartbeat-started")

        def raise_if_failed(self):
            self.checks += 1
            if self.checks >= 2:
                raise RuntimeError("historical renewal failure")

        def stop(self):
            calls.append("heartbeat-stopped")

    monkeypatch.setattr(
        utils,
        "begin_remote_instance_weight_transfer",
        lambda seed_url, **_kwargs: session,
    )
    monkeypatch.setattr(
        utils,
        "release_remote_instance_weight_transfer",
        lambda seed_url, transfer_id: calls.append("release") or True,
    )
    monkeypatch.setattr(utils, "RemoteInstanceWeightTransferHeartbeat", FailedHeartbeat)
    coordinator = utils.RemoteInstanceWeightTransferWorldCoordinator(
        "http://source",
        _FakeWorldGroup(
            rank=0,
            gathered_outcomes=[(False, True)] * 4,
        ),
    )

    assert coordinator.acquire() is session
    assert coordinator.ready_for_transfer(True) is True
    assert coordinator.finish(local_success=True) == (False, True)
    assert calls == ["heartbeat-started", "heartbeat-stopped", "release"]


def test_world_transfer_retries_terminal_source_release(monkeypatch) -> None:
    release_results = iter((False, False, True))
    release_calls = []
    session = _session()

    class FakeHeartbeat:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def start(self):
            pass

        def raise_if_failed(self):
            pass

        def stop(self):
            pass

    def release(seed_url, transfer_id):
        release_calls.append((seed_url, transfer_id))
        return next(release_results)

    monkeypatch.setattr(
        utils,
        "begin_remote_instance_weight_transfer",
        lambda seed_url, **_kwargs: session,
    )
    monkeypatch.setattr(utils, "release_remote_instance_weight_transfer", release)
    monkeypatch.setattr(utils, "RemoteInstanceWeightTransferHeartbeat", FakeHeartbeat)
    coordinator = utils.RemoteInstanceWeightTransferWorldCoordinator(
        "http://source", _FakeWorldGroup(rank=0)
    )

    assert coordinator.acquire() is session
    assert coordinator.ready_for_transfer(True) is True
    assert coordinator.finish(local_success=True) == (True, False)
    assert (
        coordinator.release_after_terminal_recovery(
            completion_ticket="terminal-0",
            local_terminal_status="COMPLETED",
        )
        is False
    )
    assert (
        coordinator.release_after_terminal_recovery(
            completion_ticket="terminal-0",
            local_terminal_status="COMPLETED",
        )
        is True
    )
    assert release_calls == [
        ("http://source", "transfer-1"),
        ("http://source", "transfer-1"),
        ("http://source", "transfer-1"),
    ]


def test_world_transfer_readiness_rejects_partial_world_and_runs_once(
    monkeypatch,
) -> None:
    session = _session()
    monkeypatch.setattr(
        utils,
        "begin_remote_instance_weight_transfer",
        lambda seed_url, **_kwargs: session,
    )
    monkeypatch.setattr(
        utils,
        "renew_remote_instance_weight_transfer_lease",
        lambda *args, **kwargs: utils.RemoteInstanceWeightLeaseRenewal(
            deadline_unix_sec=None
        ),
    )
    monkeypatch.setattr(
        utils, "release_remote_instance_weight_transfer", lambda *args: True
    )
    group = _FakeWorldGroup(
        rank=0,
        gathered_readiness=[True, False, True, True],
        gathered_outcomes=[(False, True), (True, True), (True, True), (True, True)],
    )
    coordinator = utils.RemoteInstanceWeightTransferWorldCoordinator(
        "http://source", group
    )

    assert coordinator.acquire() is session
    assert coordinator.ready_for_transfer(True) is False
    try:
        coordinator.ready_for_transfer(True)
    except RuntimeError as error:
        assert "already checked" in str(error)
    else:
        raise AssertionError("readiness gate must run exactly once")
    assert group.gathers[0] is True
    assert coordinator.finish(local_success=False) == (False, True)


def test_world_transfer_invalid_readiness_requires_explicit_release(
    monkeypatch,
) -> None:
    calls = []
    session = _session()
    monkeypatch.setattr(
        utils,
        "begin_remote_instance_weight_transfer",
        lambda seed_url, **_kwargs: session,
    )
    monkeypatch.setattr(
        utils,
        "renew_remote_instance_weight_transfer_lease",
        lambda *args, **kwargs: utils.RemoteInstanceWeightLeaseRenewal(
            deadline_unix_sec=None
        ),
    )
    monkeypatch.setattr(
        utils,
        "release_remote_instance_weight_transfer",
        lambda *args: calls.append("release") or True,
    )
    monkeypatch.setattr(
        utils,
        "RemoteInstanceWeightTransferHeartbeat",
        _NoopHeartbeat,
    )
    group = _FakeWorldGroup(
        rank=0,
        gathered_readiness=[True],
        gathered_outcomes=[(False, False)] * 4,
    )
    coordinator = utils.RemoteInstanceWeightTransferWorldCoordinator(
        "http://source", group
    )

    assert coordinator.acquire() is session
    assert coordinator.ready_for_transfer(True) is False
    assert coordinator.finish(local_success=False) == (False, False)
    assert calls == []


def test_world_transfer_readiness_collective_failure_requires_recovery(
    monkeypatch,
) -> None:
    calls = []
    session = _session()
    monkeypatch.setattr(
        utils,
        "begin_remote_instance_weight_transfer",
        lambda seed_url, **_kwargs: session,
    )
    monkeypatch.setattr(
        utils,
        "renew_remote_instance_weight_transfer_lease",
        lambda *args, **kwargs: utils.RemoteInstanceWeightLeaseRenewal(
            deadline_unix_sec=None
        ),
    )
    monkeypatch.setattr(
        utils,
        "release_remote_instance_weight_transfer",
        lambda *args: calls.append("release") or True,
    )
    monkeypatch.setattr(
        utils,
        "RemoteInstanceWeightTransferHeartbeat",
        _NoopHeartbeat,
    )
    group = _FakeWorldGroup(
        rank=0,
        gathered_outcomes=[(False, False)] * 4,
        readiness_error=RuntimeError("readiness collective failed"),
    )
    coordinator = utils.RemoteInstanceWeightTransferWorldCoordinator(
        "http://source", group
    )

    assert coordinator.acquire() is session
    assert coordinator.ready_for_transfer(True) is False
    assert coordinator.finish(local_success=False) == (False, False)
    assert calls == []


def test_world_transfer_unknown_completion_requires_explicit_release(
    monkeypatch,
) -> None:
    calls = []
    session = _session()

    class FakeHeartbeat:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

        def raise_if_failed(self):
            pass

        def stop(self):
            calls.append("heartbeat-stopped")

    monkeypatch.setattr(
        utils,
        "begin_remote_instance_weight_transfer",
        lambda seed_url, **_kwargs: session,
    )
    monkeypatch.setattr(
        utils,
        "release_remote_instance_weight_transfer",
        lambda seed_url, transfer_id: calls.append("release") or True,
    )
    monkeypatch.setattr(utils, "RemoteInstanceWeightTransferHeartbeat", FakeHeartbeat)
    coordinator = utils.RemoteInstanceWeightTransferWorldCoordinator(
        "http://source",
        _FakeWorldGroup(
            rank=0,
            gathered_outcomes=[(False, False)] * 4,
        ),
    )

    assert coordinator.acquire() is session
    world_success, release_success = coordinator.finish(
        local_success=False,
        local_release_safe=False,
    )

    assert world_success is False
    assert release_success is False
    assert calls == []


def test_world_transfer_session_follower_reuses_broadcast_and_never_calls_source(
    monkeypatch,
) -> None:
    session = _session()
    monkeypatch.setattr(
        utils,
        "begin_remote_instance_weight_transfer",
        lambda seed_url, **_kwargs: (_ for _ in ()).throw(
            AssertionError("only world rank zero may acquire")
        ),
    )
    monkeypatch.setattr(
        utils,
        "release_remote_instance_weight_transfer",
        lambda seed_url, transfer_id: (_ for _ in ()).throw(
            AssertionError("only world rank zero may release")
        ),
    )
    group = _FakeWorldGroup(
        rank=2,
        broadcast_session=session,
        gathered_outcomes=[
            (True, True),
            (False, True),
            (True, True),
            (True, True),
        ],
        broadcast_outcome=(False, True),
    )
    coordinator = utils.RemoteInstanceWeightTransferWorldCoordinator(
        "http://source", group
    )

    assert coordinator.acquire() is session
    world_success, release_success = coordinator.finish(local_success=True)

    assert world_success is False
    assert release_success is True


def test_world_transfer_session_rejects_invalid_collective_outcomes(
    monkeypatch,
) -> None:
    calls = []
    session = _session()

    class FakeHeartbeat:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

        def raise_if_failed(self):
            pass

        def stop(self):
            calls.append("heartbeat-stopped")

    monkeypatch.setattr(
        utils,
        "begin_remote_instance_weight_transfer",
        lambda seed_url, **_kwargs: session,
    )
    monkeypatch.setattr(
        utils,
        "release_remote_instance_weight_transfer",
        lambda seed_url, transfer_id: calls.append("release") or True,
    )
    monkeypatch.setattr(utils, "RemoteInstanceWeightTransferHeartbeat", FakeHeartbeat)
    coordinator = utils.RemoteInstanceWeightTransferWorldCoordinator(
        "http://source",
        _FakeWorldGroup(rank=0, gathered_outcomes=[(True, True)]),
    )

    assert coordinator.acquire() is session
    assert coordinator.finish(local_success=True) == (False, False)
    assert calls == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
