from __future__ import annotations

import ctypes
import hashlib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol

from sglang.srt.weight_transfer.contracts import (
    StorageWeightLocation,
    RuntimeWeightLocation,
)
from sglang.srt.weight_transfer.lowering import (
    WeightLoweringLimits,
    iter_bounded_transfer_batches,
    lowering_operation_count,
)
from sglang.srt.weight_transfer.provider import (
    WeightLoadReceipt,
    WeightLoadRequest,
    WeightProviderCapabilities,
    WeightProviderReceipt,
    WeightTransferError,
)


class StorageRangeReader(Protocol):
    def __call__(
        self,
        source: StorageWeightLocation,
        object_offset: int,
        nbytes: int,
    ) -> bytes:
        """Synchronously read one range from the selected object version."""


class StorageVersionReader(Protocol):
    def __call__(self, source: StorageWeightLocation) -> str:
        """Return a token that changes whenever readable object bytes change."""


class RuntimeRangeWriter(Protocol):
    def __call__(
        self,
        target: RuntimeWeightLocation,
        target_offset: int,
        payload: bytes,
    ) -> None:
        """Return only after the write has completed or terminally failed."""


_DEFAULT_CHECKSUM_CHUNK_BYTES = 8 * 1024 * 1024


class CheckpointProviderState(str, Enum):
    PREPARED = "prepared"
    SUBMITTED = "submitted"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RELEASED = "released"


@dataclass(frozen=True)
class CheckpointLoadStats:
    batch_count: int
    operation_count: int
    max_batch_operations: int
    max_batch_bytes: int


@dataclass
class _PreparedCheckpointLoad:
    request: WeightLoadRequest
    verified_versions: dict[tuple, str]
    released: bool = False


@dataclass
class _CheckpointSubmission:
    prepared: _PreparedCheckpointLoad
    receipt: WeightLoadReceipt | None
    cancelled: bool = False


class CheckpointStorageToRuntimeProvider:
    """Synchronous, backend-neutral checkpoint/object-range loader."""

    name = "checkpoint"
    requires_runtime_attestation = True

    def __init__(
        self,
        *,
        range_reader: StorageRangeReader | None = None,
        source_version_reader: StorageVersionReader | None = None,
        target_writer: RuntimeRangeWriter | None = None,
        lowering_limits: WeightLoweringLimits | None = None,
        checksum_chunk_bytes: int = _DEFAULT_CHECKSUM_CHUNK_BYTES,
    ) -> None:
        if type(checksum_chunk_bytes) is not int or checksum_chunk_bytes <= 0:
            raise ValueError("checksum_chunk_bytes must be a positive integer")
        if range_reader is None:
            self._range_reader = self._read_local_range
            self._source_version_reader = (
                source_version_reader or self._read_local_version
            )
        else:
            self._range_reader = range_reader
            self._source_version_reader = source_version_reader
        self._target_writer = target_writer
        self.lowering_limits = lowering_limits or WeightLoweringLimits(
            max_total_operations=10_000_000,
            max_batch_operations=8192,
            max_batch_bytes=256 * 1024 * 1024,
        )
        self.checksum_chunk_bytes = checksum_chunk_bytes
        self._lifecycle: list[CheckpointProviderState] = []
        self._cuda_runtime = None

    @property
    def lifecycle(self) -> tuple[CheckpointProviderState, ...]:
        return tuple(self._lifecycle)

    def probe(self, request) -> WeightProviderCapabilities:
        del request
        return WeightProviderCapabilities(
            provider=self.name,
            load_profiles=frozenset({"storage_to_runtime"}),
            materialize_profiles=frozenset(),
            supports_nd_regions=True,
            supports_strided_regions=True,
            supports_safe_cancel=False,
            supports_completion_ticket=False,
            supports_transactional_publish=False,
            max_regions=1_000_000,
            max_segments_per_region=10_000_000,
            max_total_operations=self.lowering_limits.max_total_operations,
            max_batch_operations=self.lowering_limits.max_batch_operations,
            max_batch_bytes=self.lowering_limits.max_batch_bytes,
        )

    def prepare(self, request) -> _PreparedCheckpointLoad:
        if not isinstance(request, WeightLoadRequest):
            raise self._error(
                "checkpoint provider only supports weight load requests",
                code="INVALID_REQUEST",
                phase="prepare",
                operation_id=getattr(request, "operation_id", "unknown"),
                cleanup_required=False,
            )
        if request.profile != "storage_to_runtime" or any(
            not isinstance(region.source, StorageWeightLocation)
            for region in request.plan.regions
        ):
            raise self._error(
                "checkpoint provider requires storage source locations",
                code="INVALID_SOURCE",
                phase="prepare",
                operation_id=request.operation_id,
                cleanup_required=False,
            )
        try:
            operation_count = lowering_operation_count(
                request.plan,
                self.lowering_limits,
            )
            if operation_count > self.lowering_limits.max_total_operations:
                raise ValueError(
                    "lowering exceeds total operation limit: "
                    f"{operation_count} > "
                    f"{self.lowering_limits.max_total_operations}"
                )
            verified_versions = self._verify_declared_checksums(request)
        except WeightTransferError:
            self._lifecycle.append(CheckpointProviderState.FAILED)
            raise
        except (TypeError, ValueError) as error:
            self._lifecycle.append(CheckpointProviderState.FAILED)
            raise self._error(
                str(error),
                code="LOWERING_LIMIT_EXCEEDED",
                phase="prepare",
                operation_id=request.operation_id,
                cleanup_required=False,
            ) from error
        prepared = _PreparedCheckpointLoad(
            request=request,
            verified_versions=verified_versions,
        )
        self._lifecycle.append(CheckpointProviderState.PREPARED)
        return prepared

    def submit(self, prepared: _PreparedCheckpointLoad) -> _CheckpointSubmission:
        if not isinstance(prepared, _PreparedCheckpointLoad) or prepared.released:
            raise ValueError("checkpoint load is not prepared")
        request = prepared.request
        self._lifecycle.append(CheckpointProviderState.SUBMITTED)
        batch_count = 0
        operation_count = 0
        max_batch_operations = 0
        max_batch_bytes = 0
        wrote_target = False
        try:
            for batch in iter_bounded_transfer_batches(
                request.plan,
                self.lowering_limits,
            ):
                batch_count += 1
                operation_count += len(batch.operations)
                max_batch_operations = max(
                    max_batch_operations,
                    len(batch.operations),
                )
                max_batch_bytes = max(max_batch_bytes, batch.total_bytes)
                for operation in batch.operations:
                    source = operation.source
                    if not isinstance(source, StorageWeightLocation):
                        raise self._error(
                            "checkpoint operation has a non-storage source",
                            code="INVALID_SOURCE",
                            phase="submit",
                            operation_id=request.operation_id,
                            cleanup_required=wrote_target,
                        )
                    source_key = self._source_key(source)
                    expected_version = prepared.verified_versions.get(source_key)
                    if expected_version is None:
                        raise self._error(
                            "checkpoint source version was not captured",
                            code="SOURCE_VERSION_REQUIRED",
                            phase="submit",
                            operation_id=request.operation_id,
                            cleanup_required=wrote_target,
                        )
                    payload = self._read_verified_range(
                        source,
                        source.object_offset + operation.source_offset,
                        operation.nbytes,
                        expected_version=expected_version,
                        operation_id=request.operation_id,
                        cleanup_required=wrote_target,
                    )
                    try:
                        self._write_target(
                            operation.target,
                            operation.target_offset,
                            payload,
                        )
                    except WeightTransferError:
                        raise
                    except BaseException as error:
                        raise self._error(
                            str(error) or error.__class__.__name__,
                            code="TARGET_WRITE_FAILED",
                            phase="submit",
                            operation_id=request.operation_id,
                            cleanup_required=True,
                        ) from error
                    wrote_target = True
        except BaseException:
            self._lifecycle.append(CheckpointProviderState.FAILED)
            raise

        receipt = WeightLoadReceipt(
            operation_id=request.operation_id,
            provider=self.name,
            plan_digest=request.plan.digest,
            total_bytes=request.plan.total_bytes,
            region_count=len(request.plan.regions),
            backend_receipts=(
                CheckpointLoadStats(
                    batch_count=batch_count,
                    operation_count=operation_count,
                    max_batch_operations=max_batch_operations,
                    max_batch_bytes=max_batch_bytes,
                ),
            ),
        )
        self._lifecycle.append(CheckpointProviderState.COMPLETED)
        return _CheckpointSubmission(prepared=prepared, receipt=receipt)

    def wait(self, submission: _CheckpointSubmission) -> WeightProviderReceipt:
        if (
            not isinstance(submission, _CheckpointSubmission)
            or submission.cancelled
            or submission.receipt is None
        ):
            raise ValueError("checkpoint submission has no completed receipt")
        return submission.receipt

    def cancel(self, submission: _CheckpointSubmission) -> None:
        if not isinstance(submission, _CheckpointSubmission):
            raise ValueError("invalid checkpoint submission")
        submission.cancelled = True
        self._lifecycle.append(CheckpointProviderState.CANCELLED)

    def synchronize(self, receipt: WeightProviderReceipt) -> None:
        if not isinstance(receipt, WeightLoadReceipt) or receipt.provider != self.name:
            raise ValueError("invalid checkpoint load receipt")

    def release(
        self,
        prepared: _PreparedCheckpointLoad,
        receipt: WeightProviderReceipt | None,
    ) -> None:
        del receipt
        if not isinstance(prepared, _PreparedCheckpointLoad):
            raise ValueError("invalid prepared checkpoint load")
        if prepared.released:
            return
        prepared.released = True
        self._lifecycle.append(CheckpointProviderState.RELEASED)

    def _verify_declared_checksums(
        self,
        request: WeightLoadRequest,
    ) -> dict[tuple, str]:
        unique_locations = {}
        for region in request.plan.regions:
            source = region.source
            assert isinstance(source, StorageWeightLocation)
            unique_locations[self._source_key(source)] = source
        verified_versions = {}
        for source in unique_locations.values():
            checksum = source.checksum
            if self._source_version_reader is None:
                raise self._error(
                    "injected storage readers require source version fencing",
                    code="SOURCE_VERSION_REQUIRED",
                    phase="prepare",
                    operation_id=request.operation_id,
                    cleanup_required=False,
                )
            source_version = self._read_source_version(
                source,
                phase="prepare",
                operation_id=request.operation_id,
                cleanup_required=False,
            )
            if checksum is None:
                verified_versions[self._source_key(source)] = source_version
                continue
            expected = self._parse_checksum(
                checksum,
                operation_id=request.operation_id,
            )
            digest = hashlib.sha256()
            relative_offset = 0
            while relative_offset < source.nbytes:
                nbytes = min(
                    self.checksum_chunk_bytes,
                    source.nbytes - relative_offset,
                )
                digest.update(
                    self._read_exact(
                        source,
                        source.object_offset + relative_offset,
                        nbytes,
                        phase="prepare",
                        operation_id=request.operation_id,
                        cleanup_required=False,
                    )
                )
                relative_offset += nbytes
            if (
                self._read_source_version(
                    source,
                    phase="prepare",
                    operation_id=request.operation_id,
                    cleanup_required=False,
                )
                != source_version
            ):
                raise self._error(
                    f"source version changed while checksumming {source.object_key}",
                    code="SOURCE_VERSION_CHANGED",
                    phase="prepare",
                    operation_id=request.operation_id,
                    cleanup_required=False,
                )
            if digest.hexdigest() != expected:
                raise self._error(
                    f"checksum mismatch for {source.object_key}",
                    code="CHECKSUM_MISMATCH",
                    phase="prepare",
                    operation_id=request.operation_id,
                    cleanup_required=False,
                )
            verified_versions[self._source_key(source)] = source_version
        return verified_versions

    def _read_verified_range(
        self,
        source: StorageWeightLocation,
        object_offset: int,
        nbytes: int,
        *,
        expected_version: str,
        operation_id: str,
        cleanup_required: bool,
    ) -> bytes:
        before = self._read_source_version(
            source,
            phase="submit",
            operation_id=operation_id,
            cleanup_required=cleanup_required,
        )
        if before != expected_version:
            raise self._error(
                f"source version changed before reading {source.object_key}",
                code="SOURCE_VERSION_CHANGED",
                phase="submit",
                operation_id=operation_id,
                cleanup_required=cleanup_required,
            )
        payload = self._read_exact(
            source,
            object_offset,
            nbytes,
            phase="submit",
            operation_id=operation_id,
            cleanup_required=cleanup_required,
        )
        after = self._read_source_version(
            source,
            phase="submit",
            operation_id=operation_id,
            cleanup_required=cleanup_required,
        )
        if after != expected_version:
            raise self._error(
                f"source version changed while reading {source.object_key}",
                code="SOURCE_VERSION_CHANGED",
                phase="submit",
                operation_id=operation_id,
                cleanup_required=cleanup_required,
            )
        return payload

    def _read_source_version(
        self,
        source: StorageWeightLocation,
        *,
        phase: str,
        operation_id: str,
        cleanup_required: bool,
    ) -> str:
        assert self._source_version_reader is not None
        try:
            version = self._source_version_reader(source)
        except BaseException as error:
            raise self._error(
                str(error) or error.__class__.__name__,
                code="SOURCE_VERSION_FAILED",
                phase=phase,
                operation_id=operation_id,
                cleanup_required=cleanup_required,
            ) from error
        if type(version) is not str or not version:
            raise self._error(
                "source version reader must return a non-empty string",
                code="SOURCE_VERSION_FAILED",
                phase=phase,
                operation_id=operation_id,
                cleanup_required=cleanup_required,
            )
        return version

    def _read_exact(
        self,
        source: StorageWeightLocation,
        object_offset: int,
        nbytes: int,
        *,
        phase: str,
        operation_id: str,
        cleanup_required: bool,
    ) -> bytes:
        try:
            payload = self._range_reader(source, object_offset, nbytes)
        except BaseException as error:
            raise self._error(
                str(error) or error.__class__.__name__,
                code="SOURCE_READ_FAILED",
                phase=phase,
                operation_id=operation_id,
                cleanup_required=cleanup_required,
            ) from error
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise self._error(
                "storage range reader must return bytes-like data",
                code="SOURCE_READ_FAILED",
                phase=phase,
                operation_id=operation_id,
                cleanup_required=cleanup_required,
            )
        result = bytes(payload)
        if len(result) != nbytes:
            raise self._error(
                f"short storage range read: {len(result)} != {nbytes}",
                code="SOURCE_READ_FAILED",
                phase=phase,
                operation_id=operation_id,
                cleanup_required=cleanup_required,
            )
        return result

    @staticmethod
    def _read_local_range(
        source: StorageWeightLocation,
        object_offset: int,
        nbytes: int,
    ) -> bytes:
        path = Path(source.object_key)
        with path.open("rb") as checkpoint_file:
            checkpoint_file.seek(object_offset)
            return checkpoint_file.read(nbytes)

    @staticmethod
    def _read_local_version(source: StorageWeightLocation) -> str:
        stat = Path(source.object_key).stat()
        return ":".join(
            str(value)
            for value in (
                stat.st_dev,
                stat.st_ino,
                stat.st_size,
                stat.st_mtime_ns,
                stat.st_ctime_ns,
            )
        )

    @staticmethod
    def _source_key(source: StorageWeightLocation) -> tuple:
        return (
            source.provider,
            source.storage_id,
            source.object_key,
            source.object_offset,
            source.nbytes,
            source.checksum,
        )

    def _write_target(
        self,
        target: RuntimeWeightLocation,
        target_offset: int,
        payload: bytes,
    ) -> None:
        if (
            type(target_offset) is not int
            or target_offset < 0
            or not payload
            or target_offset > target.nbytes - len(payload)
        ):
            raise ValueError("target range exceeds runtime binding")
        if self._target_writer is not None:
            self._target_writer(target, target_offset, payload)
            return
        target_address = target.address + target_offset
        if target.device == "cpu":
            ctypes.memmove(target_address, payload, len(payload))
            return
        if target.device == "cuda" or target.device.startswith("cuda:"):
            if self._cuda_runtime is None:
                from sglang.srt.distributed.device_communicators.cuda_wrapper import (
                    CudaRTLibrary,
                )

                self._cuda_runtime = CudaRTLibrary()
            host_buffer = ctypes.create_string_buffer(payload, len(payload))
            self._cuda_runtime.cudaMemcpy(
                ctypes.c_void_p(target_address),
                ctypes.cast(host_buffer, ctypes.c_void_p),
                len(payload),
            )
            return
        raise ValueError(f"unsupported target device: {target.device}")

    def _parse_checksum(self, checksum: str, *, operation_id: str) -> str:
        prefix = "sha256:"
        digest = checksum.removeprefix(prefix)
        if (
            not checksum.startswith(prefix)
            or len(digest) != 64
            or digest != digest.lower()
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise self._error(
                "checkpoint checksum must be a canonical sha256 digest",
                code="UNSUPPORTED_CHECKSUM",
                phase="prepare",
                operation_id=operation_id,
                cleanup_required=False,
            )
        return digest

    def _error(
        self,
        message: str,
        *,
        code: str,
        phase: str,
        operation_id: str,
        cleanup_required: bool,
    ) -> WeightTransferError:
        return WeightTransferError(
            message,
            code=code,
            provider=self.name,
            phase=phase,
            operation_id=operation_id,
            retryable=False,
            completion_known=True,
            cleanup_required=cleanup_required,
        )
