from __future__ import annotations

import fcntl
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator, TypeVar

import msgspec
from sglang.srt.weight_transfer.storage import (
    InMemoryWeightStorageCatalog,
    StoredWeightSnapshot,
    WeightMaterializationAttempt,
    WeightMaterializationIntent,
    WeightRevisionHead,
    WeightRevisionState,
    WeightSnapshotPublication,
    WeightStorageRef,
)

_CATALOG_FORMAT = "sglang-weight-storage-catalog"
_CATALOG_VERSION = 1
_ResultT = TypeVar("_ResultT")


class _CatalogEnvelope(
    msgspec.Struct,
    frozen=True,
    forbid_unknown_fields=True,
):
    format: str
    version: int
    materializations: tuple[WeightMaterializationAttempt, ...]
    publications: tuple[WeightSnapshotPublication, ...]
    revision_heads: tuple[WeightRevisionHead, ...] = ()


class FileWeightStorageCatalog:
    """File-backed catalog with process-safe, durable state transitions."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        if not self.path.name:
            raise ValueError("catalog path must identify a file")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_path = self.path.with_name(f".{self.path.name}.lock")
        with self._locked():
            self._load_unlocked()

    def begin_materialization(
        self,
        materialization_id: str,
        intent: WeightMaterializationIntent,
    ) -> WeightMaterializationAttempt:
        return self._mutate(
            lambda catalog: catalog.begin_materialization(
                materialization_id,
                intent,
            )
        )

    def complete_materialization(
        self,
        materialization_id: str,
        snapshot: StoredWeightSnapshot,
    ) -> WeightMaterializationAttempt:
        return self._mutate(
            lambda catalog: catalog.complete_materialization(
                materialization_id,
                snapshot,
            )
        )

    def abort_materialization(
        self,
        materialization_id: str,
    ) -> WeightMaterializationAttempt:
        return self._mutate(
            lambda catalog: catalog.abort_materialization(materialization_id)
        )

    def set_materialization_completion_ticket(
        self,
        materialization_id: str,
        completion_ticket: str,
    ) -> WeightMaterializationAttempt:
        return self._mutate(
            lambda catalog: catalog.set_materialization_completion_ticket(
                materialization_id,
                completion_ticket,
            )
        )

    def clear_materialization_completion_ticket(
        self,
        materialization_id: str,
        completion_ticket: str,
    ) -> WeightMaterializationAttempt:
        return self._mutate(
            lambda catalog: catalog.clear_materialization_completion_ticket(
                materialization_id,
                completion_ticket,
            )
        )

    def get_materialization(
        self,
        materialization_id: str,
    ) -> WeightMaterializationAttempt | None:
        return self._read(
            lambda catalog: catalog.get_materialization(materialization_id)
        )

    def recoverable_materializations(
        self,
    ) -> tuple[WeightMaterializationAttempt, ...]:
        return self._read(lambda catalog: catalog.recoverable_materializations())

    def prepare_publish(
        self,
        publication_id: str,
        snapshot: StoredWeightSnapshot,
    ) -> WeightSnapshotPublication:
        return self._mutate(
            lambda catalog: catalog.prepare_publish(publication_id, snapshot)
        )

    def publish(self, publication_id: str) -> WeightSnapshotPublication:
        return self._mutate(lambda catalog: catalog.publish(publication_id))

    def abort(self, publication_id: str) -> WeightSnapshotPublication:
        return self._mutate(lambda catalog: catalog.abort(publication_id))

    def get_snapshot(
        self,
        ref: WeightStorageRef,
    ) -> StoredWeightSnapshot | None:
        return self._read(lambda catalog: catalog.get_snapshot(ref))

    def get_publication(
        self,
        publication_id: str,
    ) -> WeightSnapshotPublication | None:
        return self._read(lambda catalog: catalog.get_publication(publication_id))

    def recoverable_publications(
        self,
    ) -> tuple[WeightSnapshotPublication, ...]:
        return self._read(lambda catalog: catalog.recoverable_publications())

    def get_revision_head(
        self,
        model_id: str,
        revision: str,
    ) -> WeightRevisionHead | None:
        return self._read(lambda catalog: catalog.get_revision_head(model_id, revision))

    def compare_and_set_revision(
        self,
        *,
        model_id: str,
        revision: str,
        expected: WeightRevisionHead | None,
        new_ref: WeightStorageRef,
        new_state: WeightRevisionState,
    ) -> WeightRevisionHead | None:
        return self._mutate(
            lambda catalog: catalog.compare_and_set_revision(
                model_id=model_id,
                revision=revision,
                expected=expected,
                new_ref=new_ref,
                new_state=new_state,
            )
        )

    def export_materializations(
        self,
    ) -> tuple[WeightMaterializationAttempt, ...]:
        return self._read(lambda catalog: catalog.export_materializations())

    def export_publications(
        self,
    ) -> tuple[WeightSnapshotPublication, ...]:
        return self._read(lambda catalog: catalog.export_publications())

    def export_revision_heads(self) -> tuple[WeightRevisionHead, ...]:
        return self._read(lambda catalog: catalog.export_revision_heads())

    def _read(
        self,
        operation: Callable[[InMemoryWeightStorageCatalog], _ResultT],
    ) -> _ResultT:
        with self._locked():
            return operation(self._load_unlocked())

    def _mutate(
        self,
        operation: Callable[[InMemoryWeightStorageCatalog], _ResultT],
    ) -> _ResultT:
        with self._locked():
            catalog = self._load_unlocked()
            result = operation(catalog)
            try:
                self._commit_unlocked(catalog)
            except Exception:
                try:
                    committed = self._catalog_state(
                        self._load_unlocked()
                    ) == self._catalog_state(catalog)
                except Exception:
                    committed = False
                if committed:
                    return result
                raise
            return result

    @staticmethod
    def _catalog_state(
        catalog: InMemoryWeightStorageCatalog,
    ) -> tuple[tuple, tuple, tuple]:
        return (
            catalog.export_materializations(),
            catalog.export_publications(),
            catalog.export_revision_heads(),
        )

    @contextmanager
    def _locked(self) -> Iterator[None]:
        with self._lock_path.open("a+b") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _load_unlocked(self) -> InMemoryWeightStorageCatalog:
        if not self.path.exists():
            return InMemoryWeightStorageCatalog()
        payload = self.path.read_bytes()
        try:
            envelope = msgspec.json.decode(payload, type=_CatalogEnvelope)
        except (msgspec.DecodeError, TypeError, ValueError) as error:
            raise ValueError("invalid catalog file") from error
        if envelope.format != _CATALOG_FORMAT or envelope.version != _CATALOG_VERSION:
            raise ValueError(
                "unsupported catalog format or version: "
                f"{envelope.format!r} v{envelope.version!r}"
            )
        try:
            return InMemoryWeightStorageCatalog(
                materializations=envelope.materializations,
                publications=envelope.publications,
                revision_heads=envelope.revision_heads,
            )
        except (TypeError, ValueError) as error:
            raise ValueError("invalid catalog state") from error

    def _commit_unlocked(
        self,
        catalog: InMemoryWeightStorageCatalog,
    ) -> None:
        envelope = _CatalogEnvelope(
            format=_CATALOG_FORMAT,
            version=_CATALOG_VERSION,
            materializations=catalog.export_materializations(),
            publications=catalog.export_publications(),
            revision_heads=catalog.export_revision_heads(),
        )
        payload = msgspec.json.encode(envelope)
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            dir=self.path.parent,
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(file_descriptor, "wb") as temporary_file:
                temporary_file.write(payload)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, self.path)
            self._fsync_directory()
        finally:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass

    def _fsync_directory(self) -> None:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_descriptor = os.open(self.path.parent, flags)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
