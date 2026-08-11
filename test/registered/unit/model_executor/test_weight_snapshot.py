from __future__ import annotations

import pytest

from sglang.srt.model_executor.model_runner_components.weight_update_coordination import (
    coordinated_weight_update,
)
from sglang.srt.model_executor.weight_inventory_contracts import WeightInventoryError
from sglang.srt.model_executor.weight_snapshot import WeightSnapshotCoordinator


class _Clock:
    def __init__(self, value: float) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def test_expired_lease_remains_a_mutation_fence_until_explicit_release() -> None:
    clock = _Clock(100.0)
    coordinator = WeightSnapshotCoordinator(clock=clock)
    lease_id, _ = coordinator.acquire_snapshot(lease_timeout_sec=30)

    clock.advance(30)

    assert coordinator.list_snapshot_leases()[0].expired is True
    assert coordinator.has_snapshot(lease_id) is True
    assert coordinator.snapshot_is_active(lease_id) is False
    with pytest.raises(WeightInventoryError, match="lease is active"):
        coordinator.begin_update()
    with pytest.raises(WeightInventoryError, match="expired.*explicit release"):
        coordinator.renew_snapshot(lease_id, lease_timeout_sec=30)

    coordinator.release_snapshot(lease_id)
    token = coordinator.begin_update()
    coordinator.cancel_update(token)


def test_successful_update_requires_exact_generation_commit() -> None:
    coordinator = WeightSnapshotCoordinator()
    first = coordinator.begin_update()
    first_generation = coordinator.finish_update(first, success=True)

    with pytest.raises(WeightInventoryError, match="weight generation commit"):
        coordinator.begin_update()
    with pytest.raises(WeightInventoryError, match="weight generation commit"):
        coordinator.acquire_snapshot()

    coordinator.commit_weight_generation(expected_generation=first_generation)
    second = coordinator.begin_update()
    second_generation = coordinator.finish_update(second, success=True)

    with pytest.raises(WeightInventoryError, match="generation does not match"):
        coordinator.commit_weight_generation(expected_generation=first_generation)

    coordinator.commit_weight_generation(expected_generation=second_generation)
    lease_id, generation = coordinator.acquire_snapshot()
    assert generation == second_generation
    coordinator.release_snapshot(lease_id)


def test_update_fences_before_mutation_and_before_generation_publish() -> None:
    events = []
    coordinator = WeightSnapshotCoordinator(
        completion_fence=lambda: events.append(("fence", coordinator.generation))
    )

    class Updater:
        def __init__(self) -> None:
            self.begin_weight_update = coordinator.begin_update
            self.finish_weight_update = coordinator.finish_update

        @coordinated_weight_update
        def update(self):
            events.append(("mutation", coordinator.generation))
            return True, "updated"

    assert Updater().update() == (True, "updated")
    assert events == [("fence", 1), ("mutation", 1), ("fence", 1)]
    assert coordinator.generation == 2
    with pytest.raises(WeightInventoryError, match="weight generation commit"):
        coordinator.acquire_snapshot()


def test_pre_mutation_fence_failure_cancels_reservation() -> None:
    coordinator = WeightSnapshotCoordinator(
        completion_fence=lambda: (_ for _ in ()).throw(RuntimeError("drain failed"))
    )

    with pytest.raises(RuntimeError, match="drain failed"):
        coordinator.begin_update()

    lease_id, generation = coordinator.acquire_snapshot()
    assert generation == 1
    coordinator.release_snapshot(lease_id)


def test_post_mutation_fence_failure_poisons_unpublished_content() -> None:
    calls = 0

    def fence() -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("device work failed")

    coordinator = WeightSnapshotCoordinator(completion_fence=fence)
    token = coordinator.begin_update()

    with pytest.raises(RuntimeError, match="device work failed"):
        coordinator.finish_update(token, success=True)

    assert coordinator.generation == 2
    with pytest.raises(WeightInventoryError, match="full successful weight restore"):
        coordinator.acquire_snapshot()


def test_failed_update_stays_poisoned_until_complete_restore_commits() -> None:
    coordinator = WeightSnapshotCoordinator()
    failed = coordinator.begin_update()
    coordinator.finish_update(failed, success=False)

    with pytest.raises(WeightInventoryError, match="full successful weight restore"):
        coordinator.begin_update()
    with pytest.raises(WeightInventoryError, match="full successful weight restore"):
        coordinator.commit_weight_generation()

    restored = coordinator.begin_update(full_restore=True)
    restore_generation = coordinator.finish_update(restored, success=True)
    with pytest.raises(WeightInventoryError, match="weight generation commit"):
        coordinator.acquire_snapshot()
    coordinator.commit_weight_generation(expected_generation=restore_generation)

    lease_id, generation = coordinator.acquire_snapshot()
    assert generation == restore_generation
    coordinator.release_snapshot(lease_id)


def test_only_explicit_full_restore_protocol_can_clear_poison() -> None:
    coordinator = WeightSnapshotCoordinator()
    failed = coordinator.begin_update()
    coordinator.finish_update(failed, success=False)

    class Updater:
        def __init__(self) -> None:
            self.begin_weight_update = coordinator.begin_update
            self.finish_weight_update = coordinator.finish_update

        @coordinated_weight_update
        def update_weights_from_disk(
            self,
            model_path,
            load_format,
            weight_name_filter=None,
        ):
            del model_path, load_format, weight_name_filter
            return True, "restored"

        @coordinated_weight_update(full_restore=True)
        def restore_weights_with_complete_coverage_proof(self):
            return True, "restored"

    updater = Updater()
    rejected = updater.update_weights_from_disk("checkpoint", "auto")
    assert rejected[0] is False

    assert updater.restore_weights_with_complete_coverage_proof() == (
        True,
        "restored",
    )
    coordinator.commit_weight_generation()


def test_storage_relocation_changes_binding_generation_not_weight_content() -> None:
    coordinator = WeightSnapshotCoordinator()
    original_weight_generation = coordinator.weight_generation

    release = coordinator.begin_storage_transition()
    released_generation = coordinator.finish_storage_transition(
        release,
        success=True,
        storage_available=False,
    )

    assert released_generation == 2
    assert coordinator.weight_generation == original_weight_generation
    with pytest.raises(WeightInventoryError, match="storage is unavailable"):
        coordinator.acquire_snapshot()

    resume = coordinator.begin_storage_transition()
    resumed_generation = coordinator.finish_storage_transition(
        resume,
        success=True,
        storage_available=True,
    )

    assert resumed_generation == 3
    assert coordinator.weight_generation == original_weight_generation
    lease_id, binding_generation = coordinator.acquire_snapshot()
    assert binding_generation == resumed_generation
    coordinator.release_snapshot(lease_id)
