from __future__ import annotations

import msgspec
import pytest

from sglang.srt.model_executor import weight_inventory_contracts as contracts


def _topology() -> contracts.WeightParallelTopology:
    return contracts.WeightParallelTopology(
        tp_rank=1,
        tp_size=2,
        attention_tp_rank=1,
        attention_tp_size=2,
        moe_tp_rank=1,
        moe_tp_size=2,
    )


def _axes() -> tuple[contracts.LogicalParallelAxis, ...]:
    return (
        contracts.LogicalParallelAxis(kind="dp", mode="replicated"),
        contracts.LogicalParallelAxis(kind="tp", mode="split", dim=0),
        contracts.LogicalParallelAxis(kind="pp", mode="ownership"),
        contracts.LogicalParallelAxis(kind="ep", mode="replicated"),
    )


def _fragment(
    *,
    rank: contracts.WeightParallelRank | None = None,
) -> contracts.WeightPlacementInventoryFragment:
    topology = _topology()
    fragment_rank = rank or topology.rank()
    facts = {
        "tensor_id": "layers.0.weight",
        "aliases": (),
        "global_shape": (4, 3),
        "global_offset": (2, 0),
        "local_shape": (2, 3),
        "dtype": "float32",
        "itemsize": 4,
        "shard_dims": (0,),
        "parallel_axes": _axes(),
        "layer_id": 0,
        "expert_id": None,
        "layout_fingerprint": "contiguous-c-order",
        "nbytes": 24,
        "rank": fragment_rank,
    }
    return contracts.WeightPlacementInventoryFragment(
        placement_fragment_id=contracts._placement_fragment_id(**facts),
        **facts,
    )


def _placement(
    *,
    fragment: contracts.WeightPlacementInventoryFragment | None = None,
    weight_generation: int = 7,
) -> contracts.WeightPlacementInventory:
    topology = _topology()
    fragments = (fragment or _fragment(),)
    facts = {
        "model_id": "model",
        "revision": "immutable-revision",
        "weight_generation": weight_generation,
        "topology": topology,
        "fragments": fragments,
    }
    return contracts.WeightPlacementInventory(
        inventory_id=contracts._placement_id(**facts),
        participant_id=contracts._participant_id(
            model_id=facts["model_id"],
            revision=facts["revision"],
            topology=topology,
        ),
        **facts,
    )


def _binding_fragment(
    placement_fragment: contracts.WeightPlacementInventoryFragment,
    **overrides,
) -> contracts.WeightRuntimeBindingInventoryFragment:
    facts = {
        "placement_fragment_id": placement_fragment.placement_fragment_id,
        "fragment_id": "runtime-fragment",
        "address": 4096,
        "nbytes": 24,
        "storage_offset": 0,
        "itemsize": 4,
        "local_shape": (2, 3),
        "strides_bytes": (12, 4),
        "storage_address": 4096,
        "storage_nbytes": 24,
        "storage_offset_bytes": 0,
        "device": "cpu",
        "is_contiguous": True,
        "worker_id": "ephemeral-worker",
        "endpoint": "127.0.0.1:12345",
    }
    facts.update(overrides)
    return contracts.WeightRuntimeBindingInventoryFragment(**facts)


def _binding(
    placement: contracts.WeightPlacementInventory,
    *,
    fragment: contracts.WeightRuntimeBindingInventoryFragment | None = None,
) -> contracts.WeightRuntimeBindingInventory:
    return contracts.WeightRuntimeBindingInventory(
        model_id=placement.model_id,
        revision=placement.revision,
        placement_inventory_id=placement.inventory_id,
        instance_id="ephemeral-instance",
        generation=19,
        lease_id="ephemeral-lease",
        participant_id=placement.participant_id,
        fragments=(fragment or _binding_fragment(placement.fragments[0]),),
    )


def test_placement_rejects_fragment_rank_mismatch_direct() -> None:
    fragment = _fragment(rank=contracts.WeightParallelRank(tp=0))

    with pytest.raises(ValueError, match="fragment rank differs from.*topology"):
        _placement(fragment=fragment)


def test_current_sglang_inventory_rejects_empty_participants() -> None:
    """SGLang currently exports only participants that own weight fragments."""

    topology = _topology()
    facts = {
        "model_id": "model",
        "revision": "immutable-revision",
        "weight_generation": 7,
        "topology": topology,
        "fragments": (),
    }

    with pytest.raises(ValueError, match="must contain fragments"):
        contracts.WeightPlacementInventory(
            inventory_id=contracts._placement_id(**facts),
            participant_id=contracts._participant_id(
                model_id=facts["model_id"],
                revision=facts["revision"],
                topology=topology,
            ),
            **facts,
        )


def test_placement_rejects_fragment_rank_mismatch_from_wire() -> None:
    placement = _placement()
    payload = msgspec.to_builtins(placement)
    payload["fragments"] = [
        msgspec.to_builtins(_fragment(rank=contracts.WeightParallelRank(tp=0)))
    ]

    with pytest.raises(Exception, match="fragment rank differs from.*topology"):
        msgspec.json.decode(
            msgspec.json.encode(payload),
            type=contracts.WeightPlacementInventory,
        )


def test_placement_rejects_false_tp_ep_coupling() -> None:
    topology = contracts.WeightParallelTopology(
        tp_rank=1,
        tp_size=2,
        ep_rank=0,
        ep_size=2,
        attention_tp_rank=1,
        attention_tp_size=2,
    )
    rank = topology.rank()
    parallel_axes = (
        contracts.LogicalParallelAxis(kind="dp", mode="replicated"),
        contracts.LogicalParallelAxis(kind="tp", mode="split", dim=0),
        contracts.LogicalParallelAxis(kind="pp", mode="ownership"),
        contracts.LogicalParallelAxis(
            kind="ep",
            mode="coupled",
            coupled_to="tp",
        ),
    )
    fragment_facts = {
        "tensor_id": "layers.0.weight",
        "aliases": (),
        "global_shape": (4, 3),
        "global_offset": (2, 0),
        "local_shape": (2, 3),
        "dtype": "float32",
        "itemsize": 4,
        "shard_dims": (0,),
        "parallel_axes": parallel_axes,
        "layer_id": 0,
        "expert_id": None,
        "layout_fingerprint": "contiguous-c-order",
        "nbytes": 24,
        "rank": rank,
    }
    fragment = contracts.WeightPlacementInventoryFragment(
        placement_fragment_id=contracts._placement_fragment_id(**fragment_facts),
        **fragment_facts,
    )
    placement_facts = {
        "model_id": "model",
        "revision": "immutable-revision",
        "weight_generation": 7,
        "topology": topology,
        "fragments": (fragment,),
    }

    with pytest.raises(ValueError, match="coupled parallel axis differs"):
        contracts.WeightPlacementInventory(
            inventory_id=contracts._placement_id(**placement_facts),
            participant_id=contracts._participant_id(
                model_id="model",
                revision="immutable-revision",
                topology=topology,
            ),
            **placement_facts,
        )


def test_pairing_accepts_self_consistent_fragment_geometry() -> None:
    placement = _placement()

    paired = contracts.WeightPlacementBindingInventories(
        placement=placement,
        binding=_binding(placement),
    )

    assert paired.placement is placement


def test_content_generation_changes_inventory_but_not_participant_identity() -> None:
    generation_one = _placement(weight_generation=1)
    generation_seven = _placement(weight_generation=7)

    assert generation_one.participant_id == generation_seven.participant_id
    assert generation_one.inventory_id != generation_seven.inventory_id


@pytest.mark.parametrize(
    ("model_id", "revision"),
    (("", "immutable-revision"), ("model", ""), ("model", "default")),
)
def test_remote_lineage_requires_an_explicit_revision(
    model_id,
    revision,
) -> None:
    with pytest.raises(ValueError, match="content-lineage revision"):
        contracts.validate_remote_weight_lineage(
            model_id=model_id,
            revision=revision,
        )


def test_remote_source_identity_accepts_the_loaded_content() -> None:
    assert contracts.validate_remote_weight_source_identity(
        requested_model_id="model",
        requested_revision="immutable-revision",
        loaded_model_id="model",
        loaded_revision="immutable-revision",
    ) == ("model", "immutable-revision")


@pytest.mark.parametrize(
    ("loaded_model_id", "loaded_revision", "message"),
    (
        ("other-model", "immutable-revision", "does not match"),
        ("model", "other-revision", "does not match"),
        ("model", None, "content-lineage revision"),
    ),
)
def test_remote_source_identity_rejects_unattested_loaded_content(
    loaded_model_id,
    loaded_revision,
    message,
) -> None:
    with pytest.raises(contracts.WeightInventoryError, match=message):
        contracts.validate_remote_weight_source_identity(
            requested_model_id="model",
            requested_revision="immutable-revision",
            loaded_model_id=loaded_model_id,
            loaded_revision=loaded_revision,
        )


@pytest.mark.parametrize(
    "overrides",
    (
        {
            "itemsize": 2,
            "nbytes": 12,
            "strides_bytes": (6, 2),
            "storage_nbytes": 12,
        },
        {
            "local_shape": (3, 2),
            "strides_bytes": (8, 4),
        },
    ),
)
def test_pairing_rejects_fragment_geometry_mismatch(overrides) -> None:
    placement = _placement()
    binding_fragment = _binding_fragment(placement.fragments[0], **overrides)

    with pytest.raises(ValueError, match="fragment geometry differs"):
        contracts.WeightPlacementBindingInventories(
            placement=placement,
            binding=_binding(placement, fragment=binding_fragment),
        )


def test_binding_rejects_falsely_contiguous_strides() -> None:
    placement = _placement()

    with pytest.raises(ValueError, match="contiguous strides"):
        _binding_fragment(placement.fragments[0], strides_bytes=(4, 8))


def test_binding_rejects_storage_bounds_even_when_geometry_matches() -> None:
    placement = _placement()

    with pytest.raises(ValueError, match="exceeds its storage"):
        _binding_fragment(
            placement.fragments[0],
            address=4100,
            storage_offset=1,
            storage_offset_bytes=4,
        )
