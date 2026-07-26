from types import SimpleNamespace

import pytest
import torch

from sglang.srt.model_loader.remote_instance_weight_loader_utils import (
    register_memory_region_v2,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class FakeEngine:
    def __init__(self) -> None:
        self.registrations = []

    def register_memory(self, address: int, nbytes: int) -> int:
        self.registrations.append((address, nbytes))
        return 0


def parameter(address: int, nbytes: int = 256):
    return SimpleNamespace(
        data_ptr=lambda: address,
        numel=lambda: nbytes // 2,
        element_size=lambda: 2,
    )


def test_register_memory_region_covers_parameter_view_inside_allocation(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        torch.cuda.memory,
        "memory_snapshot",
        lambda: [
            {
                "blocks": [
                    {
                        "address": 0x10000,
                        "size": 0x1000,
                        "state": "active_allocated",
                    }
                ]
            }
        ],
    )
    model = SimpleNamespace(
        named_parameters=lambda: [("view.weight", parameter(0x10100))]
    )
    engine = FakeEngine()

    weight_info = register_memory_region_v2(model, engine)

    assert engine.registrations == [(0x10000, 0x1000)]
    assert weight_info == {"view.weight": (0x10100, 128, 2)}


def test_register_memory_region_rejects_uncovered_parameter(monkeypatch) -> None:
    monkeypatch.setattr(
        torch.cuda.memory,
        "memory_snapshot",
        lambda: [
            {
                "blocks": [
                    {
                        "address": 0x10000,
                        "size": 0x1000,
                        "state": "active_allocated",
                    }
                ]
            }
        ],
    )
    model = SimpleNamespace(
        named_parameters=lambda: [("missing.weight", parameter(0x30000))]
    )

    with pytest.raises(RuntimeError, match="not covered"):
        register_memory_region_v2(model, FakeEngine())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
