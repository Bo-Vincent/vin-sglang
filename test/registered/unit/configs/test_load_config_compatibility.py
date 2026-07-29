from __future__ import annotations

import pytest

from sglang.srt.configs.load_config import LoadConfig, LoadFormat
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_load_config_preserves_legacy_positional_fields() -> None:
    config = LoadConfig(
        LoadFormat.AUTO,
        None,
        {},
        None,
        None,
        -1,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        "https://modelexpress.example",
        "nixl",
    )

    assert config.modelexpress_url == "https://modelexpress.example"
    assert config.modelexpress_transport == "nixl"
    assert config.remote_instance_weight_runtime_manifest_builder is None
    assert config.remote_instance_weight_transfer_provider_factory is None
    assert config.weight_snapshot_backend_factory is None
    assert config.weight_snapshot_world_barrier is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
