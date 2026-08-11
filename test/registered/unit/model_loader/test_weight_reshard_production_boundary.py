from __future__ import annotations

import ast
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]


def _source(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def _tree(relative_path: str) -> ast.Module:
    return ast.parse(_source(relative_path), filename=relative_path)


def _defined_names(relative_path: str) -> set[str]:
    return {
        node.name
        for node in _tree(relative_path).body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _class_fields(relative_path: str, class_name: str) -> set[str]:
    for node in _tree(relative_path).body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {
                child.target.id
                for child in node.body
                if isinstance(child, ast.AnnAssign)
                and isinstance(child.target, ast.Name)
            }
    raise AssertionError(f"class not found: {relative_path}: {class_name}")


class WeightReshardProductionBoundaryTest(unittest.TestCase):
    def test_local_contract_exposes_only_inventory_types(self) -> None:
        path = "python/sglang/srt/model_executor/weight_inventory_contracts.py"
        source = _source(path)
        names = _defined_names(path)

        self.assertTrue(
            {
                "WeightPlacementInventory",
                "WeightRuntimeBindingInventory",
                "WeightPlacementBindingInventories",
            }
            <= names
        )
        self.assertNotIn("WeightRuntimeManifest", names)
        self.assertNotIn("compose_weight_runtime_manifest", names)
        for token in ("WeightTargetPlacementManifest", "format_version"):
            if token in source:
                self.fail(f"retired token remains in {path}: {token}")
        self.assertFalse(
            (
                REPO_ROOT
                / "python/sglang/srt/model_executor/weight_runtime_manifest.py"
            ).exists()
        )

    def test_public_inventory_contains_only_stable_logical_facts(self) -> None:
        path = "python/sglang/srt/model_executor/weight_inventory_contracts.py"
        fragment_fields = _class_fields(path, "WeightPlacementInventoryFragment")
        view_fields = _class_fields(path, "LogicalTensorView")

        self.assertTrue({"aliases", "shard_dims", "parallel_axes"} <= fragment_fields)
        self.assertFalse(
            {"runtime_name", "byte_offset", "partition_dim"} & fragment_fields
        )
        self.assertFalse({"partition_dim"} & view_fields)

        semantics_root = REPO_ROOT / "python/sglang/srt/model_executor/weight_semantics"
        offenders = [
            path.relative_to(REPO_ROOT).as_posix()
            for path in semantics_root.rglob("*.py")
            if "partition_dim" in path.read_text(encoding="utf-8")
        ]
        self.assertEqual(offenders, [])

    def test_inventory_identity_excludes_runtime_storage_facts_and_versions(
        self,
    ) -> None:
        contracts_path = (
            "python/sglang/srt/model_executor/weight_inventory_contracts.py"
        )
        inventory_path = "python/sglang/srt/model_executor/weight_inventory.py"
        contracts = _source(contracts_path)
        inventory = _source(inventory_path)

        self.assertIn("sglang-weight-placement-fragment", contracts)
        self.assertIn("sglang-weight-placement-inventory", contracts)
        for token in ("weight-placement-v1", "weight-placement-v2"):
            self.assertNotIn(token, contracts)
        for function_name in ("_placement_fragment_id", "_placement_id"):
            function = next(
                node
                for node in _tree(contracts_path).body
                if isinstance(node, ast.FunctionDef) and node.name == function_name
            )
            function_source = ast.unparse(function)
            for private_fact in (
                "runtime_name",
                "byte_offset",
                "instance_id",
                "worker_id",
            ):
                self.assertNotIn(private_fact, function_source)

        self.assertIn("_PhysicalFragmentLookup", inventory)
        self.assertIn("view_byte_offset", inventory)

    def test_production_wire_has_one_placement_binding_interface(self) -> None:
        paths = (
            "python/sglang/srt/entrypoints/http_server.py",
            "python/sglang/srt/managers/io_struct.py",
            "python/sglang/srt/managers/tokenizer_control_mixin.py",
            "python/sglang/srt/managers/scheduler_components/weight_updater.py",
            "python/sglang/srt/model_loader/remote_instance_weight_loader_utils.py",
            "python/sglang/srt/model_loader/loader.py",
        )
        retired = (
            "runtime_v1",
            "placement_binding_v1",
            "manifest_format",
            "weight_runtime_manifests",
        )
        for path in paths:
            source = _source(path)
            for token in retired:
                with self.subTest(path=path, token=token):
                    if token in source:
                        self.fail(f"retired token remains in {path}: {token}")

    def test_loader_depends_only_on_backend_neutral_protocol(self) -> None:
        loader_path = "python/sglang/srt/model_loader/loader.py"
        loader_source = _source(loader_path)
        loader_tree = _tree(loader_path)
        imported_modules = {
            node.module
            for node in ast.walk(loader_tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }

        self.assertIn(
            "sglang.srt.model_loader.weight_reshard_backend",
            imported_modules,
        )
        self.assertNotIn("mooncake.reshard.weight", imported_modules)
        for symbol in (
            "MooncakeCanonicalReshardAdapter",
            "MooncakeTransferEngineReader",
            "MemoryRegistrationLease",
            "bind_logical_transfer_plan",
            "plan_placement_transfer_to_local_target",
            "TransferCompletionUnknownError",
            "TransferEngineError",
        ):
            if symbol in loader_source:
                self.fail(f"loader directly depends on backend symbol: {symbol}")

    def test_backend_neutral_protocol_and_mooncake_backend_are_separate(self) -> None:
        neutral_path = "python/sglang/srt/model_loader/weight_reshard_backend.py"
        mooncake_path = "python/sglang/srt/model_loader/mooncake_reshard_backend.py"

        neutral_names = _defined_names(neutral_path)
        mooncake_names = _defined_names(mooncake_path)
        self.assertTrue(
            {
                "PreparedWeightReshardTransfer",
                "WeightReshardBackend",
                "WeightReshardCompletionUnknownError",
                "create_weight_reshard_backend",
            }
            <= neutral_names
        )
        self.assertIn("MooncakeWeightReshardBackend", mooncake_names)
        self.assertNotIn("mooncake.reshard.weight", _source(neutral_path))

    def test_current_factory_rejects_unimplemented_reshard_backends(self) -> None:
        from sglang.srt.model_loader.weight_reshard_backend import (
            WeightReshardBackendUnavailableError,
            create_weight_reshard_backend,
        )

        for backend_name in ("nccl", "modelexpress"):
            with self.subTest(backend_name=backend_name):
                with self.assertRaisesRegex(
                    WeightReshardBackendUnavailableError,
                    "unsupported weight reshard backend",
                ):
                    create_weight_reshard_backend(backend_name)

    def test_model_runner_never_recomposes_combined_runtime_manifest(self) -> None:
        runner = _source("python/sglang/srt/model_executor/model_runner.py")
        transporter = _source(
            "python/sglang/srt/model_executor/model_runner_components/"
            "remote_instance_weight_transporter.py"
        )
        load_utils = _source(
            "python/sglang/srt/model_executor/model_runner_components/"
            "load_model_utils.py"
        )

        for token in (
            "compose_weight_runtime_manifest",
            "get_remote_instance_weight_runtime_manifest(",
            "build_remote_instance_target_weight_runtime_manifest(",
            ".snapshot(",
        ):
            if token in runner:
                self.fail(f"model_runner retains combined interface: {token}")
        self.assertIn("validate_runtime_binding_inventory_addresses", transporter)
        self.assertNotIn("validate_runtime_manifest_addresses", transporter)
        self.assertNotIn("remote_instance_weight_runtime_manifest_builder", load_utils)

    def test_target_binding_uses_the_real_update_and_activation_coordinator(
        self,
    ) -> None:
        runner = _source("python/sglang/srt/model_executor/model_runner.py")

        self.assertNotIn("coordinator=WeightSnapshotCoordinator()", runner)
        self.assertIn("coordinator=self.weight_snapshot_coordinator", runner)
        self.assertIn("adopt_weight_generation_from_snapshot", runner)
        self.assertLess(
            runner.index("self.init_weight_snapshot_coordinator()"),
            runner.index("self.initialize()"),
        )
        self.assertLess(
            runner.index("self.initialize()"),
            runner.index("self.init_weight_updater()"),
        )

    def test_source_export_attests_the_loaded_model_identity(self) -> None:
        runner = _source("python/sglang/srt/model_executor/model_runner.py")
        loader = _source("python/sglang/srt/model_loader/loader.py")

        self.assertIn("validate_remote_weight_source_identity", runner)
        self.assertIn("self.server_args.get_weight_reshard_resource_id()", runner)
        tokenizer = _source("python/sglang/srt/managers/tokenizer_control_mixin.py")
        self.assertIn(
            "model_id=self.server_args.get_weight_reshard_resource_id()", tokenizer
        )
        self.assertIn("loaded_revision=self.model_config.revision", runner)
        self.assertIn("load_config.weight_reshard_resource_id", loader)

    def test_canonical_translation_is_confined_to_adapter_and_backend(self) -> None:
        allowed = {
            "python/sglang/srt/model_executor/mooncake_reshard_adapter.py",
            "python/sglang/srt/model_loader/mooncake_reshard_backend.py",
        }
        roots = (
            REPO_ROOT / "python/sglang/srt/model_executor",
            REPO_ROOT / "python/sglang/srt/model_loader",
        )
        offenders = []
        for root in roots:
            for path in root.rglob("*.py"):
                relative = path.relative_to(REPO_ROOT).as_posix()
                if relative in allowed:
                    continue
                if "mooncake.reshard.weight" in path.read_text(encoding="utf-8"):
                    offenders.append(relative)
        self.assertEqual(offenders, [])

    def test_tests_do_not_import_retired_manifest_owner_or_manager(self) -> None:
        tests_root = REPO_ROOT / "test/registered/unit"
        offenders = []
        for path in tests_root.rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            for token in (
                "model_executor." + "weight_runtime_manifest",
                "WeightRuntime" + "ManifestManager",
                "create_weight_" + "runtime_manifest_manager",
                "create_sglang_weight_" + "runtime_manifest_manager",
            ):
                if token in source:
                    offenders.append((path.relative_to(REPO_ROOT).as_posix(), token))
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
