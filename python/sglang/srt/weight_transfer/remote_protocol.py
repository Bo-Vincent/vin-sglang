from __future__ import annotations

RUNTIME_MANIFEST_V1 = "runtime_v1"
PLACEMENT_BINDING_V1 = "placement_binding_v1"

HF_REVISION_V1 = "hf_revision_v1"
ARTIFACT_WEIGHT_VERSION_V1 = "artifact_weight_version_v1"


def validate_manifest_revision_semantics(
    manifest_format: str,
    revision_semantics: str,
) -> None:
    if manifest_format not in {RUNTIME_MANIFEST_V1, PLACEMENT_BINDING_V1}:
        raise ValueError(f"unsupported source manifest format: {manifest_format}")
    if revision_semantics not in {
        HF_REVISION_V1,
        ARTIFACT_WEIGHT_VERSION_V1,
    }:
        raise ValueError(
            f"unsupported manifest revision semantics: {revision_semantics}"
        )
