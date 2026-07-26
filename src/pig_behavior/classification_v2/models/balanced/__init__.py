"""Balanced causal main-model research scaffolding for classification_v2.

This subpackage is model-research scaffolding only. It never reads production
run roots, never builds production datasets, and never publishes canonical
artifacts. It reuses the repository's existing schema authorities
(``features.motion_schema`` and ``spatial_sequence_export``) instead of
declaring a second copy of the feature order.
"""

from pig_behavior.classification_v2.models.balanced.contracts import (
    BATCH_CONTRACT_CHECKS,
    MOTION_FEATURE_NAMES,
    MOTION_SCHEMA_DIMENSION,
    NUMERIC_GROUP_NAMES,
    SPATIAL_PREDICTIVE_DIMENSION,
    BatchContract,
    ContractCheck,
    ContractReport,
    ModelBatch,
    SequenceSegment,
    TensorContractError,
    numeric_group_feature_names,
    require_batch,
    validate_batch,
)
from pig_behavior.classification_v2.models.balanced.registry import (
    BALANCED_MODEL_NAMES,
    ModelSpec,
    build_model,
    model_spec,
    model_spec_contract,
)

__all__ = [
    "BALANCED_MODEL_NAMES",
    "BATCH_CONTRACT_CHECKS",
    "MOTION_FEATURE_NAMES",
    "MOTION_SCHEMA_DIMENSION",
    "NUMERIC_GROUP_NAMES",
    "SPATIAL_PREDICTIVE_DIMENSION",
    "BatchContract",
    "ContractCheck",
    "ContractReport",
    "ModelBatch",
    "ModelSpec",
    "SequenceSegment",
    "TensorContractError",
    "build_model",
    "model_spec",
    "model_spec_contract",
    "numeric_group_feature_names",
    "require_batch",
    "validate_batch",
]
