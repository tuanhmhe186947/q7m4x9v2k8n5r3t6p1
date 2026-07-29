"""Canonical scientific tracking-method contracts.

Historical aliases belong in provenance documents, not in this active
registry. Internal engine names such as ``realtime`` are implementation
details and are recorded only inside the corresponding contract.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True, slots=True)
class TrackingMethodContract:
    """Immutable scientific identity and execution contract for one method."""

    method_id: str
    canonical_version: str
    scientific_role: str
    scientific_status: str
    prediction_authority_type: str
    prediction_authority_path: str
    prediction_authority_hash: str
    execution_authority_status: tuple[str, ...]
    entry_point: str
    detector_contract: str
    tracker_contract: str
    state_lifecycle: str
    future_frame_policy: str
    causal: bool
    development_evaluation_eligible: bool
    runtime_benchmark_eligible: bool
    deployment_eligible: bool
    unseen_execution_eligible: str
    recommended_runtime_use: str
    known_limitations: tuple[str, ...]
    provenance_authority_path: str
    stage_graph: tuple[str, ...]
    export_contract: str
    artifact_authority: tuple[str, ...]
    unseen_authorization_status: str


_BYTETRACK_RAW = TrackingMethodContract(
    method_id="bytetrack_raw",
    canonical_version="current_b0",
    scientific_role="CURRENT_EXECUTABLE_BYTETRACK_BASELINE",
    scientific_status="ACTIVE_EXECUTABLE_BASELINE",
    prediction_authority_type="FROZEN_EXECUTABLE_PREDICTION_SET",
    prediction_authority_path=(
        "outputs/tracking/frozen_predictions_standard_v2_20260728_retry1/"
        "B0_bytetrack_raw/predictions"
    ),
    prediction_authority_hash=(
        "13d9226c36141264cc33e4b498d38e5f3eaa9891cf32bc4c8fb87b01fd27d576"
    ),
    execution_authority_status=("ESTABLISHED",),
    entry_point="scripts/run_tracking_mode.py --mode bytetrack_raw",
    detector_contract=(
        "LIVE_YOLO_TRACK; per-frame detector input; profile-bound confidence "
        "and ByteTrack YAML; no detector-cache replay"
    ),
    tracker_contract=(
        "Ultralytics ByteTrack output parsed into fixed project identities "
        "without accepted hybrid repair flags"
    ),
    state_lifecycle=(
        "one YOLO/tracker instance per video; persist=True across frames; "
        "no cross-video state"
    ),
    future_frame_policy="NONE",
    causal=True,
    development_evaluation_eligible=True,
    runtime_benchmark_eligible=True,
    deployment_eligible=True,
    unseen_execution_eligible="PENDING_SEPARATE_PREFLIGHT",
    recommended_runtime_use="YES",
    known_limitations=(
        "Current executable baseline is not identical to the archived "
        "historical raw artifact.",
        "Complete-method comparisons include detector and producer semantics.",
    ),
    provenance_authority_path=(
        "docs/tracking/method_standardization/"
        "BYTETRACK_RAW_AUTHORITY_RESOLUTION_20260730.json"
    ),
    stage_graph=(
        "VIDEO_DECODE",
        "LIVE_YOLO_TRACK",
        "BYTETRACK_PERSISTENT_STATE",
        "DETECTION_PARSE",
        "FIXED_EIGHT_INITIALIZATION",
        "RAW_PROJECT_TRACK_OUTPUT",
        "FINAL_CVAT_XML",
    ),
    export_contract="DETERMINISTIC_PROJECT_JSON_COCO_AND_CVAT_VIDEO_XML",
    artifact_authority=(
        "docs/tracking/three_mode_historical_reconstruction/"
        "B0_HISTORICAL_RECONSTRUCTION_AUTHORITY_20260729.json",
        "docs/tracking/reconciliation/"
        "STATE_8_DEVELOPMENT_EVALUATION_AUTHORITY_20260729.json",
    ),
    unseen_authorization_status="DEVELOPMENT_ONLY_NOT_PRIMARY_RQ2_UNSEEN",
)

_HYBRID_BYTETRACK = TrackingMethodContract(
    method_id="hybrid_bytetrack",
    canonical_version="historical_h5b_h4_final",
    scientific_role="HISTORICAL_OFFLINE_DEVELOPMENT_CHAMPION",
    scientific_status="HISTORICAL_ARTIFACT_AUTHORITY_ESTABLISHED",
    prediction_authority_type="SURVIVING_HISTORICAL_FINAL_XML_SET",
    prediction_authority_path=(
        "outputs/tracking/historical_h5b_h4_frozen_predictions_20260728/"
        "predictions"
    ),
    prediction_authority_hash=(
        "36c3bdd3f6d92c0c5336590dbf4c8822402d718ec47df2b137cef862339d5b8a"
    ),
    execution_authority_status=(
        "HISTORICAL_ARTIFACT_AUTHORITY_ESTABLISHED",
        "ALGORITHMIC_LINEAGE_RECOVERED",
        "EXACT_NUMERICAL_RUNTIME_NOT_RECOVERED",
    ),
    entry_point="scripts/run_tracking_mode.py --mode hybrid_bytetrack",
    detector_contract=(
        "HISTORICAL_LIVE_YOLO_TRACK; det_conf=0.20; max_raw_detections=64; "
        "low-confidence rows preserved through historical parsing semantics"
    ),
    tracker_contract=(
        "complete accepted hybrid_bytetrack lineage with ByteTrack-specific "
        "owner, occlusion, re-entry, identity, Hidden, geometry, H5b, and H4 "
        "mechanisms"
    ),
    state_lifecycle=(
        "one YOLO/tracker instance per video; persist=True across frames; "
        "post-video accepted stages may use future frames; no cross-video state"
    ),
    future_frame_policy="POST_VIDEO_ALLOWED_BY_ACCEPTED_LINEAGE_ONLY",
    causal=False,
    development_evaluation_eligible=True,
    runtime_benchmark_eligible=False,
    deployment_eligible=False,
    unseen_execution_eligible="NO",
    recommended_runtime_use="NO",
    known_limitations=(
        "Exact numerical historical runtime is not recovered.",
        "Surviving final XMLs are the development prediction authority.",
        "Must not be represented by standardized B1 predictions.",
    ),
    provenance_authority_path=(
        "docs/tracking/method_standardization/"
        "CANONICAL_TRACKING_METHOD_AUTHORITY_20260730.json"
    ),
    stage_graph=(
        "LIVE_YOLO_TRACK",
        "BYTETRACK_PERSISTENT_STATE",
        "DETECTION_PARSE_AND_LOW_CONF_FLOW",
        "FIXED_EIGHT_INITIALIZATION",
        "RAW_OWNER_AND_LOST_REACQUIRE_GUARDS",
        "HIDDEN_OWNER_GUARD",
        "OCCLUSION_REENTRY_GUARDS",
        "RAW_PROJECT_TRACK_OUTPUT",
        "OFFLINE_IDENTITY_SWAP_GUARD",
        "TEMPORAL_BBOX_REFINEMENT",
        "OVERLAP_HIDDEN_ISLAND_STABILIZATION",
        "LOCAL_EPISODE_LONG_PAIR_REPAIRS",
        "SUFFIX_PAIR_SWAP_REPAIR",
        "OVERLAP_SMALL_BOX_SUPPRESSION",
        "H5B_HIDDEN_SUFFIX_OVERLAP_PERSISTENCE",
        "REALTIME_MOTION_PAIR_STABILIZER",
        "NEAR_WALL_HIDDEN_GEOMETRY",
        "FAR_CAMERA_GEOMETRY_DURING_H5B",
        "H5B_CVAT_EXPORT",
        "H4_FAR_CAMERA_GEOMETRY_REPLAY",
        "FINAL_CVAT_XML",
    ),
    export_contract=(
        "HISTORICAL_ACCEPTED_FINAL_CVAT_XML_SEMANTICS_PLUS_PROJECT_EXPORTS"
    ),
    artifact_authority=(
        "docs/tracking/historical_hybrid_best_recovery/"
        "HISTORICAL_HYBRID_BEST_LINEAGE_RECOVERY_AUTHORITY_20260729.json",
        "docs/tracking/historical_hybrid_best_recovery/"
        "HISTORICAL_HYBRID_ACCEPTED_ALGORITHM_GRAPH_20260729.json",
        "historical final XML set: 20260719_h5b_h4_full13_combined_v2",
        "docs/tracking/reconciliation/"
        "STATE_8_DEVELOPMENT_EVALUATION_AUTHORITY_20260729.json",
    ),
    unseen_authorization_status=(
        "DEVELOPMENT_ARTIFACT_ONLY_EXACT_RUNTIME_UNAVAILABLE"
    ),
)

_REALTIME_FAST = TrackingMethodContract(
    method_id="realtime_fast",
    canonical_version="v1",
    scientific_role="FROZEN_CAUSAL_REALTIME_PRIMARY",
    scientific_status="FROZEN_CAUSAL_REALTIME_PRIMARY",
    prediction_authority_type="FROZEN_EXECUTABLE_PREDICTION_SET",
    prediction_authority_path=(
        "outputs/tracking/current_main_baseline_20260728/predictions"
    ),
    prediction_authority_hash=(
        "fd2d4f3dec0710d1c9eecba9308247a7b226dd34a4a02a9cb89f17acb22bbbfe"
    ),
    execution_authority_status=("ESTABLISHED",),
    entry_point="scripts/run_tracking_mode.py --mode realtime_fast",
    detector_contract=(
        "profile-bound YOLO predict cadence; detect_every_n_frames=2; "
        "max_raw_detections=32"
    ),
    tracker_contract=(
        "causal fixed-identity association with realtime competitor and "
        "unassigned-track guards; no ByteTrack internal state"
    ),
    state_lifecycle=(
        "one causal runtime state per video; skipped frames use motion "
        "prediction; no future-frame or cross-video state"
    ),
    future_frame_policy="CAUSAL_ZERO_DELAY",
    causal=True,
    development_evaluation_eligible=True,
    runtime_benchmark_eligible=True,
    deployment_eligible=True,
    unseen_execution_eligible="PENDING_SEPARATE_PREFLIGHT",
    recommended_runtime_use="YES",
    known_limitations=(
        "Development authority is frozen on the 13-video population.",
        "Unseen execution requires a separate preflight freeze.",
    ),
    provenance_authority_path=(
        "docs/tracking/method_standardization/"
        "CANONICAL_TRACKING_METHOD_AUTHORITY_20260730.json"
    ),
    stage_graph=(
        "VIDEO_DECODE",
        "CADENCED_YOLO_PREDICT",
        "CAUSAL_ASSOCIATION",
        "VISIBLE_COMPETITOR_GUARDS",
        "SKIP_FRAME_MOTION_PREDICTION",
        "ZERO_DELAY_EXPORT",
    ),
    export_contract="DETERMINISTIC_PROJECT_JSON_COCO_AND_CVAT_VIDEO_XML",
    artifact_authority=(
        "docs/tracking/three_mode_historical_reconstruction/"
        "R0_HISTORICAL_RECONSTRUCTION_AUTHORITY_20260729.json",
        "docs/tracking/CURRENT_MAIN_R0_BASELINE_AUTHORITY_20260728.json",
        "docs/tracking/reconciliation/"
        "STATE_8_DEVELOPMENT_EVALUATION_AUTHORITY_20260729.json",
    ),
    unseen_authorization_status="PENDING_SEPARATE_PREFLIGHT",
)

_RF_HYBRID = TrackingMethodContract(
    method_id="rf_hybrid",
    canonical_version="v1",
    scientific_role="FROZEN_MIXED_TRANSFER_ABLATION",
    scientific_status="FROZEN_MIXED_TRANSFER_ABLATION",
    prediction_authority_type="FROZEN_DEVELOPMENT_ABLATION_PREDICTIONS",
    prediction_authority_path=(
        "outputs/tracking/reconciliation_state8_development_20260729_run2/"
        "predictions/rf_hybrid"
    ),
    prediction_authority_hash=(
        "8ecf2a42bedced243232dbf8c537d1177c0abed77bb43dfa74c117185d2e528e"
    ),
    execution_authority_status=(
        "DEVELOPMENT_EVALUATION_AUTHORITY_ESTABLISHED",
        "TRANSFER_SIGNAL_MIXED",
    ),
    entry_point="scripts/run_tracking_mode.py --mode rf_hybrid",
    detector_contract=(
        "identical realtime_fast detector evidence and cadence; no detector "
        "rerun after the frozen realtime_fast tracklet boundary"
    ),
    tracker_contract=(
        "frozen realtime_fast causal tracklets followed only by the "
        "predeclared portable and RF-native transfer stage set; no ByteTrack "
        "internal state or raw ByteTrack IDs"
    ),
    state_lifecycle=(
        "one causal realtime_fast state per video; immutable raw tracklet "
        "snapshot; post-video transfer state; no cross-video state"
    ),
    future_frame_policy=(
        "POST_VIDEO_ALLOWED_FOR_DECLARED_OFFLINE_TRANSFER_STAGES"
    ),
    causal=False,
    development_evaluation_eligible=True,
    runtime_benchmark_eligible=False,
    deployment_eligible=False,
    unseen_execution_eligible="NO",
    recommended_runtime_use="NO",
    known_limitations=(
        "Mixed transfer result: lower IDSW but worse HOTA, IDF1, and "
        "wrong-identity exposure than realtime_fast.",
        "Not a quality or deployment upgrade.",
        "rf_hybrid v2 is a rejected provenance-only candidate.",
    ),
    provenance_authority_path=(
        "docs/tracking/method_standardization/"
        "CANONICAL_TRACKING_METHOD_AUTHORITY_20260730.json"
    ),
    stage_graph=(
        "CADENCED_YOLO_PREDICT",
        "CAUSAL_ASSOCIATION",
        "FROZEN_REALTIME_FAST_TRACKLETS",
        "OFFLINE_IDENTITY_SWAP_GUARD",
        "TEMPORAL_BBOX_REFINEMENT",
        "OVERLAP_HIDDEN_ISLAND_STABILIZATION",
        "LOCAL_EPISODE_LONG_PAIR_REPAIRS",
        "SUFFIX_PAIR_SWAP_REPAIR",
        "OVERLAP_SMALL_BOX_SUPPRESSION",
        "H5B_HIDDEN_SUFFIX_OVERLAP_PERSISTENCE",
        "REALTIME_MOTION_PAIR_STABILIZER",
        "NEAR_WALL_HIDDEN_GEOMETRY",
        "FAR_CAMERA_GEOMETRY_DURING_H5B",
        "RF_HYBRID_CHANGE_LEDGER",
        "FINAL_CVAT_XML",
    ),
    export_contract=(
        "FROZEN_REALTIME_FAST_OUTPUT_PLUS_RF_HYBRID_OUTPUT_LEDGER_AND_CVAT_XML"
    ),
    artifact_authority=(
        "docs/tracking/reconciliation/"
        "RF_HYBRID_STAGE_PORTABILITY_20260729.csv",
        "docs/tracking/reconciliation/"
        "STATE_6_RF_HYBRID_PORTABILITY_AUTHORITY_20260729.json",
        "docs/tracking/reconciliation/"
        "STATE_8_DEVELOPMENT_EVALUATION_AUTHORITY_20260729.json",
    ),
    unseen_authorization_status="NO",
)

SCIENTIFIC_METHOD_REGISTRY: Mapping[str, TrackingMethodContract] = (
    MappingProxyType(
        {
            contract.method_id: contract
            for contract in (
                _BYTETRACK_RAW,
                _HYBRID_BYTETRACK,
                _REALTIME_FAST,
                _RF_HYBRID,
            )
        }
    )
)

ACTIVE_SCIENTIFIC_METHOD_IDS = tuple(SCIENTIFIC_METHOD_REGISTRY)

PROVENANCE_ALIASES: Mapping[str, Mapping[str, str]] = MappingProxyType(
    {
        "B0": MappingProxyType(
            {"kind": "PROVENANCE_ALIAS", "canonical_method_id": "bytetrack_raw"}
        ),
        "B1": MappingProxyType(
            {"kind": "FORENSIC_ONLY", "canonical_method_id": ""}
        ),
        "R0": MappingProxyType(
            {"kind": "PROVENANCE_ALIAS", "canonical_method_id": "realtime_fast"}
        ),
        "realtime": MappingProxyType(
            {"kind": "COMPATIBILITY_ONLY", "canonical_method_id": "realtime_fast"}
        ),
        "R1": MappingProxyType(
            {"kind": "PROVENANCE_ALIAS", "canonical_method_id": "rf_hybrid"}
        ),
        "historical_h5b_h4": MappingProxyType(
            {"kind": "PROVENANCE_ALIAS", "canonical_method_id": "hybrid_bytetrack"}
        ),
        "hybrid_bytetrack_best": MappingProxyType(
            {"kind": "COMPATIBILITY_ONLY", "canonical_method_id": "hybrid_bytetrack"}
        ),
        "archived_historical_bytetrack_raw": MappingProxyType(
            {"kind": "HISTORICAL_ARTIFACT", "canonical_method_id": ""}
        ),
        "standardized_b1": MappingProxyType(
            {"kind": "FORENSIC_ONLY", "canonical_method_id": ""}
        ),
        "hybrid_bytetrack_best_recovered": MappingProxyType(
            {"kind": "REJECTED_CANDIDATE", "canonical_method_id": ""}
        ),
        "rf_hybrid_v2": MappingProxyType(
            {"kind": "REJECTED_CANDIDATE", "canonical_method_id": ""}
        ),
        "rf_native_hybrid_candidate": MappingProxyType(
            {"kind": "REJECTED_CANDIDATE", "canonical_method_id": ""}
        ),
    }
)

_FORBIDDEN_ACTIVE_IDS = frozenset(PROVENANCE_ALIASES)


def validate_method_registry() -> None:
    """Raise ``ValueError`` when active method identity is contaminated."""

    expected = (
        "bytetrack_raw",
        "hybrid_bytetrack",
        "realtime_fast",
        "rf_hybrid",
    )
    if ACTIVE_SCIENTIFIC_METHOD_IDS != expected:
        raise ValueError("active scientific method IDs are not canonical")
    if _FORBIDDEN_ACTIVE_IDS.intersection(SCIENTIFIC_METHOD_REGISTRY):
        raise ValueError("provenance alias exposed as active method")
    for contract in SCIENTIFIC_METHOD_REGISTRY.values():
        required = (
            contract.method_id,
            contract.canonical_version,
            contract.scientific_role,
            contract.scientific_status,
            contract.prediction_authority_type,
            contract.prediction_authority_path,
            contract.prediction_authority_hash,
            contract.execution_authority_status,
            contract.provenance_authority_path,
        )
        if not all(required):
            raise ValueError(f"incomplete authority metadata: {contract.method_id}")


validate_method_registry()


def get_scientific_method(method_id: str) -> TrackingMethodContract:
    """Return one canonical scientific method contract."""

    return SCIENTIFIC_METHOD_REGISTRY[method_id]


__all__ = [
    "ACTIVE_SCIENTIFIC_METHOD_IDS",
    "PROVENANCE_ALIASES",
    "SCIENTIFIC_METHOD_REGISTRY",
    "TrackingMethodContract",
    "get_scientific_method",
    "validate_method_registry",
]
