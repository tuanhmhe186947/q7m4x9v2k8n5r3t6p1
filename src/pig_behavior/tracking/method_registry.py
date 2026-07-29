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
    scientific_role: str
    entry_point: str
    detector_contract: str
    tracker_contract: str
    state_lifecycle: str
    future_frame_policy: str
    stage_graph: tuple[str, ...]
    export_contract: str
    artifact_authority: tuple[str, ...]
    execution_authority_status: tuple[str, ...]
    unseen_authorization_status: str


_BYTETRACK_RAW = TrackingMethodContract(
    method_id="bytetrack_raw",
    scientific_role="ORIGINAL_BASELINE",
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
    ),
    execution_authority_status=("EXECUTABLE_DEVELOPMENT_BASELINE",),
    unseen_authorization_status="NOT_AUTHORIZED",
)

_HYBRID_BYTETRACK = TrackingMethodContract(
    method_id="hybrid_bytetrack",
    scientific_role="COMPLETE_OPTIMIZED_OFFLINE_METHOD",
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
    ),
    execution_authority_status=(
        "HISTORICAL_ARTIFACT_AUTHORITY_ESTABLISHED",
        "ALGORITHMIC_LINEAGE_RECOVERED",
        "EXACT_NUMERICAL_RUNTIME_NOT_RECOVERED",
    ),
    unseen_authorization_status="NOT_AUTHORIZED",
)

_REALTIME_FAST = TrackingMethodContract(
    method_id="realtime_fast",
    scientific_role="CAUSAL_REALTIME_METHOD",
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
    ),
    execution_authority_status=("FROZEN_CAUSAL_EXECUTION_CONTRACT",),
    unseen_authorization_status="NOT_AUTHORIZED",
)

SCIENTIFIC_METHOD_REGISTRY: Mapping[str, TrackingMethodContract] = (
    MappingProxyType(
        {
            contract.method_id: contract
            for contract in (
                _BYTETRACK_RAW,
                _HYBRID_BYTETRACK,
                _REALTIME_FAST,
            )
        }
    )
)

ACTIVE_SCIENTIFIC_METHOD_IDS = tuple(SCIENTIFIC_METHOD_REGISTRY)


def get_scientific_method(method_id: str) -> TrackingMethodContract:
    """Return one canonical scientific method contract."""

    return SCIENTIFIC_METHOD_REGISTRY[method_id]


__all__ = [
    "ACTIVE_SCIENTIFIC_METHOD_IDS",
    "SCIENTIFIC_METHOD_REGISTRY",
    "TrackingMethodContract",
    "get_scientific_method",
]
