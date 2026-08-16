import json
from pathlib import Path

import pandas as pd

from pig_behavior.classification_v2.training.post_s1_host_binding import (
    ensure_post_s1_t6_host_binding,
)
from pig_behavior.classification_v2.training.remote_input_resolution import (
    load_remote_input_authority,
)


base = Path("/home/zeus/c2v2_overnight_20260812")
runtime = base / "runtime"
bootstrap = base / "bootstrap_31d983920b287eccb170145ee0b93c9fa15e9587"
route = runtime / (
    "docs/classification_v2/corrected_pooled_route_20260806/"
    "next_phase_20260806_r2"
)
bundle = runtime / (
    "outputs/classification_v2/post_s1_cpu_host_bindings_20260812/"
    "t6_binding_host_bound_v3h/stage1_temporal_rgb_bindings.json"
)
bundle_text = bundle.read_text()
payload, _ = json.JSONDecoder().raw_decode(bundle_text.lstrip())
roles = pd.read_csv(
    runtime
    / (
        "outputs/classification_v2/post_s1_cpu_host_bindings_20260812/"
        "t6_binding_host_bound_v3h/post_s1_cpu_20260812_t6/"
        "stage1_window_context.csv"
    ),
    usecols=["window_id", "stage1_role"],
    low_memory=False,
)
roles = roles.rename(columns={"stage1_role": "primary_s1_role"})
runtime_binding_path = runtime / (
    "outputs/classification_v2/post_s1_cpu_host_bindings_20260812/"
    "runtime_input_binding.json"
)
runtime_binding = json.loads(runtime_binding_path.read_text())
registration = bootstrap / (
    "code/docs/classification_v2/corrected_pooled_route_20260806/"
    "next_phase_20260806_r2/cvat_source_registration_authority_20260811.json"
)
binding_path = runtime / (
    "outputs/classification_v2/post_s1_cpu_host_bindings_20260812/"
    "t6_binding_host_bound_v4/post_s1_cpu_20260812_t6/"
    "post_s1_host_binding.json"
)
result = ensure_post_s1_t6_host_binding(
    binding_path=binding_path,
    canonical_code_sha="31d983920b287eccb170145ee0b93c9fa15e9587",
    input_authority=load_remote_input_authority(
        route / "remote_input_root_contract_20260811.json"
    ),
    runtime_input_binding=runtime_binding,
    media_root=Path("/inputs"),
    rgb_source_root=Path("/inputs/reviewed_rgb_v1"),
    t6_population_authority_sha256=(
        "9daf6a3bda89678c2b3ddd6ba9a0132fa1be16b583bfafb1cc76eb610ff8b1e4"
    ),
    t6_population_provenance_hashes=payload["views"]["T6"]["provenance_hashes"],
    requested_roles=roles,
    input_resolution=64,
    cvat_source_registration_path=registration,
)
print(
    json.dumps(
        {
            "status": "PASS",
            "binding_path": str(result.binding_path),
            "binding_sha256": result.binding_sha256,
            "regenerated": result.regenerated,
            "role_counts": result.payload["scientific_identity"]["role_counts"],
            "coverage": dict(result.rgb.coverage),
        },
        indent=2,
    )
)
