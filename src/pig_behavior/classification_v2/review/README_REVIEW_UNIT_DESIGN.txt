classification_v2 review-unit design
====================================

Canonical rule:
- review_unit_id is the key for human review decisions.
- window_id remains the key for training windows.
- Do not review every 6/8/12/16 training window manually.
- The default queue selects evidence-bearing or policy-required native units.
- A lineage claiming complete legacy 16f behavior review must build with
  --include-all-retained-legacy-units and pass --require-complete-legacy.

Review unit types:
- legacy_burst_16: one legacy recovered burst/pig/tracklet, display all 16 frames.
- cvat_interval_6: one CVAT anchor interval, display all 6 frames anchor..anchor+5.

Workflow:
1) Build window-level spatial-temporal outputs first:
   outputs/classification_v2/sequence_features/temporal_label_intervals.csv
   outputs/classification_v2/sequence_features/sequence_window_manifest.csv
   outputs/classification_v2/sequence_features/sequence_window_features.csv

2) Build normal review templates if you want window-level reasons:
   python scripts/classification_v2/01_review_units_gui/build_behavior_review_templates.py

3) Build canonical review units:
   python scripts/classification_v2/01_review_units_gui/classification_v2_build_review_units.py

4) Create interaction review unit shortlist:
   python scripts/classification_v2/01_review_units_gui/make_interaction_review_unit_shortlist.py

5) Test GUI by source:
   python scripts/classification_v2/01_review_units_gui/run_interaction_review_unit_gui_pilot.py ^
     --fresh --source-type legacy_recovered --max-items 5
   python scripts/classification_v2/01_review_units_gui/run_interaction_review_unit_gui_pilot.py ^
     --fresh --source-type cvat_tracking_xml --max-items 5

Outputs:
- outputs/classification_v2/review_units/review_unit_manifest.csv
- outputs/classification_v2/review_units/*_review_unit_template.csv
- outputs/classification_v2/review_units/interaction_review_unit_shortlist.csv
- outputs/classification_v2/review_policy/interaction_review_unit_gui_pilot/
  behavior_unit_review_decisions.csv

Important:
- Legacy GUI uses crop_path directly from spatiotemporal_frame_features_enhanced.csv.
- CVAT GUI uses video frames and crops by bbox.
- The GUI writes unit-level decisions, not expanded training-window decisions.
- A later apply step should apply unit decisions to temporal intervals/enhanced
  frame labels and then rebuild sequence windows 6/8/12/16.
