# Scenario

Compare two prediction files only after proving they use the same fold manifest,
native-unit universe, aggregation policy, label order, and checkpoint lineage.
Report unequal support as FAIL instead of intersecting away missing predictions.
