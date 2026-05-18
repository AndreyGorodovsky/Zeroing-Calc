Explain the zeroing calculation pipeline step by step as it applies to: $ARGUMENTS

Read the source code from cell index 0 of Calc.ipynb (parse the notebook JSON to extract it). Then explain the full pipeline for the given context — covering these stages in order:
1. `_register_to` — image alignment
2. `_to_gray_norm` — CLAHE normalisation
3. `estimate_group_center_from_diff_avg_robust` — shot hole detection
4. `estimate_px_per_cm_from_grid` — scale calibration
5. `pixels_to_clicks` — offset to adjustment conversion

If $ARGUMENTS names a specific function or stage, go deep on that one. If $ARGUMENTS is empty, give an overview of all stages.
