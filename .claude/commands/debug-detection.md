Help debug why shot hole detection is failing or producing unexpected results for: $ARGUMENTS

Read the source from Calc.ipynb cell 0. Focus on `estimate_group_center_from_diff_avg_robust` and the full `process_shot` pipeline.

Walk through each stage of the detection logic:
1. Was alignment (`_register_to`) likely to succeed? What could cause it to fall back to pure translation?
2. Is the abs-diff producing meaningful signal, or would CLAHE/bilateral filtering wash out the holes?
3. Does the contour filter's area range match what's expected for the given image/scale?
4. Is the fallback (intensity-weighted centroid) being triggered, and if so, why?

If $ARGUMENTS includes a file path or specific test name, look at that test's inputs and trace the expected values through each stage. Suggest concrete parameter tweaks or checks to isolate the issue.
