Add a new test case to the appropriate test file for: $ARGUMENTS

First read the source from Calc.ipynb cell 0. Then look at the existing tests in tests/ to understand the custom pass/fail harness pattern used (no pytest — tests use a minimal custom runner).

Write a new test that:
- Follows the exact same harness pattern as existing tests in that file
- Tests the function or scenario described in: $ARGUMENTS
- Uses synthetic images from images/synth/ or images/synth_centers/ if real images aren't needed
- Includes a clear test name and a one-line comment explaining what property is being verified

Append the test to the most relevant existing test file (test_calc.py, test_synth.py, or test_alignment.py). Do not create a new file.
