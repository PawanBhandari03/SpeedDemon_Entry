# Track C: performance log

Every step from the reference to your final version, each with the live-board run ID that measured it. Judges check that the run IDs exist and that each change moved the way you say it did.

| Step | Change | Run ID | T (s) | M (MB) | Points |
|---|---|---|---|---|---|
| 0 | The reference, unchanged | | 15.0 | 367 | 0 |
| 1 | C++ rewrite: mmap, quote-parity chunking so threads can start at real record boundaries, CSV state machine, flat region table, per-thread totals, sorted-key distinct count. Local hash check passes on sample and full. | commit 9ab7648 | 0.145 | 46 | 55.1 |
| 2 | Profiled the phases locally. Counting distinct names was the second-largest phase (65-90 ms): it sorted 3.3M (group, hash) keys per thread and again per group. Now each thread appends the name hash to a per-group list while parsing, and each group is de-duplicated with a small open-addressing set (linear time, no sort): 65-90 ms -> 10 ms. Reading regions.csv took 19 ms because every region did a binary search over the codes; a code is three letters A-Z, so it now indexes a 26^3 table directly: 19 ms -> 5 ms. The note's hash was computed and never used; it is skipped now. Whole run locally: about 165-185 ms -> 120-130 ms. | | | | |
