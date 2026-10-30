# Track C: performance log

Every step from the reference to your final version, each with the live-board run ID that measured it. Judges check that the run IDs exist and that each change moved the way you say it did.

| Step | Change | Run ID | T (s) | M (MB) | Points |
|---|---|---|---|---|---|
| 0 | The reference, unchanged | | 15.0 | 367 | 0 |
| 1 | C++ rewrite: mmap, quote-parity chunking so threads can start at real record boundaries, CSV state machine, flat region table, per-thread totals, sorted-key distinct count. Local hash check passes on sample and full. | | | | |
