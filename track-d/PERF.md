# Track D: performance log

Every step from the reference to your final version, each with the live-board run ID that measured it. Judges check that the run IDs exist and that each change moved the way you say it did.

| Step | Change | Run ID | T (s) | M (MB) | Points |
|---|---|---|---|---|---|
| 0 | The reference, unchanged | | 72.69 | | 0 |
| 1 | C++ rewrite: parallel parse, CSR graph, Dijkstra per query with early stop, queries in parallel. Local hash check passes on sample and full. | commit 88dd68c | 1.657 | 245 | 41.1 |
| 2 | A target with no in-edge is unreachable, so answer -1 without searching. Before this, each such query settled the whole 1.7M-node main body. | | | | |
