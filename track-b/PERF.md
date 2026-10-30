# Track B: performance log

Every step from the reference to your final version, each with the live-board run ID that measured it. Judges check that the run IDs exist and that each change moved the way you say it did.

| Step | Change | Run ID | T (s) | M (MB) | Points |
|---|---|---|---|---|---|
| 0 | The reference, unchanged | | 56.9 | 2199 | 0 |
| 1 | C++ rewrite: index only the query terms, parallel tokenization with a hash lookup per token, sorted posting lists by construction, bitsets for common terms, phrase check by re-tokenizing the candidates. Local hash check passes on sample and full. | | | | |
