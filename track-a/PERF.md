# Track A: performance log

Every step from the reference to your final version, each with the live-board run ID that measured it. Judges check that the run IDs exist and that each change moved the way you say it did.

| Step | Change | Run ID | T (s) | M (MB) | Points |
|---|---|---|---|---|---|
| 0 | The reference, unchanged | | 32.05 | 228 | 0 |
| 1 | C++ rewrite: mmap, one chunk per thread, hand-parsed timestamps, counting sort, nth_element percentiles. Local hash check passes on tiny and sample. | commit 88dd68c | 0.104 | 59 | 62.8 |
| 2 | Profiled the phases locally: parsing was about 80% of the run, and inside it `days_from_civil` (several integer divisions) ran on every line. Neighbouring lines share their hour, so the date is now recomputed only when the `YYYY-MM-DDTHH` prefix changes. The endpoint's end is found 8 bytes at a time (SWAR) instead of byte by byte. Parse phase locally: 98 ms -> 46 ms. The bucket counting sort and the percentile/formatting step, both single-threaded before, now run on every thread, and the output is written per thread instead of through one 10 MB string. Tried replacing MAP_POPULATE with per-thread MADV_POPULATE_READ: no measurable gain, reverted. | commit 9ab7648 | 0.075 | 59 | 65.9 |
| 3 | Each worker now counts its records per (local endpoint, hour) while parsing, so the separate pass that renamed every record to a global endpoint id and counted it is gone; the scatter indexes the worker's own count array directly. The count arrays are calloc'd and sized by the local endpoint table, so only the part in use is real memory. Peak memory measured locally from the cgroup's `memory.peak`: 61 MB -> 51 MB. | | | | |
| 4 | Dropped MAP_POPULATE: it made one thread map the whole input before the workers started. Now every worker faults in the pages of its own chunk, in parallel. Output hash unchanged on tiny, sample and full; the board run is the timing measurement. | | | | |
| 5 | The average bucket holds only about 25 latencies, where three nth_element calls cost more than just sorting. Buckets of at most 64 values are now sorted with std::sort and read at the three ranks; larger ones keep the selections. Best-of-9, three alternating rounds: 102-107 ms -> 96-104 ms (about 4%). | | | | |
