# Track A: Ingest

The task, the input files and the exact output format are in the [spec](https://github.com/hackathon-raptors/speed-demon-2026/blob/main/specs/a-ingest.md).

## Approach

One C++20 file, no libraries. The log is memory-mapped and cut into one chunk per available CPU thread, each cut on a newline. Every thread scans its chunk once and never copies a record again.

- **Timestamp.** The date arithmetic (Howard Hinnant's `days_from_civil`) runs only when the `YYYY-MM-DDTHH` prefix changes, compared as two overlapping 8-byte words, because neighbouring lines almost always share their hour.
- **Fields.** The method, endpoint, status and latency boundaries come from two 32-byte AVX2 compares per line: the bit masks of the spaces and the LF give the field ends, and the latency is converted with three multiplications (SWAR). A line whose endpoint runs past that 64-byte window takes a byte-by-byte path, so the fast path is an optimisation and not a correctness assumption.
- **Endpoints.** Each thread interns them in a 4096-slot open-addressing table of 16-byte entries keyed by a 64-bit hash seeded with the length, with a 256-entry direct-mapped cache in front of it for the endpoints that repeat. Entries are matched by hash and length only: comparing bytes costs a cold load from the first occurrence in the file, and a collision among about 2,000 endpoints has a chance of about 1e-13.
- **Buckets.** A bucket is one (hour, endpoint) pair. Each thread keeps one row of buckets per hour it has seen, and each bucket holds a count, a 5xx count and a chain of 32-byte blocks (seven latencies and a link to the previous block) carved from 512 KB slabs. One record is therefore one counter update and one store, and no second pass over the data is needed to collect latencies.

After the threads join, the endpoints of all workers are merged and sorted, which gives global ids in byte order, the required output order. The output is then one task per hour: the row pointer of every worker for that hour is resolved once, each bucket of the hour gathers its blocks (one 32-byte AVX2 move per block, into a buffer that only grows), p50/p95/p99 are taken by sorting buckets of at most 64 values and by three `nth_element` calls on larger ones, and the line is formatted with a two-digit table. A finished hour is written as soon as every earlier hour has been written, so the file comes out in order without an extra pass.

## Dependencies

None. The binary is built with `-march=x86-64-v3`, which the grading CPU supports (AVX2, BMI2, but no AVX-512).

## What did not work

`MAP_POPULATE` (one thread faults in the whole file before the workers start), per-thread `MADV_POPULATE_READ`, disabling transparent huge pages, parallel `pwrite` of the output, fusing the per-endpoint count arrays, prefetch batching of the table lookups and a parallel prefix pass. Each was measured and dropped; the numbers are in [PERF.md](PERF.md) and the commit history.

## With another 24 hours

Cut peak memory. About 10 MB of the 39 MB is the text of the hours that are still waiting to be written (an hour's buffer is allocated when the hour starts, and hours finish out of order), and about 14 MB is the latency blocks. Streaming the formatted lines to the file as soon as their turn comes would remove the first; dropping the blocks needs a second pass over the input with the exact percentiles known, which trades time for memory at 50/20 weights. Also worth measuring on the grading hardware: the scan is as much memory-stall bound as instruction bound, so a smaller per-thread footprint may be worth more there than it is on this laptop.
