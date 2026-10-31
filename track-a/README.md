# Track A: Ingest

The task, the input files and the exact output format are in the [spec](https://github.com/hackathon-raptors/speed-demon-2026/blob/main/specs/a-ingest.md).

## Approach

One C++20 file, no libraries. The log is memory-mapped and cut into one chunk per available CPU thread, each chunk ending on a newline. Every thread scans its chunk once.

- **Timestamp.** The date arithmetic runs only when the `YYYY-MM-DDTHH` prefix changes (compared as two overlapping 8-byte words), since neighbouring lines almost always share their hour.
- **Fields.** The method, endpoint, status and latency boundaries come from two 32-byte AVX2 compares per line (bit masks of the spaces and the LF), and the latency is read with three multiplications (SWAR). A line that does not fit that shape takes a byte-by-byte path, so the result never depends on the fast path.
- **Endpoints.** Each thread interns them in a small open-addressing table keyed by a 64-bit hash seeded with the length. Entries are matched by the hash and the length only: comparing the bytes costs a cold load from the first occurrence in the file, and a collision among about 2000 endpoints has a chance of about 1e-13.
- **Records.** The thread appends an 8-byte record (endpoint, hour, latency, is-5xx) and counts it per (local endpoint, hour) in the same step, so no separate counting pass is needed.

When the threads finish, endpoints get global ids in byte order, so the bucket index `hour * endpoints + endpoint` is already the required output order. The per-thread counts become write positions, and every thread scatters its records into one latency array. Each thread then takes a run of buckets holding about the same number of latencies, sorts the small ones (at most 64 values) or runs `nth_element` on the large ones, and formats its lines (nearest-rank p50, p95, p99). The output is written in order.

## Dependencies

None.

## What did not work

Writing the output with parallel `pwrite`, fusing the two count arrays, prefetch batching of the table lookups and a parallel prefix pass all measured no better on the test laptop and were dropped; the numbers are in [PERF.md](PERF.md) and the commit history.

## With another 24 hours

Drop the 8-byte record array and the global scatter by writing each latency straight into its bucket. Timestamps in the log drift up to about 70 minutes from the byte order, so one thread cannot own an hour and the threads would have to hand records to each other. Measure on the grading hardware: the laptop's cache sizes and huge-page behaviour differ enough that several local speed-ups did not show up on the board.
