# Track A: Ingest

The task, the input files and the exact output format are in the [spec](https://github.com/hackathon-raptors/speed-demon-2026/blob/main/specs/a-ingest.md).

## Approach

One C++20 file, no libraries. The log is memory-mapped and cut into one chunk per available CPU thread, each chunk ending on a newline. Every thread scans its chunk once with hand-written parsing: the hour index comes straight from the timestamp digits (no date library), the endpoint is interned in a small hash table, and the thread appends a compact 8-byte record (endpoint, hour, latency, is-5xx).

When the threads finish, endpoints get global ids in byte order, so a bucket index of `hour * endpoints + endpoint` is already the required output order. A counting sort groups the latencies per (hour, endpoint) bucket, then `nth_element` selects p99, p95 and p50 (nearest rank) without sorting the whole bucket, spread over the threads. The output is formatted by hand into one buffer and written once.

## Dependencies

None.

## With another 24 hours

Use SIMD to find newlines and spaces, read the file in chunks to lower peak memory, and parse the timestamp with fewer branches.
