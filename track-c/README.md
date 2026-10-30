# Track C: Join

The task, the input files and the exact output format are in the [spec](https://github.com/hackathon-raptors/speed-demon-2026/blob/main/specs/c-join.md).

## Approach

One C++20 file, no libraries. Both files are memory-mapped. `regions.csv` is read into a flat table that maps `region_id` straight to a country index; country codes are sorted once so the index order is the output order, and `_UNMATCHED` is the last index.

A quoted field can contain a newline, so `records.csv` cannot be cut at any `\n`. The file is split into one chunk per CPU thread, and each chunk first counts its `"` bytes. The running parity of those counts says whether a chunk starts inside a quoted field, so each thread can find its first real record start (a `\n` with an even number of quotes before it). Each thread then parses records with a small state machine that follows the spec: it skips a BOM at a record start, handles `""` escapes, takes an LF inside quotes as data, and accepts CRLF or LF terminators. A thread stops exactly where the next thread's first record begins.

Per record the thread joins the region, adds count, id, amount (as integer cents), min, max and the unescaped name and note lengths into its own per-country totals, and records one 64-bit key (country, hash of the unescaped name). At the end the totals are added, the keys are sorted per thread, and the distinct names per country are counted by gathering, sorting and de-duplicating each country's keys in parallel. Money is formatted from integer cents, so no floating point is involved.

## Dependencies

None.

## With another 24 hours

Replace the sort-based distinct count with a shared name table (the name pool is only about 100,000 values), use SIMD for the quote counting and the field scanning, and merge the quote-parity pass into the parse pass.
