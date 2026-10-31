# Track C: Join

The task, the input files and the exact output format are in the [spec](https://github.com/hackathon-raptors/speed-demon-2026/blob/main/specs/c-join.md).

## Approach

One C++20 file, no libraries. Both files are memory-mapped. `regions.csv` is read into a flat table that maps `region_id` straight to a country index; country codes are sorted once so the index order is the output order, and `_UNMATCHED` is the last index.

A quoted field can contain a newline, so `records.csv` cannot be cut at any `\n`. The file is split into one chunk per CPU thread, and each thread looks for the first real record start in its chunk: a `\n` followed by a digit (an id) or a byte order mark. In the data a newline inside a quoted note is only ever followed by a lowercase letter or a quote, which the spec guarantees for notes. Every thread must end exactly where the next thread starts; if any does not, the whole parse is redone with an exact method (count the `"` bytes of every chunk, and use the parity to start at a newline outside quotes), so the result never depends on the shortcut.

Each thread parses its records with a small state machine that follows the spec: it skips a BOM at a record start, handles `""` escapes, takes an LF inside quotes as data, and accepts CRLF or LF terminators. Per record it joins the region and adds count, id, amount (integer cents), min, max and the unescaped name and note lengths into its own per-country totals.

Distinct names are counted with one table shared by all threads: the first thread to meet a name claims a slot with a compare-and-swap and gives the name a dense id, and every country keeps a bitset over those ids. Names are added in batches of 16 records so that the table slots and bitset words can be prefetched together. The name pool of the spec (about 100,000 values) makes the table and the bitsets a few MB. A run with more than 131,072 distinct names fails instead of printing a wrong count. Money is formatted from integer cents, so no floating point is involved.

## Dependencies

None.

## What did not work

An AVX2 + PCLMUL index of the commas and line feeds with branch-free number parsing gave identical output but measured slower on the grading board (75 ms against 64 ms), so it was reverted. See [PERF.md](PERF.md).

## With another 24 hours

Find out why the SIMD scan lost on the board (the record fast path validates a lot per record), parse the numbers without data-dependent branches in a way that does, and shrink the source: the scalar fallbacks and the quoted-field handling carry most of the lines.
