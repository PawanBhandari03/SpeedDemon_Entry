# Track B: Index

The task, the input files and the exact output format are in the [spec](https://github.com/hackathon-raptors/speed-demon-2026/blob/main/specs/b-index.md).

## Approach

One C++20 file, no libraries. The spec allows indexing only the terms the queries use, so `queries.txt` is parsed first and every distinct term gets a 64-bit key and an id in a small open-addressing table.

The corpus is memory-mapped and cut into one chunk of lines per CPU thread; every chunk after the first starts at a document id that is a multiple of 64, so threads never share a bitset word. Each thread tokenizes its lines with a 256-entry byte-class table (a-z, 0-9 and A-Z folded to lower case are token bytes, everything else separates), hashing each token as it is read, and looks the hash up in the term table. The table keeps its keys in their own dense array (1 MB for the 75,747 query terms of seed 1), so the probe that almost every token makes stays in the L2 cache.

A short sample (the first 1/32 of every chunk) estimates each term's document count first. Terms expected in more than 4,096 documents are dense: the scan sets their bit in a per-term bitset directly. For the other terms it records (term, document) once per document; because each thread owns a contiguous range of documents, the records are scattered into one flat posting array and every term's list comes out sorted without any sorting. The line offset of every document is kept for phrase checks.

A BOOL query runs left to right on a working bitset: AND, OR and NOT against a dense term are word-wise operations, and against a rare term they touch only that term's list. The count, the id sum and the ten smallest ids are read from the final bitset.

A PHRASE query takes the term with the fewest documents, tests each of its documents against the other terms (bitset test or binary search), and for the survivors tokenizes the document again, comparing each token's key with the phrase's keys through a ring of the last k keys, so no positions are stored and no table lookup is needed. Queries are answered in parallel and the output is joined in order.

## Dependencies

None.

## With another 24 hours

Replace the double tokenization with an indexed pass that keeps positions only for the few phrase terms, intersect lists with galloping and SIMD instead of per-document probes, and shrink the pass-1 record buffers to cut peak memory.
