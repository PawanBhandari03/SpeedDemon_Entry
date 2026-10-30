# Track B: Index

The task, the input files and the exact output format are in the [spec](https://github.com/hackathon-raptors/speed-demon-2026/blob/main/specs/b-index.md).

## Approach

One C++20 file, no libraries. The spec allows indexing only the terms the queries use, so `queries.txt` is parsed first and every distinct term gets a 64-bit key and an id in a small open-addressing table.

The corpus is memory-mapped and cut into one chunk of lines per CPU thread. Each thread tokenizes its lines with a 256-entry byte-class table (a-z, 0-9 and A-Z folded to lower case are token bytes, everything else separates), hashing each token as it is read, and looks the hash up in the term table. For a query term it records (term, document) once per document. Because each thread owns a contiguous range of documents, the records can be scattered into one flat posting array and every term's list comes out sorted without any sorting. The line offset of every document is kept for phrase checks.

Terms that occur in more than 4,096 documents also get a bitset over the document ids. A BOOL query runs left to right on a working bitset: AND, OR and NOT against a dense term are word-wise operations, and against a rare term they touch only that term's list. The count, the id sum and the ten smallest ids are read from the final bitset.

A PHRASE query takes the term with the fewest documents, tests each of its documents against the other terms (bitset test or binary search), and for the survivors tokenizes the document again and looks for the exact id sequence, so no positions are stored. Queries are answered in parallel and the output is joined in order.

## Dependencies

None.

## With another 24 hours

Replace the double tokenization with an indexed pass that keeps positions only for the few phrase terms, intersect lists with galloping and SIMD instead of per-document probes, and shrink the pass-1 record buffers to cut peak memory.
