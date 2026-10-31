# Track B: Index

The task, the input files and the exact output format are in the [spec](https://github.com/hackathon-raptors/speed-demon-2026/blob/main/specs/b-index.md).

## Approach

One C++20 file, no libraries. The spec allows indexing only the terms the queries use, so `queries.txt` is parsed first and every distinct term gets a 64-bit key and an id in a small open-addressing table. The table keeps its keys in a dense array of their own, so the probe that most tokens make stays in cache.

The corpus is memory-mapped and cut into one chunk of lines per CPU thread; every chunk after the first starts at a document id that is a multiple of 64, so two threads never write the same bitset word. Each thread tokenizes its lines with a 256-entry byte-class table (a-z, 0-9 and A-Z folded to lower case are token bytes), and hashes each token eight bytes at a time.

A short sample (the first 1/32 of every chunk) estimates each term's document count. Terms expected in more than 4,096 documents are dense, and the rest are rare:

- **Dense terms** (the common words, most of the corpus tokens) have their own tiny table that is checked first. The thread keeps one word per dense term for the 64 documents in progress and writes those words into the term's bitset once per 64 documents. Setting a bit in the bitsets directly would be a cache miss and a TLB miss per (document, term).
- **Rare terms** record (term, document) once per document in a 4-byte stream. The rarest-looking term of every phrase also keeps each occurrence with its file offset. Because each thread owns a contiguous range of documents, the streams are scattered into one flat posting array and every term's list comes out sorted without sorting.

A BOOL query runs left to right on a working bitset: AND, OR and NOT against a dense term are word-wise operations, and against a rare term they touch only that term's list. The count, the id sum and the ten smallest ids come from the final bitset.

A PHRASE query is driven by its rarest term. A document is a candidate when it holds every term (a bitset test or a binary search per term). For a rare driver, only the tokens around each stored occurrence are compared with the phrase's keys, at every phrase position the driver can take. When every term is common, the candidate document is tokenized and compared through a ring of the last k keys. Queries are answered in parallel and the output is joined in order.

## Dependencies

None.

## What did not work

A bit filter in front of the term table, evaluating BOOL queries on sorted lists, an AVX2 tokenizer, a prefetch queue for the table lookups, a sparser table, and a lazily allocated posting array (less memory on the test laptop, more on the board) were all measured and dropped. See [PERF.md](PERF.md).

## With another 24 hours

Cut the memory peak at the posting scatter (the term streams and the posting array exist at the same time) by splitting the streams by term range and freeing each after it is scattered, and intersect lists with galloping and SIMD instead of per-document probes.
