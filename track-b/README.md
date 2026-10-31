# Track B: Index

The task, the input files and the exact output format are in the [spec](https://github.com/hackathon-raptors/speed-demon-2026/blob/main/specs/b-index.md).

## Approach

One C++20 file, no libraries. The spec allows indexing only the terms the queries use, so `queries.txt` is parsed first and every distinct term gets a 64-bit key and an id in a small open-addressing table. The table keeps its keys in a dense array of their own, so the probe that most tokens make stays in cache.

The corpus is memory-mapped and cut into one chunk of lines per CPU thread; every chunk after the first starts at a document id that is a multiple of 64, so two threads never write the same bitset word. Each thread tokenizes its lines 32 bytes at a time: AVX2 compares give a mask of the token bytes (a-z, 0-9, and A-Z folded to lower case) and one of the line feeds, and the token boundaries are the bits where the mask changes. The last 32 bytes of the file are finished byte by byte with a 256-entry class table. Each token is hashed eight bytes at a time.

A short sample (the first 1/32 of every chunk) estimates each term's document count. Terms expected in more than 4,096 documents are dense, and the rest are rare:

- **Dense terms** (the common words, most of the corpus tokens) have their own tiny table that is checked first. The thread keeps one word per dense term for the 64 documents in progress and writes those words into the term's bitset once per 64 documents. Setting a bit in the bitsets directly would be a cache miss and a TLB miss per (document, term).
- **Rare terms** record (term, document) once per document in a 4-byte stream. The rarest-looking term of every phrase also keeps each occurrence with its file offset. Because each thread owns a contiguous range of documents, the streams are scattered into one flat posting array and every term's list comes out sorted without sorting. The terms are split into 4 groups by id (a stream holds term / 4 in 16 bits), so the posting lists are built, and the streams freed, one group at a time. Every big array is its own `mmap` (`mallopt`), so freed memory goes back to the kernel instead of staying in a thread's arena; together these took the peak from 151 MB to 90 MB.

A BOOL query runs left to right on a working bitset: AND, OR and NOT against a dense term are word-wise operations, and against a rare term they touch only that term's list. The count, the id sum and the ten smallest ids come from the final bitset.

A PHRASE query is driven by its rarest term. A document is a candidate when it holds every term (a bitset test or a binary search per term). For a rare driver, only the tokens around each stored occurrence are compared with the phrase's keys, at every phrase position the driver can take. When every term is common, the candidate document is tokenized and compared through a ring of the last k keys. Queries are answered in parallel and the output is joined in order.

## Dependencies

None.

## What did not work

A bit filter in front of the term table, evaluating BOOL queries on sorted lists, a prefetch queue for the table lookups, a sparser table, and a lazily allocated posting array (less memory on the test laptop, more on the board) were all measured and dropped. See [PERF.md](PERF.md).

## With another 24 hours

Pass 1 is still about 55% of the run, at roughly 200 cycles per token. Cut off at stages it splits into tokenizing and hashing, the common-word table, the dictionary lookup and the per-term record, each of a similar size. A branch-free record path and a smaller per-thread term state are the next things to try, and the dense-term bitsets (23 MB) could be stored as lists when their document count is low.
