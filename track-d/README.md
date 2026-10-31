# Track D: Traverse

The task, the input files and the exact output format are in the [spec](https://github.com/hackathon-raptors/speed-demon-2026/blob/main/specs/d-traverse.md).

## Approach

One C++20 file, no libraries. `graph.txt` is memory-mapped, cut into one chunk per available CPU thread at newline boundaries, and the edges are parsed in parallel with hand-written integer parsing.

- **CSR build without atomics.** While parsing, each thread appends its edges to a bucket per range of tail nodes. The owner of a range then counts and places that range's edges with plain writes, and a prefix sum turns the counts into one offsets array and one flat array of (head, weight) arcs. Atomic counters were the first version; their cache misses cannot overlap, and the build was a fifth of the run.
- **Renumbering.** The input ids are a random shuffle, so a search jumps all over memory. The nodes are renumbered in breadth-first order from a node of the main component (a level-synchronous BFS on all threads, claiming nodes with compare-and-swap), so nodes that are close on the map are close in memory. Any visiting order is correct, because it only renames nodes.
- **Queries.** The 2,000 queries are answered in parallel. Each thread owns one distance array (allocated with `calloc` and never filled, so untouched pages cost nothing) and runs Dijkstra until the target is settled, resetting only the nodes it touched. Distances are 32-bit with a radix heap; a search whose distance would reach 2^32 - 1 is redone with 64-bit distances and a binary heap, since a path can be longer. A target with no in-edge is answered `-1` without a search. Unreachable targets print `-1`.

## Dependencies

None.

## What did not work

Disabling transparent huge pages for the process lowered the memory a little but made every run slower, so it was reverted. See [PERF.md](PERF.md).

## With another 24 hours

Search from both ends at once for the long queries (it needs a reverse graph, about 47 MB more), cut the second CSR copy that the renumbering makes, and look at the unreachable-target queries, which settle the whole main body.
