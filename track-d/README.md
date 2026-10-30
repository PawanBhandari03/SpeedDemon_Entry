# Track D: Traverse

The task, the input files and the exact output format are in the [spec](https://github.com/hackathon-raptors/speed-demon-2026/blob/main/specs/d-traverse.md).

## Approach

One C++20 file, no libraries. `graph.txt` is memory-mapped, cut into one chunk per available CPU thread at newline boundaries, and the edges are parsed in parallel with hand-written integer parsing. A counting sort on the tail node turns the edge list into a compressed-sparse-row graph: one offsets array and one flat array of (head, weight) arcs, so a node's out-edges sit next to each other in memory.

The 2,000 queries are answered in parallel. Each thread owns one distance array and runs a binary-heap Dijkstra that stops as soon as the target is settled. A search records the nodes it touched and resets only those, so nothing clears the whole 1.7M-node array between queries. Distances are 64-bit, since a path can exceed 2^32 - 1. Unreachable targets print `-1`.

## Dependencies

None.

## With another 24 hours

Parallelise the CSR build, use a radix or 4-ary heap, search from both ends at once for the long queries, and cut the work of the unreachable-target queries, which settle the whole main body.
