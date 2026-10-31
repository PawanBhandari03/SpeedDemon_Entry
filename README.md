# Speed Demon 2026 — team BobTheBuilder

Entry for [Speed Demon 2026](https://speeddemon.dev) (Hackathon Raptors, 30 October – 2 November 2026). Four tracks, one C++20 file each, no libraries, one `Dockerfile` per track built from the published allowlist.

| Track | Task | Approach | Source |
|---|---|---|---|
| A · Ingest | 3.1M log lines → per-hour, per-endpoint counts, p50/p95/p99, error rate | memory-mapped parallel scan, AVX2 field split, per-bucket latency blocks, per-hour output tasks | [track-a](track-a/) |
| B · Index | 250k documents, 50k boolean and phrase queries | index only the query terms, bitsets for the dense ones, offset-checked phrase verification | [track-b](track-b/) |
| C · Join | 200 MB hostile CSV joined to a region table | quote-aware chunking, per-thread aggregates, shared dense name ids with per-group bitsets | [track-c](track-c/) |
| D · Traverse | 5M-edge graph, 2,000 shortest-path queries | atomic-free CSR build, BFS renumbering, per-thread Dijkstra with a radix heap | [track-d](track-d/) |

Each track folder holds the submission: `main.cpp`, the `Dockerfile` that builds it offline, `.speed-demon.toml`, a `README.md` describing the approach, and `PERF.md`, the optimisation log with the live-board run IDs that measured each step.

## Correctness

Every track was checked against the organisers' published output hashes at all three scales before each push, and against the Python reference where a tiny-scale hash is not published:

```
track   tiny      sample    full
A       OK        OK        OK
B       OK        OK        OK
C       OK        OK        OK
D       OK        OK        OK
```

`check.py` from the starter does the same check under the board's sandbox settings:

```
python3 check.py a --data data/a --expected <sha256>
```

## How the work was measured

The board is the only authority, and it is noisy at this scale (the same Track A binary has scored between 54 and 74 ms), so changes were chosen by counting work rather than by timing on a laptop: `callgrind` instruction counts on the seed-1 sample, plus the cgroup's `memory.peak`. The last round cut Track A by 22% and Track C by 21% of instructions; the board moved about 4% and 13% respectively, which is why both tracks' logs now discuss memory stalls as much as instruction counts.

## Licence

MIT, see [LICENSE](LICENSE). The reference implementations this repository started from are copyright Hackathon Raptors CIC, MIT licensed.
