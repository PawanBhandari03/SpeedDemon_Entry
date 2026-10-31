// Speed Demon 2026, Track D (Traverse): graph.txt + queries.txt -> distances.txt
//
// Plan: mmap graph.txt, cut it into one chunk per thread at newline boundaries and parse the
// edges in parallel. A counting sort turns the edge list into a compressed-sparse-row graph
// (one flat array of (head, weight) arcs, grouped by tail); the threads build it together with
// atomic counters. The 2,000 queries are then answered in parallel, each by a Dijkstra search
// that stops when the target is settled. A search only resets the nodes it touched, so no
// per-query clearing of the whole graph. Distances are kept in 32 bits, which halves the
// per-thread arrays; the rare search that would pass 2^32 - 1 is redone with 64-bit distances.
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

struct Edge { uint32_t u, v, w; };
struct Arc { uint32_t to, w; };

constexpr int64_t kOverflow = -2;

struct Mapped {
    const char* p = nullptr;
    size_t n = 0;
};

Mapped map_file(const std::string& path) {
    Mapped m;
    const int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) return m;
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size == 0) { close(fd); return m; }
    m.n = static_cast<size_t>(st.st_size);
    void* p = mmap(nullptr, m.n, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (p == MAP_FAILED) { m.n = 0; return m; }
    m.p = static_cast<const char*>(p);
    return m;
}

inline uint64_t read_uint(const char*& p) {
    uint64_t v = 0;
    while (*p >= '0') v = v * 10 + static_cast<uint64_t>(*p++ - '0');
    return v;
}

int thread_count() {
    cpu_set_t set;
    if (sched_getaffinity(0, sizeof set, &set) == 0) {
        const int n = CPU_COUNT(&set);
        if (n > 0) return n;
    }
    return static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
}

// All threads wait here until every one has arrived. Spinning is cheap: each thread has its own
// hardware thread, and a level of the search takes microseconds.
struct SpinBarrier {
    explicit SpinBarrier(int n) : n(n) {}
    void wait() {
        const int g = gen.load(std::memory_order_acquire);
        if (count.fetch_add(1, std::memory_order_acq_rel) == n - 1) {
            count.store(0, std::memory_order_relaxed);
            gen.fetch_add(1, std::memory_order_release);
        } else {
            while (gen.load(std::memory_order_acquire) == g) __builtin_ia32_pause();
        }
    }
    const int n;
    std::atomic<int> count{0}, gen{0};
};

template <class F>
void parallel(int n, F&& f) {
    std::vector<std::thread> th;
    for (int i = 1; i < n; ++i) th.emplace_back(f, i);
    f(0);
    for (auto& t : th) t.join();
}

// A zeroed array whose pages are committed only when written, by whichever thread writes them
// (calloc maps big blocks lazily), instead of being zero-filled by one thread up front.
template <class T>
struct LazyArray {
    T* p;
    explicit LazyArray(size_t n) : p(static_cast<T*>(calloc(n ? n : 1, sizeof(T)))) {}
    ~LazyArray() { free(p); }
    LazyArray(const LazyArray&) = delete;
    LazyArray& operator=(const LazyArray&) = delete;
    T& operator[](size_t i) { return p[i]; }
    T* data() { return p; }
    void swap(LazyArray& o) { std::swap(p, o.p); }
};

// Monotone priority queue for 32-bit keys (a radix heap). An entry (key << 32 | node) sits in
// the bucket of the highest bit in which its key differs from the last key popped; Dijkstra
// never pushes a key below that, so only the lowest non-empty bucket is ever redistributed.
struct RadixHeap {
    std::vector<uint64_t> b[33];
    uint32_t last = 0;
    size_t size = 0;

    static int bucket(uint32_t k, uint32_t last) { return k == last ? 0 : 32 - __builtin_clz(k ^ last); }
    void push(uint64_t e) {
        b[bucket(static_cast<uint32_t>(e >> 32), last)].push_back(e);
        ++size;
    }
    uint64_t pop() {
        if (b[0].empty()) {
            int i = 1;
            while (b[i].empty()) ++i;
            last = static_cast<uint32_t>(*std::min_element(b[i].begin(), b[i].end()) >> 32);
            for (uint64_t e : b[i]) b[bucket(static_cast<uint32_t>(e >> 32), last)].push_back(e);
            b[i].clear();
        }
        --size;
        const uint64_t e = b[0].back();
        b[0].pop_back();
        return e;
    }
    void clear() {
        for (auto& v : b) v.clear();
        last = 0;
        size = 0;
    }
};

// Dijkstra on a reusable distance array of type D (uint32_t or uint64_t). With 32-bit distances
// the queue is a radix heap; the rare 64-bit search uses a binary heap of (distance, node).
template <class D>
struct Searcher {
    static constexpr D kInf = std::numeric_limits<D>::max();
    static constexpr bool kNarrow = sizeof(D) == 4;
    using Entry = std::pair<uint64_t, uint32_t>;

    const uint32_t* off;
    const Arc* arcs;
    // Stored as distance + 1, so 0 means "not reached yet". The array comes from calloc and is
    // never filled: pages the searches do not reach are never touched and cost no memory.
    D* dist;
    std::vector<uint32_t> touched;
    RadixHeap radix;
    std::vector<Entry> heap;

    Searcher(const uint32_t* o, const Arc* a, size_t n) : off(o), arcs(a), dist(static_cast<D*>(calloc(n, sizeof(D)))) {}
    Searcher(const Searcher&) = delete;
    ~Searcher() { free(dist); }

    // Distance from s to t, -1 if unreachable, or kOverflow if D is too small for this search.
    int64_t run(uint32_t s, uint32_t t) {
        if (s == t) return 0;
        const auto cmp = std::greater<Entry>();
        auto push = [&](uint64_t d, uint32_t v) {
            if constexpr (kNarrow) radix.push(d << 32 | v);
            else { heap.emplace_back(d, v); std::push_heap(heap.begin(), heap.end(), cmp); }
        };
        int64_t result = -1;
        dist[s] = 1;
        touched.push_back(s);
        push(0, s);
        while (kNarrow ? radix.size != 0 : !heap.empty()) {
            uint64_t d;
            uint32_t u;
            if constexpr (kNarrow) {
                const uint64_t e = radix.pop();
                d = e >> 32;
                u = static_cast<uint32_t>(e);
            } else {
                std::pop_heap(heap.begin(), heap.end(), cmp);
                d = heap.back().first;
                u = heap.back().second;
                heap.pop_back();
            }
            if (d + 1 > dist[u]) continue;  // stale entry
            if (u == t) { result = static_cast<int64_t>(d); break; }
            bool overflow = false;
            for (uint32_t i = off[u], e = off[u + 1]; i < e; ++i) {
                const uint64_t nd = d + arcs[i].w;
                const uint32_t v = arcs[i].to;
                if (nd + 1 >= kInf) { overflow = true; break; }
                const D cur = dist[v];
                if (cur == 0 || nd + 1 < cur) {
                    if (cur == 0) touched.push_back(v);
                    dist[v] = static_cast<D>(nd + 1);
                    push(nd, v);
                }
            }
            if (overflow) { result = kOverflow; break; }
        }
        for (uint32_t v : touched) dist[v] = 0;
        touched.clear();
        radix.clear();
        heap.clear();
        return result;
    }
};

inline char* put_i64(char* o, int64_t v) {
    if (v < 0) { *o++ = '-'; *o++ = '1'; return o; }
    char tmp[20];
    int n = 0;
    uint64_t x = static_cast<uint64_t>(v);
    do { tmp[n++] = static_cast<char>('0' + x % 10); x /= 10; } while (x);
    while (n) *o++ = tmp[--n];
    return o;
}

}  // namespace

int main(int argc, char** argv) {
    std::string data = "/data", out = "/out";
    for (int i = 1; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "--data")) data = argv[i + 1];
        else if (!strcmp(argv[i], "--out")) out = argv[i + 1];
    }

    const Mapped g = map_file(data + "/graph.txt");
    if (!g.p) return 1;
    const char* hp = g.p;
    const uint64_t n = read_uint(hp);
    ++hp;
    const uint64_t m = read_uint(hp);
    ++hp;  // first edge line
    const char* const end = g.p + g.n;

    // Parse the edges, one chunk per thread.
    const int nt = (end - hp) < (1 << 20) ? 1 : thread_count();
    std::vector<const char*> cut(nt + 1);
    cut[0] = hp;
    cut[nt] = end;
    for (int i = 1; i < nt; ++i) {
        const char* p = hp + (end - hp) * i / nt;
        const char* nl = static_cast<const char*>(memchr(p - 1, '\n', static_cast<size_t>(end - (p - 1))));
        cut[i] = nl ? nl + 1 : end;
    }
    // Every thread appends the edges it parses to a bucket per range of tail nodes. The owner of
    // a range later builds that range's part of the CSR with plain writes: no atomic updates, whose
    // cache misses cannot overlap, and every thread writes only inside its own ranges.
    const int R = nt;
    std::vector<std::vector<Edge>> parts(static_cast<size_t>(nt) * R);
    std::vector<uint8_t> has_in(n, 0);  // a node with no in-edge can only be reached from itself
    parallel(nt, [&](int i) {
        const char* p = cut[i];
        const char* e = std::max(cut[i], cut[i + 1]);
        const size_t per = static_cast<size_t>(e - p) / 12 / R;
        for (int r = 0; r < R; ++r) parts[static_cast<size_t>(i) * R + r].reserve(per + per / 4 + 16);
        while (p < e) {
            const uint32_t u = static_cast<uint32_t>(read_uint(p)); ++p;
            const uint32_t to = static_cast<uint32_t>(read_uint(p)); ++p;
            const uint32_t w = static_cast<uint32_t>(read_uint(p)); ++p;
            parts[static_cast<size_t>(i) * R + static_cast<size_t>(static_cast<uint64_t>(u) * R / n)].push_back({u, to, w});
            std::atomic_ref<uint8_t>(has_in[to]).store(1, std::memory_order_relaxed);
        }
    });

    // CSR build: out-degree counts, prefix sums, then each edge takes a slot in its tail's block.
    // The order of arcs inside a block does not change any distance.
    std::vector<uint32_t> off(n + 2, 0);
    parallel(nt, [&](int r) {
        for (int t = 0; t < nt; ++t)
            for (const Edge& e : parts[static_cast<size_t>(t) * R + r]) ++off[e.u + 2];
    });
    for (uint64_t i = 2; i < n + 2; ++i) off[i] += off[i - 1];
    // off[u + 1] is now the start of u's block; claiming slots advances it to the block end,
    // which is the start of u + 1's block.
    LazyArray<Arc> arcs(m);
    parallel(nt, [&](int r) {
        for (int t = 0; t < nt; ++t) {
            auto& part = parts[static_cast<size_t>(t) * R + r];
            for (const Edge& e : part) arcs[off[e.u + 1]++] = {e.v, e.w};
            std::vector<Edge>().swap(part);
        }
    });

    // Renumber the nodes in breadth-first order. The input ids are a random shuffle, so a
    // search jumps all over the arrays; after renumbering, nodes that are close on the map
    // are close in memory and a search stays inside a few cache-friendly ranges.
    // The main body is one strongly connected component, so a single search from one of its
    // nodes covers nearly everything; it runs level by level on all threads. Any visiting order
    // is correct (it only renames nodes), so the threads need not agree on one.
    constexpr uint32_t kUnset = UINT32_MAX, kClaimed = UINT32_MAX - 1;
    std::vector<uint32_t> order(n);  // new id -> old id
    std::vector<uint32_t> new_id(n, kUnset);
    size_t filled = 0;
    uint32_t root = 0;
    while (root < n && (off[root + 1] == off[root] || !has_in[root])) ++root;
    if (root < n) {
        order[0] = root;
        new_id[root] = 0;
        std::vector<std::vector<uint32_t>> found(nt);
        std::vector<size_t> base(nt + 1);
        size_t lo = 0, hi = 1;  // the current level is order[lo, hi)
        SpinBarrier bar(nt);
        parallel(nt, [&](int i) {
            auto& mine = found[i];
            while (lo < hi) {
                for (size_t x = lo + (hi - lo) * i / nt, b = lo + (hi - lo) * (i + 1) / nt; x < b; ++x)
                    for (uint32_t j = off[order[x]], e = off[order[x] + 1]; j < e; ++j) {
                        std::atomic_ref<uint32_t> id(new_id[arcs[j].to]);
                        uint32_t unset = kUnset;
                        if (id.load(std::memory_order_relaxed) == kUnset &&
                            id.compare_exchange_strong(unset, kClaimed, std::memory_order_relaxed))
                            mine.push_back(arcs[j].to);
                    }
                bar.wait();
                if (i == 0) {
                    base[0] = hi;
                    for (int t = 0; t < nt; ++t) base[t + 1] = base[t] + found[t].size();
                }
                bar.wait();
                for (size_t j = 0; j < mine.size(); ++j) {
                    order[base[i] + j] = mine[j];
                    new_id[mine[j]] = static_cast<uint32_t>(base[i] + j);
                }
                mine.clear();
                bar.wait();
                if (i == 0) { lo = hi; hi = base[nt]; }
                bar.wait();
            }
        });
        filled = hi;
    }
    // What the root cannot reach (islands, isolated nodes, nodes with no in-edge) follows, one
    // small search at a time.
    for (uint32_t r = 0; r < n; ++r) {
        if (new_id[r] != kUnset) continue;
        new_id[r] = static_cast<uint32_t>(filled);
        order[filled++] = r;
        for (size_t head = filled - 1; head < filled; ++head)
            for (uint32_t j = off[order[head]], e = off[order[head] + 1]; j < e; ++j) {
                const uint32_t v = arcs[j].to;
                if (new_id[v] == kUnset) { new_id[v] = static_cast<uint32_t>(filled); order[filled++] = v; }
            }
    }
    {
        std::vector<uint32_t> off2(n + 1, 0);
        std::vector<uint8_t> has_in2(n);
        for (uint32_t x = 0; x < n; ++x) {
            off2[x + 1] = off2[x] + (off[order[x] + 1] - off[order[x]]);
            has_in2[x] = has_in[order[x]];
        }
        LazyArray<Arc> arcs2(m);
        parallel(nt, [&](int i) {
            for (uint64_t x = n * i / nt; x < n * (i + 1) / nt; ++x) {
                uint32_t pos = off2[x];
                for (uint32_t j = off[order[x]], e = off[order[x] + 1]; j < e; ++j) arcs2[pos++] = {new_id[arcs[j].to], arcs[j].w};
            }
        });
        off.swap(off2);
        arcs.swap(arcs2);
        has_in.swap(has_in2);
    }
    std::vector<uint32_t>().swap(order);

    // Queries.
    std::vector<std::pair<uint32_t, uint32_t>> queries;
    const Mapped q = map_file(data + "/queries.txt");
    if (q.p) {
        const char* p = q.p;
        const char* e = q.p + q.n;
        while (p < e) {
            const uint32_t s = static_cast<uint32_t>(read_uint(p)); ++p;
            const uint32_t t = static_cast<uint32_t>(read_uint(p)); ++p;
            queries.emplace_back(new_id[s], new_id[t]);
        }
    }
    std::vector<uint32_t>().swap(new_id);  // only the queries needed the renumbering table

    std::vector<int64_t> answer(queries.size());
    {
        const int qt = static_cast<int>(std::min<size_t>(thread_count(), std::max<size_t>(1, queries.size())));
        std::atomic<size_t> next{0};
        parallel(qt, [&](int) {
            Searcher<uint32_t> s32(off.data(), arcs.data(), n);
            std::unique_ptr<Searcher<uint64_t>> s64;  // made only if some search overflows 32 bits
            for (size_t k; (k = next.fetch_add(1)) < queries.size();) {
                const auto [src, dst] = queries[k];
                // Without this check the search settles everything src can reach before giving up.
                if (src != dst && !has_in[dst]) { answer[k] = -1; continue; }
                answer[k] = s32.run(src, dst);
                if (answer[k] == kOverflow) {
                    if (!s64) s64 = std::make_unique<Searcher<uint64_t>>(off.data(), arcs.data(), n);
                    answer[k] = s64->run(src, dst);
                }
            }
        });
    }

    std::string result;
    result.reserve(queries.size() * 12);
    char buf[32];
    for (int64_t a : answer) {
        char* o = put_i64(buf, a);
        *o++ = '\n';
        result.append(buf, static_cast<size_t>(o - buf));
    }
    FILE* f = fopen((out + "/distances.txt").c_str(), "wb");
    if (!f) return 1;
    fwrite(result.data(), 1, result.size(), f);
    fclose(f);
    return 0;
}
