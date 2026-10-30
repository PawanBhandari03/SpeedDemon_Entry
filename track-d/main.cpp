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
    void* p = mmap(nullptr, m.n, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
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

// Dijkstra with a binary heap on a reusable distance array of type D (uint32_t or uint64_t).
// With 32-bit distances a heap entry is (distance << 32 | node) in one integer, so a
// comparison is one integer compare.
template <class D>
struct Searcher {
    static constexpr D kInf = std::numeric_limits<D>::max();
    using Entry = std::conditional_t<sizeof(D) == 4, uint64_t, std::pair<uint64_t, uint32_t>>;

    const uint32_t* off;
    const Arc* arcs;
    std::vector<D> dist;
    std::vector<uint32_t> touched;
    std::vector<Entry> heap;

    Searcher(const uint32_t* o, const Arc* a, size_t n) : off(o), arcs(a), dist(n, kInf) {}

    static Entry entry(uint64_t d, uint32_t v) {
        if constexpr (sizeof(D) == 4) return d << 32 | v;
        else return {d, v};
    }

    // Distance from s to t, -1 if unreachable, or kOverflow if D is too small for this search.
    int64_t run(uint32_t s, uint32_t t) {
        if (s == t) return 0;
        const auto cmp = std::greater<Entry>();
        int64_t result = -1;
        dist[s] = 0;
        touched.push_back(s);
        heap.push_back(entry(0, s));
        while (!heap.empty()) {
            std::pop_heap(heap.begin(), heap.end(), cmp);
            uint64_t d;
            uint32_t u;
            if constexpr (sizeof(D) == 4) { d = heap.back() >> 32; u = static_cast<uint32_t>(heap.back()); }
            else { d = heap.back().first; u = heap.back().second; }
            heap.pop_back();
            if (d > dist[u]) continue;  // stale entry
            if (u == t) { result = static_cast<int64_t>(d); break; }
            for (uint32_t i = off[u], e = off[u + 1]; i < e; ++i) {
                const uint64_t nd = d + arcs[i].w;
                const uint32_t v = arcs[i].to;
                if (nd >= kInf) { result = kOverflow; heap.clear(); break; }
                if (nd < dist[v]) {
                    if (dist[v] == kInf) touched.push_back(v);
                    dist[v] = static_cast<D>(nd);
                    heap.push_back(entry(nd, v));
                    std::push_heap(heap.begin(), heap.end(), cmp);
                }
            }
        }
        for (uint32_t v : touched) dist[v] = kInf;
        touched.clear();
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
    std::vector<std::vector<Edge>> parts(nt);
    parallel(nt, [&](int i) {
        const char* p = cut[i];
        const char* e = std::max(cut[i], cut[i + 1]);
        auto& v = parts[i];
        v.reserve(static_cast<size_t>(e - p) / 12 + 16);
        while (p < e) {
            const uint32_t u = static_cast<uint32_t>(read_uint(p)); ++p;
            const uint32_t to = static_cast<uint32_t>(read_uint(p)); ++p;
            const uint32_t w = static_cast<uint32_t>(read_uint(p)); ++p;
            v.push_back({u, to, w});
        }
    });

    // CSR build, all threads at once: out-degree counts, prefix sums, then each edge claims a
    // slot in its tail's block. The order of arcs inside a block does not change any distance.
    std::vector<uint32_t> off(n + 2, 0);
    std::vector<uint8_t> has_in(n, 0);  // a node with no in-edge can only be reached from itself
    parallel(nt, [&](int i) {
        for (const Edge& e : parts[i]) {
            std::atomic_ref<uint32_t>(off[e.u + 2]).fetch_add(1, std::memory_order_relaxed);
            std::atomic_ref<uint8_t>(has_in[e.v]).store(1, std::memory_order_relaxed);
        }
    });
    for (uint64_t i = 2; i < n + 2; ++i) off[i] += off[i - 1];
    // off[u + 1] is now the start of u's block; claiming slots advances it to the block end,
    // which is the start of u + 1's block.
    std::vector<Arc> arcs(m);
    parallel(nt, [&](int i) {
        for (const Edge& e : parts[i]) arcs[std::atomic_ref<uint32_t>(off[e.u + 1]).fetch_add(1, std::memory_order_relaxed)] = {e.v, e.w};
        std::vector<Edge>().swap(parts[i]);
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
        std::vector<Arc> arcs2(m);
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
