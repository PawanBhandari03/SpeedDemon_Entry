// Speed Demon 2026, Track D (Traverse): graph.txt + queries.txt -> distances.txt
//
// Plan: mmap graph.txt, cut it into one chunk per thread at newline boundaries and parse the
// edges in parallel. A counting sort turns the edge list into a compressed-sparse-row graph
// (one flat array of (head, weight) arcs, grouped by tail). The 2,000 queries are then
// answered in parallel, each by a Dijkstra search that stops when the target is settled. A
// search only resets the nodes it touched, so no per-query clearing of the whole graph.
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <thread>
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

constexpr uint64_t kInf = ~0ull;

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

// Dijkstra with a binary heap on a reusable distance array.
struct Searcher {
    const uint32_t* off;
    const Arc* arcs;
    std::vector<uint64_t> dist;
    std::vector<uint32_t> touched;
    std::vector<std::pair<uint64_t, uint32_t>> heap;

    Searcher(const uint32_t* o, const Arc* a, size_t n) : off(o), arcs(a), dist(n, kInf) {}

    int64_t run(uint32_t s, uint32_t t) {
        if (s == t) return 0;
        const auto cmp = std::greater<std::pair<uint64_t, uint32_t>>();
        int64_t result = -1;
        dist[s] = 0;
        touched.push_back(s);
        heap.emplace_back(0, s);
        while (!heap.empty()) {
            std::pop_heap(heap.begin(), heap.end(), cmp);
            const auto [d, u] = heap.back();
            heap.pop_back();
            if (d > dist[u]) continue;  // stale entry
            if (u == t) { result = static_cast<int64_t>(d); break; }
            for (uint32_t i = off[u], e = off[u + 1]; i < e; ++i) {
                const uint64_t nd = d + arcs[i].w;
                const uint32_t v = arcs[i].to;
                if (nd < dist[v]) {
                    if (dist[v] == kInf) touched.push_back(v);
                    dist[v] = nd;
                    heap.emplace_back(nd, v);
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
    {
        std::vector<std::thread> th;
        for (int i = 0; i < nt; ++i)
            th.emplace_back([&, i] {
                const char* p = cut[i];
                const char* e = std::max(cut[i], cut[i + 1]);
                auto& v = parts[i];
                v.reserve(static_cast<size_t>(e - p) / 12 + 16);
                while (p < e) {
                    const uint32_t u = static_cast<uint32_t>(read_uint(p)); ++p;
                    const uint32_t w2 = static_cast<uint32_t>(read_uint(p)); ++p;
                    const uint32_t wt = static_cast<uint32_t>(read_uint(p)); ++p;
                    v.push_back({u, w2, wt});
                }
            });
        for (auto& t : th) t.join();
    }

    // CSR build.
    std::vector<uint32_t> off(n + 2, 0);
    std::vector<uint8_t> has_in(n, 0);  // a node with no in-edge can only be reached from itself
    for (const auto& part : parts)
        for (const Edge& e : part) {
            ++off[e.u + 2];
            has_in[e.v] = 1;
        }
    for (uint64_t i = 2; i < n + 2; ++i) off[i] += off[i - 1];
    // off[u + 1] is now the start of u's block; scattering advances it to the block end,
    // which is the start of u + 1's block.
    std::vector<Arc> arcs(m);
    for (const auto& part : parts)
        for (const Edge& e : part) arcs[off[e.u + 1]++] = {e.v, e.w};
    parts.clear();
    parts.shrink_to_fit();

    // Queries.
    std::vector<std::pair<uint32_t, uint32_t>> queries;
    const Mapped q = map_file(data + "/queries.txt");
    if (q.p) {
        const char* p = q.p;
        const char* e = q.p + q.n;
        while (p < e) {
            const uint32_t s = static_cast<uint32_t>(read_uint(p)); ++p;
            const uint32_t t = static_cast<uint32_t>(read_uint(p)); ++p;
            queries.emplace_back(s, t);
        }
    }

    std::vector<int64_t> answer(queries.size());
    {
        const int qt = static_cast<int>(std::min<size_t>(thread_count(), std::max<size_t>(1, queries.size())));
        std::atomic<size_t> next{0};
        std::vector<std::thread> th;
        for (int i = 0; i < qt; ++i)
            th.emplace_back([&] {
                Searcher s(off.data(), arcs.data(), n);
                for (size_t k; (k = next.fetch_add(1)) < queries.size();) {
                    const auto [src, dst] = queries[k];
                    // Without this check the search settles everything src can reach before giving up.
                    answer[k] = (src != dst && !has_in[dst]) ? -1 : s.run(src, dst);
                }
            });
        for (auto& t : th) t.join();
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
