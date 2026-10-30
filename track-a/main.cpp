// Speed Demon 2026, Track A (Ingest): requests.log -> summary.txt
//
// Plan: mmap the file, cut it into one chunk per thread at newline boundaries, and let every
// thread scan its chunk once with hand-written parsing. A thread only appends compact
// (endpoint, hour, latency, is_error) records. After the threads join we give every
// endpoint a global id in byte order, bucket the latencies by (hour, endpoint) with a counting
// sort, pick p50/p95/p99 with nth_element, and print. The bucket index is hour-major, so
// walking the buckets in order is already the output order.
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

constexpr int kHours = 72;  // the span is fixed by the spec: 2026-10-30T18 .. 2026-11-02T17

struct Rec {
    uint16_t ep;   // thread-local endpoint id
    uint8_t hour;  // 0..71 since 2026-10-30T18:00Z
    uint8_t err;   // 1 if status is 5xx
    uint32_t lat;
};

// Days since 1970-01-01 for a civil date (Howard Hinnant's algorithm).
int64_t days_from_civil(int64_t y, unsigned m, unsigned d) {
    y -= m <= 2;
    const int64_t era = (y >= 0 ? y : y - 399) / 400;
    const unsigned yoe = static_cast<unsigned>(y - era * 400);
    const unsigned doy = (153 * (m > 2 ? m - 3 : m + 9) + 2) / 5 + d - 1;
    const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return era * 146097 + static_cast<int64_t>(doe) - 719468;
}

inline int dig2(const char* p) { return (p[0] - '0') * 10 + (p[1] - '0'); }

// Open-addressing table: endpoint bytes -> small id. The bytes stay in the mmap.
struct EpTable {
    struct Slot {
        const char* p = nullptr;
        uint32_t len = 0;
        uint32_t id = 0;
        uint64_t h = 0;
    };
    static constexpr size_t kCap = 8192;
    std::vector<Slot> slots = std::vector<Slot>(kCap);
    std::vector<std::string_view> names;

    static uint64_t hash(const char* p, size_t n) {
        uint64_t h = 0x9E3779B97F4A7C15ull ^ n;
        while (n >= 8) {
            uint64_t w;
            memcpy(&w, p, 8);
            h = (h ^ w) * 0xff51afd7ed558ccdull;
            h ^= h >> 32;
            p += 8;
            n -= 8;
        }
        uint64_t w = 0;
        memcpy(&w, p, n);
        h = (h ^ w) * 0xc4ceb9fe1a85ec53ull;
        return h ^ (h >> 29);
    }

    uint16_t intern(const char* p, size_t n) {
        const uint64_t h = hash(p, n);
        size_t i = h & (kCap - 1);
        for (;; i = (i + 1) & (kCap - 1)) {
            Slot& s = slots[i];
            if (!s.p) {
                s = {p, static_cast<uint32_t>(n), static_cast<uint32_t>(names.size()), h};
                names.emplace_back(p, n);
                return static_cast<uint16_t>(s.id);
            }
            if (s.h == h && s.len == n && memcmp(s.p, p, n) == 0) return static_cast<uint16_t>(s.id);
        }
    }
};

struct Worker {
    EpTable table;
    std::vector<Rec> recs;
    bool bad = false;
};

void scan(const char* b, const char* e, int64_t base_hour, Worker& w) {
    w.recs.reserve(static_cast<size_t>(e - b) / 70 + 16);
    while (b < e) {
        // ts: 2026-10-30T18:07:18.569411Z  (always 27 bytes)
        const int64_t y = (b[0] - '0') * 1000 + (b[1] - '0') * 100 + dig2(b + 2);
        const int64_t abs_hour = days_from_civil(y, dig2(b + 5), dig2(b + 8)) * 24 + dig2(b + 11);
        const int64_t rel = abs_hour - base_hour;
        if (rel < 0 || rel >= kHours) w.bad = true;
        const char* q = b + 28;                 // method
        while (*q != ' ') ++q;
        const char* ep = ++q;                   // endpoint
        while (*q != ' ') ++q;
        const size_t ep_len = static_cast<size_t>(q - ep);
        ++q;
        const uint8_t err = (*q == '5');        // status is always three digits
        q += 4;
        uint32_t lat = 0;
        while (*q != '\n') lat = lat * 10 + static_cast<uint32_t>(*q++ - '0');
        w.recs.push_back({w.table.intern(ep, ep_len), static_cast<uint8_t>(rel), err, lat});
        b = q + 1;
    }
}

inline char* put_u64(char* o, uint64_t v) {
    char tmp[20];
    int n = 0;
    do { tmp[n++] = static_cast<char>('0' + v % 10); v /= 10; } while (v);
    while (n) *o++ = tmp[--n];
    return o;
}

int thread_count() {
    cpu_set_t set;
    if (sched_getaffinity(0, sizeof set, &set) == 0) {
        const int n = CPU_COUNT(&set);
        if (n > 0) return n;
    }
    return static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
}

}  // namespace

int main(int argc, char** argv) {
    std::string data = "/data", out = "/out";
    for (int i = 1; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "--data")) data = argv[i + 1];
        else if (!strcmp(argv[i], "--out")) out = argv[i + 1];
    }

    const int fd = open((data + "/requests.log").c_str(), O_RDONLY);
    if (fd < 0) return 1;
    struct stat st;
    if (fstat(fd, &st) != 0) return 1;
    const size_t size = static_cast<size_t>(st.st_size);

    std::string result;
    if (size > 0) {
        const char* base = static_cast<const char*>(mmap(nullptr, size, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0));
        if (base == MAP_FAILED) return 1;
        const int64_t base_hour = days_from_civil(2026, 10, 30) * 24 + 18;

        // One chunk per thread, cut just after a newline.
        const int nt = size < (1u << 20) ? 1 : thread_count();
        std::vector<const char*> cut(nt + 1);
        cut[0] = base;
        cut[nt] = base + size;
        for (int i = 1; i < nt; ++i) {
            const char* p = base + size * i / nt;
            const char* nl = static_cast<const char*>(memchr(p - 1, '\n', static_cast<size_t>(base + size - (p - 1))));
            cut[i] = nl ? nl + 1 : base + size;
        }
        std::vector<Worker> workers(nt);
        {
            std::vector<std::thread> th;
            for (int i = 0; i < nt; ++i)
                th.emplace_back([&, i] { scan(cut[i], std::max(cut[i], cut[i + 1]), base_hour, workers[i]); });
            for (auto& t : th) t.join();
        }
        for (auto& w : workers) if (w.bad) return 1;

        // Global endpoint ids, assigned in byte order.
        std::vector<std::string_view> eps;
        for (auto& w : workers) eps.insert(eps.end(), w.table.names.begin(), w.table.names.end());
        std::sort(eps.begin(), eps.end());
        eps.erase(std::unique(eps.begin(), eps.end()), eps.end());
        const size_t E = eps.size();
        std::vector<std::vector<uint32_t>> remap(nt);
        for (int i = 0; i < nt; ++i)
            for (auto sv : workers[i].table.names)
                remap[i].push_back(static_cast<uint32_t>(std::lower_bound(eps.begin(), eps.end(), sv) - eps.begin()));

        // Counting sort of latencies into (hour, endpoint) buckets.
        const size_t G = static_cast<size_t>(kHours) * E;
        std::vector<uint32_t> start(G + 1, 0), errs(G, 0);
        for (int i = 0; i < nt; ++i)
            for (const Rec& r : workers[i].recs) {
                const size_t g = static_cast<size_t>(r.hour) * E + remap[i][r.ep];
                ++start[g + 1];
                errs[g] += r.err;
            }
        for (size_t g = 0; g < G; ++g) start[g + 1] += start[g];
        std::vector<uint32_t> lats(start[G]), fill(start.begin(), start.end() - 1);
        for (int i = 0; i < nt; ++i)
            for (const Rec& r : workers[i].recs)
                lats[fill[static_cast<size_t>(r.hour) * E + remap[i][r.ep]]++] = r.lat;

        // Percentiles (nearest rank), spread over the threads.
        std::vector<uint32_t> p50(G), p95(G), p99(G);
        {
            std::atomic<size_t> next{0};
            std::vector<std::thread> th;
            for (int i = 0; i < nt; ++i)
                th.emplace_back([&] {
                    constexpr size_t kBatch = 256;
                    for (size_t g0; (g0 = next.fetch_add(kBatch)) < G;)
                        for (size_t g = g0; g < std::min(G, g0 + kBatch); ++g) {
                            const size_t n = start[g + 1] - start[g];
                            if (!n) continue;
                            uint32_t* v = lats.data() + start[g];
                            const size_t r99 = (99 * n + 99) / 100, r95 = (95 * n + 99) / 100, r50 = (50 * n + 99) / 100;
                            // Each selection leaves the smaller ranks in front, so the next one searches less.
                            std::nth_element(v, v + r99 - 1, v + n);
                            p99[g] = v[r99 - 1];
                            std::nth_element(v, v + r95 - 1, v + r99);
                            p95[g] = v[r95 - 1];
                            std::nth_element(v, v + r50 - 1, v + r95);
                            p50[g] = v[r50 - 1];
                        }
                });
            for (auto& t : th) t.join();
        }

        // Output.
        char hourstr[kHours][24];
        for (int h = 0; h < kHours; ++h) {
            const int64_t t = base_hour + h;
            int64_t z = (t / 24) + 719468;  // days since 0000-03-01
            const int64_t era = (z >= 0 ? z : z - 146096) / 146097;
            const unsigned doe = static_cast<unsigned>(z - era * 146097);
            const unsigned yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
            const unsigned doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
            const unsigned mp = (5 * doy + 2) / 153;
            const unsigned d = doy - (153 * mp + 2) / 5 + 1;
            const unsigned m = mp < 10 ? mp + 3 : mp - 9;
            const int64_t yr = static_cast<int64_t>(yoe) + era * 400 + (m <= 2);
            snprintf(hourstr[h], sizeof hourstr[h], "%04lld-%02u-%02uT%02lldZ", static_cast<long long>(yr), m, d,
                     static_cast<long long>(t % 24));
        }
        result.reserve(static_cast<size_t>(start[G]) * 4 + (1 << 20));
        std::vector<char> line(512);
        for (size_t g = 0; g < G; ++g) {
            const uint64_t n = start[g + 1] - start[g];
            if (!n) continue;
            const char* hs = hourstr[g / E];  // "YYYY-MM-DDTHHZ" -> needs ":00:00Z"
            char* o = line.data();
            memcpy(o, hs, 13); o += 13;       // YYYY-MM-DDTHH
            memcpy(o, ":00:00Z ", 8); o += 8;
            const std::string_view name = eps[g % E];
            memcpy(o, name.data(), name.size()); o += name.size();
            *o++ = ' '; o = put_u64(o, n);
            *o++ = ' '; o = put_u64(o, p50[g]);
            *o++ = ' '; o = put_u64(o, p95[g]);
            *o++ = ' '; o = put_u64(o, p99[g]);
            *o++ = ' '; o = put_u64(o, errs[g]);
            *o++ = ' '; o = put_u64(o, static_cast<uint64_t>(errs[g]) * 1000000ull / n);
            *o++ = '\n';
            result.append(line.data(), static_cast<size_t>(o - line.data()));
        }
    }

    const std::string path = out + "/summary.txt";
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return 1;
    fwrite(result.data(), 1, result.size(), f);
    fclose(f);
    return 0;
}
