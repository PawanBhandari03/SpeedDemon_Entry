// Speed Demon 2026, Track A (Ingest): requests.log -> summary.txt
//
// Plan: map the file, cut it into one chunk per thread at newline boundaries, and let every
// thread scan its chunk once with hand-written parsing. A thread only appends compact
// (endpoint, hour, latency, is_error) records. After the threads join we give every
// endpoint a global id in byte order and bucket the latencies by (hour, endpoint) with a
// counting sort that every thread helps with. The bucket index is hour-major, so walking the
// buckets in order is already the output order: each thread takes a run of buckets, picks
// p50/p95/p99 with nth_element and formats its lines, and the runs are written in order.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <immintrin.h>
#include <fcntl.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/prctl.h>
#include <unistd.h>

namespace {

constexpr int kHours = 72;  // the span is fixed by the spec: 2026-10-30T18 .. 2026-11-02T17

struct Rec {
    uint16_t ep;   // thread-local endpoint id, then the global one
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
            // The full 64-bit hash (seeded with the length) decides: comparing the bytes costs a cold load
            // from the first occurrence of the endpoint in the file, and two of the ~2000 endpoints
            // colliding has a chance of about 1e-13.
            if (s.h == h && s.len == n) return static_cast<uint16_t>(s.id);
        }
    }
};

// A worker counts its records per (local endpoint, hour) while parsing, so no extra pass over the
// records is needed. calloc keeps the untouched part of the arrays out of memory.
struct Worker {
    EpTable table;
    std::vector<Rec> recs;
    uint32_t* cursor = static_cast<uint32_t*>(calloc(EpTable::kCap * kHours, 4));  // count, then write position
    uint32_t* errs = static_cast<uint32_t*>(calloc(EpTable::kCap * kHours, 4));    // 5xx count
    std::string out;
    bool bad = false;
};

// Index of the first space byte in the 8 bytes w, or 8 if there is none.
inline unsigned space_in(uint64_t w) {
    const uint64_t x = w ^ 0x2020202020202020ull;
    const uint64_t t = (x - 0x0101010101010101ull) & ~x & 0x8080808080808080ull;
    return t ? static_cast<unsigned>(__builtin_ctzll(t)) / 8 : 8;
}

// Splits "METHOD ENDPOINT STATUS LATENCY<LF>" at q, which has at least 64 readable bytes, with
// two 32-byte compares instead of byte loops: the bits of the space and LF masks give the field
// ends. Returns the pointer just past the LF, or nullptr when the line does not fit in the
// window or has an unexpected shape (the caller then parses it byte by byte).
inline const char* fast_fields(const char* q, const char*& ep, size_t& ep_len, uint8_t& err, uint32_t& lat) {
    const __m256i lo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q));
    const __m256i hi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + 32));
    const __m256i space = _mm256_set1_epi8(' ');
    const __m256i lf = _mm256_set1_epi8('\n');
    const uint64_t spaces = static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(lo, space))) |
                            static_cast<uint64_t>(static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(hi, space)))) << 32;
    const uint64_t lfs = static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(lo, lf))) |
                         static_cast<uint64_t>(static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(hi, lf)))) << 32;
    if (!lfs) return nullptr;
    const unsigned end = static_cast<unsigned>(__builtin_ctzll(lfs));  // index of the LF
    uint64_t sp = spaces & ((uint64_t(1) << end) - 1);                // spaces before it
    if (__builtin_popcountll(sp) != 3) return nullptr;
    const unsigned s1 = static_cast<unsigned>(__builtin_ctzll(sp));
    sp &= sp - 1;
    const unsigned s2 = static_cast<unsigned>(__builtin_ctzll(sp));
    sp &= sp - 1;
    const unsigned s3 = static_cast<unsigned>(__builtin_ctzll(sp));
    const unsigned digits = end - s3 - 1;  // latency digits
    if (digits == 0 || digits > 8 || s3 != s2 + 4) return nullptr;
    ep = q + s1 + 1;
    ep_len = s2 - s1 - 1;
    err = (q[s2 + 1] == '5');
    // The last 8 bytes before the LF hold the digits at their top end; clear the bytes in front
    // and turn the 8 digits (leading zeros included) into a number with three multiplications.
    uint64_t v;
    memcpy(&v, q + end - 8, 8);
    v &= ~uint64_t(0) << ((8 - digits) * 8);
    v = ((v & 0x0F0F0F0F0F0F0F0Full) * 2561) >> 8;
    v = ((v & 0x00FF00FF00FF00FFull) * 6553601) >> 16;
    v = ((v & 0x0000FFFF0000FFFFull) * 42949672960001ull) >> 32;
    lat = static_cast<uint32_t>(v);
    return q + end + 1;
}

void scan(const char* b, const char* e, int64_t base_hour, Worker& w) {
    w.recs.reserve(static_cast<size_t>(e - b) / 70 + 16);
    // Neighbouring lines almost always share their hour, so the date arithmetic runs only when
    // the "YYYY-MM-DDTHH" prefix changes (compared as two overlapping 8-byte words).
    uint64_t prev_a = 0, prev_b = 0;
    int64_t rel = 0;
    while (b < e) {
        // ts: 2026-10-30T18:07:18.569411Z  (always 27 bytes)
        uint64_t ta, tb;
        memcpy(&ta, b, 8);
        memcpy(&tb, b + 5, 8);
        if (ta != prev_a || tb != prev_b) {
            prev_a = ta;
            prev_b = tb;
            const int64_t y = (b[0] - '0') * 1000 + (b[1] - '0') * 100 + dig2(b + 2);
            rel = days_from_civil(y, dig2(b + 5), dig2(b + 8)) * 24 + dig2(b + 11) - base_hour;
            if (rel < 0 || rel >= kHours) w.bad = true;
        }
        const char* q = b + 28;  // method
        const char* ep;
        size_t ep_len;
        uint8_t err;
        uint32_t lat;
        const char* next = e - q >= 64 ? fast_fields(q, ep, ep_len, err, lat) : nullptr;
        if (!next) {
            while (*q != ' ') ++q;
            ep = ++q;                   // endpoint, scanned 8 bytes at a time
            for (;;) {
                if (e - q < 8) { while (*q != ' ') ++q; break; }
                uint64_t word;
                memcpy(&word, q, 8);
                const unsigned k = space_in(word);
                q += k;
                if (k < 8) break;
            }
            ep_len = static_cast<size_t>(q - ep);
            ++q;
            err = (*q == '5');          // status is always three digits
            q += 4;
            lat = 0;
            while (*q != '\n') lat = lat * 10 + static_cast<uint32_t>(*q++ - '0');
            next = q + 1;
        }
        const uint16_t id = w.table.intern(ep, ep_len);
        w.recs.push_back({id, static_cast<uint8_t>(rel), err, lat});
        ++w.cursor[size_t(id) * kHours + static_cast<size_t>(rel)];
        w.errs[size_t(id) * kHours + static_cast<size_t>(rel)] += err;
        b = next;
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

// Runs f(i) on threads 0..n-1 and waits for all of them.
template <class F>
void parallel(int n, F&& f) {
    std::vector<std::thread> th;
    for (int i = 1; i < n; ++i) th.emplace_back(f, i);
    f(0);
    for (auto& t : th) t.join();
}

}  // namespace

int main(int argc, char** argv) {
    // Huge pages would commit 2 MB for every sparsely touched page of the big arrays; ask for 4 KB pages.
    prctl(PR_SET_THP_DISABLE, 1, 0, 0, 0);
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
    FILE* f = fopen((out + "/summary.txt").c_str(), "wb");
    if (!f) return 1;
    if (size == 0) { fclose(f); return 0; }

    const char* base = static_cast<const char*>(mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0));
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
    parallel(nt, [&](int i) { scan(cut[i], std::max(cut[i], cut[i + 1]), base_hour, workers[i]); });
    for (auto& w : workers) if (w.bad) return 1;

    // Global endpoint ids, assigned in byte order.
    std::vector<std::string_view> eps;
    for (auto& w : workers) eps.insert(eps.end(), w.table.names.begin(), w.table.names.end());
    std::sort(eps.begin(), eps.end());
    eps.erase(std::unique(eps.begin(), eps.end()), eps.end());
    const size_t E = eps.size();
    const size_t G = static_cast<size_t>(kHours) * E;

    // Bucket g = hour * E + global endpoint. Add up the workers' counts per bucket, then turn each
    // worker's count into its write position inside the bucket (workers in order).
    std::vector<std::vector<uint32_t>> remap(nt);
    for (int i = 0; i < nt; ++i)
        for (auto sv : workers[i].table.names)
            remap[i].push_back(static_cast<uint32_t>(std::lower_bound(eps.begin(), eps.end(), sv) - eps.begin()));
    std::vector<uint32_t> start(G + 1, 0), errs(G, 0);
    for (int i = 0; i < nt; ++i)
        for (size_t lep = 0; lep < remap[i].size(); ++lep)
            for (size_t h = 0; h < kHours; ++h) {
                const size_t g = h * E + remap[i][lep];
                start[g + 1] += workers[i].cursor[lep * kHours + h];
                errs[g] += workers[i].errs[lep * kHours + h];
            }
    for (size_t g = 0; g < G; ++g) start[g + 1] += start[g];
    std::vector<uint32_t> running(start.begin(), start.end() - 1);
    for (int i = 0; i < nt; ++i)
        for (size_t lep = 0; lep < remap[i].size(); ++lep)
            for (size_t h = 0; h < kHours; ++h) {
                uint32_t& c = workers[i].cursor[lep * kHours + h];
                const uint32_t n = c;
                c = running[h * E + remap[i][lep]];
                running[h * E + remap[i][lep]] += n;
            }
    std::vector<uint32_t> lats(start[G]);
    parallel(nt, [&](int i) {
        Worker& w = workers[i];
        for (const Rec& r : w.recs) lats[w.cursor[size_t(r.ep) * kHours + r.hour]++] = r.lat;
        std::vector<Rec>().swap(w.recs);
    });

    char hourstr[kHours][16];
    for (int h = 0; h < kHours; ++h) {
        const int64_t t = base_hour + h;
        const int64_t z = (t / 24) + 719468;  // days since 0000-03-01
        const int64_t era = (z >= 0 ? z : z - 146096) / 146097;
        const unsigned doe = static_cast<unsigned>(z - era * 146097);
        const unsigned yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
        const unsigned doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        const unsigned mp = (5 * doy + 2) / 153;
        const unsigned d = doy - (153 * mp + 2) / 5 + 1;
        const unsigned m = mp < 10 ? mp + 3 : mp - 9;
        const int64_t yr = static_cast<int64_t>(yoe) + era * 400 + (m <= 2);
        snprintf(hourstr[h], sizeof hourstr[h], "%04lld-%02u-%02uT%02lld", static_cast<long long>(yr), m, d,
                 static_cast<long long>(t % 24));
    }

    // Each worker takes a run of buckets holding about the same number of latencies, picks the
    // percentiles (nearest rank) and formats its lines.
    std::vector<size_t> bound(nt + 1, G);
    bound[0] = 0;
    for (int i = 1; i < nt; ++i)
        bound[i] = static_cast<size_t>(std::lower_bound(start.begin(), start.end(), static_cast<uint32_t>(uint64_t(start[G]) * i / nt)) - start.begin());
    parallel(nt, [&](int i) {
        std::string& o = workers[i].out;
        o.reserve((bound[i + 1] - bound[i]) * 90);
        char line[256];
        for (size_t g = bound[i]; g < bound[i + 1]; ++g) {
            const size_t n = start[g + 1] - start[g];
            if (!n) continue;
            uint32_t* v = lats.data() + start[g];
            const size_t r99 = (99 * n + 99) / 100, r95 = (95 * n + 99) / 100, r50 = (50 * n + 99) / 100;
            if (n <= 64) {
                // Most buckets are small: a plain sort of a few values beats three selections.
                std::sort(v, v + n);
            } else {
                // Each selection leaves the smaller ranks in front, so the next one searches less.
                std::nth_element(v, v + r99 - 1, v + n);
                std::nth_element(v, v + r95 - 1, v + r99);
                std::nth_element(v, v + r50 - 1, v + r95);
            }
            char* p = line;
            memcpy(p, hourstr[g / E], 13); p += 13;  // YYYY-MM-DDTHH
            memcpy(p, ":00:00Z ", 8); p += 8;
            const std::string_view name = eps[g % E];
            memcpy(p, name.data(), name.size()); p += name.size();
            *p++ = ' '; p = put_u64(p, n);
            *p++ = ' '; p = put_u64(p, v[r50 - 1]);
            *p++ = ' '; p = put_u64(p, v[r95 - 1]);
            *p++ = ' '; p = put_u64(p, v[r99 - 1]);
            *p++ = ' '; p = put_u64(p, errs[g]);
            *p++ = ' '; p = put_u64(p, static_cast<uint64_t>(errs[g]) * 1000000ull / n);
            *p++ = '\n';
            o.append(line, static_cast<size_t>(p - line));
        }
    });
    for (auto& w : workers) fwrite(w.out.data(), 1, w.out.size(), f);
    fclose(f);
    return 0;
}
