// Speed Demon 2026, Track A (Ingest): requests.log -> summary.txt
//
// Plan: map the file, cut it into one chunk per thread at newline boundaries, and let every
// thread scan its chunk once with hand-written parsing. A thread files each latency into small
// blocks chained per (hour, endpoint) bucket, so nothing is copied again later. After the threads
// join we give every endpoint a global id in byte order. The output is hour-major, so each hour is
// one task: it gathers the blocks of all threads for every endpoint, picks p50/p95/p99 with a
// sort or nth_element, formats the lines, and the hours are written in order.
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <immintrin.h>
#include <fcntl.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

constexpr int kHours = 72;  // the span is fixed by the spec: 2026-10-30T18 .. 2026-11-02T17

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
//
// Two levels, because the endpoint of a line is a Zipf draw: a 256-entry cache in front holds the
// endpoints that actually repeat (4 KB, always in L1), and only a miss walks the 4096-slot table.
// A slot is 16 bytes, so the table is 64 KB and a miss is an L2 access rather than an L3 one.
struct EpTable {
    struct Slot {
        uint64_t h = 0;
        uint32_t id = 0;
        uint32_t len = 0;
    };
    struct Line {
        uint64_t h = 0;  // 0 marks an empty entry; intern() keeps hashes odd
        uint32_t id = 0;
        uint32_t len = 0;
    };
    static constexpr size_t kCap = 4096;    // the catalogue is 1,842 endpoints, so 45% full
    static constexpr size_t kCache = 256;   // 4 KB of L1
    std::vector<Slot> slots = std::vector<Slot>(kCap);
    std::vector<Line> cache = std::vector<Line>(kCache);
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
        return (h ^ (h >> 29)) | 1;
    }

    uint16_t intern(const char* p, size_t n) {
        const uint64_t h = hash(p, n);
        Line& c = cache[h & (kCache - 1)];
        if (c.h == h && c.len == n) return static_cast<uint16_t>(c.id);
        size_t i = h & (kCap - 1);
        for (;; i = (i + 1) & (kCap - 1)) {
            Slot& s = slots[i];
            if (!s.h) {
                const uint32_t id = static_cast<uint32_t>(names.size());
                s = {h, id, static_cast<uint32_t>(n)};
                names.emplace_back(p, n);
                c = {h, id, static_cast<uint32_t>(n)};
                return static_cast<uint16_t>(id);
            }
            // The full 64-bit hash (seeded with the length) decides: comparing the bytes costs a cold load
            // from the first occurrence of the endpoint in the file, and two of the ~2000 endpoints
            // colliding has a chance of about 1e-13.
            if (s.h == h && s.len == n) {
                c = {h, s.id, s.len};
                return static_cast<uint16_t>(s.id);
            }
        }
    }
};

// A bucket is one (hour, endpoint) pair. A worker keeps, per bucket, a count, the 5xx count and a chain of
// small blocks that hold its latencies: 7 values and a link to the previous block, 32 bytes. Blocks come
// from big slabs, so one record costs one counter update and one store, and no record is ever copied
// again: the percentile step reads the blocks of every worker for a bucket straight from where they are.
constexpr size_t kBlockValues = 7;
constexpr unsigned kSlabShift = 14;  // 16384 blocks (512 KB) per slab
constexpr size_t kSlabBlocks = size_t(1) << kSlabShift;

struct Cell {
    uint32_t count = 0, errs = 0;
    uint32_t tail = 0;  // block with the newest values (block 0 is never handed out)
};

struct Worker {
    EpTable table;
    int8_t slot_of[kHours];               // row of each hour this worker has seen, -1 if none
    std::vector<std::vector<Cell>> rows;  // [row of an hour][local endpoint]; a chunk spans only a few hours
    std::vector<uint32_t*> slabs;
    size_t used = kSlabBlocks;  // blocks taken from the newest slab
    bool bad = false;

    Worker() { memset(slot_of, -1, sizeof slot_of); }
    Worker(const Worker&) = delete;
    Worker& operator=(const Worker&) = delete;
    ~Worker() {
        for (uint32_t* s : slabs) free(s);
    }

    uint32_t* block(uint32_t id) const { return slabs[id >> kSlabShift] + size_t(id & (kSlabBlocks - 1)) * 8; }

    uint32_t new_block() {
        if (used == kSlabBlocks) {
            slabs.push_back(static_cast<uint32_t*>(aligned_alloc(64, kSlabBlocks * 32)));
            used = slabs.size() == 1 ? 1 : 0;
        }
        return static_cast<uint32_t>(((slabs.size() - 1) << kSlabShift) | used++);
    }

    void add(size_t hour, uint16_t ep, uint32_t lat, uint8_t err) {
        int s = slot_of[hour];
        if (s < 0) {
            s = static_cast<int>(rows.size());
            slot_of[hour] = static_cast<int8_t>(s);
            rows.emplace_back();
        }
        std::vector<Cell>& row = rows[static_cast<size_t>(s)];
        if (ep >= row.size()) row.resize(std::max<size_t>(ep + 1, row.size() * 2));
        Cell& c = row[ep];
        const size_t pos = c.count % kBlockValues;
        uint32_t* blk;
        if (pos == 0) {
            const uint32_t b = new_block();
            blk = block(b);
            blk[kBlockValues] = c.tail;
            c.tail = b;
        } else {
            blk = block(c.tail);
        }
        blk[pos] = lat;
        ++c.count;
        c.errs += err;
    }

    // The cell of (hour, local endpoint), or null when this worker saw no such record.
    const Cell* cell(size_t hour, uint32_t local_ep) const {
        const int s = slot_of[hour];
        if (s < 0 || local_ep == UINT32_MAX) return nullptr;
        const std::vector<Cell>& row = rows[static_cast<size_t>(s)];
        return local_ep < row.size() ? &row[local_ep] : nullptr;
    }
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
            if (rel < 0 || rel >= kHours) {
                w.bad = true;  // the run fails below; keep the index valid until then
                rel = 0;
            }
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
        w.add(static_cast<size_t>(rel), w.table.intern(ep, ep_len), lat, err);
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

    // Global endpoint ids, assigned in byte order, and for every worker the map from the global id
    // back to its local one (UINT32_MAX when the worker never saw the endpoint).
    std::vector<std::string_view> eps;
    for (auto& w : workers) eps.insert(eps.end(), w.table.names.begin(), w.table.names.end());
    std::sort(eps.begin(), eps.end());
    eps.erase(std::unique(eps.begin(), eps.end()), eps.end());
    const size_t E = eps.size();
    std::vector<std::vector<uint32_t>> local_of(nt, std::vector<uint32_t>(E, UINT32_MAX));
    for (int i = 0; i < nt; ++i)
        for (size_t lep = 0; lep < workers[i].table.names.size(); ++lep)
            local_of[i][static_cast<size_t>(std::lower_bound(eps.begin(), eps.end(), workers[i].table.names[lep]) - eps.begin())] =
                static_cast<uint32_t>(lep);

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

    // One task per hour, handed out to the threads in turn. For every endpoint the task collects the
    // latencies of all workers, picks the percentiles (nearest rank) and formats the line. The output
    // order is hour-major with the endpoints in byte order, so a finished hour is written as soon as all
    // the hours before it are, and its text is freed.
    std::vector<std::string> hour_out(kHours);
    std::vector<uint8_t> hour_done(kHours, 0);
    std::mutex write_mutex;
    int next_write = 0;
    std::atomic<int> next_hour{0};
    parallel(nt, [&](int) {
        std::vector<uint32_t> vals;
        char line[256];
        for (int h; (h = next_hour.fetch_add(1)) < kHours;) {
            std::string& o = hour_out[static_cast<size_t>(h)];
            o.reserve(E * 80);
            for (size_t e = 0; e < E; ++e) {
                size_t n = 0;
                uint64_t errs = 0;
                for (int i = 0; i < nt; ++i)
                    if (const Cell* c = workers[i].cell(static_cast<size_t>(h), local_of[i][e])) {
                        n += c->count;
                        errs += c->errs;
                    }
                if (!n) continue;
                vals.resize(n);
                size_t k = 0;
                for (int i = 0; i < nt; ++i) {
                    const Cell* c = workers[i].cell(static_cast<size_t>(h), local_of[i][e]);
                    if (!c || !c->count) continue;
                    size_t left = c->count;
                    size_t take = (left - 1) % kBlockValues + 1;  // the newest block may be partly filled
                    for (uint32_t id = c->tail; left;) {
                        const uint32_t* blk = workers[i].block(id);
                        memcpy(&vals[k], blk, take * sizeof(uint32_t));
                        k += take;
                        left -= take;
                        id = blk[kBlockValues];
                        take = kBlockValues;
                    }
                }
                uint32_t* v = vals.data();
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
                memcpy(p, hourstr[h], 13); p += 13;  // YYYY-MM-DDTHH
                memcpy(p, ":00:00Z ", 8); p += 8;
                const std::string_view name = eps[e];
                memcpy(p, name.data(), name.size()); p += name.size();
                *p++ = ' '; p = put_u64(p, n);
                *p++ = ' '; p = put_u64(p, v[r50 - 1]);
                *p++ = ' '; p = put_u64(p, v[r95 - 1]);
                *p++ = ' '; p = put_u64(p, v[r99 - 1]);
                *p++ = ' '; p = put_u64(p, errs);
                *p++ = ' '; p = put_u64(p, errs * 1000000ull / n);
                *p++ = '\n';
                o.append(line, static_cast<size_t>(p - line));
            }
            std::lock_guard<std::mutex> lock(write_mutex);
            hour_done[static_cast<size_t>(h)] = 1;
            for (; next_write < kHours && hour_done[static_cast<size_t>(next_write)]; ++next_write) {
                std::string& done = hour_out[static_cast<size_t>(next_write)];
                fwrite(done.data(), 1, done.size(), f);
                std::string().swap(done);
            }
        }
    });
    fclose(f);
    return 0;
}
