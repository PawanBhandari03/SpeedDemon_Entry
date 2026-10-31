// Speed Demon 2026, Track C (Join): records.csv + regions.csv -> aggregates.txt
//
// Plan: a quoted field may contain a newline, so the file cannot be cut at any '\n'. Instead
// every chunk first counts its '"' bytes. The running parity of those counts says whether a
// chunk starts inside a quoted field, so each thread can find the first real record start in
// its chunk (a '\n' seen with an even number of quotes before it) and then parse records with a
// proper CSV state machine until it reaches the next thread's first record. Each record is
// joined to its country through a flat region_id -> group table. Per-group sums are kept per
// thread and added at the end. Distinct names are counted with one shared table of name
// hashes and a per-group bitset over its slots.
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include <immintrin.h>
#include <fcntl.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

constexpr uint16_t kNoGroup = 0xFFFF;

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

int thread_count() {
    cpu_set_t set;
    if (sched_getaffinity(0, sizeof set, &set) == 0) {
        const int n = CPU_COUNT(&set);
        if (n > 0) return n;
    }
    return static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
}

uint64_t hash_bytes(const char* p, size_t n) {
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

struct Agg {
    uint64_t count = 0, id_sum = 0, name_bytes = 0, note_bytes = 0;
    int64_t sum = 0, min = INT64_MAX, max = INT64_MIN;
};

struct Local {
    std::vector<Agg> agg;
    std::string scratch;
};

// Distinct names. The spec draws names from a pool of 100,000 values, so every distinct name
// gets a dense id from one table shared by all threads (compare-and-swap on the key), and each
// group keeps a bitset over the ids it has seen. Memory is a few MB instead of one hash per
// record, and a group's bitset is small enough to stay in cache.
struct NameSet {
    static constexpr size_t kCap = size_t(1) << 18;      // table slots
    static constexpr size_t kMaxNames = size_t(1) << 17;  // ids; the table is half empty at most
    static constexpr size_t kWords = kMaxNames / 64;
    uint64_t* keys = static_cast<uint64_t*>(calloc(kCap, sizeof(uint64_t)));
    uint32_t* ids = static_cast<uint32_t*>(calloc(kCap, sizeof(uint32_t)));  // id + 1 once published
    std::atomic<uint32_t> next{0};
    std::atomic<bool> overflow{false};
    uint64_t* bits = nullptr;  // groups x kMaxNames bits

    void init(size_t groups) { bits = static_cast<uint64_t*>(calloc(groups * kWords, sizeof(uint64_t))); }

    uint32_t id_of(uint64_t h) {
        h |= 1;  // 0 marks an empty slot
        for (size_t i = h & (kCap - 1);; i = (i + 1) & (kCap - 1)) {
            std::atomic_ref<uint64_t> key(keys[i]);
            uint64_t cur = key.load(std::memory_order_relaxed);
            if (cur == 0 && key.compare_exchange_strong(cur, h, std::memory_order_relaxed)) {
                const uint32_t id = next.fetch_add(1, std::memory_order_relaxed);
                std::atomic_ref<uint32_t>(ids[i]).store(id + 1, std::memory_order_release);
                return id;
            }
            if (cur == h) {  // the key may have been stored just now by another thread: wait for its id
                std::atomic_ref<uint32_t> published(ids[i]);
                uint32_t v;
                while (!(v = published.load(std::memory_order_acquire))) {}
                return v - 1;
            }
        }
    }
    void prefetch_slot(uint64_t h) const { __builtin_prefetch(&keys[(h | 1) & (kCap - 1)]); }
    void prefetch_bit(size_t group, uint32_t id) const { __builtin_prefetch(&bits[group * kWords + (id < kMaxNames ? id : 0) / 64]); }
    void set(size_t group, uint32_t id) {
        if (id >= kMaxNames) { overflow.store(true, std::memory_order_relaxed); id = 0; }
        uint64_t& w = bits[group * kWords + id / 64];
        const uint64_t bit = 1ull << (id % 64);
        if (!(std::atomic_ref<uint64_t>(w).load(std::memory_order_relaxed) & bit))
            std::atomic_ref<uint64_t>(w).fetch_or(bit, std::memory_order_relaxed);
    }
    void reset(size_t groups) {
        memset(keys, 0, kCap * sizeof(uint64_t));
        memset(ids, 0, kCap * sizeof(uint32_t));
        memset(bits, 0, groups * kWords * sizeof(uint64_t));
        next.store(0);
    }
    uint64_t distinct(size_t group) const {
        uint64_t c = 0;
        for (size_t w = 0; w < kWords; ++w) c += static_cast<uint64_t>(__builtin_popcountll(bits[group * kWords + w]));
        return c;
    }
};

// Reads the digits at p and moves p to the first non-digit.
inline uint64_t read_digits(const char*& p) {
    uint64_t v = 0;
    while (static_cast<unsigned>(*p - '0') < 10) v = v * 10 + static_cast<uint64_t>(*p++ - '0');
    return v;
}

// Scans one text field starting at p. Sets the unescaped length (and the hash of the unescaped
// value when hash is not null) and moves p past the field, to the ',' after it. A quoted field
// ends at the first '"' that is not part of a "" pair.
inline void scan_text(const char*& p, const char* end, Local& L, size_t& len, uint64_t* hash) {
    if (*p != '"') {
        const char* s = p;
        while (*p != ',') ++p;
        len = static_cast<size_t>(p - s);
        if (hash) *hash = hash_bytes(s, len);
        return;
    }
    const char* s = ++p;
    size_t escapes = 0;
    const char* q = s;
    for (;;) {
        q = static_cast<const char*>(memchr(q, '"', static_cast<size_t>(end - q)));
        if (q[1] != '"') break;
        ++escapes;
        q += 2;
    }
    p = q + 1;
    const size_t raw = static_cast<size_t>(q - s);
    len = raw - escapes;
    if (!hash) return;
    if (!escapes) {
        *hash = hash_bytes(s, raw);
        return;
    }
    L.scratch.clear();
    for (const char* x = s; x < q; ++x) {
        L.scratch.push_back(*x);
        if (*x == '"') ++x;  // second quote of the pair
    }
    *hash = hash_bytes(L.scratch.data(), L.scratch.size());
}

// One record's parsed fields.
struct Fields {
    uint64_t id = 0, name_hash = 0, region = 0;
    size_t name_len = 0, note_len = 0;
    int64_t cents = 0;
};

inline uint64_t load8(const char* p) {
    uint64_t w;
    memcpy(&w, p, 8);
    return w;
}

// Value of the len (1..8) digits at s, or UINT64_MAX when a byte is not a digit. Reads the 8
// bytes that end with the digits, clears the ones in front, and turns the 8 digits (leading
// zeros included) into a number with three multiplications instead of a loop with a data-
// dependent exit.
inline uint64_t parse_digits(const char* s, size_t len) {
    const uint64_t keep = ~uint64_t(0) << ((8 - len) * 8);
    uint64_t v = (load8(s + len - 8) & keep) | (0x3030303030303030ull & ~keep);
    if (((v + 0x4646464646464646ull) | (v - 0x3030303030303030ull)) & 0x8080808080808080ull) return UINT64_MAX;
    v = ((v & 0x0F0F0F0F0F0F0F0Full) * 2561) >> 8;
    v = ((v & 0x00FF00FF00FF00FFull) * 6553601) >> 16;
    return ((v & 0x0000FFFF0000FFFFull) * 42949672960001ull) >> 32;
}

// Number of '"' bytes among the n bytes at s. May read up to 7 bytes past the end.
inline unsigned count_quotes(const char* s, size_t n) {
    unsigned c = 0;
    for (; n; s += 8, n -= std::min<size_t>(n, 8)) {
        uint64_t x = load8(s) ^ 0x2222222222222222ull;
        if (n < 8) x |= ~uint64_t(0) << (n * 8);  // bytes past the field count as "not a quote"
        c += static_cast<unsigned>(__builtin_popcountll(~(((x & 0x7F7F7F7F7F7F7F7Full) + 0x7F7F7F7F7F7F7F7Full) | x | 0x7F7F7F7F7F7F7F7Full)));
    }
    return c;
}

// Offsets (from p) of every ',' and LF outside quotes in the n bytes at p, n a multiple of 64.
// p must be outside quotes. A bit per byte says whether it is inside quotes: the running xor of the
// quote bits, computed with a carry-less multiplication by all ones. A doubled quote inside a
// field toggles twice, so it needs no special case. Returns the number of offsets.
inline size_t index_structure(const char* p, size_t n, uint32_t* out) {
    const __m256i quote = _mm256_set1_epi8('"'), comma = _mm256_set1_epi8(','), lf = _mm256_set1_epi8('\n');
    const __m128i ones = _mm_set1_epi8(-1);
    uint64_t inside_at_start = 0;  // all ones when the block starts inside quotes
    size_t k = 0;
    for (size_t b = 0; b < n; b += 64) {
        const __m256i lo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + b));
        const __m256i hi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + b + 32));
        auto mask = [&](__m256i c) {
            return static_cast<uint64_t>(static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(lo, c)))) |
                   static_cast<uint64_t>(static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(hi, c)))) << 32;
        };
        const uint64_t q = mask(quote);
        const uint64_t inside =
            static_cast<uint64_t>(_mm_cvtsi128_si64(_mm_clmulepi64_si128(_mm_cvtsi64_si128(static_cast<long long>(q)), ones, 0))) ^ inside_at_start;
        inside_at_start = static_cast<uint64_t>(static_cast<int64_t>(inside) >> 63);
        uint64_t sep = (mask(comma) | mask(lf)) & ~inside;
        while (sep) {
            out[k++] = static_cast<uint32_t>(b + static_cast<size_t>(__builtin_ctzll(sep)));
            sep &= sep - 1;
        }
    }
    return k;
}

// Reads one record byte by byte. Handles whatever the fast path does not: a byte order mark,
// numbers of an unexpected shape, records that do not fit a window, the end of the file.
const char* scalar_record(const char* p, const char* end, Local& L, Fields& f) {
    if (static_cast<uint8_t>(p[0]) == 0xEF && static_cast<uint8_t>(p[1]) == 0xBB && static_cast<uint8_t>(p[2]) == 0xBF)
        p += 3;  // byte order mark at the start of a record
    f.id = read_digits(p);
    ++p;
    scan_text(p, end, L, f.name_len, &f.name_hash);
    ++p;
    scan_text(p, end, L, f.note_len, nullptr);  // only the note's length matters
    ++p;
    f.region = read_digits(p);
    ++p;
    const bool neg = *p == '-';
    p += neg;
    const uint64_t whole = read_digits(p);
    const uint64_t frac = static_cast<uint64_t>(p[1] - '0') * 10 + static_cast<uint64_t>(p[2] - '0');
    p += 3;
    if (*p == '\r') ++p;
    if (p < end && *p == '\n') ++p;
    f.cents = neg ? -static_cast<int64_t>(whole * 100 + frac) : static_cast<int64_t>(whole * 100 + frac);
    return p;
}

// Length of a text field's value, and the hash of it when hash is not null. [b, e) is the field
// as written (quotes included). False when the field is not shaped as expected.
inline bool text_field(const char* b, const char* e, Local& L, size_t& len, uint64_t* hash) {
    if (e == b || *b != '"') {
        len = static_cast<size_t>(e - b);
        if (hash) *hash = hash_bytes(b, len);
        return true;
    }
    if (e - b < 2 || e[-1] != '"') return false;
    const char* inner = b + 1;
    const size_t raw = static_cast<size_t>(e - 1 - inner);
    const unsigned quotes = count_quotes(inner, raw);
    if (quotes & 1) return false;
    len = raw - quotes / 2;
    if (hash) {
        if (!quotes) {
            *hash = hash_bytes(inner, raw);
        } else {
            L.scratch.clear();
            for (const char* x = inner; x < e - 1; ++x) {
                L.scratch.push_back(*x);
                if (*x == '"') ++x;  // second quote of the pair
            }
            *hash = hash_bytes(L.scratch.data(), L.scratch.size());
        }
    }
    return true;
}

// Reads one record whose five separators (four commas, then its LF) are at offsets s[0..4] from
// base. Returns the start of the next record, or nullptr when anything is not as expected, in
// which case the caller parses the record byte by byte.
inline const char* fast_record(const char* base, const uint32_t* s, const char* rec, Local& L, Fields& f) {
    const char *c0 = base + s[0], *c1 = base + s[1], *c2 = base + s[2], *c3 = base + s[3], *lf = base + s[4];
    if (*c0 != ',' || *c1 != ',' || *c2 != ',' || *c3 != ',' || *lf != '\n') return nullptr;
    const size_t id_len = static_cast<size_t>(c0 - rec), region_len = static_cast<size_t>(c3 - c2 - 1);
    if (id_len < 1 || id_len > 8 || region_len < 1 || region_len > 8) return nullptr;
    f.id = parse_digits(rec, id_len);
    f.region = parse_digits(c2 + 1, region_len);
    if (f.id == UINT64_MAX || f.region == UINT64_MAX) return nullptr;
    if (!text_field(c0 + 1, c1, L, f.name_len, &f.name_hash) || !text_field(c1 + 1, c2, L, f.note_len, nullptr)) return nullptr;
    // amount: [-]digits '.' two digits, then an optional CR
    const char* a = c3 + 1;
    const char* ae = lf;
    if (ae > a && ae[-1] == '\r') --ae;
    if (ae - a < 4 || ae[-3] != '.') return nullptr;
    const bool neg = *a == '-';
    a += neg;
    const size_t whole_len = static_cast<size_t>(ae - 3 - a);
    if (whole_len < 1 || whole_len > 8) return nullptr;
    const uint64_t whole = parse_digits(a, whole_len);
    const unsigned d1 = static_cast<unsigned>(ae[-2] - '0'), d2 = static_cast<unsigned>(ae[-1] - '0');
    if (whole == UINT64_MAX || d1 > 9 || d2 > 9) return nullptr;
    const int64_t cents = static_cast<int64_t>(whole * 100 + d1 * 10 + d2);
    f.cents = neg ? -cents : cents;
    return lf + 1;
}

constexpr size_t kWindow = 16384;  // bytes of the file whose structure is indexed at a time

const char* parse_range(const char* p, const char* stop, const char* end, const std::vector<uint16_t>& region_group,
                        uint16_t unmatched, Local& L, NameSet& names) {
    // The names of 16 records are added together: the table slots and the bitset words are cache
    // misses, and issuing all the prefetches first lets them overlap instead of stalling one by one.
    constexpr int kBatch = 16;
    uint64_t batch_hash[kBatch];
    uint32_t batch_id[kBatch];
    uint16_t batch_group[kBatch];
    int batched = 0;
    auto flush = [&] {
        for (int k = 0; k < batched; ++k) names.prefetch_slot(batch_hash[k]);
        for (int k = 0; k < batched; ++k) {
            batch_id[k] = names.id_of(batch_hash[k]);
            names.prefetch_bit(batch_group[k], batch_id[k]);
        }
        for (int k = 0; k < batched; ++k) names.set(batch_group[k], batch_id[k]);
        batched = 0;
    };
    auto commit = [&](const Fields& f) {
        uint16_t g = f.region < region_group.size() ? region_group[f.region] : kNoGroup;
        if (g == kNoGroup) g = unmatched;
        Agg& a = L.agg[g];
        ++a.count;
        a.id_sum += f.id;
        a.sum += f.cents;
        a.min = std::min(a.min, f.cents);
        a.max = std::max(a.max, f.cents);
        a.name_bytes += f.name_len;
        a.note_bytes += f.note_len;
        batch_hash[batched] = f.name_hash;
        batch_group[batched] = g;
        if (++batched == kBatch) flush();
    };
    std::vector<uint32_t> index(kWindow);
    Fields f;
    while (p < stop) {
        // Fast path: find the separators of a whole window at once, then read the records in it.
        // The window ends 64 bytes before the end of the file, so no read goes past it.
        size_t done = 0;
        if (static_cast<size_t>(end - p) >= kWindow + 64) {
            const size_t m = index_structure(p, kWindow, index.data());
            const char* rec = p;
            for (size_t si = 0; rec < stop && si + 5 <= m; si += 5) {
                const char* next = fast_record(p, index.data() + si, rec, L, f);
                if (!next) break;
                commit(f);
                rec = next;
                ++done;
            }
            p = rec;
        }
        if (!done && p < stop) {  // a record the fast path did not take, or the end of the file
            p = scalar_record(p, end, L, f);
            commit(f);
        }
    }
    flush();
    return p;
}

void put_cents(std::string& out, int64_t c) {
    if (c < 0) { out += '-'; c = -c; }
    out += std::to_string(c / 100);
    out += '.';
    out += static_cast<char>('0' + (c % 100) / 10);
    out += static_cast<char>('0' + c % 10);
}

}  // namespace

int main(int argc, char** argv) {
    std::string data = "/data", outdir = "/out";
    for (int i = 1; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "--data")) data = argv[i + 1];
        else if (!strcmp(argv[i], "--out")) outdir = argv[i + 1];
    }

    // regions.csv: "region_id,country" header, then one "id,CCC" line per region.
    const Mapped reg = map_file(data + "/regions.csv");
    if (!reg.p) return 1;
    // A code is three letters A-Z, so it indexes a 26^3 table directly, and that index order is
    // the bytewise order of the codes.
    std::vector<std::pair<uint32_t, uint16_t>> regions;  // (region_id, code index)
    std::vector<uint16_t> code_group(26 * 26 * 26, kNoGroup);
    uint32_t max_id = 0;
    {
        const char* p = static_cast<const char*>(memchr(reg.p, '\n', reg.n)) + 1;
        const char* e = reg.p + reg.n;
        while (p < e) {
            const uint32_t id = static_cast<uint32_t>(read_digits(p));
            ++p;
            const uint16_t code = static_cast<uint16_t>((p[0] - 'A') * 676 + (p[1] - 'A') * 26 + (p[2] - 'A'));
            p += 4;  // code and LF
            regions.emplace_back(id, code);
            code_group[code] = 0;
            max_id = std::max(max_id, id);
        }
    }
    std::vector<uint16_t> codes;  // group -> code index
    for (uint16_t c = 0; c < code_group.size(); ++c)
        if (code_group[c] != kNoGroup) { code_group[c] = static_cast<uint16_t>(codes.size()); codes.push_back(c); }
    const uint16_t unmatched = static_cast<uint16_t>(codes.size());
    std::vector<uint16_t> region_group(static_cast<size_t>(max_id) + 1, kNoGroup);
    for (auto& r : regions) region_group[r.first] = code_group[r.second];
    const size_t groups = codes.size() + 1;

    const Mapped rec = map_file(data + "/records.csv");
    std::vector<Local> locals;
    NameSet names;
    names.init(groups);
    int nt = 1;
    if (rec.p) {
        const char* const base = rec.p;
        const char* const end = base + rec.n;
        nt = rec.n < (1u << 20) ? 1 : thread_count();
        locals.resize(nt);
        for (auto& L : locals) L.agg.assign(groups, Agg());

        std::vector<const char*> chunk(nt + 1);
        for (int i = 0; i <= nt; ++i) chunk[i] = base + rec.n * static_cast<size_t>(i) / static_cast<size_t>(nt);
        const char* const hdr = static_cast<const char*>(memchr(base, '\n', rec.n)) + 1;  // skip BOM + header

        // Parses every thread's records from the given first-record starts. Returns true when
        // each thread stopped exactly where the next one started, i.e. the starts were right.
        auto parse_all = [&](const std::vector<const char*>& starts) {
            std::vector<const char*> ends(nt);
            std::vector<std::thread> th;
            for (int i = 0; i < nt; ++i)
                th.emplace_back([&, i] {
                    ends[i] = starts[i] < starts[i + 1]
                                  ? parse_range(starts[i], starts[i + 1], end, region_group, unmatched, locals[i], names)
                                  : starts[i];
                });
            for (auto& t : th) t.join();
            for (int i = 0; i < nt; ++i)
                if (ends[i] != starts[i + 1]) return false;
            return true;
        };

        // Fast start finder. A newline inside a quoted note is always followed by a lowercase
        // letter or a quote, and a record really starts after a newline followed by a digit (an
        // id) or a byte order mark (spec 2.2: ids are [1-9][0-9]*, notes are lowercase words).
        std::vector<const char*> starts(nt + 1, end);
        starts[0] = hdr;
        for (int i = 1; i < nt; ++i)
            for (const char* x = std::max(chunk[i], hdr); x + 1 < end; ++x)
                if (*x == '\n' && (static_cast<unsigned>(static_cast<uint8_t>(x[1]) - '0') < 10 || static_cast<uint8_t>(x[1]) == 0xEF)) {
                    starts[i] = x + 1;
                    break;
                }
        bool ok = parse_all(starts);

        if (!ok) {
            // Fallback, exact for any input: count the '"' bytes of every chunk, so the parity
            // says whether a chunk starts inside quotes, and start at a '\n' with an even count.
            for (auto& L : locals) L.agg.assign(groups, Agg());
            names.reset(groups);
            std::vector<uint8_t> parity(nt, 0);
            {
                std::vector<std::thread> th;
                for (int i = 0; i < nt; ++i)
                    th.emplace_back([&, i] {
                        size_t c = 0;
                        for (const char* x = chunk[i]; x < chunk[i + 1]; ++x) c += (*x == '"');
                        parity[i] = c & 1;
                    });
                for (auto& t : th) t.join();
            }
            std::fill(starts.begin(), starts.end(), end);
            starts[0] = hdr;
            uint8_t inside = parity[0];
            for (int i = 1; i < nt; ++i) {
                uint8_t q = inside;
                for (const char* x = chunk[i]; x < end; ++x) {
                    if (*x == '"') q ^= 1;
                    else if (*x == '\n' && !q) { starts[i] = x + 1; break; }
                }
                inside ^= parity[i];
            }
            parse_all(starts);
        }
    }

    if (names.overflow.load()) return 1;  // more distinct names than the spec allows

    // Add up the per-thread sums.
    std::vector<Agg> total(groups);
    for (auto& L : locals)
        for (size_t g = 0; g < groups; ++g) {
            const Agg& a = L.agg[g];
            Agg& t = total[g];
            t.count += a.count;
            t.id_sum += a.id_sum;
            t.sum += a.sum;
            t.min = std::min(t.min, a.min);
            t.max = std::max(t.max, a.max);
            t.name_bytes += a.name_bytes;
            t.note_bytes += a.note_bytes;
        }

    std::string out;
    for (size_t g = 0; g < groups; ++g) {
        const Agg& a = total[g];
        if (!a.count) continue;
        if (g == unmatched) out += "_UNMATCHED";
        else {
            out += static_cast<char>('A' + codes[g] / 676);
            out += static_cast<char>('A' + codes[g] / 26 % 26);
            out += static_cast<char>('A' + codes[g] % 26);
        }
        out += ' ' + std::to_string(a.count) + ' ' + std::to_string(a.id_sum) + ' ';
        put_cents(out, a.sum);
        out += ' ';
        put_cents(out, a.min);
        out += ' ';
        put_cents(out, a.max);
        out += ' ' + std::to_string(names.distinct(g)) + ' ' + std::to_string(a.name_bytes) + ' ' + std::to_string(a.note_bytes) + '\n';
    }
    FILE* f = fopen((outdir + "/aggregates.txt").c_str(), "wb");
    if (!f) return 1;
    fwrite(out.data(), 1, out.size(), f);
    fclose(f);
    return 0;
}
