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

#include <fcntl.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/prctl.h>
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
    while (p < stop) {
        if (static_cast<uint8_t>(p[0]) == 0xEF && static_cast<uint8_t>(p[1]) == 0xBB && static_cast<uint8_t>(p[2]) == 0xBF)
            p += 3;  // byte order mark at the start of a record
        const uint64_t id = read_digits(p);
        ++p;
        size_t name_len, note_len;
        uint64_t name_hash;
        scan_text(p, end, L, name_len, &name_hash);
        ++p;
        scan_text(p, end, L, note_len, nullptr);  // only the note's length matters
        ++p;
        const uint64_t region = read_digits(p);
        ++p;
        const bool neg = *p == '-';
        p += neg;
        const uint64_t whole = read_digits(p);
        const uint64_t frac = static_cast<uint64_t>(p[1] - '0') * 10 + static_cast<uint64_t>(p[2] - '0');
        p += 3;
        if (*p == '\r') ++p;
        if (p < end && *p == '\n') ++p;
        const int64_t cents = neg ? -static_cast<int64_t>(whole * 100 + frac) : static_cast<int64_t>(whole * 100 + frac);

        uint16_t g = region < region_group.size() ? region_group[region] : kNoGroup;
        if (g == kNoGroup) g = unmatched;
        Agg& a = L.agg[g];
        ++a.count;
        a.id_sum += id;
        a.sum += cents;
        a.min = std::min(a.min, cents);
        a.max = std::max(a.max, cents);
        a.name_bytes += name_len;
        a.note_bytes += note_len;
        batch_hash[batched] = name_hash;
        batch_group[batched] = g;
        if (++batched == kBatch) flush();
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
    // Huge pages would commit 2 MB for every sparsely touched page of the big arrays; ask for 4 KB pages.
    prctl(PR_SET_THP_DISABLE, 1, 0, 0, 0);
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
