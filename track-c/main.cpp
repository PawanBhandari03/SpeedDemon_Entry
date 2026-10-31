#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <string>
#include <thread>
#include <vector>
#include <fcntl.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
namespace {
constexpr uint8_t kNoGroup = 0xFF;
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
inline uint64_t hash_bytes_padded(const char* p, size_t n) {
    uint64_t h = 0x9E3779B97F4A7C15ull ^ n;
    while (n >= 8) {
        uint64_t w;
        memcpy(&w, p, 8);
        h = (h ^ w) * 0xff51afd7ed558ccdull;
        h ^= h >> 32;
        p += 8;
        n -= 8;
    }
    uint64_t w;
    memcpy(&w, p, 8);
    w &= (n == 0) ? 0 : (~uint64_t(0) >> ((8 - n) * 8));
    h = (h ^ w) * 0xc4ceb9fe1a85ec53ull;
    return h ^ (h >> 29);
}
inline const char* find_quote(const char* p, const char* end) {
    const __m256i q = _mm256_set1_epi8('"');
    while (p + 32 <= end) {
        const unsigned m = static_cast<unsigned>(
            _mm256_movemask_epi8(_mm256_cmpeq_epi8(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p)), q)));
        if (m) return p + __builtin_ctz(m);
        p += 32;
    }
    return static_cast<const char*>(memchr(p, '"', static_cast<size_t>(end - p)));
}
struct Agg {
    uint64_t count = 0, id_sum = 0, name_bytes = 0, note_bytes = 0;
    int64_t sum = 0, min = INT64_MAX, max = INT64_MIN;
};
struct Local {
    std::vector<Agg> agg;
    const std::vector<uint8_t>* region_group = nullptr;  // region id -> group (2 MB table)
};
struct NameSet {
    static constexpr size_t kCap = size_t(1) << 18;      // table slots
    static constexpr size_t kMaxNames = size_t(1) << 17;  // ids; the table is half empty at most
    static constexpr size_t kWords = kMaxNames / 64;
    struct Slot {
        uint64_t h;
        uint32_t id;
        uint32_t pad;
    };
    Slot* slots = static_cast<Slot*>(calloc(kCap, sizeof(Slot)));
    std::atomic<uint32_t> next{0};
    std::atomic<bool> overflow{false};
    uint64_t* bits = nullptr;  // groups x kMaxNames bits
    void init(size_t groups) { bits = static_cast<uint64_t*>(calloc(groups * kWords, sizeof(uint64_t))); }
    uint32_t id_of(uint64_t h) {
        h |= 1;  // 0 marks an empty slot
        for (size_t i = h & (kCap - 1);; i = (i + 1) & (kCap - 1)) {
            std::atomic_ref<uint64_t> key(slots[i].h);
            uint64_t cur = key.load(std::memory_order_relaxed);
            if (cur == 0 && key.compare_exchange_strong(cur, h, std::memory_order_relaxed)) {
                const uint32_t id = next.fetch_add(1, std::memory_order_relaxed);
                std::atomic_ref<uint32_t>(slots[i].id).store(id + 1, std::memory_order_release);
                return id;
            }
            if (cur == h) {  // the key may have been stored just now by another thread: wait for its id
                std::atomic_ref<uint32_t> published(slots[i].id);
                uint32_t v;
                while (!(v = published.load(std::memory_order_acquire))) {}
                return v - 1;
            }
        }
    }
    void prefetch_slot(uint64_t h) const { __builtin_prefetch(&slots[(h | 1) & (kCap - 1)].h); }
    void prefetch_bit(size_t group, uint32_t id) const { __builtin_prefetch(&bits[group * kWords + (id >> 6)]); }
    void set(size_t group, uint32_t id) {
        if (id >= kMaxNames) { overflow.store(true, std::memory_order_relaxed); id = 0; }
        uint64_t& w = bits[group * kWords + (id >> 6)];
        const uint64_t bit = 1ull << (id % 64);
        if (!(std::atomic_ref<uint64_t>(w).load(std::memory_order_relaxed) & bit))
            std::atomic_ref<uint64_t>(w).fetch_or(bit, std::memory_order_relaxed);
    }
    void reset(size_t groups) {
        memset(slots, 0, kCap * sizeof(Slot));
        memset(bits, 0, groups * kWords * sizeof(uint64_t));
        next.store(0);
    }
    uint64_t distinct(size_t group) const {
        uint64_t c = 0;
        for (size_t w = 0; w < kWords; ++w) c += static_cast<uint64_t>(__builtin_popcountll(bits[group * kWords + w]));
        return c;
    }
};
inline uint64_t read_digits(const char*& p) {
    uint64_t w;
    memcpy(&w, p, 8);
    const uint64_t x = w ^ 0x3030303030303030ull;              // digit byte -> 0x00..0x09
    const uint64_t bad = ((x + 0x7676767676767676ull) | x) & 0x8080808080808080ull;
    if (bad && !(bad & 0xFFull)) {                             // 1 to 7 digits, then a non-digit
        const unsigned n = static_cast<unsigned>(__builtin_ctzll(bad)) / 8;
        uint64_t v = (w << ((8 - n) * 8)) & 0x0F0F0F0F0F0F0F0Full;
        v = ((v * 2561) >> 8);
        v = ((v & 0x00FF00FF00FF00FFull) * 6553601) >> 16;
        v = ((v & 0x0000FFFF0000FFFFull) * 42949672960001ull) >> 32;
        p += n;
        return v;
    }
    uint64_t v = 0;
    while (static_cast<unsigned>(*p - '0') < 10) v = v * 10 + static_cast<uint64_t>(*p++ - '0');
    return v;
}
__attribute__((noinline)) const char* read_region_amount_scalar(const char* p, uint64_t& region,
                                                               int64_t& cents) {
    uint64_t r = 0;
    while (static_cast<unsigned>(*p - '0') < 10) r = r * 10 + static_cast<uint64_t>(*p++ - '0');
    ++p;  // comma
    const bool neg = *p == '-';
    p += neg;
    uint64_t whole = 0;
    while (static_cast<unsigned>(*p - '0') < 10) whole = whole * 10 + static_cast<uint64_t>(*p++ - '0');
    ++p;  // dot
    const uint64_t frac = static_cast<uint64_t>(p[0] - '0') * 10 + static_cast<uint64_t>(p[1] - '0');
    p += 2;
    if (*p == '\r') ++p;
    if (*p == '\n') ++p;
    region = r;
    const uint64_t v = whole * 100 + frac;
    cents = neg ? -static_cast<int64_t>(v) : static_cast<int64_t>(v);
    return p;
}
inline const char* read_region_amount(const char* p, const char* end, uint64_t& region, int64_t& cents) {
    if (end - p < 18) return read_region_amount_scalar(p, region, cents);
    const __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
    const __m128i d = _mm_sub_epi8(v, _mm_set1_epi8('0'));
    const __m128i isdig = _mm_cmpeq_epi8(_mm_min_epu8(d, _mm_set1_epi8(9)), d);
    const unsigned dm = static_cast<unsigned>(_mm_movemask_epi8(isdig));
    const unsigned rn = static_cast<unsigned>(__builtin_ctz(~dm));  // region digits, then the comma
    if (rn < 1 || rn > 7) return read_region_amount_scalar(p, region, cents);
    const bool neg = p[rn + 1] == '-';
    const unsigned astart = rn + 1 + static_cast<unsigned>(neg);
    const unsigned wn = static_cast<unsigned>(__builtin_ctz((~dm) >> astart));  // whole digits, then '.'
    const unsigned dot = astart + wn;
    if (wn < 1 || wn > 5 || dot > 14) return read_region_amount_scalar(p, region, cents);
    uint64_t wr = static_cast<uint64_t>(_mm_cvtsi128_si64(v));  // the region's eight bytes
    uint64_t wa;
    memcpy(&wa, p + astart, 8);
    wr <<= (8 - rn) * 8;  // the comma moves out of the word and digits land in the high bytes
    wa <<= (8 - wn) * 8;
    wr &= 0x0F0F0F0F0F0F0F0Full;  // ASCII digit -> value; the zero padding stays zero
    wa &= 0x0F0F0F0F0F0F0F0Full;
    const __m128i packed = _mm_set_epi64x(static_cast<long long>(wr), static_cast<long long>(wa));
    const __m128i w16 = _mm_maddubs_epi16(packed, _mm_set1_epi16(0x010A));     // [10,1] per pair
    const __m128i w32 = _mm_madd_epi16(w16, _mm_set1_epi32(0x00010064));       // [100,1] per dword
    const uint32_t whole = static_cast<uint32_t>(_mm_cvtsi128_si32(w32)) * 10000u +
                           static_cast<uint32_t>(_mm_extract_epi32(w32, 1));
    region = static_cast<uint32_t>(_mm_extract_epi32(w32, 2)) * 10000u +
             static_cast<uint32_t>(_mm_extract_epi32(w32, 3));
    const uint64_t frac = static_cast<uint64_t>(p[dot + 1] - '0') * 10 + static_cast<uint64_t>(p[dot + 2] - '0');
    p += dot + 3;
    if (*p == '\r') ++p;
    if (*p == '\n') ++p;
    const uint64_t v64 = static_cast<uint64_t>(whole) * 100 + frac;
    cents = neg ? -static_cast<int64_t>(v64) : static_cast<int64_t>(v64);
    return p;
}
inline void scan_text(const char*& p, const char* end, Local& L, size_t& len, uint64_t* hash) {
    if (*p != '"') {
        const char* s = p;
        while (*p != ',') ++p;
        len = static_cast<size_t>(p - s);
        if (hash) *hash = hash_bytes_padded(s, len);
        return;
    }
    const char* s = ++p;
    size_t escapes = 0;
    const char* q = s;
    for (;;) {
        q = find_quote(q, end);
        if (q[1] != '"') break;
        ++escapes;
        q += 2;
    }
    p = q + 1;
    const size_t raw = static_cast<size_t>(q - s);
    len = raw - escapes;
    if (hash) *hash = hash_bytes_padded(s, raw);
}
const char* parse_range(const char* p, const char* stop, const char* end, uint8_t unmatched, Local& L,
                        NameSet& names) {
    constexpr int kBatch = 64;
    struct Pending {
        uint64_t id, region, name_hash;
        int64_t cents;
        uint32_t name_len, note_len;
    } pend[kBatch];
    uint64_t batch_hash[kBatch];
    uint32_t batch_id[kBatch];
    uint16_t batch_group[kBatch];
    int batched = 0;
    auto flush = [&] {
        const std::vector<uint8_t>& rg = *L.region_group;
        for (int k = 0; k < batched; ++k)
            if (pend[k].region < rg.size()) __builtin_prefetch(&rg[pend[k].region]);
        for (int k = 0; k < batched; ++k) {
            const Pending& r = pend[k];
            uint8_t g = r.region < rg.size() ? rg[r.region] : kNoGroup;
            g = g == kNoGroup ? unmatched : g;
            Agg& a = L.agg[g];
            ++a.count;
            a.id_sum += r.id;
            a.sum += r.cents;
            a.min = std::min(a.min, r.cents);
            a.max = std::max(a.max, r.cents);
            a.name_bytes += r.name_len;
            a.note_bytes += r.note_len;
            batch_hash[k] = r.name_hash;
            batch_group[k] = g;
        }
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
        uint64_t region;
        int64_t cents;
        p = read_region_amount(p, end, region, cents);
        pend[batched] = {id, region, name_hash, cents, static_cast<uint32_t>(name_len), static_cast<uint32_t>(note_len)};
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
    std::string data = "/data", outdir = "/out";
    for (int i = 1; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "--data")) data = argv[i + 1];
        else if (!strcmp(argv[i], "--out")) outdir = argv[i + 1];
    }
    const Mapped reg = map_file(data + "/regions.csv");
    if (!reg.p) return 1;
    std::vector<std::pair<uint32_t, uint16_t>> regions;  // (region_id, code index)
    std::vector<uint8_t> code_group(26 * 26 * 26, kNoGroup);
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
    std::vector<uint16_t> codes;  // group -> code index (the code itself is 26^3, so halfwords)
    for (uint16_t c = 0; c < code_group.size(); ++c)
        if (code_group[c] != kNoGroup) { code_group[c] = static_cast<uint8_t>(codes.size()); codes.push_back(c); }
    if (codes.size() >= 254) return 1;  // the spec fixes 240 countries; more would not fit a byte
    const uint8_t unmatched = static_cast<uint8_t>(codes.size());
    std::vector<uint8_t> region_group(static_cast<size_t>(max_id) + 1, kNoGroup);
    for (auto& r : regions) region_group[r.first] = code_group[r.second];
    std::vector<std::pair<uint32_t, uint16_t>>().swap(regions);
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
        for (auto& L : locals) { L.agg.assign(groups, Agg()); L.region_group = &region_group; }
        std::vector<const char*> chunk(nt + 1);
        for (int i = 0; i <= nt; ++i) chunk[i] = base + rec.n * static_cast<size_t>(i) / static_cast<size_t>(nt);
        const char* const hdr = static_cast<const char*>(memchr(base, '\n', rec.n)) + 1;  // skip BOM + header
        auto parse_all = [&](const std::vector<const char*>& starts) {
            std::vector<const char*> ends(nt);
            std::vector<std::thread> th;
            for (int i = 0; i < nt; ++i)
                th.emplace_back([&, i] {
                    ends[i] = starts[i] < starts[i + 1]
                                  ? parse_range(starts[i], starts[i + 1], end, unmatched, locals[i], names)
                                  : starts[i];
                });
            for (auto& t : th) t.join();
            for (int i = 0; i < nt; ++i)
                if (ends[i] != starts[i + 1]) return false;
            return true;
        };
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
