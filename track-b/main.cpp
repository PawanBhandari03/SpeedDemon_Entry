// Speed Demon 2026, Track B (Index): docs.txt + queries.txt -> results.txt
//
// Plan: read the queries first and index only the words they use. One parallel pass over the
// corpus then finds those words (a hash lookup per token) and records, per word, the sorted list
// of documents containing it. Words in many documents also get a bitset, so AND/OR/NOT on them
// are word-wise operations. A BOOL query is evaluated left to right on a bitset of document
// ids. A PHRASE query takes the documents that contain all its words (rarest word first) and
// checks each one by tokenizing that document again and looking for the exact sequence.
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
#include <unistd.h>

namespace {

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

// Token bytes: a-z and 0-9 map to themselves, A-Z fold to lower case, everything else is a separator (0).
uint8_t g_class[256];

void init_class() {
    for (int c = 0; c < 256; ++c) {
        if (c >= 'a' && c <= 'z') g_class[c] = static_cast<uint8_t>(c);
        else if (c >= 'A' && c <= 'Z') g_class[c] = static_cast<uint8_t>(c + 32);
        else if (c >= '0' && c <= '9') g_class[c] = static_cast<uint8_t>(c);
    }
}

constexpr uint64_t kFnvBasis = 0xcbf29ce484222325ull;
constexpr uint64_t kFnvPrime = 0x100000001b3ull;

inline uint64_t finish_key(uint64_t h, uint32_t len) {
    h ^= static_cast<uint64_t>(len) * 0x9E3779B97F4A7C15ull;
    h ^= h >> 29;
    h *= 0xbf58476d1ce4e5b9ull;
    h ^= h >> 32;
    return h ? h : 1;
}

// Calls f(key, start) for every token of one document line starting at p (the id, a TAB, then
// the text); start points at the token's first byte. Returns a pointer just past the line's LF.
template <class F>
inline const char* tokenize_line(const char* p, const char* end, F&& f) {
    while (p < end && *p != '\t') ++p;
    ++p;
    uint64_t h = kFnvBasis;
    uint32_t len = 0;
    for (; p < end; ++p) {
        const uint8_t c = g_class[static_cast<uint8_t>(*p)];
        if (c) {
            h = (h ^ c) * kFnvPrime;
            ++len;
        } else {
            if (len) { f(finish_key(h, len), p - len); h = kFnvBasis; len = 0; }
            if (*p == '\n') return p + 1;
        }
    }
    if (len) f(finish_key(h, len), p - len);
    return end;
}

// True when the k keys in want are k consecutive tokens of the text [begin, ...) and the m-th
// of them starts at `at`: walks back m tokens from `at`, then hashes k tokens forward.
inline bool window_match(const char* begin, const char* at, const uint64_t* want, uint32_t k, uint32_t m) {
    const char* p = at;
    for (uint32_t r = 0; r < m; ++r) {
        const char* q = p - 1;
        while (q >= begin && !g_class[static_cast<uint8_t>(*q)]) --q;  // separators before the token
        if (q < begin) return false;
        while (q >= begin && g_class[static_cast<uint8_t>(*q)]) --q;   // the token itself
        p = q + 1;
    }
    for (uint32_t i = 0; i < k; ++i) {
        while (!g_class[static_cast<uint8_t>(*p)]) {
            if (*p == '\n') return false;
            ++p;
        }
        uint64_t h = kFnvBasis;
        uint32_t len = 0;
        for (uint8_t c; (c = g_class[static_cast<uint8_t>(*p)]); ++p, ++len) h = (h ^ c) * kFnvPrime;
        if (finish_key(h, len) != want[i]) return false;
    }
    return true;
}

// Open-addressing table from token key to term id. Most corpus tokens are probed, so the probe
// walks a dense array of keys alone (8 bytes a slot, a quarter of it in each cache line pair)
// and reads the id array only on a hit.
struct Dict {
    std::vector<uint64_t> keys;
    std::vector<uint32_t> ids;
    size_t mask = 0;

    void init(size_t terms) {
        size_t cap = 16;
        while (cap < terms + terms / 2) cap <<= 1;
        keys.assign(cap, 0);
        ids.assign(cap, 0);
        mask = cap - 1;
    }
    void insert(uint64_t key, uint32_t id) {
        size_t i = key & mask;
        while (keys[i]) i = (i + 1) & mask;
        keys[i] = key;
        ids[i] = id;
    }
    // Returns the term id, or UINT32_MAX when the key is not a query term.
    inline uint32_t find(uint64_t key) const {
        for (size_t i = key & mask;; i = (i + 1) & mask) {
            if (keys[i] == key) return ids[i];
            if (!keys[i]) return UINT32_MAX;
        }
    }
};

struct Worker {
    std::vector<uint64_t> records;  // (term << 32) | doc, one per occurrence of a rare term, in file order
    std::vector<uint32_t> offs;     // file offset of each of those occurrences
    std::vector<uint32_t> count;    // occurrences per term seen by this worker
    std::vector<uint32_t> last;     // last document seen for each dense term, to set its bit once
};

struct TermSet {
    const uint32_t* list;
    uint32_t n;
    const uint64_t* bits;  // null for a rare term
};

struct Query {
    bool phrase;
    uint32_t first, count;  // slice of qterm / qop
};

constexpr uint32_t kDenseDf = 4096;  // a term in more documents than this also gets a bitset

inline char* put_u64(char* o, uint64_t v) {
    char tmp[20];
    int n = 0;
    do { tmp[n++] = static_cast<char>('0' + v % 10); v /= 10; } while (v);
    while (n) *o++ = tmp[--n];
    return o;
}

}  // namespace

int main(int argc, char** argv) {
    std::string data = "/data", outdir = "/out";
    for (int i = 1; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "--data")) data = argv[i + 1];
        else if (!strcmp(argv[i], "--out")) outdir = argv[i + 1];
    }
    init_class();

    const Mapped qf = map_file(data + "/queries.txt");
    const Mapped docs = map_file(data + "/docs.txt");
    if (!qf.p || !docs.p) return 1;

    // ---- Queries: parse into flat arrays of term keys and operators.
    std::vector<Query> queries;
    std::vector<uint64_t> qterm;  // key first, replaced by the term id below
    std::vector<uint8_t> qop;     // operator in front of this term: 0 AND, 1 OR, 2 NOT (unused for the first)
    {
        const char* p = qf.p;
        const char* const e = qf.p + qf.n;
        while (p < e) {
            Query q;
            q.phrase = p[0] == 'P';
            q.first = static_cast<uint32_t>(qterm.size());
            p += q.phrase ? 8 : 5;  // "PHRASE \"" or "BOOL "
            bool want_term = true;
            uint8_t pending = 0;
            while (p < e && *p != '\n') {
                if (*p == ' ') { ++p; continue; }
                if (*p == '"') { ++p; continue; }
                const char* s = p;
                uint64_t h = kFnvBasis;
                while (p < e && *p != ' ' && *p != '\n' && *p != '"') { h = (h ^ static_cast<uint8_t>(*p)) * kFnvPrime; ++p; }
                const uint32_t len = static_cast<uint32_t>(p - s);
                if (q.phrase || want_term) {
                    qterm.push_back(finish_key(h, len));
                    qop.push_back(pending);
                    want_term = false;
                } else {
                    pending = len == 3 && s[0] == 'A' ? 0 : (s[0] == 'O' ? 1 : 2);
                    want_term = true;
                }
            }
            ++p;
            q.count = static_cast<uint32_t>(qterm.size()) - q.first;
            queries.push_back(q);
        }
    }
    std::vector<uint64_t> keys(qterm);
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    const uint32_t Q = static_cast<uint32_t>(keys.size());
    Dict dict;
    dict.init(Q);
    for (uint32_t i = 0; i < Q; ++i) dict.insert(keys[i], i);
    std::vector<uint32_t> qid(qterm.size());
    for (size_t i = 0; i < qterm.size(); ++i) qid[i] = static_cast<uint32_t>(std::lower_bound(keys.begin(), keys.end(), qterm[i]) - keys.begin());

    // ---- Corpus: number of documents from the last line's id.
    const char* const dbase = docs.p;
    const char* const dend = docs.p + docs.n;
    uint32_t N;
    {
        const char* last = dend - 1;                      // the final LF
        while (last > dbase && last[-1] != '\n') --last;  // start of the last line
        uint32_t id = 0;
        while (*last != '\t') id = id * 10 + static_cast<uint32_t>(*last++ - '0');
        N = id + 1;
    }
    const size_t W = (N + 63) / 64;
    std::vector<uint32_t> doc_off(N);

    // ---- Chunks of whole lines, one per thread. Every chunk after the first starts at a
    // document whose id is a multiple of 64, so two threads never write the same bitset word.
    const int nt = docs.n < (1u << 20) ? 1 : thread_count();
    std::vector<const char*> cut(nt + 1);
    cut[0] = dbase;
    cut[nt] = dend;
    for (int i = 1; i < nt; ++i) {
        const char* p = dbase + docs.n * static_cast<size_t>(i) / static_cast<size_t>(nt);
        const char* nl = static_cast<const char*>(memchr(p - 1, '\n', static_cast<size_t>(dend - (p - 1))));
        p = nl ? nl + 1 : dend;
        while (p < dend) {
            uint32_t id = 0;
            for (const char* x = p; *x != '\t'; ++x) id = id * 10 + static_cast<uint32_t>(*x - '0');
            if (id % 64 == 0) break;
            p = static_cast<const char*>(memchr(p, '\n', static_cast<size_t>(dend - p))) + 1;
        }
        cut[i] = std::max(p, cut[i - 1]);
    }
    std::vector<Worker> workers(nt);

    // ---- Estimate every term's document count from the first 1/32 of each chunk. A term
    // expected in more than kDenseDf documents is "dense": the scan sets its bits directly
    // instead of recording (term, document) pairs, which are most of the pairs.
    std::vector<uint32_t> sampled(nt, 0);
    parallel(nt, [&](int i) {
        Worker& w = workers[i];
        w.count.assign(Q, 0);
        w.last.assign(Q, UINT32_MAX);
        const char* p = cut[i];
        const char* const e = std::min(std::max(cut[i], cut[i + 1]), cut[i] + (cut[i + 1] - cut[i]) / 32);
        uint32_t doc = 0;
        while (p < e) {
            p = tokenize_line(p, dend, [&](uint64_t key, const char*) {
                const uint32_t t = dict.find(key);
                if (t != UINT32_MAX && w.last[t] != doc) { w.last[t] = doc; ++w.count[t]; }
            });
            ++doc;
        }
        sampled[i] = doc;
    });
    uint64_t sample_docs = 0;
    for (uint32_t s : sampled) sample_docs += s;
    std::vector<int32_t> dense_of(Q, -1);
    uint32_t dense_count = 0;
    for (uint32_t t = 0; t < Q; ++t) {
        uint64_t seen = 0;
        for (auto& w : workers) seen += w.count[t];
        if (sample_docs && seen * N > uint64_t(kDenseDf) * sample_docs) dense_of[t] = static_cast<int32_t>(dense_count++);
    }
    std::vector<uint64_t> dense_bits(static_cast<size_t>(dense_count) * W, 0);

    // ---- Pass 1: find the query terms in every document.
    parallel(nt, [&](int i) {
        Worker& w = workers[i];
        std::fill(w.count.begin(), w.count.end(), 0);
        std::fill(w.last.begin(), w.last.end(), UINT32_MAX);
        const char* p = cut[i];
        const char* const e = std::max(cut[i], cut[i + 1]);
        w.records.reserve(static_cast<size_t>(e - p) / 40 + 16);
        w.offs.reserve(static_cast<size_t>(e - p) / 40 + 16);
        while (p < e) {
            uint32_t doc = 0;
            for (const char* x = p; *x != '\t'; ++x) doc = doc * 10 + static_cast<uint32_t>(*x - '0');
            doc_off[doc] = static_cast<uint32_t>(p - dbase);
            p = tokenize_line(p, dend, [&](uint64_t key, const char* at) {
                const uint32_t t = dict.find(key);
                if (t == UINT32_MAX) return;
                if (dense_of[t] >= 0) {
                    if (w.last[t] != doc) {
                        w.last[t] = doc;
                        dense_bits[static_cast<size_t>(dense_of[t]) * W + (doc >> 6)] |= 1ull << (doc & 63);
                    }
                    return;
                }
                // A rare term keeps every occurrence with its offset, so a phrase check can start
                // right at it instead of re-reading the whole document.
                ++w.count[t];
                w.records.push_back((static_cast<uint64_t>(t) << 32) | doc);
                w.offs.push_back(static_cast<uint32_t>(at - dbase));
            });
        }
    });

    // ---- Posting lists of the other terms: every worker's records go to their term's slice,
    // already in document order.
    std::vector<uint32_t> start(Q + 1, 0);
    for (uint32_t t = 0; t < Q; ++t) {
        uint32_t run = start[t];
        for (int i = 0; i < nt; ++i) {
            const uint32_t c = workers[i].count[t];
            workers[i].count[t] = run;  // becomes this worker's write cursor for the term
            run += c;
        }
        start[t + 1] = run;
    }
    std::vector<uint32_t> postings(start[Q]), post_off(start[Q]);
    parallel(nt, [&](int i) {
        Worker& w = workers[i];
        for (size_t j = 0; j < w.records.size(); ++j) {
            const uint32_t at = w.count[w.records[j] >> 32]++;
            postings[at] = static_cast<uint32_t>(w.records[j]);
            post_off[at] = w.offs[j];
        }
        std::vector<uint64_t>().swap(w.records);
        std::vector<uint32_t>().swap(w.offs);
    });
    std::vector<uint32_t> df(Q);
    parallel(nt, [&](int i) {
        for (uint32_t t = Q * uint64_t(i) / nt; t < Q * uint64_t(i + 1) / nt; ++t) {
            if (dense_of[t] < 0) { df[t] = start[t + 1] - start[t]; continue; }
            const uint64_t* b = dense_bits.data() + static_cast<size_t>(dense_of[t]) * W;
            uint32_t c = 0;
            for (size_t k = 0; k < W; ++k) c += static_cast<uint32_t>(__builtin_popcountll(b[k]));
            df[t] = c;
        }
    });
    auto term_set = [&](uint32_t t) {
        TermSet s;
        s.list = postings.data() + start[t];
        s.n = df[t];
        s.bits = dense_of[t] >= 0 ? dense_bits.data() + static_cast<size_t>(dense_of[t]) * W : nullptr;
        return s;
    };

    // ---- Answer the queries in parallel.
    std::vector<std::string> results(queries.size());
    {
        std::atomic<size_t> next{0};
        std::vector<std::thread> th;
        for (int ti = 0; ti < nt; ++ti)
            th.emplace_back([&] {
                std::vector<uint64_t> A(W), B(W);
                char buf[256];
                constexpr size_t kBatch = 64;
                for (size_t q0; (q0 = next.fetch_add(kBatch)) < queries.size();)
                    for (size_t qi = q0; qi < std::min(queries.size(), q0 + kBatch); ++qi) {
                        const Query& q = queries[qi];
                        const uint32_t* ids = qid.data() + q.first;
                        uint64_t count = 0, sum = 0;
                        uint32_t first10[10];
                        int nfirst = 0;
                        auto take = [&](uint32_t d) {
                            ++count;
                            sum += d;
                            if (nfirst < 10) first10[nfirst++] = d;
                        };
                        if (!q.phrase) {
                            uint64_t* R = A.data();
                            uint64_t* T = B.data();
                            const TermSet s0 = term_set(ids[0]);
                            if (s0.bits) memcpy(R, s0.bits, W * 8);
                            else {
                                memset(R, 0, W * 8);
                                for (uint32_t j = 0; j < s0.n; ++j) R[s0.list[j] >> 6] |= 1ull << (s0.list[j] & 63);
                            }
                            for (uint32_t k = 1; k < q.count; ++k) {
                                const TermSet s = term_set(ids[k]);
                                const uint8_t op = qop[q.first + k];
                                if (op == 0) {  // AND
                                    if (s.bits) for (size_t w = 0; w < W; ++w) R[w] &= s.bits[w];
                                    else {
                                        memset(T, 0, W * 8);
                                        for (uint32_t j = 0; j < s.n; ++j) {
                                            const uint32_t d = s.list[j];
                                            T[d >> 6] |= R[d >> 6] & (1ull << (d & 63));
                                        }
                                        std::swap(R, T);
                                    }
                                } else if (op == 1) {  // OR
                                    if (s.bits) for (size_t w = 0; w < W; ++w) R[w] |= s.bits[w];
                                    else for (uint32_t j = 0; j < s.n; ++j) R[s.list[j] >> 6] |= 1ull << (s.list[j] & 63);
                                } else {  // NOT
                                    if (s.bits) for (size_t w = 0; w < W; ++w) R[w] &= ~s.bits[w];
                                    else for (uint32_t j = 0; j < s.n; ++j) R[s.list[j] >> 6] &= ~(1ull << (s.list[j] & 63));
                                }
                            }
                            for (size_t w = 0; w < W; ++w)
                                for (uint64_t bits = R[w]; bits; bits &= bits - 1) take(static_cast<uint32_t>(w * 64 + static_cast<unsigned>(__builtin_ctzll(bits))));
                        } else {
                            // Documents holding every term, driven by the rarest term, then an exact check of each.
                            uint32_t drv = ids[0];
                            for (uint32_t k = 1; k < q.count; ++k)
                                if (df[ids[k]] < df[drv]) drv = ids[k];
                            const TermSet sd = term_set(drv);
                            const uint32_t k = q.count;
                            uint64_t want[8];
                            for (uint32_t m = 0; m < k; ++m) want[m] = keys[ids[m]];
                            auto has_others = [&](uint32_t d) {
                                for (uint32_t m = 0; m < k; ++m) {
                                    if (ids[m] == drv) continue;
                                    const TermSet s = term_set(ids[m]);
                                    if (!(s.bits ? (s.bits[d >> 6] >> (d & 63)) & 1 : std::binary_search(s.list, s.list + s.n, d))) return false;
                                }
                                return true;
                            };
                            if (!sd.bits) {
                                // A rare driver lists every occurrence with its offset: check only the
                                // k tokens around each one, at every phrase position the driver takes.
                                const uint32_t* offs = post_off.data() + start[drv];
                                uint32_t cur = UINT32_MAX;
                                bool cur_ok = false, cur_found = false;
                                for (uint32_t j = 0; j < sd.n; ++j) {
                                    const uint32_t d = sd.list[j];
                                    if (d != cur) { cur = d; cur_found = false; cur_ok = has_others(d); }
                                    if (!cur_ok || cur_found) continue;
                                    const char* begin = dbase + doc_off[d];
                                    while (*begin != '\t') ++begin;
                                    ++begin;  // the text starts after the TAB
                                    for (uint32_t m = 0; m < k && !cur_found; ++m)
                                        if (ids[m] == drv) cur_found = window_match(begin, dbase + offs[j], want, k, m);
                                    if (cur_found) take(d);
                                }
                            } else {
                                // Every term is common (rare in the query mix): scan each candidate
                                // document, comparing token keys with a ring of the last k keys.
                                for (size_t w = 0; w < W; ++w)
                                    for (uint64_t bits = sd.bits[w]; bits; bits &= bits - 1) {
                                        const uint32_t d = static_cast<uint32_t>(w * 64 + static_cast<unsigned>(__builtin_ctzll(bits)));
                                        if (!has_others(d)) continue;
                                        uint64_t ring[8];
                                        uint32_t pos = 0, seen = 0;
                                        bool found = false;
                                        tokenize_line(dbase + doc_off[d], dend, [&](uint64_t key, const char*) {
                                            ring[pos] = key;
                                            pos = pos + 1 == k ? 0 : pos + 1;  // now the oldest of the last k
                                            if (found || ++seen < k || key != want[k - 1]) return;
                                            uint32_t m = 0;
                                            for (uint32_t r = pos; m < k && ring[r] == want[m]; r = r + 1 == k ? 0 : r + 1) ++m;
                                            found = m == k;
                                        });
                                        if (found) take(d);
                                    }
                            }
                        }
                        char* o = put_u64(buf, count);
                        *o++ = ' ';
                        o = put_u64(o, sum);
                        for (int k = 0; k < nfirst; ++k) { *o++ = ' '; o = put_u64(o, first10[k]); }
                        *o++ = '\n';
                        results[qi].assign(buf, static_cast<size_t>(o - buf));
                    }
            });
        for (auto& t : th) t.join();
    }

    std::string out;
    out.reserve(queries.size() * 32);
    for (auto& r : results) out += r;
    FILE* f = fopen((outdir + "/results.txt").c_str(), "wb");
    if (!f) return 1;
    fwrite(out.data(), 1, out.size(), f);
    fclose(f);
    return 0;
}
