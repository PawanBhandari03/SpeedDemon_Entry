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

// A zeroed array whose pages are committed only when they are written (calloc maps big blocks lazily).
template <class T>
struct LazyArray {
    T* p;
    explicit LazyArray(size_t n) : p(static_cast<T*>(calloc(n ? n : 1, sizeof(T)))) {}
    ~LazyArray() { free(p); }
    LazyArray(const LazyArray&) = delete;
    LazyArray& operator=(const LazyArray&) = delete;
    T& operator[](size_t i) { return p[i]; }
    T* data() { return p; }
    const T* data() const { return p; }
};

// Gives the whole pages of an array's consumed front back to the kernel, in steps of 1 MB.
// `upto` is the address up to which pages were already given back.
inline void release_pages(const void* base, size_t consumed_bytes, uintptr_t& upto) {
    const uintptr_t first = (reinterpret_cast<uintptr_t>(base) + 4095) & ~uintptr_t(4095);
    const uintptr_t hi = (reinterpret_cast<uintptr_t>(base) + consumed_bytes) & ~uintptr_t(4095);
    if (upto < first) upto = first;
    if (hi >= upto + (1u << 20)) {
        madvise(reinterpret_cast<void*>(upto), hi - upto, MADV_DONTNEED);
        upto = hi;
    }
}

// Runs f(i) on threads 0..n-1 and waits for all of them.
template <class F>
void parallel(int n, F&& f) {
    std::vector<std::thread> th;
    for (int i = 1; i < n; ++i) th.emplace_back(f, i);
    f(0);
    for (auto& t : th) t.join();
}

// Byte kinds: 1 for token bytes (a-z, A-Z, 0-9), 2 for LF, 0 for every other byte (a separator).
uint8_t g_kind[256];
uint8_t g_class[256];  // token bytes only: nonzero for a-z, A-Z, 0-9 (used when walking backwards)

void init_class() {
    for (int c = 0; c < 256; ++c) {
        const bool tok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9');
        g_kind[c] = tok ? 1 : (c == '\n' ? 2 : 0);
        g_class[c] = tok ? 1 : 0;
    }
}

inline uint64_t load8(const char* p) {
    uint64_t w;
    memcpy(&w, p, 8);
    return w;
}

// Key of the n token bytes at s (a-z, A-Z or 0-9; A-Z count as lower case, since OR 0x20 maps
// A-Z to a-z and leaves digits and a-z alone). Eight bytes are hashed at a time instead of one,
// and `limit` is the end of the buffer, so the last token never reads past it.
inline uint64_t token_key(const char* s, size_t n, const char* limit) {
    constexpr uint64_t kFold = 0x2020202020202020ull;
    uint64_t h = 0x9E3779B97F4A7C15ull ^ (n * 0xff51afd7ed558ccdull);
    while (n > 8) {
        h = (h ^ (load8(s) | kFold)) * 0xc4ceb9fe1a85ec53ull;
        h ^= h >> 32;
        s += 8;
        n -= 8;
    }
    const uint64_t mask = n == 8 ? ~0ull : (1ull << (n * 8)) - 1;
    uint64_t w = 0;
    if (s + 8 <= limit) w = load8(s);
    else memcpy(&w, s, n);
    h = (h ^ ((w | kFold) & mask)) * 0xff51afd7ed558ccdull;
    h ^= h >> 29;
    h *= 0xbf58476d1ce4e5b9ull;
    h ^= h >> 32;
    return h ? h : 1;
}

// Calls f(key, start) for every token of one document line starting at p (the id, a TAB, then
// the text); start points at the token's first byte. Returns a pointer just past the line's LF.
// Every line ends with an LF, so the loops need no end check.
template <class F>
inline const char* tokenize_line(const char* p, const char* end, F&& f) {
    while (*p != '\t') ++p;
    ++p;
    for (;;) {
        uint8_t k;
        while ((k = g_kind[static_cast<uint8_t>(*p)]) == 0) ++p;  // separators
        if (k == 2) return p + 1;
        const char* s = p;
        while (g_kind[static_cast<uint8_t>(*++p)] == 1) {}
        f(token_key(s, static_cast<size_t>(p - s), end), s);
    }
}

// True when the k keys in want are k consecutive tokens of the text [begin, ...) and the m-th
// of them starts at `at`: walks back m tokens from `at`, then hashes k tokens forward.
inline bool window_match(const char* begin, const char* at, const char* limit, const uint64_t* want, uint32_t k, uint32_t m) {
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
        const char* s = p;
        while (g_class[static_cast<uint8_t>(*p)]) ++p;
        if (token_key(s, static_cast<size_t>(p - s), limit) != want[i]) return false;
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
    std::vector<uint32_t> terms;    // rare terms, once per document, in document order
    std::vector<uint32_t> doc_end;  // per document of this worker: end of its run in terms
    uint32_t first_doc = 0;
    std::vector<uint64_t> occ;      // (term << 32) | doc for every occurrence of a phrase-driver term
    std::vector<uint32_t> offs;     // file offset of each of those occurrences
    std::vector<uint32_t> count;    // list entries per term seen by this worker
    std::vector<uint32_t> last;     // last document seen for each term, to record a document once
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
                while (p < e && *p != ' ' && *p != '\n' && *p != '"') ++p;
                const uint32_t len = static_cast<uint32_t>(p - s);
                if (q.phrase || want_term) {
                    qterm.push_back(token_key(s, len, e));
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
    std::vector<uint64_t> est(Q, 0);  // sampled document count per term
    uint32_t dense_count = 0;
    for (uint32_t t = 0; t < Q; ++t) {
        for (auto& w : workers) est[t] += w.count[t];
        if (sample_docs && est[t] * N > uint64_t(kDenseDf) * sample_docs) dense_of[t] = static_cast<int32_t>(dense_count++);
    }
    std::vector<uint64_t> dense_bits(static_cast<size_t>(dense_count) * W, 0);
    // The rarest-looking term of each phrase will drive its check, which needs every occurrence
    // and its offset. Every other rare term only needs its list of documents.
    std::vector<uint8_t> positional(Q, 0);
    for (const Query& q : queries) {
        if (!q.phrase) continue;
        uint32_t best = qid[q.first];
        for (uint32_t k = 1; k < q.count; ++k)
            if (est[qid[q.first + k]] < est[best]) best = qid[q.first + k];
        if (dense_of[best] < 0) positional[best] = 1;
    }

    // The dense terms are the common words: they make up most corpus tokens. They get their own
    // tiny table (a few KB, so it stays in the L1 cache) that is checked first, and a per-worker
    // "last document" array indexed by dense number, so a common token touches only L1-sized data.
    size_t hot_cap = 16;
    while (hot_cap < size_t(dense_count) * 2) hot_cap <<= 1;
    std::vector<uint64_t> hot_keys(hot_cap, 0);
    std::vector<uint32_t> hot_dense(hot_cap, 0);
    for (uint32_t t = 0; t < Q; ++t) {
        if (dense_of[t] < 0) continue;
        size_t s = keys[t] & (hot_cap - 1);
        while (hot_keys[s]) s = (s + 1) & (hot_cap - 1);
        hot_keys[s] = keys[t];
        hot_dense[s] = static_cast<uint32_t>(dense_of[t]);
    }
    const size_t hot_mask = hot_cap - 1;
    // ---- Pass 1: find the query terms in every document.
    parallel(nt, [&](int i) {
        Worker& w = workers[i];
        std::fill(w.count.begin(), w.count.end(), 0);
        std::fill(w.last.begin(), w.last.end(), UINT32_MAX);
        // The dense terms' bits of the 64 documents in progress: one word per term, which stays
        // in the L1 cache, written to the big bitsets once per 64 documents. Setting a bit straight
        // in the bitsets is a cache and TLB miss per (document, dense term), because the bitsets
        // of two terms are W words apart. A chunk starts at a multiple of 64, so this thread owns
        // every word it writes.
        std::vector<uint64_t> block(dense_count, 0);
        uint32_t block_no = 0;
        auto flush_block = [&] {
            for (uint32_t d = 0; d < dense_count; ++d)
                if (block[d]) {
                    dense_bits[static_cast<size_t>(d) * W + block_no] = block[d];
                    block[d] = 0;
                }
        };
        struct TermState { uint32_t last, count; };  // last document and entries of a rare term, in one cache line
        std::vector<TermState> st(Q, TermState{UINT32_MAX, 0});
        const char* p = cut[i];
        const char* const e = std::max(cut[i], cut[i + 1]);
        w.terms.reserve(static_cast<size_t>(e - p) / 64 + 16);
        bool first = true;
        while (p < e) {
            uint32_t doc = 0;
            for (const char* x = p; *x != '\t'; ++x) doc = doc * 10 + static_cast<uint32_t>(*x - '0');
            if (first) { w.first_doc = doc; block_no = doc >> 6; first = false; }
            if ((doc >> 6) != block_no) { flush_block(); block_no = doc >> 6; }
            doc_off[doc] = static_cast<uint32_t>(p - dbase);
            p = tokenize_line(p, dend, [&](uint64_t key, const char* at) {
                for (size_t s = key & hot_mask; hot_keys[s]; s = (s + 1) & hot_mask) {
                    if (hot_keys[s] != key) continue;
                    block[hot_dense[s]] |= 1ull << (doc & 63);
                    return;
                }
                const uint32_t t = dict.find(key);
                if (t == UINT32_MAX) return;
                if (positional[t]) {
                    ++st[t].count;
                    w.occ.push_back((static_cast<uint64_t>(t) << 32) | doc);
                    w.offs.push_back(static_cast<uint32_t>(at - dbase));
                    return;
                }
                if (st[t].last == doc) return;
                st[t].last = doc;
                ++st[t].count;
                w.terms.push_back(t);
            });
            w.doc_end.push_back(static_cast<uint32_t>(w.terms.size()));
        }
        if (!first) flush_block();
        for (uint32_t t = 0; t < Q; ++t) w.count[t] = st[t].count;
    });

    // ---- Posting lists of the rare terms. The phrase-driver terms come first, so their offsets
    // fit in one array of exactly their size. Each worker's entries go to their term's slice,
    // already in document order.
    std::vector<uint32_t> start(Q + 1, 0);
    uint32_t run = 0, positional_total = 0;
    for (int pass = 0; pass < 2; ++pass) {
        for (uint32_t t = 0; t < Q; ++t) {
            if (positional[t] != (pass == 0)) continue;
            start[t] = run;
            for (int i = 0; i < nt; ++i) {
                const uint32_t c = workers[i].count[t];
                workers[i].count[t] = run;  // becomes this worker's write cursor for the term
                run += c;
            }
        }
        if (pass == 0) positional_total = run;
    }
    LazyArray<uint32_t> postings(run), post_off(positional_total);
    std::vector<uint32_t> df(Q);
    parallel(nt, [&](int i) {
        Worker& w = workers[i];
        // The consumed front of each stream goes back to the kernel while the postings fill up, so
        // the streams and the postings are never both fully resident.
        uintptr_t up_terms = 0, up_occ = 0, up_offs = 0;
        for (size_t k = 0, r = 0; k < w.doc_end.size(); ++k) {
            for (; r < w.doc_end[k]; ++r) postings[w.count[w.terms[r]]++] = w.first_doc + static_cast<uint32_t>(k);
            if ((k & 4095) == 0) release_pages(w.terms.data(), r * sizeof(uint32_t), up_terms);
        }
        for (size_t j = 0; j < w.occ.size(); ++j) {
            const uint32_t at = w.count[w.occ[j] >> 32]++;
            postings[at] = static_cast<uint32_t>(w.occ[j]);
            post_off[at] = w.offs[j];
            if ((j & 32767) == 0) {
                release_pages(w.occ.data(), j * sizeof(uint64_t), up_occ);
                release_pages(w.offs.data(), j * sizeof(uint32_t), up_offs);
            }
        }
        std::vector<uint32_t>().swap(w.terms);
        std::vector<uint32_t>().swap(w.doc_end);
        std::vector<uint64_t>().swap(w.occ);
        std::vector<uint32_t>().swap(w.offs);
    });
    // After the scatter the last worker's cursor of a term is the end of the term's slice.
    parallel(nt, [&](int i) {
        for (uint32_t t = Q * uint64_t(i) / nt; t < Q * uint64_t(i + 1) / nt; ++t) {
            if (dense_of[t] < 0) { df[t] = workers[nt - 1].count[t] - start[t]; continue; }
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
                            // The driver is the phrase's rarest term that kept its occurrences; when
                            // every term is common there is none, and the rarest term's bitset drives.
                            uint32_t drv = UINT32_MAX;
                            for (uint32_t k = 0; k < q.count; ++k)
                                if (positional[ids[k]] && (drv == UINT32_MAX || df[ids[k]] < df[drv])) drv = ids[k];
                            if (drv == UINT32_MAX) {
                                drv = ids[0];
                                for (uint32_t k = 1; k < q.count; ++k)
                                    if (df[ids[k]] < df[drv]) drv = ids[k];
                            }
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
                                        if (ids[m] == drv) cur_found = window_match(begin, dbase + offs[j], dend, want, k, m);
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
